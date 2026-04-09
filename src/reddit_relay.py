"""External relay for Reddit data collected by the Devvit app.

The relay gives the Python discovery pipeline a stable HTTP surface it can call
directly, while the Devvit app mirrors normalized Reddit payloads into the same
contract. Query endpoints return cached payloads; on cache misses the Python
runtime falls back to its legacy/public Reddit path.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlencode

import aiohttp
from aiohttp import web

from src.runtime.paths import resolve_project_path

try:
    from reddit_bridge import normalize_reddit_item
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_bridge import normalize_reddit_item


logger = logging.getLogger(__name__)


def _resolve_env(value: str | None) -> str:
    raw = (value or "").strip()
    if raw.startswith("${") and raw.endswith("}"):
        return os.getenv(raw[2:-1], "")
    return raw


def _canonical_url(url: str) -> str:
    raw = url.strip()
    if raw.startswith("/"):
        raw = f"https://reddit.com{raw}"
    parsed = urlparse(raw)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower() or "reddit.com"
    if netloc == "www.reddit.com":
        netloc = "reddit.com"
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


class RedditRelayStore:
    """SQLite-backed cache for normalized Reddit query results."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _with_schema_retry(self, operation):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
            logger.warning("reddit relay store missing schema at %s; recreating tables", self.db_path)
            self._ensure_schema()
            return operation()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reddit_search_cache (
                    cache_key TEXT PRIMARY KEY,
                    subreddit TEXT NOT NULL,
                    query TEXT NOT NULL,
                    sort TEXT NOT NULL,
                    cursor TEXT NOT NULL,
                    next_cursor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    collected_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reddit_thread_cache (
                    url TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    collected_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reddit_comments_cache (
                    url TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    collected_at INTEGER NOT NULL
                );
                """
            )

    @staticmethod
    def _search_key(subreddit: str, query: str, sort: str, cursor: str) -> str:
        return json.dumps(
            {
                "subreddit": subreddit.strip().lower(),
                "query": query.strip().lower(),
                "sort": sort.strip().lower(),
                "cursor": cursor.strip(),
            },
            sort_keys=True,
        )

    def put_search(
        self,
        *,
        subreddit: str,
        query: str,
        sort: str,
        cursor: str,
        items: list[dict[str, Any]],
        next_cursor: str,
        collected_at: int | None = None,
    ) -> None:
        payload = json.dumps(items)
        collected_at = int(collected_at or time.time())
        cache_key = self._search_key(subreddit, query, sort, cursor)
        def _op() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reddit_search_cache (
                        cache_key, subreddit, query, sort, cursor, next_cursor, payload_json, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        next_cursor = excluded.next_cursor,
                        payload_json = excluded.payload_json,
                        collected_at = excluded.collected_at
                    """,
                    (cache_key, subreddit, query, sort, cursor, next_cursor, payload, collected_at),
                )

        self._with_schema_retry(_op)

    def get_search(
        self,
        *,
        subreddit: str,
        query: str,
        sort: str,
        cursor: str,
        limit: int,
    ) -> dict[str, Any] | None:
        cache_key = self._search_key(subreddit, query, sort, cursor)
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT payload_json, next_cursor, collected_at
                    FROM reddit_search_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()

        row = self._with_schema_retry(_op)
        if row is None:
            return None
        try:
            raw_payload = json.loads(row["payload_json"])
        except Exception:
            logger.warning(
                "reddit relay get_search failed to decode cached payload subreddit=%s query=%s sort=%s cursor=%s",
                subreddit,
                query,
                sort,
                cursor,
            )
            raw_payload = []
        items = _normalize_cached_search_items(raw_payload)
        return {
            "items": items[:limit],
            "next_cursor": row["next_cursor"],
            "collected_at": row["collected_at"],
        }

    def has_fresh_search(
        self,
        *,
        subreddit: str,
        query: str,
        sort: str = "relevance",
        cursor: str = "",
        max_age_seconds: int = 21600,
    ) -> bool:
        cache_key = self._search_key(subreddit, query, sort, cursor)
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT collected_at
                    FROM reddit_search_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()

        row = self._with_schema_retry(_op)
        if row is None:
            return False
        return int(time.time()) - int(row["collected_at"]) <= max_age_seconds

    def has_search(
        self,
        *,
        subreddit: str,
        query: str,
        sort: str = "relevance",
        cursor: str = "",
    ) -> bool:
        cache_key = self._search_key(subreddit, query, sort, cursor)
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT 1
                    FROM reddit_search_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()

        row = self._with_schema_retry(_op)
        return row is not None

    async def fetch_live_search(
        self,
        *,
        subreddit: str,
        query: str,
        limit: int = 2,
        sort: str = "relevance",
        cursor: str = "",
    ) -> dict[str, Any] | None:
        """Fetch live from Reddit API on cache miss."""
        import asyncio
        params = {
            "q": query,
            "sort": sort,
            "t": "year",
            "limit": str(min(limit, 25)),
        }
        if cursor:
            params["after"] = cursor
        if subreddit != "all":
            params["restrict_sr"] = "on"

        subreddit_path = subreddit if subreddit != "all" else "all"
        url = f"https://www.reddit.com/r/{subreddit_path}/search.json?{urlencode(params)}"

        headers = {
            "User-Agent": "AutoResearcher/1.0 (Python;aiohttp)",
            "Accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.warning("live reddit search failed status=%s url=%s", resp.status, url)
                        return None
                    data = await resp.json()
        except asyncio.TimeoutError:
            logger.warning("live reddit search timeout url=%s", url)
            return None
        except Exception as exc:
            logger.warning("live reddit search error url=%s error=%s", url, exc)
            return None

        children = data.get("data", {}).get("children", [])
        items = []
        for child in children:
            post = child.get("data", {})
            if not post.get("id"):
                continue
            items.append({
                "id": post.get("id", ""),
                "kind": "post",
                "subreddit": post.get("subreddit", ""),
                "title": post.get("title", ""),
                "body": post.get("selftext", "") or post.get("body", ""),
                "author": post.get("author", ""),
                "permalink": post.get("permalink", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "created_utc": post.get("created_utc", 0),
                "source_type": "reddit",
                "post_id": post.get("id", ""),
                "parent_id": "",
            })

        next_cursor = data.get("data", {}).get("after", "") or ""
        collected_at = int(time.time())

        if items:
            self.put_search(
                subreddit=subreddit,
                query=query,
                sort=sort,
                cursor=cursor,
                next_cursor=next_cursor,
                items=items,
                collected_at=collected_at,
            )

        return {
            "items": items,
            "next_cursor": next_cursor,
            "collected_at": collected_at,
        }

    async def fetch_live_thread(
        self,
        *,
        url: str,
        comment_limit: int = 50,
        depth: int = 4,
    ) -> dict[str, Any] | None:
        """Fetch a Reddit thread directly and hydrate thread/comment caches on miss."""
        canonical_url = _canonical_url(url)
        json_url = f"{canonical_url}.json?limit={min(max(comment_limit, 1), 50)}&depth={max(depth, 1)}&raw_json=1"
        headers = {
            "User-Agent": "AutoResearcher/1.0 (Python;aiohttp)",
            "Accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(json_url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        logger.warning("live reddit thread failed status=%s url=%s", resp.status, canonical_url)
                        return None
                    data = await resp.json()
        except asyncio.TimeoutError:
            logger.warning("live reddit thread timeout url=%s", canonical_url)
            return None
        except Exception as exc:
            logger.warning("live reddit thread error url=%s error=%s", canonical_url, exc)
            return None

        if not isinstance(data, list) or len(data) < 2:
            logger.warning("live reddit thread bad payload url=%s", canonical_url)
            return None

        post_children = data[0].get("data", {}).get("children", [])
        if not post_children:
            return None
        post_data = post_children[0].get("data", {})
        post_id = str(post_data.get("id", "") or "")
        if not post_id:
            return None

        permalink = str(post_data.get("permalink", "") or "")
        canonical_permalink = _canonical_url(permalink or canonical_url)
        post_item = {
            "id": post_id,
            "kind": "post",
            "subreddit": str(post_data.get("subreddit", "") or ""),
            "title": str(post_data.get("title", "") or ""),
            "body": str(post_data.get("selftext", "") or post_data.get("body", "") or ""),
            "author": str(post_data.get("author", "") or ""),
            "permalink": canonical_permalink,
            "score": int(post_data.get("score", 0) or 0),
            "num_comments": int(post_data.get("num_comments", 0) or 0),
            "created_utc": int(post_data.get("created_utc", 0) or 0),
            "source_type": "reddit",
            "post_id": post_id,
            "parent_id": "",
        }

        comment_items: list[dict[str, Any]] = []

        def _walk_comments(children: list[dict[str, Any]]) -> None:
            for child in children:
                if len(comment_items) >= min(max(comment_limit, 1), 50):
                    return
                if child.get("kind") != "t1":
                    continue
                comment = child.get("data", {}) or {}
                comment_id = str(comment.get("id", "") or "")
                body = str(comment.get("body", "") or "")
                if not comment_id or not body.strip():
                    replies = comment.get("replies")
                    if isinstance(replies, dict):
                        _walk_comments(replies.get("data", {}).get("children", []) or [])
                    continue
                comment_items.append(
                    {
                        "id": comment_id,
                        "kind": "comment",
                        "subreddit": str(comment.get("subreddit", post_item["subreddit"]) or post_item["subreddit"]),
                        "title": "",
                        "body": body,
                        "author": str(comment.get("author", "") or ""),
                        "permalink": canonical_permalink,
                        "score": int(comment.get("score", 0) or 0),
                        "num_comments": 0,
                        "created_utc": int(comment.get("created_utc", 0) or 0),
                        "source_type": "reddit",
                        "post_id": post_id,
                        "parent_id": str(comment.get("parent_id", f"t3_{post_id}") or f"t3_{post_id}"),
                    }
                )
                replies = comment.get("replies")
                if isinstance(replies, dict):
                    _walk_comments(replies.get("data", {}).get("children", []) or [])

        _walk_comments(data[1].get("data", {}).get("children", []) or [])

        collected_at = int(time.time())
        self.put_thread(url=canonical_permalink, post=post_item, comments=comment_items, collected_at=collected_at)
        self.put_comments(url=canonical_permalink, items=comment_items, collected_at=collected_at)
        return self.get_thread(url=canonical_permalink)

    def put_thread(
        self,
        *,
        url: str,
        post: dict[str, Any] | None,
        comments: list[dict[str, Any]],
        collected_at: int | None = None,
    ) -> None:
        collected_at = int(collected_at or time.time())
        payload = json.dumps({"post": post, "comments": comments})
        def _op() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reddit_thread_cache (url, payload_json, collected_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        collected_at = excluded.collected_at
                    """,
                    (_canonical_url(url), payload, collected_at),
                )

        self._with_schema_retry(_op)

    def get_thread(self, *, url: str) -> dict[str, Any] | None:
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    "SELECT payload_json, collected_at FROM reddit_thread_cache WHERE url = ?",
                    (_canonical_url(url),),
                ).fetchone()

        row = self._with_schema_retry(_op)
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["collected_at"] = row["collected_at"]
        return payload

    def has_thread(self, *, url: str) -> bool:
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    "SELECT 1 FROM reddit_thread_cache WHERE url = ?",
                    (_canonical_url(url),),
                ).fetchone()

        row = self._with_schema_retry(_op)
        return row is not None

    def put_comments(
        self,
        *,
        url: str,
        items: list[dict[str, Any]],
        collected_at: int | None = None,
    ) -> None:
        collected_at = int(collected_at or time.time())
        def _op() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reddit_comments_cache (url, payload_json, collected_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        collected_at = excluded.collected_at
                    """,
                    (_canonical_url(url), json.dumps(items), collected_at),
                )

        self._with_schema_retry(_op)

    def get_comments(self, *, url: str, limit: int) -> dict[str, Any] | None:
        def _op():
            with self._connect() as conn:
                return conn.execute(
                    "SELECT payload_json, collected_at FROM reddit_comments_cache WHERE url = ?",
                    (_canonical_url(url),),
                ).fetchone()

        row = self._with_schema_retry(_op)
        if row is None:
            return None
        items = json.loads(row["payload_json"])
        return {"items": items[:limit], "collected_at": row["collected_at"]}


STORE_KEY = web.AppKey("store", RedditRelayStore)
AUTH_TOKEN_KEY = web.AppKey("auth_token", str)
ALLOW_NO_AUTH_KEY = web.AppKey("allow_no_auth", bool)


def _json_error(status: int, code: str, message: str, **extra: str) -> web.Response:
    payload = {"ok": False, "error_code": code, "error": message}
    payload.update(extra)
    return web.json_response(payload, status=status)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    if request.path == "/api/health":
        return await handler(request)

    token = request.app[AUTH_TOKEN_KEY]
    allow_no_auth = request.app[ALLOW_NO_AUTH_KEY]

    if not token:
        if allow_no_auth:
            logger.warning("Relay running without auth (allow_no_auth=True)")
            return await handler(request)
        # No token configured and allow_no_auth is False — reject all requests
        return _json_error(401, "auth_failed", "no token configured and allow_no_auth is false")

    authorization = request.headers.get("Authorization", "")
    expected = f"Bearer {token}"
    if not hmac.compare_digest(authorization, expected):
        return _json_error(401, "auth_failed", "missing or invalid bearer token")
    return await handler(request)


def _normalize_items(items: list[dict[str, Any]], *, expected_kind: str | None = None) -> list[dict[str, Any]]:
    return [normalize_reddit_item(item, expected_kind=expected_kind) for item in items]


def _repair_reddit_item(item: dict[str, Any], *, expected_kind: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("cached reddit item is not a dict")
    repaired = dict(item)
    repaired.setdefault("kind", expected_kind)
    repaired.setdefault("source_type", "reddit")
    repaired.setdefault("subreddit", "")
    repaired.setdefault("title", "")
    repaired.setdefault("body", "")
    repaired.setdefault("author", "")
    repaired.setdefault("permalink", "")
    repaired.setdefault("score", 0)
    repaired.setdefault("num_comments", 0)
    repaired.setdefault("created_utc", 0)
    repaired.setdefault("parent_id", "")
    if expected_kind == "post":
        repaired.setdefault("post_id", repaired.get("id", ""))
    else:
        repaired.setdefault("post_id", "")
    return normalize_reddit_item(repaired, expected_kind=expected_kind)


def _normalize_cached_thread_payload(payload: dict[str, Any]) -> dict[str, Any]:
    post = payload.get("post")
    comments = payload.get("comments", [])
    normalized_post = _repair_reddit_item(post, expected_kind="post") if isinstance(post, dict) and post else None
    normalized_comments: list[dict[str, Any]] = []
    for comment in comments if isinstance(comments, list) else []:
        try:
            normalized_comments.append(_repair_reddit_item(comment, expected_kind="comment"))
        except Exception:
            continue
    return {"post": normalized_post, "comments": normalized_comments}


def _normalize_cached_search_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        try:
            normalized_items.append(_repair_reddit_item(item, expected_kind="post"))
        except Exception:
            continue
    return normalized_items


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "service": "autoresearch-reddit-relay",
            "auth_enabled": bool(request.app[AUTH_TOKEN_KEY]),
            "allow_no_auth": request.app[ALLOW_NO_AUTH_KEY],
        }
    )


async def ingest_search(request: web.Request) -> web.Response:
    body = await request.json()
    items = _normalize_items(body.get("items", []), expected_kind="post")
    store = request.app[STORE_KEY]
    store.put_search(
        subreddit=str(body.get("subreddit", "all") or "all"),
        query=str(body.get("query", "") or ""),
        sort=str(body.get("sort", "relevance") or "relevance"),
        cursor=str(body.get("cursor", "") or ""),
        items=items,
        next_cursor=str(body.get("next_cursor", "") or ""),
        collected_at=int(body.get("collected_at") or time.time()),
    )
    return web.json_response({"ok": True, "stored": len(items)})


async def ingest_thread(request: web.Request) -> web.Response:
    body = await request.json()
    post = body.get("post")
    comments = body.get("comments", [])
    normalized_post = normalize_reddit_item(post, expected_kind="post") if isinstance(post, dict) and post else None
    normalized_comments = _normalize_items(comments, expected_kind="comment")
    store = request.app[STORE_KEY]
    store.put_thread(
        url=str(body.get("url", "") or ""),
        post=normalized_post,
        comments=normalized_comments,
        collected_at=int(body.get("collected_at") or time.time()),
    )
    return web.json_response({"ok": True, "stored_comments": len(normalized_comments)})


async def ingest_comments(request: web.Request) -> web.Response:
    body = await request.json()
    items = _normalize_items(body.get("items", []), expected_kind="comment")
    store = request.app[STORE_KEY]
    store.put_comments(
        url=str(body.get("url", "") or ""),
        items=items,
        collected_at=int(body.get("collected_at") or time.time()),
    )
    return web.json_response({"ok": True, "stored": len(items)})


async def cached_search(request: web.Request) -> web.Response:
    body = await request.json()
    limit = min(max(int(body.get("limit", 2) or 2), 1), 25)
    store = request.app[STORE_KEY]
    subreddit = str(body.get("subreddit", "all") or "all")
    query = str(body.get("query", "") or "")
    sort = str(body.get("sort", "relevance") or "relevance")
    cursor = str(body.get("cursor", "") or "")
    try:
        result = store.get_search(
            subreddit=subreddit,
            query=query,
            sort=sort,
            cursor=cursor,
            limit=limit,
        )
    except Exception as exc:
        logger.warning(
            "reddit relay cached_search failed subreddit=%s query=%s sort=%s cursor=%s error=%s",
            subreddit,
            query,
            sort,
            cursor,
            exc,
        )
        return _json_error(500, "cached_search_bad_shape", "cached search payload could not be normalized", items=[], next_cursor="")
    if result is None:
        # Try live search on cache miss
        result = await store.fetch_live_search(
            subreddit=subreddit,
            query=query,
            limit=limit,
            sort=sort,
            cursor=cursor,
        )
        if result is None:
            # Surface this as a cache miss so clients can warm/retry without poisoning the bridge circuit.
            return _json_error(404, "no_cached_result", "no cached search result", items=[], next_cursor="")

    return web.json_response(
        {
            "ok": True,
            "items": result["items"],
            "next_cursor": result["next_cursor"],
            "collected_at": result["collected_at"],
        }
    )


async def cached_thread(request: web.Request) -> web.Response:
    body = await request.json()
    store = request.app[STORE_KEY]
    url = str(body.get("url", "") or "")
    comment_limit = min(max(int(body.get("comment_limit", 50) or 50), 1), 50)
    depth = max(int(body.get("depth", 4) or 4), 1)
    result = store.get_thread(url=url)
    if result is None:
        result = await store.fetch_live_thread(url=url, comment_limit=comment_limit, depth=depth)
    if result is None:
        return _json_error(404, "no_cached_result", "no cached thread result", post=None, comments=[])
    try:
        normalized = _normalize_cached_thread_payload(result)
    except Exception as exc:
        logger.warning("reddit relay cached_thread normalization failed url=%s error=%s", url, exc)
        return _json_error(500, "cached_thread_bad_shape", "cached thread payload could not be normalized", post=None, comments=[])
    return web.json_response(
        {
            "ok": True,
            "post": normalized.get("post"),
            "comments": normalized.get("comments", []),
            "collected_at": result["collected_at"],
        }
    )


async def cached_comments(request: web.Request) -> web.Response:
    body = await request.json()
    limit = min(max(int(body.get("limit", 8) or 8), 1), 50)
    store = request.app[STORE_KEY]
    url = str(body.get("url", "") or "")
    result = store.get_comments(url=url, limit=limit)
    if result is None:
        await store.fetch_live_thread(url=url, comment_limit=limit, depth=max(int(body.get("depth", 4) or 4), 1))
        result = store.get_comments(url=url, limit=limit)
    if result is None:
        return _json_error(404, "no_cached_result", "no cached comments result", items=[])
    return web.json_response({"ok": True, "items": result["items"], "collected_at": result["collected_at"]})


def build_relay_app(config: dict[str, Any] | None = None) -> web.Application:
    config = config or {}
    relay_config = config.get("reddit_relay", config)
    db_path = str(resolve_project_path(relay_config.get("cache_db_path"), default="data/reddit_relay_cache.db"))
    auth_token = _resolve_env(relay_config.get("auth_token", ""))
    allow_no_auth = bool(relay_config.get("allow_no_auth", False))

    app = web.Application(middlewares=[auth_middleware])
    app[STORE_KEY] = RedditRelayStore(db_path)
    app[AUTH_TOKEN_KEY] = auth_token
    app[ALLOW_NO_AUTH_KEY] = allow_no_auth

    app.router.add_get("/api/health", health)
    app.router.add_post("/api/reddit/ingest/search-posts", ingest_search)
    app.router.add_post("/api/reddit/ingest/post-thread", ingest_thread)
    app.router.add_post("/api/reddit/ingest/comments", ingest_comments)
    app.router.add_post("/api/reddit/search-posts", cached_search)
    app.router.add_post("/api/reddit/post-thread", cached_thread)
    app.router.add_post("/api/reddit/comments", cached_comments)
    return app


async def start_relay_server(
    config: dict[str, Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> tuple[web.AppRunner, web.BaseSite]:
    config = config or {}
    relay_config = config.get("reddit_relay", config)
    app = build_relay_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    listen_host = host or relay_config.get("listen_host", "127.0.0.1")
    listen_port = int(port or relay_config.get("listen_port", 8787))
    site = web.TCPSite(runner, listen_host, listen_port)
    await site.start()
    logger.info("reddit relay listening on http://%s:%s", listen_host, listen_port)
    return runner, site


async def run_relay_server(
    config: dict[str, Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    runner, _site = await start_relay_server(config, host=host, port=port)
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    finally:
        await runner.cleanup()
