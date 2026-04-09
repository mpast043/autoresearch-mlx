"""Reddit bridge, fallback, and cache behavior for the research toolkit."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

try:
    from src.reddit_bridge import BridgeError, RedditBridgeClient
except Exception:  # pragma: no cover - supports package and direct module usage
    from reddit_bridge import BridgeError, RedditBridgeClient

try:
    from src.search_models import SearchDocument
except Exception:  # pragma: no cover - supports package and direct module usage
    from search_models import SearchDocument


class RedditTransport:
    """Encapsulates Reddit bridge, fallback, cache, and warming behavior."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        reddit_bridge: RedditBridgeClient,
        reddit_mode: str,
        node_bin: str | None,
        readonly_script: Path,
        user_agent: str,
        run_json_command: Callable[..., Any],
        compact_text: Callable[[str, int], str],
        normalize_query: Callable[[str], str],
        request_get: Callable[..., Any],
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.reddit_bridge = reddit_bridge
        self.reddit_mode = reddit_mode
        self.node_bin = node_bin
        self.readonly_script = readonly_script
        self.user_agent = user_agent
        self._run_json_command = run_json_command
        self._compact_text = compact_text
        self._normalize_query = normalize_query
        self._request_get = request_get
        self._logger = logger
        bridge_config = config.get("reddit_bridge", {}) if isinstance(config.get("reddit_bridge", {}), dict) else {}
        self.bridge_seed_on_miss = bool(bridge_config.get("seed_on_miss", True))
        self.search_cache: dict[tuple[str, str, int, str, str], list[SearchDocument]] = {}
        self.thread_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl: float = 1800.0  # 30 minutes
        self.bridge_seed_attempted_pairs: set[tuple[str, str]] = set()
        self.metrics = {
            "reddit_mode": self.reddit_mode,
            "reddit_bridge_hits": 0,
            "reddit_bridge_misses": 0,
            "reddit_fallback_queries": 0,
            "reddit_public_direct_queries": 0,
            "reddit_validation_seed_runs": 0,
            "reddit_validation_seeded_pairs": 0,
            "reddit_validation_seed_searches": 0,
            "reddit_validation_seed_uncovered_before": 0,
            "reddit_validation_seed_uncovered_after": 0,
        }
        self.validation_seeded_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # TTL-aware cache helpers
    # ------------------------------------------------------------------

    def _cache_key_str(self, key: str) -> str:
        """Return a string representation usable as a timestamp-dict key."""
        if isinstance(key, str):
            return key
        return str(key)

    def _evict_expired_caches(self) -> None:
        """Remove entries from both caches whose TTL has expired."""
        now = time.time()
        expired: list[str] = []
        for k, ts in self._cache_timestamps.items():
            if now - ts > self._cache_ttl:
                expired.append(k)
        for k in expired:
            self._cache_timestamps.pop(k, None)
            self.search_cache.pop(k, None)  # type: ignore[arg-type]
            self.thread_cache.pop(k, None)

    def _cache_get(self, cache: dict, key: str) -> dict[str, Any] | None:
        """Get a value from *cache*, returning ``None`` if expired."""
        self._evict_expired_caches()
        key_str = self._cache_key_str(key)
        if key_str not in self._cache_timestamps:
            cache.pop(key, None)
            return None
        if time.time() - self._cache_timestamps[key_str] > self._cache_ttl:
            self._cache_timestamps.pop(key_str, None)
            cache.pop(key, None)
            return None
        return cache.get(key)

    def _cache_set(self, cache: dict, key: str, value: dict[str, Any]) -> None:
        """Store *value* in *cache* under *key* and record the timestamp."""
        key_str = self._cache_key_str(key)
        self._cache_timestamps[key_str] = time.time()
        cache[key] = value

    def _reddit_search_time_filter(self) -> str:
        """Reddit JSON search ``t`` param: hour|day|week|month|year|all."""
        raw = str(
            (self.config.get("discovery") or {}).get("reddit", {}).get("search_time_filter", "year") or "year"
        ).lower()
        allowed = {"hour", "day", "week", "month", "year", "all"}
        return raw if raw in allowed else "year"

    async def close(self) -> None:
        await self.reddit_bridge.close()

    def get_runtime_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)

    async def _maybe_seed_bridge_pair(self, *, subreddit: str, query: str) -> bool:
        if not self.bridge_seed_on_miss or not self.reddit_bridge.enabled:
            return False
        normalized_subreddit = str(subreddit).strip()
        normalized_query = self._normalize_query(query)
        if not normalized_subreddit or not normalized_query:
            return False
        cache_key = (normalized_subreddit, normalized_query)
        if cache_key in self.bridge_seed_attempted_pairs:
            return False
        self.bridge_seed_attempted_pairs.add(cache_key)
        summary = await self.warm_validation_queries(
            subreddits=[normalized_subreddit],
            queries=[normalized_query],
        )
        seeded = int(summary.get("seeded_searches", 0) or 0)
        uncovered_after = int(summary.get("uncovered_after", 0) or 0)
        if seeded > 0 or uncovered_after == 0:
            self._logger.info(
                "reddit_search warmed relay cache subreddit=%s query=%r seeded_searches=%s uncovered_after=%s",
                normalized_subreddit,
                normalized_query,
                seeded,
                uncovered_after,
            )
            return True
        return False

    async def warm_validation_queries(
        self,
        *,
        subreddits: list[str],
        queries: list[str],
    ) -> dict[str, int]:
        if not self.reddit_bridge.enabled:
            return {
                "seed_runs": 0,
                "seeded_pairs": 0,
                "seeded_searches": 0,
                "uncovered_before": 0,
                "uncovered_after": 0,
            }

        normalized_subreddits = [str(subreddit).strip() for subreddit in subreddits if str(subreddit).strip()]
        normalized_queries: list[str] = []
        for query in queries:
            normalized = self._normalize_query(query)
            if normalized and normalized not in normalized_queries:
                normalized_queries.append(normalized)
        if not normalized_subreddits or not normalized_queries:
            return {
                "seed_runs": 0,
                "seeded_pairs": 0,
                "seeded_searches": 0,
                "uncovered_before": 0,
                "uncovered_after": 0,
            }

        pairs = [
            (subreddit, query)
            for subreddit in normalized_subreddits
            for query in normalized_queries
            if (subreddit, query) not in self.validation_seeded_pairs
        ]
        if not pairs:
            return {
                "seed_runs": 0,
                "seeded_pairs": 0,
                "seeded_searches": 0,
                "uncovered_before": 0,
                "uncovered_after": 0,
            }

        try:
            try:
                from reddit_seed import RedditSeeder
            except Exception:  # pragma: no cover - supports package usage
                from src.reddit_seed import RedditSeeder

            seeder = RedditSeeder(self.config)
            before = seeder.coverage_report(subreddits=normalized_subreddits, queries=normalized_queries)
            summary = await seeder.seed(subreddits=normalized_subreddits, queries=normalized_queries)
        except Exception as exc:
            self._logger.warning("reddit validation query warm failed (%s)", exc)
            return {
                "seed_runs": 0,
                "seeded_pairs": 0,
                "seeded_searches": 0,
                "uncovered_before": 0,
                "uncovered_after": 0,
            }

        self.validation_seeded_pairs.update(pairs)
        self.metrics["reddit_validation_seed_runs"] += 1
        self.metrics["reddit_validation_seeded_pairs"] += len(pairs)
        self.metrics["reddit_validation_seed_searches"] += int(summary.cached_searches)
        self.metrics["reddit_validation_seed_uncovered_before"] += int(before.uncovered_pairs)
        self.metrics["reddit_validation_seed_uncovered_after"] += int(summary.uncovered_pairs)
        self._logger.info(
            "reddit validation queries warmed pairs=%s searches=%s uncovered_before=%s uncovered_after=%s",
            len(pairs),
            summary.cached_searches,
            before.uncovered_pairs,
            summary.uncovered_pairs,
        )
        return {
            "seed_runs": 1,
            "seeded_pairs": len(pairs),
            "seeded_searches": int(summary.cached_searches),
            "uncovered_before": int(before.uncovered_pairs),
            "uncovered_after": int(summary.uncovered_pairs),
        }

    async def search(
        self,
        *,
        subreddit: str,
        query: str,
        limit: int = 2,
        sort: str = "relevance",
    ) -> list[SearchDocument]:
        time_filter = self._reddit_search_time_filter()
        cache_key = (subreddit, query, limit, sort, time_filter)
        cached = self._cache_get(self.search_cache, cache_key)
        if cached is not None:
            return list(cached)

        if self.reddit_mode in {"bridge_with_fallback", "bridge_only"}:
            if not self.reddit_bridge.enabled:
                self.metrics["reddit_bridge_misses"] += 1
                self._logger.info(
                    "reddit_search bridge unavailable subreddit=%s query=%r mode=%s",
                    subreddit,
                    query,
                    self.reddit_mode,
                )
                if self.reddit_mode == "bridge_only":
                    self._cache_set(self.search_cache, cache_key, [])
                    return []
                # bridge_with_fallback should continue to legacy/public paths when the bridge
                # is not configured in the current environment (e.g. Codespaces).
            try:
                bridge_posts, _next_cursor = await self.reddit_bridge.search_posts(
                    subreddit=subreddit,
                    query=query,
                    limit=limit,
                    sort=sort,
                )
                docs = [
                    SearchDocument(
                        title=post.get("title", ""),
                        url=post.get("permalink", ""),
                        snippet=self._compact_text(post.get("body", ""), 500),
                        source=f"reddit/{post.get('subreddit', subreddit)}",
                    )
                    for post in bridge_posts
                    if post.get("permalink")
                ]
                self.metrics["reddit_bridge_hits"] += 1
                self._logger.info(
                    "reddit_search using bridge path subreddit=%s query=%r limit=%s results=%s mode=%s",
                    subreddit,
                    query,
                    limit,
                    len(docs),
                    self.reddit_mode,
                )
                self._cache_set(self.search_cache, cache_key, list(docs))
                return docs
            except BridgeError as exc:
                self.metrics["reddit_bridge_misses"] += 1
                if self.reddit_mode == "bridge_only":
                    self._logger.info(
                        "reddit_search bridge miss code=%s subreddit=%s query=%r mode=bridge_only; returning empty",
                        exc.code,
                        subreddit,
                        query,
                    )
                    self._cache_set(self.search_cache, cache_key, [])
                    return []
                seeded = False
                if exc.code == "no_cached_result":
                    seeded = await self._maybe_seed_bridge_pair(subreddit=subreddit, query=query)
                    if seeded:
                        try:
                            bridge_posts, _next_cursor = await self.reddit_bridge.search_posts(
                                subreddit=subreddit,
                                query=query,
                                limit=limit,
                                sort=sort,
                            )
                            docs = [
                                SearchDocument(
                                    title=post.get("title", ""),
                                    url=post.get("permalink", ""),
                                    snippet=self._compact_text(post.get("body", ""), 500),
                                    source=f"reddit/{post.get('subreddit', subreddit)}",
                                )
                                for post in bridge_posts
                                if post.get("permalink")
                            ]
                            self.metrics["reddit_bridge_hits"] += 1
                            self._logger.info(
                                "reddit_search recovered via bridge seed subreddit=%s query=%r limit=%s results=%s",
                                subreddit,
                                query,
                                limit,
                                len(docs),
                            )
                            self._cache_set(self.search_cache, cache_key, list(docs))
                            return docs
                        except BridgeError as retry_exc:
                            exc = retry_exc
                log_fn = self._logger.info if exc.code == "no_cached_result" else self._logger.warning
                log_fn(
                    "reddit_search bridge failure code=%s subreddit=%s query=%r; falling back to legacy/public path (%s)",
                    exc.code,
                    subreddit,
                    query,
                    exc.message,
                )
                self.metrics["reddit_fallback_queries"] += 1

        if self.reddit_mode == "public_direct":
            self.metrics["reddit_public_direct_queries"] += 1

        if self.node_bin and self.readonly_script.exists():
            payload = await self._run_json_command(
                [
                    self.node_bin,
                    str(self.readonly_script),
                    "search",
                    subreddit or "all",
                    query,
                    "--limit",
                    str(limit),
                ],
                timeout=20,
            )
            if payload and payload.get("ok"):
                self._logger.info(
                    "reddit_search using fallback path=readonly_script subreddit=%s query=%r limit=%s results=%s",
                    subreddit,
                    query,
                    limit,
                    len(payload.get("data", {}).get("posts", [])),
                )
                docs = [
                    SearchDocument(
                        title=post.get("title", ""),
                        url=post.get("permalink", ""),
                        snippet=post.get("selftext_snippet", "") or "",
                        source=f"reddit/{post.get('subreddit', subreddit)}",
                    )
                    for post in payload.get("data", {}).get("posts", [])
                    if post.get("permalink")
                ]
                self._cache_set(self.search_cache, cache_key, list(docs))
                return docs

        def _request() -> list[SearchDocument]:
            response = self._request_get(
                f"https://www.reddit.com/r/{subreddit}/search.json",
                params={
                    "q": query,
                    "restrict_sr": "on",
                    "sort": sort,
                    "t": time_filter,
                    "limit": limit,
                },
                timeout=15,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            payload = response.json()
            docs: list[SearchDocument] = []
            for child in payload.get("data", {}).get("children", []):
                data = child.get("data", {})
                permalink = data.get("permalink", "")
                docs.append(
                    SearchDocument(
                        title=data.get("title", ""),
                        url=f"https://reddit.com{permalink}" if permalink else "",
                        snippet=self._compact_text(data.get("selftext", ""), 500),
                        source=f"reddit/{subreddit}",
                    )
                )
            return [doc for doc in docs if doc.url]

        try:
            docs = await asyncio.to_thread(_request)
            self._logger.info(
                "reddit_search using fallback path=public_json subreddit=%s query=%r limit=%s results=%s",
                subreddit,
                query,
                limit,
                len(docs),
            )
            self._cache_set(self.search_cache, cache_key, list(docs))
            return docs
        except Exception:
            return []

    async def thread_context(self, url: str) -> dict[str, Any]:
        cached = self._cache_get(self.thread_cache, url)
        if cached is not None:
            return dict(cached)

        if self.reddit_mode in {"bridge_with_fallback", "bridge_only"}:
            if not self.reddit_bridge.enabled:
                self.metrics["reddit_bridge_misses"] += 1
                self._logger.info("reddit_thread_context bridge unavailable url=%s mode=%s", url, self.reddit_mode)
                if self.reddit_mode == "bridge_only":
                    self._cache_set(self.thread_cache, url, {})
                    return {}
                # bridge_with_fallback should continue to legacy/public paths when the bridge
                # is not configured in the current environment (e.g. Codespaces).
            try:
                # Fetch 50 comments sorted by rising to get corroboration
                payload = await self.reddit_bridge.get_post_thread(url=url, comment_limit=50, depth=4)
                comments = [item.get("body", "") for item in payload.get("comments", []) if item.get("body")]
                # Get comment metadata for corroboration scoring
                comment_metadata = [
                    {
                        "body": item.get("body", "")[:200],
                        "author": item.get("author", ""),
                        "score": item.get("score", 0),
                    }
                    for item in payload.get("comments", [])[:50]
                    if item.get("body")
                ]
                result = {
                    "title": payload.get("post", {}).get("title", ""),
                    "text": self._compact_text(
                        " ".join(
                            [
                                payload.get("post", {}).get("title", ""),
                                payload.get("post", {}).get("body", ""),
                                *comments,
                            ]
                        ),
                        2500,
                    ),
                    "description": self._compact_text(payload.get("post", {}).get("body", ""), 900),
                    "comments": comments[:50],
                    "comment_metadata": comment_metadata,
                }
                self.metrics["reddit_bridge_hits"] += 1
                self._cache_set(self.thread_cache, url, result)
                return result
            except BridgeError as exc:
                self.metrics["reddit_bridge_misses"] += 1
                if self.reddit_mode == "bridge_only":
                    self._logger.info(
                        "reddit_thread_context bridge miss code=%s url=%s mode=bridge_only; returning empty",
                        exc.code,
                        url,
                    )
                    self._cache_set(self.thread_cache, url, {})
                    return {}
                self._logger.info("reddit_thread_context bridge failure code=%s url=%s (%s)", exc.code, url, exc.message)
                self.metrics["reddit_fallback_queries"] += 1

        if self.reddit_mode == "public_direct":
            self.metrics["reddit_public_direct_queries"] += 1

        if self.node_bin and self.readonly_script.exists():
            payload = await self._run_json_command(
                [
                    self.node_bin,
                    str(self.readonly_script),
                    "thread",
                    url,
                    "--commentLimit",
                    "50",
                    "--depth",
                    "4",
                ],
                timeout=25,
            )
            if payload and payload.get("ok"):
                data = payload.get("data", {})
                post = data.get("post", {})
                raw_comments = data.get("comments", [])
                comments = [
                    comment.get("body_snippet", "")
                    for comment in raw_comments
                    if comment.get("body_snippet")
                ]
                # Build comment metadata for corroboration scoring
                comment_metadata = [
                    {
                        "body": comment.get("body_snippet", "")[:200],
                        "author": comment.get("author", ""),
                        "score": comment.get("score", 0),
                    }
                    for comment in raw_comments[:50]
                    if comment.get("body_snippet")
                ]
                result = {
                    "title": post.get("title", ""),
                    "text": self._compact_text(
                        " ".join([post.get("title", ""), post.get("selftext_snippet", ""), *comments]),
                        2500,
                    ),
                    "description": self._compact_text(post.get("selftext_snippet", ""), 900),
                    "comments": comments,
                    "comment_metadata": comment_metadata,
                }
                self._cache_set(self.thread_cache, url, result)
                self._logger.info(
                    "reddit_thread_context using fallback path=readonly_script url=%s comments=%s",
                    url,
                    len(comments),
                )
                return result

        # Fallback to public JSON API
        self._logger.info("reddit_thread_context falling back to public_json url=%s node_bin=%s script_exists=%s",
            url, bool(self.node_bin), self.readonly_script.exists() if hasattr(self.readonly_script, 'exists') else 'N/A')
        json_url = url.rstrip("/") + "/.json"

        def _request() -> dict[str, Any]:
            response = self._request_get(
                json_url,
                params={"limit": 50},
                timeout=15,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            payload = response.json()
            post = payload[0]["data"]["children"][0]["data"]
            raw_comments = []
            comment_metadata = []
            for child in payload[1]["data"].get("children", []):
                if child.get("kind") != "t1":
                    continue
                data = child.get("data", {})
                body = data.get("body", "")
                if body and len(raw_comments) < 50:
                    raw_comments.append(body)
                    comment_metadata.append({
                        "body": body[:200],
                        "author": data.get("author", ""),
                        "score": data.get("score", 0),
                    })
                if len(raw_comments) >= 50:
                    break
            text = self._compact_text(" ".join([post.get("title", ""), post.get("selftext", ""), *raw_comments]), 2500)
            return {
                "title": post.get("title", ""),
                "text": text,
                "description": self._compact_text(post.get("selftext", ""), 900),
                "comments": raw_comments,
                "comment_metadata": comment_metadata,
            }

        try:
            payload = await asyncio.to_thread(_request)
            self._cache_set(self.thread_cache, url, payload)
            self._logger.info(
                "reddit_thread_context using fallback path=public_json url=%s comments=%s",
                url,
                len(payload.get("comments", [])),
            )
            return payload
        except Exception:
            return {"title": "", "text": "", "description": "", "comments": []}
