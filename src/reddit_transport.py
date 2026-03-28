"""Reddit bridge, fallback, and cache behavior for the research toolkit."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

try:
    from reddit_bridge import BridgeError, RedditBridgeClient
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_bridge import BridgeError, RedditBridgeClient

try:
    from search_models import SearchDocument
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.search_models import SearchDocument


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
        self.search_cache: dict[tuple[str, str, int, str, str], list[SearchDocument]] = {}
        self.thread_cache: dict[str, dict[str, Any]] = {}
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

    async def warm_validation_queries(
        self,
        *,
        subreddits: list[str],
        queries: list[str],
    ) -> dict[str, int]:
        if self.reddit_mode != "bridge_only" or not self.reddit_bridge.enabled:
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
            after = seeder.coverage_report(subreddits=normalized_subreddits, queries=normalized_queries)
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
        self.metrics["reddit_validation_seed_uncovered_after"] += int(after.uncovered_pairs)
        self._logger.info(
            "reddit validation queries warmed pairs=%s searches=%s uncovered_before=%s uncovered_after=%s",
            len(pairs),
            summary.cached_searches,
            before.uncovered_pairs,
            after.uncovered_pairs,
        )
        return {
            "seed_runs": 1,
            "seeded_pairs": len(pairs),
            "seeded_searches": int(summary.cached_searches),
            "uncovered_before": int(before.uncovered_pairs),
            "uncovered_after": int(after.uncovered_pairs),
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
        if cache_key in self.search_cache:
            return list(self.search_cache[cache_key])

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
                    self.search_cache[cache_key] = []
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
                self.search_cache[cache_key] = list(docs)
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
                    self.search_cache[cache_key] = []
                    return []
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
                self.search_cache[cache_key] = list(docs)
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
            self.search_cache[cache_key] = list(docs)
            return docs
        except Exception:
            return []

    async def thread_context(self, url: str) -> dict[str, Any]:
        if url in self.thread_cache:
            return dict(self.thread_cache[url])

        if self.reddit_mode in {"bridge_with_fallback", "bridge_only"}:
            if not self.reddit_bridge.enabled:
                self.metrics["reddit_bridge_misses"] += 1
                self._logger.info("reddit_thread_context bridge unavailable url=%s mode=%s", url, self.reddit_mode)
                if self.reddit_mode == "bridge_only":
                    self.thread_cache[url] = {}
                    return {}
                # bridge_with_fallback should continue to legacy/public paths when the bridge
                # is not configured in the current environment (e.g. Codespaces).
            try:
                payload = await self.reddit_bridge.get_post_thread(url=url, comment_limit=8, depth=4)
                comments = [item.get("body", "") for item in payload.get("comments", []) if item.get("body")]
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
                    "comments": comments[:6],
                }
                self.metrics["reddit_bridge_hits"] += 1
                self.thread_cache[url] = result
                return result
            except BridgeError as exc:
                self.metrics["reddit_bridge_misses"] += 1
                if self.reddit_mode == "bridge_only":
                    self._logger.info(
                        "reddit_thread_context bridge miss code=%s url=%s mode=bridge_only; returning empty",
                        exc.code,
                        url,
                    )
                    self.thread_cache[url] = {}
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
                    "8",
                    "--depth",
                    "4",
                ],
                timeout=25,
            )
            if payload and payload.get("ok"):
                data = payload.get("data", {})
                post = data.get("post", {})
                comments = [
                    comment.get("body_snippet", "")
                    for comment in data.get("comments", [])
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
                }
                self.thread_cache[url] = result
                return result

        json_url = url.rstrip("/") + "/.json"

        def _request() -> dict[str, Any]:
            response = self._request_get(
                json_url,
                params={"limit": 8},
                timeout=15,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            payload = response.json()
            post = payload[0]["data"]["children"][0]["data"]
            comments = []
            for child in payload[1]["data"].get("children", []):
                if child.get("kind") != "t1":
                    continue
                body = child.get("data", {}).get("body", "")
                if body:
                    comments.append(body)
                if len(comments) >= 6:
                    break
            text = self._compact_text(" ".join([post.get("title", ""), post.get("selftext", ""), *comments]), 2500)
            return {
                "title": post.get("title", ""),
                "text": text,
                "description": self._compact_text(post.get("selftext", ""), 900),
                "comments": comments,
            }

        try:
            payload = await asyncio.to_thread(_request)
            self.thread_cache[url] = payload
            return payload
        except Exception:
            return {"title": "", "text": "", "description": "", "comments": []}
