"""Helpers for priming the local Reddit relay cache."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    from reddit_relay import RedditRelayStore
    from research_tools import ResearchToolkit
    from search_models import SearchDocument
    from discovery_queries import reddit_discovery_subreddits, reddit_problem_keywords
    from runtime.paths import resolve_project_path
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_relay import RedditRelayStore
    from src.research_tools import ResearchToolkit
    from src.search_models import SearchDocument
    from src.discovery_queries import reddit_discovery_subreddits, reddit_problem_keywords
    from src.runtime.paths import resolve_project_path


logger = logging.getLogger(__name__)

REDDIT_POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)


@dataclass
class RedditSeedSummary:
    total_pairs: int = 0
    searched_pairs: int = 0
    skipped_fresh_pairs: int = 0
    existing_cached_pairs: int = 0
    uncovered_pairs: int = 0
    cached_searches: int = 0
    cached_threads: int = 0
    cached_comments: int = 0
    discovered_posts: int = 0
    unique_urls: int = 0
    thread_cache_hits: int = 0
    bridge_search_count: int = 0
    bridge_search_result_count: int = 0
    public_json_search_count: int = 0
    bridge_thread_count: int = 0
    bridge_thread_no_cached_count: int = 0
    public_json_thread_count: int = 0
    degraded_fallback_findings_count: int = 0
    degraded_fallback_docs_count: int = 0
    bridge_covered_pairs: int = 0
    degraded_covered_pairs: int = 0
    pairs_with_usable_bridge_docs: int = 0
    pairs_with_only_degraded_docs: int = 0
    failed_pairs: int = 0
    truly_uncovered_pairs: int = 0


def reddit_post_id_from_url(url: str) -> str:
    match = REDDIT_POST_ID_RE.search(url)
    if match:
        return match.group(1)
    slug = re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")
    return slug[:32] or "unknown"


def subreddit_from_url(url: str, fallback: str) -> str:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0].lower() == "r":
        return path_parts[1]
    return fallback


def build_post_item(
    doc: SearchDocument,
    subreddit: str,
    thread_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    post_id = reddit_post_id_from_url(doc.url)
    context = thread_context or {}
    return {
        "id": post_id,
        "kind": "post",
        "subreddit": subreddit_from_url(doc.url, subreddit),
        "title": context.get("title") or doc.title,
        "body": context.get("description") or doc.snippet,
        "author": "",
        "permalink": doc.url,
        "score": 0,
        "num_comments": len(context.get("comments", []) or []),
        "created_utc": 0,
        "source_type": "reddit",
        "post_id": post_id,
        "parent_id": "",
    }


def build_comment_items(post_item: dict[str, Any], comments: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, body in enumerate(comments):
        text = (body or "").strip()
        if not text:
            continue
        items.append(
            {
                "id": f"{post_item['post_id']}-c{index + 1}",
                "kind": "comment",
                "subreddit": post_item["subreddit"],
                "title": "",
                "body": text,
                "author": "",
                "permalink": f"{post_item['permalink']}#comment-{index + 1}",
                "score": 0,
                "num_comments": 0,
                "created_utc": 0,
                "source_type": "reddit",
                "post_id": post_item["post_id"],
                "parent_id": f"t3_{post_item['post_id']}",
            }
        )
    return items


class RedditSeeder:
    """Populate the relay cache from configured subreddit/query pairs."""

    def __init__(self, config: dict[str, Any], relay_store: RedditRelayStore | None = None, bypass_cache: bool = False) -> None:
        self.config = config
        self.discovery_config = config.get("discovery", {}).get("reddit", {})
        relay_config = config.get("reddit_relay", {})
        self.search_limit = int(self.discovery_config.get("seed_limit", 2))
        self.pair_concurrency = max(1, int(self.discovery_config.get("pair_concurrency", 1) or 1))
        max_pairs_value = self.discovery_config.get("seed_pairs")
        self.max_pairs = int(max_pairs_value) if max_pairs_value not in (None, "") else None
        self.cache_ttl_seconds = int(relay_config.get("seed_cache_ttl_seconds", 21600))
        self.bypass_cache = bypass_cache
        relay_cache_path = str(
            resolve_project_path(relay_config.get("cache_db_path"), default="data/reddit_relay_cache.db")
        )
        self.relay_store = relay_store or RedditRelayStore(
            relay_cache_path
        )

    def build_toolkit(self) -> ResearchToolkit:
        config = copy.deepcopy(self.config)
        config.setdefault("reddit_bridge", {})
        config["reddit_bridge"]["seed_on_miss"] = False
        return ResearchToolkit(config)

    def iter_pairs(
        self,
        subreddits: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        subreddits = subreddits or reddit_discovery_subreddits(self.config)
        keywords = queries or reddit_problem_keywords(self.config)
        pairs: list[tuple[str, str]] = []
        for subreddit in subreddits:
            for keyword in keywords:
                pairs.append((str(subreddit), str(keyword)))
                if self.max_pairs and len(pairs) >= self.max_pairs:
                    return pairs
        return pairs

    def coverage_report(
        self,
        subreddits: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> RedditSeedSummary:
        summary = RedditSeedSummary()
        pairs = self.iter_pairs(subreddits=subreddits, queries=queries)
        summary.total_pairs = len(pairs)
        for subreddit, query in pairs:
            # When bypass_cache is True, treat all searches as uncached
            if self.bypass_cache:
                cached = False
                fresh = False
            else:
                cached = self.relay_store.has_search(subreddit=subreddit, query=query, sort="relevance", cursor="")
                fresh = self.relay_store.has_fresh_search(
                    subreddit=subreddit,
                    query=query,
                    sort="relevance",
                    cursor="",
                    max_age_seconds=self.cache_ttl_seconds,
                )
            if cached:
                summary.existing_cached_pairs += 1
            if fresh:
                summary.skipped_fresh_pairs += 1
            if not cached:
                summary.uncovered_pairs += 1
        return summary

    async def seed(
        self,
        subreddits: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> RedditSeedSummary:
        toolkit = self.build_toolkit()
        summary = RedditSeedSummary()
        seen_urls: set[str] = set()
        seen_urls_lock = asyncio.Lock()
        pairs = self.iter_pairs(subreddits=subreddits, queries=queries)
        summary.total_pairs = len(pairs)

        async def _process_pair(subreddit: str, query: str) -> RedditSeedSummary:
            pair_summary = RedditSeedSummary(total_pairs=1)
            if self.bypass_cache:
                should_skip = False
            else:
                should_skip = self.relay_store.has_search(subreddit=subreddit, query=query, sort="relevance", cursor="")

            if should_skip:
                pair_summary.existing_cached_pairs += 1

            if not self.bypass_cache and self.relay_store.has_fresh_search(
                subreddit=subreddit,
                query=query,
                sort="relevance",
                cursor="",
                max_age_seconds=self.cache_ttl_seconds,
            ):
                pair_summary.skipped_fresh_pairs += 1
                logger.info("reddit_seed skipping fresh cache subreddit=%s query=%r", subreddit, query)
                return pair_summary

            if self.bypass_cache:
                logger.info(
                    "reddit_seed bypassing local seed cache subreddit=%s query=%r; transport still attempts bridge first and labels degraded fallback",
                    subreddit,
                    query,
                )

            pair_summary.searched_pairs += 1
            docs = await toolkit.reddit_search(subreddit, query, limit=self.search_limit)
            if not docs:
                logger.info("reddit_seed no docs subreddit=%s query=%r", subreddit, query)
                pair_summary.truly_uncovered_pairs += 1
                return pair_summary

            usable_docs = [
                doc
                for doc in docs
                if str(getattr(doc, "source_quality", "") or "normal").lower() != "degraded"
            ]
            degraded_docs = [
                doc
                for doc in docs
                if str(getattr(doc, "source_quality", "") or "normal").lower() == "degraded"
            ]
            if usable_docs:
                pair_summary.bridge_covered_pairs += 1
                pair_summary.pairs_with_usable_bridge_docs += 1
            if degraded_docs:
                pair_summary.degraded_covered_pairs += 1
            if degraded_docs and not usable_docs:
                pair_summary.pairs_with_only_degraded_docs += 1
                pair_summary.truly_uncovered_pairs += 1
                logger.info(
                    "reddit_seed degraded-only pair not cached for evidence subreddit=%s query=%r degraded_docs=%s",
                    subreddit,
                    query,
                    len(degraded_docs),
                )
                return pair_summary

            query_posts: list[dict[str, Any]] = []
            for doc in usable_docs:
                if not doc.url:
                    continue
                async with seen_urls_lock:
                    if doc.url in seen_urls:
                        continue
                    seen_urls.add(doc.url)

                if self.relay_store.has_thread(url=doc.url):
                    pair_summary.thread_cache_hits += 1

                try:
                    context = await toolkit.reddit_thread_context(doc.url)
                except Exception as exc:  # pragma: no cover - network/runtime dependent
                    logger.warning("reddit_seed thread fetch failed url=%s (%s)", doc.url, exc)
                    context = {
                        "title": doc.title,
                        "description": doc.snippet,
                        "comments": [],
                    }
                context_quality = str(context.get("source_quality", "") or "normal").lower()
                if context_quality in {"degraded", "bridge_miss"}:
                    logger.info(
                        "reddit_seed skipping unusable thread context for relay cache url=%s subreddit=%s query=%r source_quality=%s",
                        doc.url,
                        subreddit,
                        query,
                        context_quality,
                    )
                    context = {
                        "title": doc.title,
                        "description": doc.snippet,
                        "comments": [],
                        "source_quality": getattr(doc, "source_quality", "normal"),
                    }

                post_item = build_post_item(doc, subreddit=subreddit, thread_context=context)
                comment_items = build_comment_items(post_item, list(context.get("comments", []) or []))

                if context_quality not in {"degraded", "bridge_miss"}:
                    self.relay_store.put_thread(url=doc.url, post=post_item, comments=comment_items)
                    self.relay_store.put_comments(url=doc.url, items=comment_items)

                    pair_summary.cached_threads += 1
                pair_summary.cached_comments += len(comment_items)
                pair_summary.discovered_posts += 1
                query_posts.append(post_item)

            if not query_posts:
                pair_summary.failed_pairs += 1
                pair_summary.truly_uncovered_pairs += 1
                return pair_summary

            self.relay_store.put_search(
                subreddit=subreddit,
                query=query,
                sort="relevance",
                cursor="",
                items=query_posts,
                next_cursor="",
            )
            pair_summary.cached_searches += 1
            return pair_summary

        try:
            semaphore = asyncio.Semaphore(self.pair_concurrency)

            async def _run_pair(subreddit: str, query: str) -> RedditSeedSummary:
                async with semaphore:
                    return await _process_pair(subreddit, query)

            pair_summaries = await asyncio.gather(*(_run_pair(subreddit, query) for subreddit, query in pairs))
            for pair_summary in pair_summaries:
                summary.searched_pairs += pair_summary.searched_pairs
                summary.skipped_fresh_pairs += pair_summary.skipped_fresh_pairs
                summary.existing_cached_pairs += pair_summary.existing_cached_pairs
                summary.cached_searches += pair_summary.cached_searches
                summary.cached_threads += pair_summary.cached_threads
                summary.cached_comments += pair_summary.cached_comments
                summary.discovered_posts += pair_summary.discovered_posts
                summary.thread_cache_hits += pair_summary.thread_cache_hits
                summary.bridge_covered_pairs += pair_summary.bridge_covered_pairs
                summary.degraded_covered_pairs += pair_summary.degraded_covered_pairs
                summary.pairs_with_usable_bridge_docs += pair_summary.pairs_with_usable_bridge_docs
                summary.pairs_with_only_degraded_docs += pair_summary.pairs_with_only_degraded_docs
                summary.failed_pairs += pair_summary.failed_pairs
                summary.truly_uncovered_pairs += pair_summary.truly_uncovered_pairs

            summary.unique_urls = len(seen_urls)
            metrics = toolkit.get_reddit_runtime_metrics() if hasattr(toolkit, "get_reddit_runtime_metrics") else {}
            summary.bridge_search_count = int(metrics.get("bridge_search_count", 0) or 0)
            summary.bridge_search_result_count = int(metrics.get("bridge_search_result_count", 0) or 0)
            summary.public_json_search_count = int(metrics.get("public_json_search_count", 0) or 0)
            summary.bridge_thread_count = int(metrics.get("bridge_thread_count", 0) or 0)
            summary.bridge_thread_no_cached_count = int(metrics.get("bridge_thread_no_cached_count", 0) or 0)
            summary.public_json_thread_count = int(metrics.get("public_json_thread_count", 0) or 0)
            summary.degraded_fallback_findings_count = int(metrics.get("degraded_fallback_findings_count", 0) or 0)
            summary.degraded_fallback_docs_count = int(metrics.get("degraded_fallback_docs_count", 0) or 0)
            summary.uncovered_pairs = summary.truly_uncovered_pairs
            logger.info(
                "reddit_seed completed total_pairs=%s searched_pairs=%s cached_searches=%s cached_threads=%s unique_urls=%s uncovered_pairs=%s bridge_covered_pairs=%s degraded_covered_pairs=%s pairs_with_usable_bridge_docs=%s pairs_with_only_degraded_docs=%s failed_pairs=%s truly_uncovered_pairs=%s bridge_searches=%s public_json_searches=%s public_json_threads=%s degraded_fallback_findings=%s degraded_fallback_docs=%s",
                summary.total_pairs,
                summary.searched_pairs,
                summary.cached_searches,
                summary.cached_threads,
                summary.unique_urls,
                summary.uncovered_pairs,
                summary.bridge_covered_pairs,
                summary.degraded_covered_pairs,
                summary.pairs_with_usable_bridge_docs,
                summary.pairs_with_only_degraded_docs,
                summary.failed_pairs,
                summary.truly_uncovered_pairs,
                summary.bridge_search_count,
                summary.public_json_search_count,
                summary.public_json_thread_count,
                summary.degraded_fallback_findings_count,
                summary.degraded_fallback_docs_count,
            )
            return summary
        finally:
            close = getattr(toolkit, "close", None)
            if close:
                await close()
