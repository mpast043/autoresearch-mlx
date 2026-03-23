"""Helpers for priming the local Reddit relay cache."""

from __future__ import annotations

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
    from discovery_queries import reddit_problem_keywords, reddit_problem_subreddits
    from runtime.paths import resolve_project_path
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_relay import RedditRelayStore
    from src.research_tools import ResearchToolkit
    from src.search_models import SearchDocument
    from src.discovery_queries import reddit_problem_keywords, reddit_problem_subreddits
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

    def __init__(self, config: dict[str, Any], relay_store: RedditRelayStore | None = None) -> None:
        self.config = config
        self.discovery_config = config.get("discovery", {}).get("reddit", {})
        relay_config = config.get("reddit_relay", {})
        self.search_limit = int(self.discovery_config.get("seed_limit", 2))
        max_pairs_value = self.discovery_config.get("seed_pairs")
        self.max_pairs = int(max_pairs_value) if max_pairs_value not in (None, "") else None
        self.cache_ttl_seconds = int(relay_config.get("seed_cache_ttl_seconds", 21600))
        relay_cache_path = str(
            resolve_project_path(relay_config.get("cache_db_path"), default="data/reddit_relay_cache.db")
        )
        self.relay_store = relay_store or RedditRelayStore(
            relay_cache_path
        )

    def build_toolkit(self) -> ResearchToolkit:
        config = copy.deepcopy(self.config)
        config.setdefault("reddit_bridge", {})
        config["reddit_bridge"]["enabled"] = False
        config["reddit_bridge"]["mode"] = "public_direct"
        return ResearchToolkit(config)

    def iter_pairs(
        self,
        subreddits: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        subreddits = subreddits or reddit_problem_subreddits(self.config)
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
        pairs = self.iter_pairs(subreddits=subreddits, queries=queries)
        summary.total_pairs = len(pairs)
        try:
            for subreddit, query in pairs:
                if self.relay_store.has_search(subreddit=subreddit, query=query, sort="relevance", cursor=""):
                    summary.existing_cached_pairs += 1
                if self.relay_store.has_fresh_search(
                    subreddit=subreddit,
                    query=query,
                    sort="relevance",
                    cursor="",
                    max_age_seconds=self.cache_ttl_seconds,
                ):
                    summary.skipped_fresh_pairs += 1
                    logger.info("reddit_seed skipping fresh cache subreddit=%s query=%r", subreddit, query)
                    continue

                summary.searched_pairs += 1
                docs = await toolkit.reddit_search(subreddit, query, limit=self.search_limit)
                if not docs:
                    logger.info("reddit_seed no docs subreddit=%s query=%r", subreddit, query)
                    self.relay_store.put_search(
                        subreddit=subreddit,
                        query=query,
                        sort="relevance",
                        cursor="",
                        items=[],
                        next_cursor="",
                    )
                    summary.cached_searches += 1
                    continue

                query_posts: list[dict[str, Any]] = []
                for doc in docs:
                    if not doc.url or doc.url in seen_urls:
                        continue
                    seen_urls.add(doc.url)
                    if self.relay_store.has_thread(url=doc.url):
                        summary.thread_cache_hits += 1

                    try:
                        context = await toolkit.reddit_thread_context(doc.url)
                    except Exception as exc:  # pragma: no cover - network/runtime dependent
                        logger.warning("reddit_seed thread fetch failed url=%s (%s)", doc.url, exc)
                        context = {
                            "title": doc.title,
                            "description": doc.snippet,
                            "comments": [],
                        }

                    post_item = build_post_item(doc, subreddit=subreddit, thread_context=context)
                    comment_items = build_comment_items(post_item, list(context.get("comments", []) or []))

                    self.relay_store.put_thread(url=doc.url, post=post_item, comments=comment_items)
                    self.relay_store.put_comments(url=doc.url, items=comment_items)

                    summary.cached_threads += 1
                    summary.cached_comments += len(comment_items)
                    summary.discovered_posts += 1
                    query_posts.append(post_item)

                if not query_posts:
                    self.relay_store.put_search(
                        subreddit=subreddit,
                        query=query,
                        sort="relevance",
                        cursor="",
                        items=[],
                        next_cursor="",
                    )
                    summary.cached_searches += 1
                    continue

                self.relay_store.put_search(
                    subreddit=subreddit,
                    query=query,
                    sort="relevance",
                    cursor="",
                    items=query_posts,
                    next_cursor="",
                )
                summary.cached_searches += 1

            summary.unique_urls = len(seen_urls)
            summary.uncovered_pairs = max(summary.total_pairs - summary.existing_cached_pairs - summary.cached_searches, 0)
            logger.info(
                "reddit_seed completed total_pairs=%s searched_pairs=%s cached_searches=%s cached_threads=%s unique_urls=%s uncovered_pairs=%s",
                summary.total_pairs,
                summary.searched_pairs,
                summary.cached_searches,
                summary.cached_threads,
                summary.unique_urls,
                summary.uncovered_pairs,
            )
            return summary
        finally:
            close = getattr(toolkit, "close", None)
            if close:
                await close()
