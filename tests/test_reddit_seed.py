"""Tests for Reddit seed helpers."""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reddit_relay import RedditRelayStore
from reddit_seed import RedditSeeder, build_comment_items, build_post_item, reddit_post_id_from_url
from research_tools import SearchDocument
from runtime.paths import PROJECT_ROOT


def test_reddit_post_id_from_url_extracts_id():
    assert reddit_post_id_from_url("https://www.reddit.com/r/ops/comments/abc123/thread/") == "abc123"


def test_build_post_and_comment_items_shape():
    doc = SearchDocument(
        title="Manual cleanup every week",
        url="https://www.reddit.com/r/ops/comments/abc123/thread/",
        snippet="We still use spreadsheets.",
        source="reddit",
    )
    post = build_post_item(doc, subreddit="ops", thread_context={"comments": ["one", "two"]})
    comments = build_comment_items(post, ["one", "two"])

    assert post["kind"] == "post"
    assert post["post_id"] == "abc123"
    assert len(comments) == 2
    assert comments[0]["kind"] == "comment"
    assert comments[0]["post_id"] == "abc123"


def test_seed_iter_pairs_defaults_to_full_query_set_when_unset():
    seeder = RedditSeeder({"discovery": {"reddit": {}}, "reddit_relay": {"cache_db_path": tempfile.mktemp(suffix='.db')}})
    pairs = seeder.iter_pairs(subreddits=["smallbusiness", "sysadmin"], queries=["a", "b", "c"])
    assert len(pairs) == 6
    db_path = seeder.relay_store.db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_seed_coverage_report_counts_cached_and_uncovered_pairs():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        store.put_search(
            subreddit="smallbusiness",
            query="manual process",
            sort="relevance",
            cursor="",
            items=[{"id": "abc", "kind": "post", "source_type": "reddit", "post_id": "abc"}],
            next_cursor="",
        )
        seeder = RedditSeeder(
            {"discovery": {"reddit": {}}, "reddit_relay": {"cache_db_path": path}},
            relay_store=store,
        )
        summary = seeder.coverage_report(
            subreddits=["smallbusiness", "sysadmin"],
            queries=["manual process", "spreadsheet workaround"],
        )
        assert summary.total_pairs == 4
        assert summary.existing_cached_pairs == 1
        assert summary.uncovered_pairs == 3
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_seed_populates_summary_with_full_pair_counts():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        seeder = RedditSeeder(
            {"discovery": {"reddit": {"seed_limit": 1}}, "reddit_relay": {"cache_db_path": path}},
            relay_store=store,
        )

        class StubToolkit:
            async def reddit_search(self, subreddit, query, limit=2):
                slug = query.replace(" ", "-")
                return [SearchDocument(title=f"{subreddit} {query}", url=f"https://reddit.com/r/{subreddit}/comments/{slug}/test/", snippet="body", source=f"reddit/{subreddit}")]

            async def reddit_thread_context(self, url):
                return {"title": "Test", "description": "desc", "comments": ["one"]}

        seeder.build_toolkit = lambda: StubToolkit()
        summary = asyncio.run(seeder.seed(subreddits=["smallbusiness"], queries=["manual process", "spreadsheet workaround"]))

        assert summary.total_pairs == 2
        assert summary.searched_pairs == 2
        assert summary.cached_searches == 2
        assert summary.cached_threads == 2
        assert summary.unique_urls == 2
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_seed_toolkit_forces_public_direct_mode():
    path = tempfile.mktemp(suffix=".db")
    try:
        seeder = RedditSeeder(
            {"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_with_fallback"}, "reddit_relay": {"cache_db_path": path}},
        )
        toolkit = seeder.build_toolkit()
        assert toolkit.reddit_mode == "public_direct"
        assert toolkit.reddit_bridge.enabled is False
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_seed_caches_empty_search_results_for_no_doc_pairs():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        seeder = RedditSeeder(
            {"discovery": {"reddit": {"seed_limit": 1}}, "reddit_relay": {"cache_db_path": path}},
            relay_store=store,
        )

        class StubToolkit:
            async def reddit_search(self, subreddit, query, limit=2):
                return []

            async def reddit_thread_context(self, url):
                raise AssertionError("thread context should not be called for empty search results")

        seeder.build_toolkit = lambda: StubToolkit()
        summary = asyncio.run(seeder.seed(subreddits=["sysadmin"], queries=["manual reconciliation"]))

        assert summary.total_pairs == 1
        assert summary.searched_pairs == 1
        assert summary.cached_searches == 1
        assert store.has_search(subreddit="sysadmin", query="manual reconciliation", sort="relevance", cursor="")
        result = store.get_search(subreddit="sysadmin", query="manual reconciliation", sort="relevance", cursor="", limit=10)
        assert result is not None
        assert result["items"] == []
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_seed_resolves_default_relay_cache_path_from_project_root():
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        seeder = RedditSeeder({"discovery": {"reddit": {}}, "reddit_relay": {}})
        assert Path(seeder.relay_store.db_path) == PROJECT_ROOT / "data" / "reddit_relay_cache.db"
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)
