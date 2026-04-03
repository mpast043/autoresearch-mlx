"""Tests for Reddit transport bridge warming and fallback behavior."""

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.reddit_bridge import BridgeError
from src.reddit_transport import RedditTransport


class StubBridgeClient:
    def __init__(self):
        self.enabled = True
        self.calls = 0

    async def search_posts(self, *, subreddit, query, limit=2, sort="relevance"):
        self.calls += 1
        if self.calls == 1:
            raise BridgeError("no_cached_result", "cache miss", 404)
        return (
            [
                {
                    "title": "Manual reconciliation is still painful",
                    "permalink": "https://reddit.com/r/operations/comments/abc123/thread/",
                    "body": "We still do this by hand every week.",
                    "subreddit": subreddit,
                }
            ],
            "",
        )

    async def get_post_thread(self, *, url, comment_limit=8, depth=4):
        return {"post": {}, "comments": []}

    async def close(self):
        return None


def build_transport(*, bridge_client=None, reddit_mode="bridge_with_fallback") -> RedditTransport:
    return RedditTransport(
        config={"reddit_bridge": {"seed_on_miss": True}},
        reddit_bridge=bridge_client or StubBridgeClient(),
        reddit_mode=reddit_mode,
        node_bin=None,
        readonly_script=Path("/nonexistent"),
        user_agent="test-agent",
        run_json_command=lambda *args, **kwargs: None,
        compact_text=lambda text, limit: text[:limit],
        normalize_query=lambda text: " ".join(str(text).split()),
        request_get=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public fallback should not run")),
        logger=__import__("logging").getLogger("test-reddit-transport"),
    )


@pytest.mark.asyncio
async def test_warm_validation_queries_runs_in_bridge_with_fallback(monkeypatch):
    captured = {}

    class FakeSeeder:
        def __init__(self, config):
            captured["config"] = config

        def coverage_report(self, *, subreddits, queries):
            captured["before"] = (list(subreddits), list(queries))
            return type("Coverage", (), {"uncovered_pairs": 1})()

        async def seed(self, *, subreddits, queries):
            captured["seed"] = (list(subreddits), list(queries))
            return type("Summary", (), {"cached_searches": 1})()

    monkeypatch.setattr("reddit_seed.RedditSeeder", FakeSeeder)
    transport = build_transport()

    summary = await transport.warm_validation_queries(
        subreddits=["operations"],
        queries=["manual reconciliation"],
    )

    assert summary["seed_runs"] == 1
    assert summary["seeded_searches"] == 1
    assert captured["before"] == (["operations"], ["manual reconciliation"])
    assert captured["seed"] == (["operations"], ["manual reconciliation"])


@pytest.mark.asyncio
async def test_search_warms_bridge_cache_before_fallback(monkeypatch):
    transport = build_transport()
    warmed = []

    async def fake_warm_validation_queries(*, subreddits, queries):
        warmed.append((list(subreddits), list(queries)))
        return {
            "seed_runs": 1,
            "seeded_pairs": 1,
            "seeded_searches": 1,
            "uncovered_before": 1,
            "uncovered_after": 0,
        }

    monkeypatch.setattr(transport, "warm_validation_queries", fake_warm_validation_queries)

    docs = await transport.search(subreddit="operations", query="manual reconciliation", limit=1)

    assert len(docs) == 1
    assert docs[0].url == "https://reddit.com/r/operations/comments/abc123/thread/"
    assert warmed == [(["operations"], ["manual reconciliation"])]
    assert transport.metrics["reddit_fallback_queries"] == 0
