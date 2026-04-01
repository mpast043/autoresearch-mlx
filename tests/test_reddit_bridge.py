"""Tests for the Reddit bridge client contract and normalization."""

import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.reddit_bridge import BridgeError, RedditBridgeClient, normalize_reddit_item


def test_normalize_reddit_item_enforces_stable_shape():
    item = normalize_reddit_item(
        {
            "id": "abc123",
            "kind": "post",
            "subreddit": "operations",
            "title": "Manual cleanup every week",
            "body": "We still copy paste reports.",
            "author": "opslead",
            "permalink": "https://www.reddit.com/r/operations/comments/abc123/thread/",
            "score": 8,
            "num_comments": 4,
            "created_utc": 1710000000,
            "source_type": "reddit",
            "post_id": "abc123",
            "parent_id": "",
        },
        expected_kind="post",
    )

    assert item["kind"] == "post"
    assert item["source_type"] == "reddit"
    assert item["post_id"] == "abc123"


def test_normalize_reddit_item_rejects_bad_shape():
    with pytest.raises(BridgeError) as exc:
        normalize_reddit_item({"id": "abc123", "kind": "post"}, expected_kind="post")

    assert exc.value.code == "bad_response_shape"


@pytest.mark.asyncio
async def test_bridge_comments_retrieval_returns_normalized_items():
    client = RedditBridgeClient({"enabled": True, "base_url": "https://bridge.example.com"})

    async def fake_post(path, payload):
        assert path == "/api/reddit/comments"
        return {
            "ok": True,
            "items": [
                {
                    "id": "c1",
                    "kind": "comment",
                    "subreddit": "EtsySellers",
                    "title": "",
                    "body": "I resend them manually every day.",
                    "author": "seller2",
                    "permalink": "https://www.reddit.com/r/EtsySellers/comments/abc123/thread/c1",
                    "score": 3,
                    "num_comments": 0,
                    "created_utc": 1710000001,
                    "source_type": "reddit",
                    "post_id": "abc123",
                    "parent_id": "t3_abc123",
                }
            ],
        }

    client._post = fake_post
    comments = await client.get_comments(url="https://www.reddit.com/r/EtsySellers/comments/abc123/thread/")

    assert len(comments) == 1
    assert comments[0]["kind"] == "comment"
    assert comments[0]["post_id"] == "abc123"


@pytest.mark.asyncio
async def test_bridge_search_returns_items_and_cursor():
    client = RedditBridgeClient({"enabled": True, "base_url": "https://bridge.example.com"})

    async def fake_post(path, payload):
        assert path == "/api/reddit/search-posts"
        return {
            "ok": True,
            "items": [
                {
                    "id": "abc123",
                    "kind": "post",
                    "subreddit": "operations",
                    "title": "Manual cleanup every week",
                    "body": "We still copy paste reports.",
                    "author": "opslead",
                    "permalink": "https://www.reddit.com/r/operations/comments/abc123/thread/",
                    "score": 8,
                    "num_comments": 4,
                    "created_utc": 1710000000,
                    "source_type": "reddit",
                    "post_id": "abc123",
                    "parent_id": "",
                }
            ],
            "next_cursor": "t3_next",
        }

    client._post = fake_post
    items, next_cursor = await client.search_posts(subreddit="operations", query="manual cleanup", limit=1)

    assert len(items) == 1
    assert next_cursor == "t3_next"
    assert items[0]["kind"] == "post"


@pytest.mark.asyncio
async def test_bridge_client_reuses_single_bounded_session(monkeypatch):
    created_sessions = []
    created_connectors = []

    class FakeResponse:
        status = 200

        async def json(self):
            return {"ok": True, "items": [], "next_cursor": ""}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, *, timeout=None, connector=None):
            self.timeout = timeout
            self.connector = connector
            self.closed = False
            self.calls = []
            created_sessions.append(self)

        def post(self, url, json=None, headers=None):
            self.calls.append(("post", url, json, headers))
            return FakeResponse()

        async def close(self):
            self.closed = True

    class FakeConnector:
        def __init__(self, *, limit=None, enable_cleanup_closed=None):
            self.limit = limit
            self.enable_cleanup_closed = enable_cleanup_closed
            created_connectors.append(self)

    monkeypatch.setattr("reddit_bridge.aiohttp.ClientSession", FakeSession)
    monkeypatch.setattr("reddit_bridge.aiohttp.TCPConnector", FakeConnector)

    client = RedditBridgeClient(
        {"enabled": True, "base_url": "https://bridge.example.com", "connection_limit": 3}
    )

    await client.search_posts(subreddit="operations", query="manual cleanup", limit=1)
    await client.search_posts(subreddit="operations", query="manual cleanup", limit=1)
    await client.close()

    assert len(created_sessions) == 1
    assert len(created_connectors) == 1
    assert created_connectors[0].limit == 3
    assert len(created_sessions[0].calls) == 2
    assert created_sessions[0].closed is True
