"""Tests for Reddit relay cache behavior."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aiohttp.test_utils import make_mocked_request

from src.reddit_relay import (
    ALLOW_NO_AUTH_KEY,
    AUTH_TOKEN_KEY,
    STORE_KEY,
    RedditRelayStore,
    build_relay_app,
    cached_search,
    cached_thread,
)


def test_relay_store_search_round_trip():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        store.put_search(
            subreddit="operations",
            query="manual cleanup",
            sort="relevance",
            cursor="",
            items=[{"id": "abc", "kind": "post", "source_type": "reddit", "post_id": "abc"}],
            next_cursor="t3_next",
        )

        result = store.get_search(subreddit="operations", query="manual cleanup", sort="relevance", cursor="", limit=10)
        assert result is not None
        assert result["next_cursor"] == "t3_next"
        assert result["items"][0]["id"] == "abc"
        assert store.has_search(subreddit="operations", query="manual cleanup", sort="relevance", cursor="")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_relay_store_thread_and_comments_round_trip():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        url = "https://www.reddit.com/r/ops/comments/abc123/thread/"
        post = {"id": "abc123", "kind": "post", "source_type": "reddit", "post_id": "abc123"}
        comments = [{"id": "c1", "kind": "comment", "source_type": "reddit", "post_id": "abc123"}]

        store.put_thread(url=url, post=post, comments=comments)
        store.put_comments(url=url, items=comments)

        thread = store.get_thread(url=url)
        comment_result = store.get_comments(url=url, limit=10)
        assert thread is not None
        assert thread["post"]["id"] == "abc123"
        assert comment_result is not None
        assert comment_result["items"][0]["id"] == "c1"
        assert store.has_thread(url=url)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_cached_thread_repairs_legacy_shape():
    path = tempfile.mktemp(suffix=".db")
    try:
        RedditRelayStore(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO reddit_thread_cache (url, payload_json, collected_at) VALUES (?, ?, ?)",
                (
                    "https://reddit.com/r/ops/comments/abc123/thread",
                    '{"post":{"id":"abc123","title":"Legacy post","permalink":"https://reddit.com/r/ops/comments/abc123/thread"},"comments":[{"id":"c1","body":"legacy comment","permalink":"https://reddit.com/r/ops/comments/abc123/thread#c1","parent_id":"t3_abc123"}]}',
                    1,
                ),
            )
        app = build_relay_app({"reddit_relay": {"cache_db_path": path, "allow_no_auth": True}})
        request = make_mocked_request("POST", "/api/reddit/post-thread", app=app, headers={"Content-Type": "application/json"})

        async def fake_json():
            return {"url": "https://reddit.com/r/ops/comments/abc123/thread/"}

        request.json = fake_json
        response = sys.modules["asyncio"].run(cached_thread(request))
        assert response.status == 200
        body = response.text
        assert '"ok": true' in body.lower()
        assert '"source_type": "reddit"' in body
        assert '"post_id": "abc123"' in body
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_cached_search_repairs_legacy_shape():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO reddit_search_cache (cache_key, subreddit, query, sort, cursor, next_cursor, payload_json, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    store._search_key("ops", "manual cleanup", "relevance", ""),
                    "ops",
                    "manual cleanup",
                    "relevance",
                    "",
                    "",
                    '[{"id":"abc123","title":"Legacy search post","permalink":"https://reddit.com/r/ops/comments/abc123/thread"}]',
                    1,
                ),
            )
        app = build_relay_app({"reddit_relay": {"cache_db_path": path, "allow_no_auth": True}})
        request = make_mocked_request("POST", "/api/reddit/search-posts", app=app, headers={"Content-Type": "application/json"})

        async def fake_json():
            return {"subreddit": "ops", "query": "manual cleanup", "limit": 5}

        request.json = fake_json
        response = sys.modules["asyncio"].run(cached_search(request))
        assert response.status == 200
        body = response.text
        assert '"ok": true' in body.lower()
        assert '"source_type": "reddit"' in body
        assert '"kind": "post"' in body
        assert '"post_id": "abc123"' in body
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_cached_search_handles_invalid_json_payload():
    path = tempfile.mktemp(suffix=".db")
    try:
        store = RedditRelayStore(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO reddit_search_cache (cache_key, subreddit, query, sort, cursor, next_cursor, payload_json, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    store._search_key("ops", "manual cleanup", "relevance", ""),
                    "ops",
                    "manual cleanup",
                    "relevance",
                    "",
                    "",
                    "{not-valid-json",
                    1,
                ),
            )
        app = build_relay_app({"reddit_relay": {"cache_db_path": path, "allow_no_auth": True}})
        request = make_mocked_request("POST", "/api/reddit/search-posts", app=app, headers={"Content-Type": "application/json"})

        async def fake_json():
            return {"subreddit": "ops", "query": "manual cleanup", "limit": 5}

        request.json = fake_json
        response = sys.modules["asyncio"].run(cached_search(request))
        assert response.status == 200
        body = response.text
        assert '"ok": true' in body.lower()
        assert '"items": []' in body
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_build_relay_app_default_db_path_is_project_rooted():
    app = build_relay_app({"reddit_relay": {"allow_no_auth": True}})
    try:
        expected = str((Path(__file__).resolve().parents[1] / "data" / "reddit_relay_cache.db").resolve())
        assert app[STORE_KEY].db_path == expected
    finally:
        db_path = Path(app[STORE_KEY].db_path)
        if db_path.exists():
            db_path.unlink()
