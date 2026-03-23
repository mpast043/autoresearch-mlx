"""Optional HTTP bridge client for Reddit discovery via a Devvit-hosted service."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import aiohttp


def _resolve_env(value: str | None) -> str:
    raw = (value or "").strip()
    match = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", raw)
    if match:
        return os.getenv(match.group(1), "")
    return raw


@dataclass
class BridgeError(Exception):
    """Structured bridge failure used to drive transparent fallback behavior."""

    code: str
    message: str
    status: int = 0

    def __str__(self) -> str:
        return self.message


def normalize_reddit_item(item: dict[str, Any], *, expected_kind: str | None = None) -> dict[str, Any]:
    normalized = {
        "id": str(item.get("id", "") or ""),
        "kind": str(item.get("kind", "") or ""),
        "subreddit": str(item.get("subreddit", "") or ""),
        "title": str(item.get("title", "") or ""),
        "body": str(item.get("body", "") or ""),
        "author": str(item.get("author", "") or ""),
        "permalink": str(item.get("permalink", "") or ""),
        "score": int(item.get("score", 0) or 0),
        "num_comments": int(item.get("num_comments", 0) or 0),
        "created_utc": int(item.get("created_utc", 0) or 0),
        "source_type": str(item.get("source_type", "") or ""),
        "post_id": str(item.get("post_id", "") or ""),
        "parent_id": str(item.get("parent_id", "") or ""),
    }
    if expected_kind and normalized["kind"] != expected_kind:
        raise BridgeError("bad_response_shape", f"expected {expected_kind} item but got {normalized['kind']!r}")
    if not normalized["id"] or normalized["source_type"] != "reddit":
        raise BridgeError("bad_response_shape", "bridge item is missing required reddit identity fields")
    if normalized["kind"] == "post" and not normalized["post_id"]:
        normalized["post_id"] = normalized["id"]
    return normalized


class RedditBridgeClient:
    """Small async client for the read-only Reddit bridge endpoints."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.base_url = _resolve_env(config.get("base_url", "")).rstrip("/")
        self.auth_token = _resolve_env(config.get("auth_token", ""))
        self.timeout_seconds = float(config.get("timeout_seconds", 20))
        self.enabled = bool(config.get("enabled", False) and self.base_url)

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise BridgeError("bridge_disabled", "reddit bridge is disabled")
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    headers=self._auth_headers(),
                ) as response:
                    try:
                        body = await response.json()
                    except aiohttp.ContentTypeError as exc:
                        raise BridgeError(
                            "bad_response_shape",
                            "bridge returned non-json response",
                            response.status,
                        ) from exc
        except TimeoutError as exc:
            raise BridgeError("bridge_timeout", "reddit bridge request timed out") from exc
        except aiohttp.ClientError as exc:
            raise BridgeError("bridge_unavailable", f"reddit bridge request failed: {exc}") from exc

        if response.status == 401:
            raise BridgeError("auth_failed", "reddit bridge rejected the bearer token", 401)
        if response.status >= 500:
            message = body.get("error") if isinstance(body, dict) else "bridge upstream error"
            raise BridgeError("upstream_reddit_failure", str(message), response.status)
        if response.status >= 400:
            message = body.get("error") if isinstance(body, dict) else "bridge bad request"
            code = body.get("error_code") if isinstance(body, dict) else "bridge_bad_request"
            raise BridgeError(str(code), str(message), response.status)
        if not isinstance(body, dict):
            raise BridgeError("bad_response_shape", "bridge returned invalid response shape", response.status)
        if not body.get("ok"):
            raise BridgeError(
                str(body.get("error_code", "bridge_error")),
                str(body.get("error", "bridge returned error")),
                response.status,
            )
        return body

    async def search_posts(
        self,
        *,
        subreddit: str,
        query: str,
        limit: int = 2,
        sort: str = "relevance",
        cursor: str = "",
    ) -> tuple[list[dict[str, Any]], str]:
        payload = await self._post(
            "/api/reddit/search-posts",
            {
                "subreddit": subreddit,
                "query": query,
                "limit": limit,
                "sort": sort,
                "cursor": cursor,
            },
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise BridgeError("bad_response_shape", "search response is missing items")
        return (
            [normalize_reddit_item(item, expected_kind="post") for item in items],
            str(payload.get("next_cursor", "") or ""),
        )

    async def get_post_thread(
        self,
        *,
        url: str,
        comment_limit: int = 8,
        depth: int = 4,
    ) -> dict[str, Any]:
        payload = await self._post(
            "/api/reddit/post-thread",
            {
                "url": url,
                "comment_limit": comment_limit,
                "depth": depth,
            },
        )
        post = payload.get("post")
        comments = payload.get("comments")
        if not isinstance(post, dict) or not isinstance(comments, list):
            raise BridgeError("bad_response_shape", "thread response is missing post or comments")
        return {
            "post": normalize_reddit_item(post, expected_kind="post"),
            "comments": [normalize_reddit_item(comment, expected_kind="comment") for comment in comments],
        }

    async def get_comments(
        self,
        *,
        url: str,
        limit: int = 8,
        depth: int = 4,
    ) -> list[dict[str, Any]]:
        payload = await self._post(
            "/api/reddit/comments",
            {
                "url": url,
                "limit": limit,
                "depth": depth,
            },
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise BridgeError("bad_response_shape", "comments response is missing items")
        return [normalize_reddit_item(item, expected_kind="comment") for item in items]

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            raise BridgeError("bridge_disabled", "reddit bridge is disabled")
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{self.base_url}/api/health",
                    headers=self._auth_headers(),
                ) as response:
                    payload = await response.json()
        except TimeoutError as exc:
            raise BridgeError("bridge_timeout", "reddit bridge health check timed out") from exc
        except aiohttp.ClientError as exc:
            raise BridgeError("bridge_unavailable", f"reddit bridge health check failed: {exc}") from exc
        if response.status >= 400:
            raise BridgeError("bridge_unhealthy", "reddit bridge health check failed", response.status)
        if not isinstance(payload, dict):
            raise BridgeError("bad_response_shape", "health endpoint returned invalid response")
        return payload
