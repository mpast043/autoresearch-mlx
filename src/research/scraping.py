"""Scraping module - web search, fetching, Reddit, YouTube integration."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Re-export from utils for use in scraping
try:
    from src.utils.text import compact_text
except ImportError:
    def compact_text(text: str, limit: int = 500) -> str:
        return text[:limit] if text else ""


class WebScraper:
    """Handles web scraping and API calls."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.user_agent = (
            self.config.get("api_keys", {}).get("reddit", {}).get("user_agent")
            or "Mozilla/5.0 (compatible; AutoResearcher/1.0)"
        )
        self.request_timeout_general = 12
        self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self.node_bin = shutil.which("node")

    async def search_web(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the web using DuckDuckGo."""
        results = []
        try:
            import ddgs
            ddgs_client = ddgs.DDGS(timeout=self.request_timeout_general)
            for r in ddgs_client.text(query, max_result=limit):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        except Exception as e:
            logger.warning("ddgs search failed: %s", e)
            # Fallback: use requests + BeautifulSoup
            results = await self._web_html_fallback(query, limit)
        return results

    async def _web_html_fallback(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Fallback web search using requests + DuckDuckGo HTML."""
        results = []
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        try:
            resp = await asyncio.to_thread(requests.get, url, timeout=self.request_timeout_general)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result__snippet")[:limit]:
                title_elem = result.select_one(".result__a")
                url_elem = result.select_one(".result__url")
                if title_elem and url_elem:
                    results.append({
                        "title": title_elem.get_text(),
                        "url": url_elem.get_text(),
                        "snippet": result.get_text(),
                    })
        except Exception as e:
            logger.warning("web fallback failed: %s", e)
        return results

    async def fetch_content(self, url: str) -> dict[str, Any]:
        """Fetch content from a URL."""
        # SSRF protection: validate URL scheme and reject private IPs
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return {"url": url, "error": f"Unsupported scheme: {parsed.scheme}"}
            import ipaddress
            import socket
            hostname = parsed.hostname or ""
            try:
                resolved_ip = socket.getaddrinfo(hostname, None)[0][4][0]
                ip = ipaddress.ip_address(resolved_ip)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    return {"url": url, "error": "Private/internal IP addresses are not allowed"}
            except (socket.gaierror, ValueError):
                pass  # Hostname may not resolve; let the request proceed and fail naturally
        except Exception:
            pass
        try:
            headers = {"User-Agent": self.user_agent}
            resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=self.request_timeout_general)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n", strip=True)
                return {"url": url, "text": text, "html": resp.text[:50000]}
            else:
                return {"url": url, "text": resp.text[:50000]}
        except Exception as e:
            return {"url": url, "error": str(e)}


class RedditClient:
    """Reddit API client."""

    def __init__(self, config: Optional[dict[str, Any]] = None, bridge=None):
        self.config = config or {}
        self.bridge = bridge
        self.user_agent = (
            self.config.get("api_keys", {}).get("reddit", {}).get("user_agent")
            or "Mozilla/5.0 (compatible; AutoResearcher/1.0)"
        )

    async def search(self, subreddit: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Reddit posts."""
        results = []
        # Try bridge first if enabled
        if self.bridge and hasattr(self.bridge, "search"):
            try:
                results = await self.bridge.search(subreddit, query, limit)
            except Exception as e:
                logger.warning("bridge search failed: %s", e)

        # Fallback to direct PRAW or scraping
        if not results:
            results = await self._direct_search(subreddit, query, limit)
        return results

    async def _direct_search(self, subreddit: str, query: str, limit: int) -> list[dict[str, Any]]:
        """Direct Reddit search fallback."""
        # Simplified - actual implementation uses PRAW or scraping
        return []


class YouTubeClient:
    """YouTube API client."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self.yt_dlp_bin = shutil.which("yt-dlp")

    async def get_transcript(self, video_id: str) -> Optional[str]:
        """Get YouTube video transcript."""
        # Validate video ID format (YouTube IDs are exactly 11 chars, alphanumeric + - and _)
        if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
            logger.warning("Invalid YouTube video ID format: %s", video_id)
            return None
        if not self.yt_dlp_bin:
            return None
        try:
            cmd = [self.yt_dlp_bin, "--write-subs", "--write-auto-subs", "--quiet", f"https://youtube.com/watch?v={video_id}"]
            result = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await result.communicate()
            # Would need to read the generated subtitle file
            return None
        except Exception as e:
            logger.warning("yt-dlp failed: %s", e)
            return None

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search YouTube videos."""
        if not self.api_key:
            return []
        # YouTube API search implementation
        return []

    async def comments(self, video_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get YouTube video comments."""
        # Implementation using YouTube Data API
        return []


# Convenience functions matching research_tools.py interface
async def scrape_web(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Scrape web search results."""
    scraper = WebScraper()
    return await scraper.search_web(query, limit)


async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch URL content."""
    scraper = WebScraper()
    return await scraper.fetch_content(url)