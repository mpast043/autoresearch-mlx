"""Jina Reader for web content extraction.

Provides markdown extraction from any URL using Jina's free reader API.
No API key required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from src.utils.circuit_breaker import CircuitOpenError, get_breaker

logger = logging.getLogger(__name__)

JINA_READER_URL = "https://r.jina.ai"


@dataclass
class JinaReadResult:
    """Result from Jina Reader."""
    url: str
    title: str
    markdown: str
    success: bool
    error: Optional[str] = None


async def read_url(url: str, timeout: float = 30.0) -> JinaReadResult:
    """Fetch a URL and extract its content as markdown.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds

    Returns:
        JinaReadResult with extracted content
    """
    if not url:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error="Empty URL",
        )

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error=f"Invalid URL: {url}",
        )
    if parsed.scheme not in ("http", "https"):
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error=f"Unsupported scheme: {parsed.scheme}",
        )
    # SSRF protection: reject private/internal IPs
    try:
        import ipaddress
        import socket
        hostname = parsed.hostname or ""
        resolved_ip = socket.getaddrinfo(hostname, None)[0][4][0]
        ip = ipaddress.ip_address(resolved_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return JinaReadResult(
                url=url,
                title="",
                markdown="",
                success=False,
                error="Private/internal IP addresses are not allowed",
            )
    except (socket.gaierror, ValueError):
        pass  # Hostname may not resolve; let the request proceed and fail naturally

    breaker = get_breaker("jina_reader", failure_threshold=5, recovery_timeout=60.0)
    try:
        return await breaker.call(_do_read_url, url, timeout)
    except CircuitOpenError:
        return JinaReadResult(
            url=url, title="", markdown="", success=False,
            error="Circuit breaker open — too many recent Jina failures",
        )
    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        error_msg = "Timeout" if isinstance(e, httpx.TimeoutException) else f"HTTP {e.response.status_code}"
        return JinaReadResult(
            url=url, title="", markdown="", success=False, error=error_msg,
        )
    except Exception as e:
        return JinaReadResult(
            url=url, title="", markdown="", success=False, error=str(e),
        )


async def _do_read_url(url: str, timeout: float = 30.0) -> JinaReadResult:
    """Actual HTTP call to Jina — called through the circuit breaker.

    Transient errors (timeouts, 5xx) are re-raised so the circuit breaker
    can track failures. Client errors (4xx) are returned as failures without
    tripping the breaker.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{JINA_READER_URL}/{url}")
            response.raise_for_status()
            # Response is plain text with title in header
            text = response.text
            title = ""
            markdown = text
            # Extract title from response
            if text.startswith("Title:"):
                lines = text.split("\n", 2)
                if len(lines) >= 1 and lines[0].startswith("Title:"):
                    title = lines[0][6:].strip()
                if len(lines) >= 3:
                    markdown = lines[2].strip()
            return JinaReadResult(
                url=url,
                title=title,
                markdown=markdown,
                success=True,
            )
    except httpx.TimeoutException:
        logger.warning("Jina Reader timeout for %s", url)
        raise  # Let circuit breaker track this failure
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            logger.warning("Jina Reader server error for %s: %s", url, e)
            raise  # Server error — trip the breaker
        logger.warning("Jina Reader HTTP error for %s: %s", url, e)
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error=f"HTTP {e.response.status_code}",
        )
    except Exception as e:
        logger.warning("Jina Reader error for %s: %s", url, e)
        raise  # Connection errors — trip the breaker


def read_url_sync(url: str, timeout: float = 30.0) -> JinaReadResult:
    """Synchronous version of read_url."""
    if not url:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error="Empty URL",
        )

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error=f"Invalid URL: {url}",
        )

    try:
        response = httpx.get(f"{JINA_READER_URL}/{url}", timeout=timeout)
        response.raise_for_status()
        text = response.text
        title = ""
        markdown = text
        if text.startswith("Title:"):
            lines = text.split("\n", 2)
            if len(lines) >= 1 and lines[0].startswith("Title:"):
                title = lines[0][6:].strip()
            if len(lines) >= 3:
                markdown = lines[2].strip()
        return JinaReadResult(
            url=url,
            title=title,
            markdown=markdown,
            success=True,
        )
    except httpx.TimeoutException:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error="Timeout",
        )
    except Exception as e:
        return JinaReadResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error=str(e),
        )


async def batch_read(urls: list[str], *, max_concurrent: int = 5, timeout: float = 30.0) -> list[JinaReadResult]:
    """Fetch multiple URLs in parallel.

    Args:
        urls: List of URLs to fetch
        max_concurrent: Max concurrent requests
        timeout: Per-request timeout

    Returns:
        List of JinaReadResult in same order as input URLs
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_fetch(url: str) -> JinaReadResult:
        async with semaphore:
            return await read_url(url, timeout)

    results = await asyncio.gather(*[limited_fetch(url) for url in urls], return_exceptions=True)

    output: list[JinaReadResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output.append(JinaReadResult(
                url=urls[i],
                title="",
                markdown="",
                success=False,
                error=str(result),
            ))
        else:
            output.append(result)

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.jina_reader <URL>")
        sys.exit(1)

    result = read_url_sync(sys.argv[1])
    if result.success:
        print(f"# {result.title}\n\n{result.markdown[:500]}")
    else:
        print(f"Error: {result.error}")