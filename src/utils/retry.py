"""Retry utilities with exponential backoff.

This module provides retry functionality with exponential backoff for handling
transient errors like network timeouts and rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class TransientError(Exception):
    """Exception for errors that may succeed when retried."""


class PermanentError(Exception):
    """Exception for errors that will not resolve with retry."""


def _calculate_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Calculate exponential backoff delay with jitter."""
    return min(
        base_delay * 2 ** (attempt - 1) + random.uniform(0, 1),
        max_delay,
    )


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Execute a function with exponential backoff retry logic (sync version)."""
    attempt = 0
    func_name = getattr(func, "__name__", repr(func))

    while True:
        try:
            return func()
        except PermanentError:
            logger.error("Permanent error occurred, not retrying: %s", func_name)
            raise
        except TransientError as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Max retries (%d) exceeded for %s: %s",
                    max_retries,
                    func_name,
                    exc,
                )
                raise

            delay = _calculate_delay(attempt, base_delay, max_delay)
            logger.warning(
                "Transient error on attempt %d/%d for %s: %s. Retrying in %.2f seconds...",
                attempt,
                max_retries,
                func_name,
                exc,
                delay,
            )
            time.sleep(delay)


async def retry_with_backoff_async(
    func: Callable[..., Any],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Execute an async function with exponential backoff retry logic."""
    attempt = 0
    func_name = getattr(func, "__name__", repr(func))

    while True:
        try:
            return await func()
        except PermanentError:
            logger.error("Permanent error occurred, not retrying: %s", func_name)
            raise
        except TransientError as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Max retries (%d) exceeded for %s: %s",
                    max_retries,
                    func_name,
                    exc,
                )
                raise

            delay = _calculate_delay(attempt, base_delay, max_delay)
            logger.warning(
                "Transient error on attempt %d/%d for %s: %s. Retrying in %.2f seconds...",
                attempt,
                max_retries,
                func_name,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


def retry_decorator(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorate a callable with retry-with-backoff behavior (sync version)."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def wrapped_func() -> Any:
                return func(*args, **kwargs)

            return retry_with_backoff(
                wrapped_func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_decorator_async(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorate an async callable with retry-with-backoff behavior."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async def wrapped_func() -> Any:
                return await func(*args, **kwargs)

            return await retry_with_backoff_async(
                wrapped_func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )

        return wrapper  # type: ignore[return-value]

    return decorator