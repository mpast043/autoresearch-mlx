"""Retry utilities with exponential backoff.

This module provides retry functionality with exponential backoff for handling
transient errors like network timeouts and rate limits.
"""

from __future__ import annotations

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


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Execute a function with exponential backoff retry logic."""
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

            delay = min(
                base_delay * 2 ** (attempt - 1) + random.uniform(0, 1),
                max_delay,
            )
            logger.warning(
                "Transient error on attempt %d/%d for %s: %s. Retrying in %.2f seconds...",
                attempt,
                max_retries,
                func_name,
                exc,
                delay,
            )
            time.sleep(delay)


def retry_decorator(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorate a callable with retry-with-backoff behavior."""

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
