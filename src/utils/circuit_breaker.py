"""Circuit breaker for external service calls.

Prevents cascading failures by opening the circuit after a threshold of
consecutive failures, then allowing periodic probe requests to test recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = auto()      # Normal operation — requests pass through
    OPEN = auto()        # Circuit tripped — requests fail fast
    HALF_OPEN = auto()   # Probing — one request allowed to test recovery


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and the call is rejected."""


class CircuitBreaker:
    """Thread-safe async circuit breaker with configurable thresholds.

    States:
        CLOSED  — Normal operation. Track failures; trip to OPEN after
                  ``failure_threshold`` consecutive failures.
        OPEN    — All calls fail fast with CircuitOpenError. After
                  ``recovery_timeout`` seconds, transition to HALF_OPEN.
        HALF_OPEN — One probe request is allowed. If it succeeds, close
                    the circuit. If it fails, re-open.

    Usage::

        breaker = CircuitBreaker("reddit_api", failure_threshold=5, recovery_timeout=30)

        async def fetch():
            return await breaker.call(external_api_request, arg1, arg2)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition from OPEN to HALF_OPEN)."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        Raises CircuitOpenError if the circuit is OPEN.
        """
        current_state = self.state
        if current_state == CircuitState.OPEN:
            logger.warning("Circuit %s is OPEN — rejecting call", self.name)
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        async with self._lock:
            # Re-check state after acquiring lock (another coroutine may have transitioned)
            current_state = self.state
            if current_state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous *func* through the circuit breaker via to_thread."""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            logger.warning("Circuit %s is OPEN — rejecting call", self.name)
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        async with self._lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")

        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    logger.info("Circuit %s: HALF_OPEN → CLOSED (probe succeeded)", self.name)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._success_count = 0

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit %s: HALF_OPEN → OPEN (probe failed)", self.name)
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                logger.warning(
                    "Circuit %s: CLOSED → OPEN (%d consecutive failures)",
                    self.name,
                    self._failure_count,
                )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        logger.info("Circuit %s: manually reset to CLOSED", self.name)

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self.state.name}, "
            f"failures={self._failure_count})"
        )


# ─── Pre-wired breakers for known external services ──────────────────────────

_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    First call creates it with *kwargs*; subsequent calls return the same instance.
    """
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, **kwargs)
    return _breakers[name]


def reset_all() -> None:
    """Reset all registered circuit breakers (useful in tests)."""
    for breaker in _breakers.values():
        breaker.reset()