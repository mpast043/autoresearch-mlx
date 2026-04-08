"""Tests for the circuit breaker utility."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_breaker,
    reset_all,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_successful_call_stays_closed(self):
        cb = CircuitBreaker("test")
        func = AsyncMock(return_value="ok")
        result = await cb.call(func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_increments_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED  # Not yet at threshold

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await cb.call(func)

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        fail_func = AsyncMock(side_effect=RuntimeError("fail"))
        ok_func = AsyncMock(return_value="ok")

        with pytest.raises(RuntimeError):
            await cb.call(fail_func)
        assert cb.failure_count == 1

        await cb.call(ok_func)
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        fail_func = AsyncMock(side_effect=RuntimeError("fail"))
        ok_func = AsyncMock(return_value="ok")

        with pytest.raises(RuntimeError):
            await cb.call(fail_func)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        result = await cb.call(ok_func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        fail_func = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError):
            await cb.call(fail_func)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            await cb.call(fail_func)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_call_sync(self):
        cb = CircuitBreaker("test")
        func = Mock(return_value="sync_ok")
        result = await cb.call_sync(func)
        assert result == "sync_ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_call_sync_rejects_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        func = Mock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call_sync(func)
        with pytest.raises(CircuitOpenError):
            await cb.call_sync(func)


class TestGetBreaker:
    """Tests for the get_breaker registry."""

    def setup_method(self):
        reset_all()

    def test_returns_same_instance(self):
        cb1 = get_breaker("reddit")
        cb2 = get_breaker("reddit")
        assert cb1 is cb2

    def test_different_names_different_instances(self):
        cb1 = get_breaker("reddit")
        cb2 = get_breaker("github")
        assert cb1 is not cb2

    @pytest.mark.asyncio
    async def test_reset_all(self):
        cb = get_breaker("test", failure_threshold=1)
        func = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            await cb.call(func)
        assert cb.state == CircuitState.OPEN

        reset_all()
        assert cb.state == CircuitState.CLOSED