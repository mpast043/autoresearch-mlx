"""Base agent class with lifecycle management."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from src.messaging import Message, MessageBus, MessageQueue, MessageType, create_message


class AgentStatus(Enum):
    """Status states for an agent lifecycle."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, message_queue: Optional[MessageBus | MessageQueue] = None):
        self.name = name
        self.status = AgentStatus.IDLE
        self._message_queue = message_queue if message_queue else MessageBus()
        self._error_count = 0
        self._max_errors = 3
        self._task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._processing_count = 0

    async def start(self) -> None:
        if self.status == AgentStatus.RUNNING:
            raise RuntimeError(f"Agent {self.name} is already running")

        self.status = AgentStatus.RUNNING
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.status = AgentStatus.STOPPED
        self._pause_event.set()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def reset(self) -> None:
        """Reset agent from ERROR state back to IDLE, clearing error count."""
        if self.status != AgentStatus.ERROR:
            return
        self._error_count = 0
        self.status = AgentStatus.IDLE
        self._pause_event.set()

    async def pause(self) -> None:
        if self.status != AgentStatus.RUNNING:
            return

        self.status = AgentStatus.PAUSED
        self._pause_event.clear()

    async def resume(self) -> None:
        if self.status != AgentStatus.PAUSED:
            raise RuntimeError(f"Agent {self.name} is not paused (current status: {self.status.value})")

        self.status = AgentStatus.RUNNING
        self._pause_event.set()

    async def _run_loop(self) -> None:
        while self.status in (AgentStatus.RUNNING, AgentStatus.PAUSED):
            try:
                await self._pause_event.wait()

                if self.status == AgentStatus.STOPPED:
                    break

                message = await self._message_queue.receive(self.name)

                self._processing_count += 1
                try:
                    await self.process(message)
                finally:
                    self._processing_count = max(0, self._processing_count - 1)

            except asyncio.CancelledError:
                break
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Agent %s error in run loop", self.name)
                self._error_count += 1
                if self._error_count >= self._max_errors:
                    self.status = AgentStatus.ERROR
                    break

    @abstractmethod
    async def process(self, message) -> Dict[str, Any]:
        pass

    def busy_count(self) -> int:
        return int(self._processing_count)

    async def send_result(self, to_agent: str, result: Dict[str, Any]) -> None:
        msg = create_message(
            from_agent=self.name,
            to_agent=to_agent,
            msg_type=MessageType.RESULT,
            payload=result,
            priority=3,
        )
        await self._message_queue.put(msg)

    async def send_message(
        self,
        to_agent: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 3,
    ) -> None:
        msg = create_message(
            from_agent=self.name,
            to_agent=to_agent,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        await self._message_queue.put(msg)
