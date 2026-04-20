from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(Enum):
    FINDING = "FINDING"
    FINDING_UNSEEDED = "FINDING_UNSEEDED"  # skip evidence stage, route to validation directly
    EVIDENCE = "EVIDENCE"
    VALIDATION = "VALIDATION"
    BUILD_BRIEF = "BUILD_BRIEF"
    BUILD_PREP = "BUILD_PREP"
    IDEA = "IDEA"
    BUILD_REQUEST = "BUILD_REQUEST"
    RESOURCE_REQUEST = "RESOURCE_REQUEST"
    RESULT = "RESULT"
    ERROR = "ERROR"
    SECURITY_SCAN = "SECURITY_SCAN"
    DOC_GENERATION = "DOC_GENERATION"
    HEALTH_CHECK = "HEALTH_CHECK"


@dataclass
class Message:
    msg_id: str
    from_agent: str
    to_agent: str
    msg_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int


class MessageQueue:
    """Legacy single-queue implementation kept for backward compatibility."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, Message]] = asyncio.PriorityQueue()
        self._counter = 0
        self._selection_lock = asyncio.Lock()

    async def put(self, message: Message) -> None:
        self._counter += 1
        await self._queue.put((message.priority, self._counter, message))

    async def get(self) -> Message:
        _, _, message = await self._queue.get()
        return message

    async def get_for_agent(self, agent_name: str) -> Optional[Message]:
        async with self._selection_lock:
            if self._queue.empty():
                return None

            held: list[tuple[int, int, Message]] = []
            target: Optional[Message] = None

            while not self._queue.empty():
                item = self._queue.get_nowait()
                message = item[2]
                if target is None and message.to_agent == agent_name:
                    target = message
                    break
                held.append(item)

            for item in held:
                self._queue.put_nowait(item)

            return target

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


class MessageBus:
    """Per-agent queue implementation for O(1) retrieval and natural backpressure."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.PriorityQueue[tuple[int, int, Message]]] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

    async def _get_queue(self, agent_name: str) -> asyncio.PriorityQueue[tuple[int, int, Message]]:
        """Get or create queue for an agent."""
        if agent_name not in self._queues:
            async with self._lock:
                if agent_name not in self._queues:
                    self._queues[agent_name] = asyncio.PriorityQueue()
        return self._queues[agent_name]

    async def send(self, message: Message) -> None:
        """Send a message to its recipient's queue with priority."""
        queue = await self._get_queue(message.to_agent)
        async with self._lock:
            self._counter += 1
            counter = self._counter
        await queue.put((message.priority, counter, message))

    async def put(self, message: Message) -> None:
        """Alias for send() for backward compatibility."""
        await self.send(message)

    async def get_for_agent(self, agent_id: str) -> Optional[Message]:
        """Get next message for agent, or None if queue empty."""
        queue = await self._get_queue(agent_id)
        try:
            item = queue.get_nowait()
            if isinstance(item, tuple):
                return item[2]
            return item
        except asyncio.QueueEmpty:
            return None

    async def receive(self, agent_name: str) -> Message:
        """Receive a message from agent's queue (blocks until available)."""
        queue = await self._get_queue(agent_name)
        item = await queue.get()
        if isinstance(item, tuple):
            return item[2]
        return item

    async def receive_nowait(self, agent_name: str) -> Optional[Message]:
        """Receive immediately or return None if empty."""
        queue = await self._get_queue(agent_name)
        try:
            item = queue.get_nowait()
            if isinstance(item, tuple):
                return item[2]
            return item
        except asyncio.QueueEmpty:
            return None

    def empty(self, agent_name: str | None = None) -> bool:
        """Check if agent's queue is empty. If agent_name is None, check if ALL queues are empty."""
        if agent_name is not None:
            return agent_name not in self._queues or self._queues[agent_name].empty()
        # Check if ALL queues are empty
        return all(q.empty() for q in self._queues.values())

    def qsize(self, agent_name: str | None = None) -> int:
        """Get size of agent's queue. If agent_name is None, return total across all queues."""
        if agent_name is not None:
            if agent_name not in self._queues:
                return 0
            return self._queues[agent_name].qsize()
        return sum(q.qsize() for q in self._queues.values())

    async def register_agent(self, agent_name: str) -> None:
        """Register an agent to ensure its queue exists."""
        await self._get_queue(agent_name)

    def registered_agents(self) -> list[str]:
        """List all registered agents."""
        return list(self._queues.keys())


def create_message(
    from_agent: str,
    to_agent: str,
    msg_type: MessageType,
    payload: Dict[str, Any],
    priority: int = 3,
) -> Message:
    return Message(
        msg_id=str(uuid.uuid4()),
        from_agent=from_agent,
        to_agent=to_agent,
        msg_type=msg_type,
        payload=payload,
        timestamp=datetime.now(),
        priority=priority,
    )


# Compatibility aliases for older code and docs.
AgentMessage = Message
