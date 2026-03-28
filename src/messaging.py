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
MessageBus = MessageQueue
