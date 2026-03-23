"""Tests for message queue and protocol."""

import asyncio
import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from messaging import MessageType, MessageQueue, create_message


def test_create_message_generates_uuid_and_timestamp():
    msg = create_message(
        from_agent="agent_a",
        to_agent="agent_b",
        msg_type=MessageType.FINDING,
        payload={"key": "value"},
        priority=3,
    )

    assert msg.msg_id
    assert isinstance(msg.msg_id, str)
    assert len(msg.msg_id) == 36
    assert isinstance(msg.timestamp, datetime)
    assert msg.from_agent == "agent_a"
    assert msg.to_agent == "agent_b"
    assert msg.msg_type == MessageType.FINDING
    assert msg.payload == {"key": "value"}
    assert msg.priority == 3


def test_message_type_enum_values():
    assert MessageType.FINDING.value == "FINDING"
    assert MessageType.VALIDATION.value == "VALIDATION"
    assert MessageType.BUILD_BRIEF.value == "BUILD_BRIEF"
    assert MessageType.BUILD_PREP.value == "BUILD_PREP"
    assert MessageType.RESULT.value == "RESULT"


def test_queue_priority_ordering_and_fifo():
    async def scenario():
        queue = MessageQueue()
        await queue.put(create_message("a", "b", MessageType.RESULT, {"name": "low"}, priority=5))
        await queue.put(create_message("a", "b", MessageType.RESULT, {"name": "high"}, priority=1))
        await queue.put(create_message("a", "b", MessageType.RESULT, {"name": "medium"}, priority=3))

        first = await queue.get()
        second = await queue.get()
        third = await queue.get()
        return first, second, third

    first, second, third = asyncio.run(scenario())
    assert first.payload["name"] == "high"
    assert second.payload["name"] == "medium"
    assert third.payload["name"] == "low"


def test_get_for_agent_selects_targeted_message():
    async def scenario():
        queue = MessageQueue()
        await queue.put(create_message("a", "x", MessageType.RESULT, {"id": 1}, priority=3))
        await queue.put(create_message("a", "target", MessageType.RESULT, {"id": 2}, priority=2))
        await queue.put(create_message("a", "y", MessageType.RESULT, {"id": 3}, priority=1))
        target = await queue.get_for_agent("target")
        remaining = [await queue.get(), await queue.get()]
        return target, remaining

    target, remaining = asyncio.run(scenario())
    assert target is not None
    assert target.payload["id"] == 2
    assert sorted(item.payload["id"] for item in remaining) == [1, 3]
