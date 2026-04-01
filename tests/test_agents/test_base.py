"""Tests for base agent lifecycle behavior."""

import asyncio
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.agents.base import AgentStatus, BaseAgent
from src.messaging import MessageType, create_message


class DummyAgent(BaseAgent):
    def __init__(self):
        super().__init__("dummy")
        self.processed = []

    async def process(self, message):
        self.processed.append(message.payload)
        return {"ok": True}


def test_agent_start_pause_resume_stop_lifecycle():
    async def scenario():
        agent = DummyAgent()
        await agent.start()
        assert agent.status == AgentStatus.RUNNING
        await agent.pause()
        assert agent.status == AgentStatus.PAUSED
        await agent.resume()
        assert agent.status == AgentStatus.RUNNING
        await agent.stop()
        assert agent.status == AgentStatus.STOPPED

    asyncio.run(scenario())


def test_agent_processes_targeted_message():
    async def scenario():
        agent = DummyAgent()
        await agent.start()
        await agent._message_queue.put(
            create_message("sender", "dummy", MessageType.RESULT, {"id": 1}, priority=1)
        )
        await asyncio.sleep(0.2)
        await agent.stop()
        return agent.processed

    processed = asyncio.run(scenario())
    assert processed == [{"id": 1}]


def test_agent_skips_non_targeted_message_without_starving_targeted_work():
    async def scenario():
        agent = DummyAgent()
        await agent.start()
        await agent._message_queue.put(
            create_message("sender", "someone_else", MessageType.RESULT, {"id": 99}, priority=1)
        )
        await agent._message_queue.put(
            create_message("sender", "dummy", MessageType.RESULT, {"id": 2}, priority=2)
        )
        await asyncio.sleep(0.2)
        remaining = await agent._message_queue.get_for_agent("someone_else")
        await agent.stop()
        return agent.processed, remaining

    processed, remaining = asyncio.run(scenario())
    assert processed == [{"id": 2}]
    assert remaining is not None
    assert remaining.payload == {"id": 99}
