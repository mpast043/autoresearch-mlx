"""Tests for ideation agent database integration."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from src.agents.ideation import IdeationAgent
from src.database import Database, Finding, Validation
from src.messaging import MessageBus


@pytest.fixture
def temp_db():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        yield db
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_ideation_generates_research_brief_with_db_validation_accessor(temp_db: Database):
    queue = MessageBus()
    agent = IdeationAgent(temp_db, message_queue=queue, config={})

    finding_id = temp_db.insert_finding(
        Finding(
            source="reddit-problem/accounting",
            source_url="https://reddit.com/r/accounting/comments/1",
            product_built="Stripe to QuickBooks reconciliation drift",
            outcome_summary="Finance ops manually rebuild payout ledgers before monthly close.",
            content_hash="ideation-finding",
            status="promoted",
            finding_kind="problem_signal",
            source_class="pain_signal",
            evidence={"run_id": "test-run"},
        )
    )
    validation_id = temp_db.insert_validation(
        Validation(
            finding_id=finding_id,
            run_id="test-run",
            overall_score=0.72,
            passed=True,
            evidence={
                "cluster": {"label": "Stripe/QBO drift", "summary": {"segment": "finance operations"}},
                "opportunity_scorecard": {"total_score": 0.72, "decision": "promote"},
                "scores": {"feasibility_score": 0.61},
                "market_gap_state": "underserved",
            },
        )
    )

    sent_messages: list[dict] = []

    async def _capture_send_message(**kwargs):
        sent_messages.append(kwargs)

    agent.send_message = _capture_send_message  # type: ignore[method-assign]

    result = asyncio.run(
        agent._generate_idea(
            {
                "validation_id": validation_id,
                "finding_id": finding_id,
                "opportunity_id": 0,
                "passed": True,
                "selection_status": "",
                "build_brief_id": 0,
            }
        )
    )

    assert result["success"] is True
    ideas = temp_db.get_ideas(limit=5)
    assert ideas
    assert ideas[0].title.endswith("Brief")
    assert sent_messages
