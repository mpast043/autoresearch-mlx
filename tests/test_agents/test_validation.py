"""Tests for the validation agent."""

import asyncio
import os
import sys
import tempfile

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.agents.base import AgentStatus
from src.agents.validation import ValidationAgent
from src.database import Database, Finding
from src.messaging import MessageQueue, MessageType, create_message


class MockValidationAgent(ValidationAgent):
    """Mock validation agent for testing with controlled gate scores."""

    def __init__(self, db, message_queue=None, mock_scores=None, config=None):
        super().__init__(db, message_queue, config=config)
        self.mock_scores = mock_scores or {"market": 0.7, "technical": 0.7, "distribution": 0.7}

    async def _check_market_proof(self, evidence_scores):
        return self.mock_scores.get("market", 0.7)

    async def _check_technical_feasibility(self, evidence_scores):
        return self.mock_scores.get("technical", 0.7)

    async def _check_distribution(self, evidence_scores):
        return self.mock_scores.get("distribution", 0.7)


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


def test_default_initialization(temp_db):
    agent = ValidationAgent(temp_db)

    assert agent.name == "validation"
    assert agent.status == AgentStatus.IDLE
    assert agent.db is temp_db
    assert agent._message_queue is not None
    assert agent.market_weight == 0.4
    assert agent.technical_weight == 0.35
    assert agent.distribution_weight == 0.25
    assert agent.gate_threshold == 0.6
    assert agent.overall_threshold == 0.7
    assert agent.promotion_threshold == 0.62
    assert agent.park_threshold == 0.48


def test_decision_thresholds_follow_validation_config(temp_db):
    agent = ValidationAgent(
        temp_db,
        config={
            "validation": {"decisions": {"promote_score": 0.7, "park_score": 0.5}},
            "orchestration": {"promotion_threshold": 0.9, "park_threshold": 0.2},
        },
    )

    assert agent.promotion_threshold == 0.7
    assert agent.park_threshold == 0.5


def test_top_level_validation_promotion_threshold_used(temp_db):
    agent = ValidationAgent(
        temp_db,
        config={"validation": {"promotion_threshold": 0.66, "park_threshold": 0.42}},
    )
    assert agent.promotion_threshold == 0.66
    assert agent.park_threshold == 0.42


def test_validate_finding_fails_when_missing_finding_id(temp_db):
    agent = ValidationAgent(temp_db, MessageQueue())

    result = asyncio.run(agent._validate_finding({}))

    assert result["success"] is False
    assert "finding_id" in result["error"]


def test_set_weights_updates_weights(temp_db):
    agent = ValidationAgent(temp_db, MessageQueue())

    result = asyncio.run(
        agent.process(
            create_message(
                from_agent="test",
                to_agent="validation",
                msg_type=MessageType.RESULT,
                payload={"command": "set_weights", "market_weight": 0.5, "technical_weight": 0.3, "distribution_weight": 0.2},
            )
        )
    )

    assert result == {"market_weight": 0.5, "technical_weight": 0.3, "distribution_weight": 0.2}


def test_process_validation_message_sends_orchestrator_result(temp_db):
    queue = MessageQueue()
    agent = MockValidationAgent(temp_db, queue)
    finding = Finding(
        source="test",
        source_url="https://example.com/thread",
        entrepreneur="Ops lead",
        product_built="Restore keeps failing",
        outcome_summary="Teams keep retrying restores manually when environments stay unreachable.",
        finding_kind="pain_point",
        source_class="pain_signal",
        status="qualified",
    )
    finding_id = temp_db.insert_finding(finding)

    result = asyncio.run(
        agent.process(
            create_message(
                from_agent="evidence",
                to_agent="validation",
                msg_type=MessageType.EVIDENCE,
                payload={
                    "finding_id": finding_id,
                    "source_class": "pain_signal",
                    "corroboration": {"corroboration_score": 0.7, "evidence_sufficiency": 0.65},
                    "market_enrichment": {"operational_buyer_score": 0.7, "cost_pressure_score": 0.6},
                },
            )
        )
    )

    assert result["success"] is True
    assert result["validation_id"] > 0
    assert result["decision"] in {"park", "kill", "promote"}
    persisted = temp_db.get_finding(finding_id)
    assert persisted is not None
    assert "high_leverage" in (persisted.evidence or {})
    signals = temp_db.get_raw_signals_by_finding(finding_id)
    assert signals
    assert "high_leverage" in (signals[0].metadata or {})

    queued = asyncio.run(queue.get_for_agent("orchestrator"))
    assert queued is not None
    assert queued.msg_type == MessageType.VALIDATION
    assert queued.payload["validation_id"] == result["validation_id"]


def test_validation_persists_recurrence_runtime_fields_for_reporting(temp_db):
    queue = MessageQueue()
    agent = ValidationAgent(temp_db, queue)
    finding = Finding(
        source="test",
        source_url="https://example.com/thread",
        entrepreneur="Ops lead",
        product_built="Restore keeps failing",
        outcome_summary="Teams keep retrying restores manually when environments stay unreachable.",
        finding_kind="pain_point",
        source_class="pain_signal",
        status="qualified",
    )
    finding_id = temp_db.insert_finding(finding)

    async def fake_validate_problem(**_kwargs):
        return {
            "problem_score": 0.24,
            "value_score": 0.49,
            "feasibility_score": 0.62,
            "solution_gap_score": 0.58,
            "saturation_score": 0.41,
            "evidence": {
                "recurrence_state": "timeout",
                "recurrence_timeout": True,
                "competitor_timeout": False,
                "recurrence_gap_reason": "recurrence_budget_timeout",
                "queries_considered": ['"restore failed" backup'],
                "queries_executed": [],
                "recurrence_budget_profile": {"query_limit": 2, "subreddit_limit": 2},
                "matched_docs_by_source": {
                    "web": [
                        {
                            "source_family": "web",
                            "source": "web",
                            "query_text": "spreadsheet cleanup workflow",
                            "normalized_url": "https://ops.example.com/manual-spreadsheet-cleanup",
                            "title": "Spreadsheet cleanup workflow still manual",
                            "snippet": "Teams still rely on manual csv cleanup after spreadsheet imports break.",
                            "match_class": "strong",
                        }
                    ]
                },
                "partial_docs_by_source": {
                    "web": [
                        {
                            "source_family": "web",
                            "source": "web",
                            "query_text": "manual handoff workflow",
                            "normalized_url": "https://ops.example.com/manual-handoff-workflow",
                            "title": "Manual handoff workflow creates follow-up misses",
                            "snippet": "Teams copy updates across files and lose track of ownership.",
                            "match_class": "partial",
                        }
                    ]
                },
            },
        }

    agent.toolkit.validate_problem = fake_validate_problem

    result = asyncio.run(
        agent.process(
            create_message(
                from_agent="evidence",
                to_agent="validation",
                msg_type=MessageType.EVIDENCE,
                payload={"finding_id": finding_id, "source_class": "pain_signal"},
            )
        )
    )

    assert result["success"] is True
    review = temp_db.get_validation_review(run_id=temp_db.active_run_id)
    assert len(review) == 1
    assert review[0]["recurrence_timeout"] is True
    assert review[0]["recurrence_state"] == "timeout"
    assert review[0]["recurrence_gap_reason"] == "recurrence_budget_timeout"
    assert review[0]["queries_executed"] == []
    assert review[0]["recurrence_budget_profile"]["query_limit"] == 2
    assert review[0]["matched_docs_by_source"]["web"][0]["normalized_url"] == "https://ops.example.com/manual-spreadsheet-cleanup"
    assert review[0]["matched_docs_by_source"]["web"][0]["match_class"] == "strong"
    assert review[0]["reviewable_recurrence_matches_by_source"]["web"][0]["match_class"] == "strong"
    assert review[0]["reviewable_recurrence_matches_by_source"]["web"][1]["match_class"] == "partial"
