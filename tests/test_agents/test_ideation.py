"""Tests for ideation agent database integration."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from src.agents.ideation import IdeationAgent
from src.database import Database, Finding, Idea, Opportunity, OpportunityCluster, Validation, ValidationExperiment
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
                "market_gap": {"market_gap": "underserved", "solution_gap_score": 0.72},
                "selection_status": "prototype_candidate",
                "selection_reason": "validated_selection_gate",
                "selection_gate": {"eligible": True, "reasons": ["multi_family_support"], "blocked_by": []},
                "counterevidence": [
                    {"claim": "The pain is rare or isolated.", "status": "contradicted", "summary": "Frequency score 0.72 clears the recurrence bar."}
                ],
                "opportunity_evaluation": {
                    "schema_version": "opportunity_evaluation_v1",
                    "policy": {"decision": "promote", "decision_reason": "validated_selection_gate"},
                    "selection": {
                        "selection_status": "prototype_candidate",
                        "selection_reason": "validated_selection_gate",
                        "selection_checks": {"eligible": True, "reasons": ["multi_family_support"], "blocked_by": []},
                    },
                    "evidence": {
                        "market_gap_state": "underserved",
                        "validation_plan": {"test_type": "workflow_walkthrough"},
                        "counterevidence": [
                            {
                                "claim": "The pain is rare or isolated.",
                                "status": "contradicted",
                                "summary": "Frequency score 0.72 clears the recurrence bar.",
                            }
                        ],
                    },
                    "inputs": {"validation": {"overall_score": 0.72}},
                },
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
    spec = ideas[0].spec
    assert spec["schema_version"] == "research_spec_v1"
    assert spec["artifact_type"] == "research_spec"
    assert spec["opportunity_scorecard"]["total_score"] == 0.72
    assert spec["opportunity_evaluation"]["schema_version"] == "opportunity_evaluation_v1"
    assert spec["market_gap"]["solution_gap_score"] == 0.72
    assert spec["selection_status"] == "prototype_candidate"
    assert spec["selection_gate"]["eligible"] is True
    assert spec["counterevidence"][0]["summary"] == "Frequency score 0.72 clears the recurrence bar."
    assert spec["source_validation"]["decision"] == "promote"
    assert sent_messages


def test_ideation_uses_validation_experiment_plan_property(temp_db: Database):
    queue = MessageBus()
    agent = IdeationAgent(temp_db, message_queue=queue, config={})

    finding_id = temp_db.insert_finding(
        Finding(
            source="reddit-problem/accounting",
            source_url="https://reddit.com/r/accounting/comments/2",
            product_built="QuickBooks reconciliation backlog",
            outcome_summary="The owner spends every weekend cleaning up imports.",
            content_hash="ideation-experiment-plan",
            status="promoted",
            finding_kind="problem_signal",
            source_class="pain_signal",
            evidence={"run_id": "test-run"},
        )
    )
    cluster_id = temp_db.upsert_cluster(
        OpportunityCluster(
            label="QuickBooks cleanup backlog",
            cluster_key="quickbooks-cleanup-backlog",
            summary={"segment": "small business finance"},
        )
    )
    opportunity_id = temp_db.upsert_opportunity(
        Opportunity(
            cluster_id=cluster_id,
            title="QuickBooks reconciliation backlog",
            market_gap="Thin tooling for owner-led cleanup",
            recommendation="promote",
            status="promoted",
            selection_status="prototype_candidate",
        )
    )
    temp_db.insert_experiment(
        ValidationExperiment(
            opportunity_id=opportunity_id,
            cluster_id=cluster_id,
            test_type="workflow_walkthrough",
            hypothesis="Owners will share the exact cleanup workflow.",
            falsifier="Nobody will walk through the process.",
            smallest_test="Run 5 cleanup walkthroughs.",
            success_signal="At least 3 owners share the real spreadsheet flow.",
            failure_signal="Owners refuse to show the current process.",
        )
    )
    validation_id = temp_db.insert_validation(
        Validation(
            finding_id=finding_id,
            run_id="test-run",
            overall_score=0.79,
            passed=True,
            evidence={
                "cluster": {"label": "QuickBooks cleanup backlog", "summary": {"segment": "finance owners"}},
                "opportunity_scorecard": {"total_score": 0.79, "decision": "promote"},
                "scores": {"feasibility_score": 0.66},
                "market_gap_state": "underserved",
                "experiment_id": 1,
            },
        )
    )

    result = asyncio.run(
        agent._generate_idea(
            {
                "validation_id": validation_id,
                "finding_id": finding_id,
                "opportunity_id": opportunity_id,
                "passed": True,
                "selection_status": "",
                "build_brief_id": 0,
            }
        )
    )

    assert result["success"] is True
    idea = temp_db.get_ideas(limit=1)[0]
    assert idea.spec["validation_plan"]["test_type"] == "workflow_walkthrough"
    assert "Run 5 cleanup walkthroughs." in idea.spec["core_features"][2]


def test_ideation_updates_existing_idea_using_pattern_id_list_compatibility(temp_db: Database):
    queue = MessageBus()
    agent = IdeationAgent(temp_db, message_queue=queue, config={})

    finding_id = temp_db.insert_finding(
        Finding(
            source="reddit-problem/accounting",
            source_url="https://reddit.com/r/accounting/comments/3",
            product_built="Usage-based invoice audit",
            outcome_summary="Finance ops manually audits usage rows before invoices go out.",
            content_hash="ideation-pattern-list",
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
            overall_score=0.74,
            passed=True,
            evidence={
                "cluster": {"label": "Usage-based invoice audit", "summary": {"segment": "finance operations"}},
                "opportunity_scorecard": {"total_score": 0.74, "decision": "promote"},
                "scores": {"feasibility_score": 0.62},
                "market_gap_state": "underserved",
            },
        )
    )

    existing_id = temp_db.insert_idea(
        Idea(
            slug="usage-based-invoice-audit-brief",
            title="Usage-based invoice audit Brief",
            description="Existing brief",
            pattern_ids=json.dumps([999]),
            confidence_score=0.4,
            spec_json=json.dumps({"slug": "usage-based-invoice-audit-brief", "core_features": []}),
        )
    )

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
    assert result["refined"] is True
    assert result["idea_id"] == existing_id

    updated = temp_db.get_idea(existing_id)
    assert updated is not None
    assert updated.pattern_id_list == sorted([999, finding_id])
