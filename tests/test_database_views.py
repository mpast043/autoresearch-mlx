"""Tests for canonical-first database read-model helpers."""

from __future__ import annotations

from src.database_views import build_recent_validation_row, build_validation_review_row


def test_build_recent_validation_row_prefers_canonical_decision():
    row = {"passed": 0}
    evidence = {
        "decision": "park",
        "opportunity_evaluation": {
            "policy": {"decision": "promote"},
        },
    }

    result = build_recent_validation_row(row, evidence)

    assert result["decision"] == "promote"


def test_build_validation_review_row_prefers_canonical_snapshot_fields():
    row = {
        "id": 9,
        "run_id": "test-run",
        "finding_id": 4,
        "overall_score": 0.31,
        "market_score": 0.4,
        "technical_score": 0.5,
        "distribution_score": 0.3,
        "source": "reddit",
        "source_url": "https://example.com/thread",
        "source_class": "pain_signal",
        "validated_at": "2026-04-23T20:00:00",
        "outcome_summary": "Manual ops handoff drift",
        "product_built": "",
    }
    evidence = {
        "decision": "kill",
        "decision_reason": "stale_reason",
        "selection_status": "archive",
        "selection_reason": "stale_selection",
        "recurrence_state": "weak",
        "family_confirmation_count": 0,
        "opportunity_evaluation": {
            "inputs": {
                "validation": {"overall_score": 0.57},
            },
            "measures": {
                "dimensions": {"value_support": 0.43},
                "transition": {"composite_score": 0.29},
            },
            "evidence": {
                "recurrence_state": "supported",
                "family_confirmation_count": 2,
            },
            "policy": {
                "decision": "park",
                "decision_reason": "park_recurrence",
            },
            "selection": {
                "selection_status": "research_more",
                "selection_reason": "selection_gate_not_met",
            },
        },
    }

    result = build_validation_review_row(
        row,
        evidence=evidence,
        finding_evidence={"title": "Manual ops handoff drift"},
    )

    assert result["decision"] == "park"
    assert result["decision_reason"] == "park_recurrence"
    assert result["selection_status"] == "research_more"
    assert result["selection_reason"] == "selection_gate_not_met"
    assert result["decision_score"] == 0.0
    assert result["problem_truth_score"] == 0.0
    assert result["revenue_readiness_score"] == 0.0
    assert result["composite_score"] == 0.29
    assert result["value_support"] == 0.43
    assert result["recurrence_state"] == "supported"
    assert result["family_confirmation_count"] == 2
