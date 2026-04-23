"""Tests for canonical opportunity/research spec construction."""

from __future__ import annotations

from types import SimpleNamespace

from src.opportunity_spec import build_research_spec


def test_build_research_spec_preserves_validation_contract():
    validation = SimpleNamespace(id=123, passed=True, overall_score=0.72)
    evidence = {
        "decision": "",
        "decision_reason": "",
        "market_gap_state": "stale_market_gap_state",
        "market_gap": {"market_gap": "underserved_edge_case"},
        "opportunity_scorecard": {"decision_score": 0.42},
        "evidence_assessment": {"evidence_quality": 0.61},
        "selection_status": "research_more",
        "selection_reason": "stale_selection_reason",
        "selection_gate": {"eligible": True, "blocked_by": []},
        "counterevidence": [{"claim": "stale", "status": "supported"}],
        "opportunity_evaluation": {
            "schema_version": "opportunity_evaluation_v1",
            "inputs": {"validation": {"overall_score": 0.81}},
            "evidence": {
                "market_gap_state": "underserved_edge_case",
                "validation_plan": {"test_type": "workflow_walkthrough"},
                "counterevidence": [{"claim": "rare", "status": "contradicted"}],
            },
            "policy": {
                "decision": "promote",
                "decision_reason": "validated_selection_gate",
            },
            "selection": {
                "selection_status": "prototype_candidate",
                "selection_reason": "validated_selection_gate",
                "selection_checks": {"eligible": True, "blocked_by": []},
            },
        },
    }

    spec = build_research_spec(
        slug="stripe-qbo-drift",
        product_type="research-brief",
        problem_statement="Stripe to QBO reconciliation drift",
        value_hypothesis="Finance teams need less manual cleanup.",
        core_features=["Scorecard", "Validation plan"],
        audience="finance operations",
        monetization_strategy="Validate before build",
        source_finding_kind="problem_signal",
        validation=validation,
        evidence=evidence,
        validation_plan={"test_type": "workflow_walkthrough"},
    )

    assert spec["schema_version"] == "research_spec_v1"
    assert spec["artifact_type"] == "research_spec"
    assert spec["evidence_refresh_from_validation_id"] == 123
    assert spec["opportunity_scorecard"]["decision_score"] == 0.42
    assert spec["opportunity_evaluation"]["schema_version"] == "opportunity_evaluation_v1"
    assert spec["market_gap_state"] == "underserved_edge_case"
    assert spec["validation_plan"]["test_type"] == "workflow_walkthrough"
    assert spec["selection_status"] == "prototype_candidate"
    assert spec["selection_gate"]["eligible"] is True
    assert spec["source_validation"] == {
        "decision": "promote",
        "decision_reason": "validated_selection_gate",
        "passed": True,
        "overall_score": 0.81,
    }
