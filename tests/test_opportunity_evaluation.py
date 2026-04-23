"""Tests for the canonical opportunity evaluation contract."""

from __future__ import annotations

from src.opportunity_evaluation import (
    OPPORTUNITY_EVALUATION_SCHEMA_VERSION,
    build_opportunity_evaluation,
    v2_lite_shadow_score,
)


def test_build_opportunity_evaluation_enforces_contract_boundaries():
    evaluation = build_opportunity_evaluation(
        run_id="run-123",
        finding_id=11,
        cluster_id=22,
        opportunity_id=33,
        validation_id=44,
        source_finding_kind="pain_point",
        atom_summary={
            "segment": "finance operations",
            "user_role": "bookkeeper",
            "job_to_be_done": "close the books",
            "trigger_event": "month end",
            "failure_mode": "reconciliation drift",
            "current_workaround": "spreadsheet cleanup",
        },
        validation_inputs={
            "problem_score": 0.41,
            "solution_gap_score": 0.62,
            "saturation_score": 0.38,
            "feasibility_score": 0.67,
            "value_score": 0.55,
            "cluster_signal_count": 4,
            "cluster_atom_count": 3,
        },
        corroboration_inputs={
            "recurrence_state": "supported",
            "corroboration_score": 0.58,
            "source_family_diversity": 2,
            "cluster_source_family_diversity": 3,
            "cross_source_match_score": 0.22,
            "generalizability_class": "reusable_workflow_pain",
            "generalizability_score": 0.7,
        },
        market_enrichment_inputs={
            "buyer_intent_score": 0.52,
            "competition_score": 0.21,
            "wedge_active": True,
        },
        review_feedback_inputs={"count": 1, "labels": {"needs_more_evidence": 1}},
        measures={
            "decision_score": 0.33,
            "problem_truth_score": 0.29,
            "revenue_readiness_score": 0.36,
            "pain_severity": 0.71,
            "frequency_score": 0.42,
            "urgency_score": 0.47,
            "cost_of_inaction": 0.51,
            "workaround_density": 0.43,
            "reachability": 0.63,
            "buildability": 0.74,
            "expansion_potential": 0.44,
            "segment_concentration": 0.52,
            "dependency_risk": 0.18,
            "adoption_friction": 0.21,
            "value_support": 0.39,
            "willingness_to_pay_proxy": 0.41,
            "evidence_quality": 0.57,
            "corroboration_strength": 0.48,
            "composite_score": 0.22,
            "problem_plausibility": 0.46,
            "evidence_sufficiency": 0.51,
        },
        evidence={
            "market_gap_state": "needs_more_recurrence_evidence",
            "recurrence_state": "supported",
            "family_confirmation_count": 2,
            "counterevidence": [{"claim": "rare", "status": "contradicted"}],
            "validation_plan": {"test_type": "workflow_walkthrough"},
        },
        decision="promote",
        decision_reason="validated_selection_gate",
        promotion_threshold=0.25,
        park_threshold=0.10,
        selection_status="research_more",
        selection_reason="selection_gate_not_met",
        selection_checks={"eligible": False, "blocked_by": ["value_support_below_threshold"]},
    )

    assert evaluation["schema_version"] == OPPORTUNITY_EVALUATION_SCHEMA_VERSION
    assert sorted(evaluation.keys()) == ["evidence", "inputs", "measures", "policy", "schema_version", "selection", "shadow"]
    assert evaluation["inputs"]["ids"]["validation_id"] == 44
    assert evaluation["inputs"]["atom"]["failure_mode"] == "reconciliation drift"
    assert evaluation["inputs"]["validation"]["cluster_signal_count"] == 4
    assert evaluation["inputs"]["corroboration"]["cluster_source_family_diversity"] == 3
    assert evaluation["measures"]["scores"]["decision_score"] == 0.33
    assert evaluation["measures"]["dimensions"]["buildability"] == 0.74
    assert evaluation["measures"]["dimensions"]["workaround_density"] == 0.43
    assert evaluation["measures"]["dimensions"]["segment_concentration"] == 0.52
    assert evaluation["measures"]["transition"]["composite_score"] == 0.22
    assert evaluation["evidence"]["market_gap_state"] == "needs_more_recurrence_evidence"
    assert evaluation["policy"]["policy_checks"]["promotion_threshold"] == 0.25
    assert evaluation["selection"]["build_prep_eligible"] is True
    assert evaluation["selection"]["build_prep_route"] == "spec_draft"
    assert evaluation["shadow"]["shadow_score_v2_lite"] > 0
    assert evaluation["shadow"]["comparison_diagnostics"]["formula_version"] == "v2_lite_shadow_v1"


def test_build_opportunity_evaluation_uses_prototype_candidate_route_when_selected():
    evaluation = build_opportunity_evaluation(
        run_id="run-123",
        finding_id=1,
        cluster_id=2,
        opportunity_id=3,
        validation_id=4,
        source_finding_kind="pain_point",
        atom_summary={},
        validation_inputs={},
        corroboration_inputs={},
        market_enrichment_inputs={},
        review_feedback_inputs={},
        measures={},
        evidence={},
        decision="promote",
        decision_reason="validated_selection_gate",
        promotion_threshold=0.25,
        park_threshold=0.10,
        selection_status="prototype_candidate",
        selection_reason="validated_selection_gate",
        selection_checks={"eligible": True, "blocked_by": []},
    )

    assert evaluation["selection"]["build_prep_eligible"] is True
    assert evaluation["selection"]["build_prep_route"] == "prototype_candidate"


def test_v2_lite_shadow_score_uses_existing_proxies_and_penalties():
    shadow_score, diagnostics = v2_lite_shadow_score(
        atom_summary={"user_role": ""},
        measures={
            "pain_severity": 0.8,
            "frequency_score": 0.6,
            "urgency_score": 0.5,
            "reachability": 0.7,
            "buildability": 0.65,
            "expansion_potential": 0.55,
            "evidence_quality": 0.75,
            "segment_concentration": 0.4,
            "dependency_risk": 0.2,
            "adoption_friction": 0.3,
            "willingness_to_pay_proxy": 0.5,
            "corroboration_strength": 0.8,
        },
        corroboration_inputs={
            "family_confirmation_count": 3,
            "source_family_diversity": 2,
        },
        market_enrichment_inputs={
            "buyer_intent_score": 0.3,
            "operational_buyer_score": 0.6,
            "competition_score": 0.25,
        },
    )

    assert shadow_score == diagnostics["net_score"]
    assert diagnostics["components"]["buyer_value_proxy"] == 0.4
    assert diagnostics["components"]["corroboration_proxy"] == 0.9
    assert diagnostics["components"]["buyer_clarity_proxy"] == 0.6
    assert diagnostics["components"]["competition_headroom"] == 0.75
    assert diagnostics["penalties"]["dependency_risk"] == 0.01
    assert diagnostics["penalties"]["adoption_friction"] == 0.015
