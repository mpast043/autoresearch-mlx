"""Tests for canonical-aware gate diagnostics."""

from __future__ import annotations

from src.gate_diagnostics import build_gate_diagnostics_report, explain_validation_evidence


def test_explain_validation_evidence_prefers_canonical_snapshot():
    evidence = {
        "decision": "park",
        "selection_status": "research_more",
        "opportunity_scorecard": {
            "decision_score": 0.1,
            "frequency_score": 0.2,
            "evidence_quality": 0.3,
            "value_support": 0.2,
            "composite_score": 0.1,
        },
        "opportunity_evaluation": {
            "policy": {
                "decision": "promote",
                "decision_reason": "validated_selection_gate",
                "policy_version": "stage_decision_v1",
                "policy_checks": {"promotion_threshold": 0.25},
            },
            "selection": {
                "selection_status": "prototype_candidate",
                "selection_reason": "validated_selection_gate",
                "selection_checks": {"eligible": True, "blocked_by": []},
                "build_prep_eligible": True,
                "build_prep_route": "prototype_candidate",
            },
            "measures": {
                "scores": {
                    "decision_score": 0.31,
                    "problem_truth_score": 0.21,
                    "revenue_readiness_score": 0.41,
                },
                "dimensions": {
                    "frequency_score": 0.44,
                    "evidence_quality": 0.62,
                    "value_support": 0.58,
                },
                "transition": {"composite_score": 0.27},
            },
            "inputs": {
                "corroboration": {"source_family_diversity": 2, "generalizability_class": "reusable_workflow_pain"},
                "market_enrichment": {"wedge_active": True},
            },
            "evidence": {
                "market_gap_state": "needs_more_recurrence_evidence",
                "market_gap": {"market_gap": "needs_more_recurrence_evidence", "solution_gap_score": 0.58},
                "counterevidence": [{"claim": "rare", "status": "contradicted"}],
                "requested_specific_queries": 2,
                "generated_specific_queries": 2,
                "executed_specific_queries": 2,
                "query_origin_counts": {"finding_specific": 1},
                "attribution_scope_counts": {"finding": 1},
                "comparison_sibling_finding_id": 88,
                "comparison_scope": "same_domain",
                "query_overlap_ratio": 0.75,
                "compression_signal": "high",
            },
            "shadow": {"shadow_score_v2_lite": 0.49},
        },
    }

    detail = explain_validation_evidence(evidence, config={})

    assert detail["canonical_evaluation"]["available"] is True
    assert detail["canonical_evaluation"]["decision"] == "promote"
    assert detail["canonical_evaluation"]["selection_status"] == "prototype_candidate"
    assert detail["canonical_evaluation"]["build_prep_route"] == "prototype_candidate"
    assert detail["selection_gate"]["score_language"]["primary"]["decision_score"] == 0.31
    assert detail["selection_gate"]["score_language"]["diagnostic"]["composite_score"] == 0.27
    assert detail["evidence_summary"]["executed_specific_queries"] == 2
    assert detail["evidence_summary"]["query_origin_counts"]["finding_specific"] == 1
    assert detail["evidence_summary"]["comparison_sibling_finding_id"] == 88
    assert detail["evidence_summary"]["compression_signal"] == "high"
    assert detail["stage_decision"]["decision"]["recommendation"] in {"park", "promote", "kill"}


class _FakeDB:
    def list_validation_evidence_payloads(self, *, run_id=None, limit=500):
        return [
            {
                "validation_id": 7,
                "finding_id": 11,
                "evidence": {
                    "summary": {"problem_statement": "Canonical row"},
                    "decision": "park",
                    "selection_status": "research_more",
                    "opportunity_evaluation": {
                        "policy": {"decision": "promote", "decision_reason": "validated_selection_gate"},
                        "selection": {
                            "selection_status": "prototype_candidate",
                            "selection_reason": "validated_selection_gate",
                            "selection_checks": {"eligible": True, "blocked_by": []},
                            "build_prep_eligible": True,
                            "build_prep_route": "prototype_candidate",
                        },
                        "measures": {
                            "scores": {"decision_score": 0.31, "problem_truth_score": 0.21, "revenue_readiness_score": 0.41},
                            "dimensions": {"frequency_score": 0.44, "evidence_quality": 0.62, "value_support": 0.58},
                            "transition": {"composite_score": 0.27},
                        },
                        "inputs": {"corroboration": {}, "market_enrichment": {}},
                        "evidence": {
                            "market_gap_state": "needs_more_recurrence_evidence",
                            "market_gap": {"market_gap": "needs_more_recurrence_evidence", "solution_gap_score": 0.58},
                            "counterevidence": [{"claim": "rare", "status": "contradicted"}],
                            "requested_specific_queries": 2,
                            "generated_specific_queries": 2,
                            "executed_specific_queries": 2,
                            "query_origin_counts": {"finding_specific": 1},
                            "attribution_scope_counts": {"finding": 1},
                            "comparison_sibling_finding_id": 88,
                            "comparison_scope": "same_domain",
                            "query_overlap_ratio": 0.75,
                            "compression_signal": "high",
                        },
                        "shadow": {"shadow_score_v2_lite": 0.49},
                    },
                },
            }
        ]

    def get_active_run_id(self):
        return "run-1"

    def get_latest_run_id(self):
        return "run-1"


def test_build_gate_diagnostics_report_uses_canonical_top_level_state():
    report = build_gate_diagnostics_report(_FakeDB(), config={}, run_id="run-1", limit=5)

    assert report["canonical_evaluation_rows"] == 1
    assert report["validations"][0]["decision"] == "promote"
    assert report["validations"][0]["selection_status"] == "prototype_candidate"
    assert report["validations"][0]["selection_reason"] == "validated_selection_gate"
    evidence_summary = report["validations"][0]["diagnostics"]["evidence_summary"]
    assert evidence_summary["executed_specific_queries"] == 2
    assert evidence_summary["comparison_sibling_finding_id"] == 88
