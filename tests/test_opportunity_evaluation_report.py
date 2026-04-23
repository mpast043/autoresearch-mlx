"""Tests for shadow comparison reporting over validation evidence."""

from __future__ import annotations

from src.opportunity_evaluation_report import build_shadow_scoring_report


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list_validation_evidence_payloads(self, *, run_id=None, limit=50, all_runs=False):
        self.calls.append({"run_id": run_id, "limit": limit, "all_runs": all_runs})
        return self.rows[:limit]


def test_shadow_scoring_report_uses_all_runs_when_unscoped():
    db = _FakeDB([])

    report = build_shadow_scoring_report(db, run_id=None, limit=10)

    assert report["scope"]["mode"] == "all_runs"
    assert db.calls == [{"run_id": None, "limit": 10, "all_runs": True}]


def test_shadow_scoring_report_compares_canonical_and_recomputed_rows():
    db = _FakeDB(
        [
            {
                "validation_id": 1,
                "finding_id": 11,
                "run_id": "run-a",
                "evidence": {
                    "summary": {"problem_statement": "Promoted canonical row"},
                    "opportunity_evaluation": {
                        "policy": {"decision": "promote"},
                        "selection": {"selection_status": "prototype_candidate", "selection_reason": "validated_selection_gate"},
                        "measures": {"scores": {"decision_score": 0.31}},
                        "shadow": {
                            "shadow_score_v2_lite": 0.41,
                            "comparison_diagnostics": {"formula_version": "v2_lite_shadow_v1"},
                        },
                    },
                },
            },
            {
                "validation_id": 2,
                "finding_id": 12,
                "run_id": "run-b",
                "evidence": {
                    "summary": {"problem_statement": "Parked recomputed row"},
                    "decision": "park",
                    "selection_status": "archive",
                    "selection_reason": "selection_gate_not_met",
                    "opportunity_scorecard": {
                        "decision_score": 0.14,
                        "pain_severity": 0.77,
                        "frequency_score": 0.61,
                        "urgency_score": 0.52,
                        "reachability": 0.68,
                        "buildability": 0.7,
                        "expansion_potential": 0.55,
                        "evidence_quality": 0.73,
                        "segment_concentration": 0.57,
                        "dependency_risk": 0.12,
                        "adoption_friction": 0.11,
                        "willingness_to_pay_proxy": 0.58,
                        "corroboration_strength": 0.79,
                    },
                    "corroboration": {
                        "family_confirmation_count": 3,
                        "source_family_diversity": 3,
                    },
                    "market_enrichment": {
                        "buyer_intent_score": 0.6,
                        "operational_buyer_score": 0.7,
                        "competition_score": 0.12,
                    },
                    "cluster": {"summary": {"user_role": "ops manager"}},
                },
            },
            {
                "validation_id": 3,
                "finding_id": 13,
                "run_id": "run-c",
                "evidence": {
                    "summary": {"problem_statement": "Missing inputs row"},
                    "decision": "kill",
                },
            },
        ]
    )

    report = build_shadow_scoring_report(db, limit=10)

    assert report["coverage"]["rows_examined"] == 3
    assert report["coverage"]["rows_compared"] == 2
    assert report["coverage"]["canonical_snapshot_rows"] == 1
    assert report["coverage"]["recomputed_shadow_rows"] == 1
    assert report["coverage"]["unreconstructable_rows"] == 1
    assert report["current_state_mix"]["decision"]["promote"] == 1
    assert report["current_state_mix"]["decision"]["park"] == 1
    assert any(row["validation_id"] == 2 for row in report["top_shadow_rows"])
    disagreement = report["notable_disagreements"]["high_shadow_not_promoted"][0]
    assert disagreement["validation_id"] == 2
    assert disagreement["shadow_origin"] == "recomputed_from_validation_evidence"
