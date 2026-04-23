"""Tests for canonical opportunity evaluation backfill."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.database import (
    CorroborationRecord,
    Database,
    Finding,
    MarketEnrichment,
    Opportunity,
    OpportunityCluster,
    ProblemAtom,
    Validation,
)
from src.opportunity_evaluation_backfill import backfill_opportunity_evaluations


def test_backfill_opportunity_evaluations_reconstructs_and_writes_snapshot():
    path = Path(tempfile.mktemp(suffix=".db"))
    db = Database(str(path))
    db.init_schema()
    try:
        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/accounting",
                source_url="https://reddit.com/r/accounting/comments/backfill",
                product_built="QuickBooks CSV reconciliation drift",
                outcome_summary="Accountants manually clean import mismatches at month end.",
                content_hash="backfill-validation",
                finding_kind="problem_signal",
                source_class="pain_signal",
                status="promoted",
                evidence={"run_id": "run-1"},
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                label="QuickBooks CSV reconciliation drift",
                cluster_key="qbo-csv-drift",
                summary={"segment": "finance operations"},
                user_role="bookkeeper",
                job_to_be_done="import payout CSVs without reconciliation drift",
                signal_count=2,
                atom_count=2,
                evidence_quality=0.63,
            )
        )
        opportunity_id = db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="QuickBooks CSV reconciliation drift",
                market_gap="needs_more_recurrence_evidence",
                recommendation="promote",
                status="promoted",
                selection_status="prototype_candidate",
                selection_reason="validated_selection_gate",
                decision_score=0.33,
                problem_truth_score=0.22,
                revenue_readiness_score=0.41,
                composite_score=0.28,
                pain_severity=0.68,
                frequency_score=0.44,
                urgency_score=0.49,
                cost_of_inaction=0.52,
                reachability=0.61,
                buildability=0.73,
                expansion_potential=0.48,
                evidence_quality=0.64,
                problem_plausibility=0.57,
                value_support=0.53,
                corroboration_strength=0.58,
                evidence_sufficiency=0.55,
                willingness_to_pay_proxy=0.47,
                notes={
                    "counterevidence": [{"claim": "rare", "status": "contradicted"}],
                    "market_gap": {"market_gap": "needs_more_recurrence_evidence"},
                },
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                signal_id=1,
                raw_signal_id=1,
                finding_id=finding_id,
                cluster_key="qbo-csv-drift",
                segment="finance operations",
                user_role="bookkeeper",
                job_to_be_done="import payout CSVs without reconciliation drift",
                pain_statement="manual month-end cleanup is still required after imports",
                trigger_event="month-end close",
                failure_mode="reconciliation drift after CSV import",
                current_workaround="spreadsheet cleanup and journal-entry patching",
                current_tools="QuickBooks, spreadsheets",
                confidence=0.81,
            )
        )
        db.upsert_corroboration(
            CorroborationRecord(
                run_id="run-1",
                finding_id=finding_id,
                recurrence_state="supported",
                corroboration_score=0.57,
                evidence_sufficiency=0.6,
                query_coverage=0.5,
                independent_confirmations=2,
                source_diversity=2,
                evidence={
                    "cross_source_match_score": 0.24,
                    "generalizability_class": "reusable_workflow_pain",
                    "generalizability_score": 0.7,
                    "source_family_diversity": 2,
                    "family_confirmation_count": 2,
                },
            )
        )
        db.upsert_market_enrichment(
            MarketEnrichment(
                run_id="run-1",
                finding_id=finding_id,
                demand_score=0.41,
                buyer_intent_score=0.52,
                competition_score=0.19,
                trend_score=0.2,
                review_signal_score=0.3,
                value_signal_score=0.5,
                evidence={
                    "operational_buyer_score": 0.61,
                    "cost_pressure_score": 0.58,
                    "willingness_to_pay_signal": 0.45,
                    "multi_source_value_lift": 0.4,
                    "wedge_active": True,
                    "wedge_fit_score": 0.62,
                },
            )
        )
        validation_id = db.insert_validation(
            Validation(
                finding_id=finding_id,
                run_id="run-1",
                market_score=0.53,
                technical_score=0.66,
                distribution_score=0.49,
                overall_score=0.58,
                passed=True,
                evidence={
                    "finding_kind": "problem_signal",
                    "scores": {
                        "problem_score": 0.54,
                        "solution_gap_score": 0.59,
                        "saturation_score": 0.31,
                        "feasibility_score": 0.66,
                        "value_score": 0.51,
                    },
                    "opportunity_id": opportunity_id,
                    "cluster": {"cluster_id": cluster_id, "signal_count": 2, "atom_count": 2},
                    "decision": "promote",
                    "decision_reason": "validated_selection_gate",
                    "selection_status": "prototype_candidate",
                    "selection_reason": "validated_selection_gate",
                    "selection_gate": {"eligible": True, "blocked_by": []},
                    "market_gap_state": "needs_more_recurrence_evidence",
                },
            )
        )

        report = backfill_opportunity_evaluations(db, config={}, run_id=None, limit=10, apply=True)

        assert report["counts"]["reconstructable"] == 1
        assert report["counts"]["written"] == 1

        validation = db.get_validation(validation_id)
        assert validation is not None
        evaluation = (validation.evidence or {}).get("opportunity_evaluation", {})
        metadata = (validation.evidence or {}).get("opportunity_evaluation_backfill", {})
        assert evaluation["schema_version"] == "opportunity_evaluation_v1"
        assert evaluation["inputs"]["ids"]["opportunity_id"] == opportunity_id
        assert evaluation["inputs"]["atom"]["failure_mode"] == "reconciliation drift after CSV import"
        assert evaluation["policy"]["decision"] == "promote"
        assert evaluation["selection"]["selection_status"] == "prototype_candidate"
        assert evaluation["selection"]["build_prep_route"] == "prototype_candidate"
        assert metadata["status"] == "reconstructable"
        assert metadata["sources_used"]["measures"] == "opportunities"
    finally:
        db.close()
        path.unlink(missing_ok=True)


def test_backfill_opportunity_evaluations_marks_unreconstructable_rows_without_writing():
    path = Path(tempfile.mktemp(suffix=".db"))
    db = Database(str(path))
    db.init_schema()
    try:
        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/test",
                source_url="https://example.com/test",
                content_hash="unreconstructable-validation",
            )
        )
        validation_id = db.insert_validation(
            Validation(
                finding_id=finding_id,
                run_id="run-1",
                overall_score=0.0,
                passed=False,
                evidence={},
            )
        )

        report = backfill_opportunity_evaluations(db, config={}, run_id=None, limit=10, apply=True)

        assert report["counts"]["unreconstructable"] == 1
        assert report["counts"]["written"] == 0
        validation = db.get_validation(validation_id)
        assert validation is not None
        assert "opportunity_evaluation" not in (validation.evidence or {})
    finally:
        db.close()
        path.unlink(missing_ok=True)
