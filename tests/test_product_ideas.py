import os
import tempfile

from src.database import Database, Finding, Opportunity, OpportunityCluster, ProblemAtom, RawSignal
from src.product_ideas import generate_wedges_with_metrics


def test_generate_wedges_with_metrics_uses_live_schema_storage_model():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    try:
        db.init_schema()

        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/accounting",
                source_url="https://example.com/thread",
                product_built="QuickBooks payout reconciliation",
                outcome_summary="QuickBooks invoices do not match Stripe payouts during weekly reconciliation.",
                content_hash="wedge-test-hash",
                finding_kind="problem_signal",
                source_class="pain_signal",
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=finding_id,
                source_name="reddit-problem",
                source_type="reddit",
                source_class="pain_signal",
                source_url="https://example.com/thread",
                title="QuickBooks payout reconciliation keeps breaking",
                body_excerpt="QuickBooks invoices do not match Stripe payouts during weekly reconciliation, costing hours each week.",
                quote_text="QuickBooks invoices do not match Stripe payouts during weekly reconciliation.",
                role_hint="ops manager",
                content_hash="wedge-test-signal",
                metadata={"platform": "quickbooks"},
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                finding_id=finding_id,
                raw_signal_id=signal_id,
                signal_id=signal_id,
                cluster_key="quickbooks-payout-reconciliation",
                segment="finance ops",
                user_role="ops manager",
                job_to_be_done="reconcile weekly payouts in QuickBooks",
                trigger_event="during weekly reconciliation",
                pain_statement="QuickBooks invoices do not match Stripe payouts",
                failure_mode="QuickBooks invoices do not match Stripe payouts",
                current_workaround="manual spreadsheet reconciliation",
                cost_consequence_clues="3 hours lost per week",
                confidence=0.9,
                platform="quickbooks",
                specificity_score=0.8,
                consequence_score=0.9,
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                label="QuickBooks payout reconciliation",
                cluster_key="quickbooks-payout-reconciliation",
                segment="finance ops",
            )
        )
        db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="QuickBooks payout reconciliation",
                market_gap="Teams still reconcile Stripe payouts manually in QuickBooks.",
                recommendation="promote",
                status="promoted",
                decision_score=0.82,
                problem_truth_score=0.78,
                composite_score=0.8,
                buildability=0.7,
                corroboration_strength=0.65,
                evidence_quality=0.75,
                urgency_score=0.7,
                frequency_score=0.7,
                willingness_to_pay_proxy=0.6,
                pain_severity=0.7,
            )
        )

        ideas, candidates, stats, rejected_counts = generate_wedges_with_metrics(db, limit=5)

        assert sum(stats.values()) == 1
        assert isinstance(candidates, list)
        assert isinstance(ideas, list)
        assert isinstance(rejected_counts, dict)
    finally:
        db.close()
        os.remove(path)
