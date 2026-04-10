"""Tests for the high-leverage findings lane."""

from __future__ import annotations

import os
import tempfile

from src.database import Database, Finding, ProblemAtom, RawSignal
from src.high_leverage import (
    build_high_leverage_report,
    persist_high_leverage_assessment,
    score_high_leverage_finding,
)


def _make_finding(summary: str, *, source_class: str = "pain_signal") -> Finding:
    return Finding(
        source="reddit-problem/smallbusiness",
        source_url="https://example.com/thread",
        entrepreneur="finance ops lead",
        product_built="Finance workflow drift",
        outcome_summary=summary,
        content_hash=summary,
        status="qualified",
        finding_kind="problem_signal",
        source_class=source_class,
        evidence={
            "screening": {"accepted": True, "negative_signals": []},
            "source_classification": {"source_class": source_class, "reasons": []},
        },
    )


def _make_signal(summary: str, *, source_name: str = "reddit-problem") -> RawSignal:
    return RawSignal(
        source_name=source_name,
        source_type="forum" if "reddit" in source_name else "web",
        source_url="https://example.com/thread",
        title="Workflow drift in production",
        body_excerpt=summary,
        source_class="pain_signal",
        quote_text=summary,
        role_hint="finance ops lead",
        content_hash=summary,
        metadata={"source_class": "pain_signal"},
    )


def _make_atom(
    *,
    segment: str = "finance operations",
    role: str = "finance ops lead",
    jtbd: str = "reconcile Stripe payouts to QuickBooks invoices",
    trigger: str = "weekly close",
    failure: str = "QuickBooks invoices do not match Stripe payouts after refund adjustments",
    workaround: str = "manually rebuild the ledger in spreadsheets before exporting to accounting",
    consequence: str = "hours of cleanup and delayed close",
    specificity: float = 0.84,
    consequence_score: float = 0.7,
) -> ProblemAtom:
    return ProblemAtom(
        cluster_key="stripe_to_quickbooks_mismatch",
        segment=segment,
        user_role=role,
        job_to_be_done=jtbd,
        pain_statement=failure,
        trigger_event=trigger,
        failure_mode=failure,
        current_workaround=workaround,
        urgency_clues="weekly close deadline",
        frequency_clues="every week",
        cost_consequence_clues=consequence,
        why_now_clues="refund volume increased",
        specificity_score=specificity,
        consequence_score=consequence_score,
    )


def test_high_leverage_scores_one_family_strong_candidate():
    summary = "Stripe payouts do not match QuickBooks invoices during weekly close and finance ops rebuild the ledger manually in spreadsheets."
    finding = _make_finding(summary)
    signal = _make_signal(summary)
    atom = _make_atom()

    assessment = score_high_leverage_finding(finding, signal, atom, finding.evidence or {})

    assert assessment["status"] == "candidate"
    assert assessment["evidence_tier"] == "one_family_strong"
    assert assessment["score"] >= 0.62


def test_high_leverage_scores_multi_family_confirmation_as_confirmed():
    summary = "Deleted Shopify orders still show up in analytics exports and ops teams manually clean reports before inventory review."
    finding = _make_finding(summary)
    signal = _make_signal(summary, source_name="web-problem")
    atom = _make_atom(
        segment="shopify merchants",
        role="ecommerce operations manager",
        jtbd="export clean analytics and inventory reports",
        trigger="weekly inventory review",
        failure="Deleted orders still appear in analytics exports after cancellation",
        workaround="manually clean analytics CSV files before reporting",
        consequence="inventory and finance reporting drift every week",
    )
    evidence = {
        **(finding.evidence or {}),
        "validation": {
            "matched_results_by_source": {"web": 2, "github": 1},
            "family_confirmation_count": 2,
            "source_family_diversity": 2,
        },
        "corroboration": {"source_family_diversity": 2},
    }

    assessment = score_high_leverage_finding(finding, signal, atom, evidence)

    assert assessment["status"] == "confirmed"
    assert assessment["evidence_tier"] == "multi_family_confirmed"
    assert assessment["score"] >= 0.68


def test_high_leverage_rejects_broad_manual_work_prompt():
    summary = "Small businesses waste hours on manual work, repetitive tasks, follow-ups, and reporting."
    finding = _make_finding(summary)
    signal = _make_signal(summary)
    atom = _make_atom(
        segment="small business operations",
        role="owner",
        jtbd="do repetitive tasks",
        trigger="daily operations",
        failure="manual tasks are painful",
        workaround="keep doing the work manually",
        consequence="annoying and time consuming",
        specificity=0.28,
        consequence_score=0.2,
    )
    evidence = {
        "screening": {"accepted": False, "negative_signals": ["broad_buying_prompt_without_wedge_slice"]},
        "source_classification": {"source_class": "low_signal_summary", "reasons": ["broad_buying_prompt_without_wedge_slice"]},
    }

    assessment = score_high_leverage_finding(finding, signal, atom, evidence)

    assert assessment["band"] == "reject"
    assert assessment["status"] == "discarded"


def test_high_leverage_rejects_vendor_support_noise():
    summary = "Support ignored me and this plugin is too expensive now."
    finding = _make_finding(summary)
    signal = _make_signal(summary, source_name="wordpress-review/plugin")
    atom = _make_atom(
        segment="wordpress operators",
        role="site owner",
        jtbd="keep plugin working",
        trigger="using the plugin",
        failure="support is bad",
        workaround="complain to support",
        consequence="annoying",
        specificity=0.22,
        consequence_score=0.1,
    )
    evidence = {
        "screening": {"accepted": False, "negative_signals": ["review_product_specific_issue"]},
        "source_classification": {"source_class": "low_signal_summary", "reasons": ["vendor_specific_complaint"]},
    }

    assessment = score_high_leverage_finding(finding, signal, atom, evidence)

    assert assessment["band"] == "reject"
    assert assessment["status"] == "discarded"


def test_high_leverage_rejects_editorial_overview_content():
    summary = "What's New in Microsoft Excel's Latest Release: An Overview"
    finding = _make_finding(summary)
    signal = _make_signal(summary, source_name="web-problem")
    atom = _make_atom(
        segment="operators",
        role="operator",
        jtbd="use spreadsheets",
        trigger="latest release",
        failure="overview content",
        workaround="read the article",
        consequence="learn the release",
        specificity=0.8,
        consequence_score=0.2,
    )

    assessment = score_high_leverage_finding(finding, signal, atom, finding.evidence or {})

    assert assessment["band"] == "reject"
    assert "editorial_or_overview_content" in assessment["reasons"]


def test_high_leverage_novelty_penalizes_near_duplicate_shapes():
    summary = "Stripe payouts do not match QuickBooks invoices during weekly close and finance ops rebuild the ledger manually in spreadsheets."
    finding = _make_finding(summary)
    signal = _make_signal(summary)
    atom = _make_atom()

    baseline = score_high_leverage_finding(finding, signal, atom, finding.evidence or {}, {"recent_shapes": []})
    duplicate = score_high_leverage_finding(
        finding,
        signal,
        atom,
        finding.evidence or {},
        {
            "recent_shapes": [
                {
                    "cluster_key": "stripe_to_quickbooks_mismatch",
                    "shape_tokens": sorted({"finance", "operations", "reconcile", "stripe", "quickbooks", "spreadsheets"}),
                }
            ]
        },
    )

    assert duplicate["components"]["novelty"] < baseline["components"]["novelty"]
    assert duplicate["score"] < baseline["score"]


def test_persist_high_leverage_updates_finding_and_signal_metadata():
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)
    db.init_schema()
    try:
        finding = _make_finding("Reconciliation drift")
        finding_id = db.insert_finding(finding)
        signal = _make_signal("Reconciliation drift")
        signal.finding_id = finding_id
        signal_id = db.insert_raw_signal(signal)

        persist_high_leverage_assessment(
            db,
            finding_id=finding_id,
            signal_id=signal_id,
            assessment={"score": 0.66, "status": "candidate", "band": "strong", "reasons": ["narrow_workflow_shape"]},
        )
        persist_high_leverage_assessment(
            db,
            finding_id=finding_id,
            signal_id=signal_id,
            assessment={"score": 0.71, "status": "confirmed", "band": "strong", "reasons": ["independent_confirmation"]},
        )

        stored_finding = db.get_finding(finding_id)
        stored_signal = db.get_raw_signal(signal_id)

        assert stored_finding.evidence["high_leverage"]["score"] == 0.71
        assert stored_finding.evidence["high_leverage"]["status"] == "confirmed"
        assert stored_signal.metadata["high_leverage"]["score"] == 0.71
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_build_high_leverage_report_sorts_findings_first():
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)
    db.init_schema()
    try:
        strong_finding = _make_finding("Stripe payouts do not match QuickBooks invoices.")
        strong_finding.evidence["run_id"] = "run-1"
        strong_id = db.insert_finding(strong_finding)
        strong_signal = _make_signal("Stripe payouts do not match QuickBooks invoices.")
        strong_signal.finding_id = strong_id
        strong_signal.metadata["high_leverage"] = {"score": 0.74}
        db.insert_raw_signal(strong_signal)
        strong_atom = _make_atom()
        strong_atom.finding_id = strong_id
        db.insert_problem_atom(strong_atom)
        persist_high_leverage_assessment(
            db,
            finding_id=strong_id,
            assessment={
                "score": 0.74,
                "band": "strong",
                "status": "candidate",
                "evidence_tier": "one_family_strong",
                "reasons": ["narrow_workflow_shape"],
                "components": {"specificity": 0.84},
                "version": "high_leverage_v1",
            },
        )

        boring_finding = _make_finding("Manual tasks take too long.")
        boring_finding.evidence["run_id"] = "run-1"
        boring_id = db.insert_finding(boring_finding)
        boring_signal = _make_signal("Manual tasks take too long.")
        boring_signal.finding_id = boring_id
        db.insert_raw_signal(boring_signal)

        report = build_high_leverage_report(db, run_id="run-1", limit=5)

        assert report["findings"][0]["finding_id"] == strong_id
        assert report["findings"][0]["high_leverage_score"] == 0.74
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
