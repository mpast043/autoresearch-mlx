"""Tests for discovery suggestion helper."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database, Finding, OpportunityCluster, RawSignal
from src.discovery_suggestions import build_discovery_suggestions


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


def test_suggest_discovery_pulls_keywords_subs_themes_and_comment_phrases(temp_db):
    cid = temp_db.upsert_cluster(
        OpportunityCluster(
            label="Manual payout reconciliation keeps breaking every week",
            cluster_key="test-key",
            atom_count=3,
            job_to_be_done="keep payouts accurate without spreadsheet drift",
            trigger_summary="after pricing change",
            signal_count=2,
            evidence_quality=0.5,
            segment="ops",
            user_role="operator",
            status="active",
        )
    )
    assert cid

    finding_id = temp_db.insert_finding(
        Finding(
            source="reddit-problem/indiehackers",
            source_url="https://reddit.com/r/indiehackers/comments/abc/x",
            content_hash="sug-test-1",
            product_built="test",
            outcome_summary="pain",
            status="qualified",
            finding_kind="problem_signal",
        )
    )
    temp_db.insert_finding(
        Finding(
            source="reddit-success",
            source_url="https://reddit.com/r/saas/comments/xyz/y",
            content_hash="sug-test-2",
            product_built="We built an onboarding copilot and launched last month",
            monetization_method="$8k MRR",
            outcome_summary="Founder says they discovered churn drops after weekly usage emails.",
            status="qualified",
            finding_kind="success_signal",
        )
    )
    temp_db.insert_raw_signal(
        RawSignal(
            finding_id=finding_id,
            source_name="reddit",
            source_type="reddit-problem",
            source_url="https://reddit.com/r/indiehackers/comments/abc/x",
            title="Billing keeps breaking",
            body_excerpt="I keep reconciling Stripe and invoices manually every week. It is frustrating.",
            quote_text="We are stuck with manual reconciliation in spreadsheets.",
            content_hash="sig-test-1",
        )
    )

    out = build_discovery_suggestions(
        temp_db,
        min_atoms=2,
        limit_clusters=10,
        max_keywords=20,
        theme_keywords={
            "money": ["reconciliation", "cash flow"],
            "build": ["deployment pain"],
        },
    )
    assert "Manual payout" in out["suggested_keywords"][0] or any(
        "payout" in k.lower() for k in out["suggested_keywords"]
    )
    assert "indiehackers" in out["suggested_subreddits_from_findings"]
    assert out["suggested_keywords_by_theme"]["money"]
    assert any("manual reconciliation" in p.lower() for p in out["suggested_comment_phrases"])
    assert any("mrr" in item["claim"].lower() for item in out["suggested_money_claims"])
    assert out["suggested_money_claims_by_confidence"]["high"]
    assert out["suggested_money_claims_by_confidence"]["medium"] == []
    assert any("built" in p.lower() or "launched" in p.lower() for p in out["suggested_build_discovery_signals"])

    strict = build_discovery_suggestions(
        temp_db,
        min_atoms=2,
        limit_clusters=10,
        max_keywords=20,
        money_claim_min_confidence="high",
    )
    assert strict["suggested_money_claims"]
    assert strict["suggested_money_claims_by_confidence"]["medium"] == []
    assert strict["suggested_money_claims_by_confidence"]["low"] == []
