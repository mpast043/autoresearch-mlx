"""Tests for deterministic weak-signal extraction and scoring."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import ProblemAtom, RawSignal
from src.opportunity_engine import (
    OpportunityEngine,
    build_cluster_summary,
    build_problem_atom,
    classify_source_signal,
    plan_validation_experiment,
    qualify_problem_signal,
    score_opportunity,
)


def test_reddit_cluster_key_matches_across_subreddits_for_same_text():
    """Segment / cluster_key should not split by r/subreddit label in source."""
    body = (
        "We merge payouts manually in Excel every week. "
        "Spreadsheet reconciliation is breaking after the pricing change."
    )
    payload = {
        "source_name": "reddit-problem",
        "source_type": "forum",
        "source_url": "https://reddit.com/r/any/comments/abc/thread",
        "title": "Manual payout reconciliation",
        "body_excerpt": body,
        "quote_text": body[:120],
        "role_hint": "operator",
        "metadata_json": {},
    }
    finding_a = {
        "source": "reddit-problem/smallbusiness",
        "source_url": "https://reddit.com/r/smallbusiness/comments/a/x",
        "product_built": "Manual payout reconciliation",
        "outcome_summary": body,
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
        "tool_used": "",
    }
    finding_b = {
        **finding_a,
        "source": "reddit-problem/sysadmin",
        "source_url": "https://reddit.com/r/sysadmin/comments/b/y",
    }
    atom_a = build_problem_atom(payload, finding_a)
    atom_b = build_problem_atom(payload, finding_b)
    assert atom_a["cluster_key"] == atom_b["cluster_key"]
    assert atom_a["segment"] == atom_b["segment"]


def test_extract_problem_atom_captures_workaround_and_why_now():
    engine = OpportunityEngine()
    signal = RawSignal(
        finding_id=10,
        source_name="reddit-problem",
        source_type="reddit",
        source_url="https://reddit.com/test",
        title="We still use spreadsheets to reconcile payouts",
        body_excerpt=(
            "Every day our ops team manually copy and paste payout data into spreadsheets. "
            "It is frustrating, error-prone, and got worse after the pricing change."
        ),
        quote_text="Every day our ops team manually copy and paste payout data into spreadsheets.",
        published_at=None,
        role_hint="operator",
        timestamp_hint="",
        content_hash="abc",
        metadata={},
        id=1,
    )

    atom = engine.extract_problem_atom(signal, finding_kind="pain_point")

    assert atom is not None
    assert atom.current_workaround
    assert atom.why_now_clues
    assert atom.confidence >= 0.5


def test_build_problem_atom_keeps_workflow_distinct_and_caps_consequence_score():
    text = "QuickBooks invoices do not match Stripe payouts during weekly reconciliation costing 3 hours every week"
    payload = {
        "source_name": "reddit-problem",
        "source_type": "reddit",
        "source_url": "https://example.com/test",
        "title": text,
        "body_excerpt": text,
        "quote_text": text,
        "role_hint": "ops manager",
        "metadata_json": {},
    }
    finding = {
        "source": "reddit-problem/accounting",
        "source_url": "https://example.com/test",
        "product_built": text,
        "outcome_summary": text,
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
        "tool_used": "",
    }

    atom = build_problem_atom(payload, finding)

    assert atom["failure_mode"]
    assert atom["job_to_be_done"]
    assert atom["job_to_be_done"] != atom["failure_mode"]
    assert atom["consequence_score"] <= 1.0


def test_score_opportunity_penalizes_false_signal():
    engine = OpportunityEngine()
    atom = ProblemAtom(
        signal_id=1,
        finding_id=1,
        cluster_key="general|operator|manual",
        segment="general",
        user_role="operator",
        job_to_be_done="automate recurring operational work",
        trigger_event="",
        pain_statement="This workflow is frustrating",
        failure_mode="manual exceptions",
        current_workaround="spreadsheet",
        current_tools="Excel",
        urgency_clues="daily",
        frequency_clues="every day",
        emotional_intensity=0.7,
        cost_consequence_clues="hours lost",
        why_now_clues="pricing",
        confidence=0.8,
        atom_json=json.dumps({"supporting_quote": "This workflow is frustrating"}),
    )
    cluster = engine.build_cluster(atom, [atom], ["reddit-problem"])
    scorecard = engine.score_opportunity(
        cluster=cluster,
        atoms=[atom],
        recurrence_docs=[{"title": "doc", "url": "https://example.com", "snippet": "manual workflow"}],
        competitor_docs=[{"title": f"competitor-{idx}", "url": f"https://tool{idx}.com"} for idx in range(6)],
        counter_docs=[{"title": "already solved", "url": "https://example.com/solved", "snippet": "already solved"}],
        market_gap_state="likely_false_signal",
    )

    assert scorecard["decision"] != "promote"
    assert scorecard["total_score"] < 0.66


def test_qualify_problem_signal_rejects_promotional_success_post():
    finding_data = {
        "source": "reddit-success",
        "source_url": "https://reddit.com/r/SideProject/comments/mrr",
        "product_built": "Open-source scheduler",
        "outcome_summary": "I did it. My open-source company now makes $14k MRR.",
        "finding_kind": "success_signal",
    }
    signal_payload = {
        "title": "I did it. My open-source company now makes $14k MRR",
        "body_excerpt": "Solo founder brag post",
        "source_type": "forum",
        "metadata_json": {},
    }
    atom_payload = {
        "current_workaround": "",
        "failure_mode": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "non_problem_finding_kind" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_problem_solicitation_prompt():
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/smallbusiness/comments/1",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "What is the most annoying, manual task you have to do for work every day?",
        "body_excerpt": (
            "I'm a developer looking to build a new tool. Instead of guessing what people need, "
            "I wanted to ask directly: tell me your most annoying manual task and I might automate it."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "current_workaround": "",
        "failure_mode": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "solicitation_for_problem_examples" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_product_complaint_without_workflow_context():
    finding_data = {
        "source": "reddit-problem/airtable",
        "source_url": "https://reddit.com/r/airtable/comments/1",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Airtable should focus on improving normal features, not AI",
        "body_excerpt": (
            "I have been using Airtable for years and I'm baffled why it spends so much time on AI "
            "features instead of improving the basic spreadsheet experience."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "use Airtable",
        "trigger_event": "now",
        "pain_statement": "The product keeps shipping AI instead of core improvements",
        "failure_mode": "feature priorities feel wrong",
        "current_workaround": "spreadsheets",
        "urgency_clues": "must",
        "frequency_clues": "always",
        "cost_consequence_clues": "",
        "why_now_clues": "now",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "product_specific_complaint_without_workflow_context" in screening["negative_signals"]


def test_qualify_problem_signal_keeps_product_thread_with_real_workflow_context():
    finding_data = {
        "source": "reddit-problem/notion",
        "source_url": "https://reddit.com/r/notion/comments/2",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Does anyone else have to use 3 different tools just to collect a form response and payment?",
        "body_excerpt": (
            "I run client intake in Notion, but every booking means sending a Typeform, copying the "
            "response into my client database, sending a payment link, and manually updating records."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "collect client intake and payment without manual cleanup",
        "trigger_event": "when a new client books",
        "pain_statement": "client booking intake requires multiple tools and manual updates",
        "failure_mode": "booking workflow spans form, database, and payment link",
        "current_workaround": "manual work, copy/paste",
        "urgency_clues": "",
        "frequency_clues": "every time",
        "cost_consequence_clues": "",
        "why_now_clues": "when",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is True
    assert "product_specific_complaint_without_workflow_context" not in screening["negative_signals"]


def test_qualify_problem_signal_rejects_roi_shopping_prompt():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/roi",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "What was the first automation you paid for, and was it worth the money?",
        "body_excerpt": (
            "I'm considering outsourcing an automation project and trying to gauge the ROI. "
            "What was the first automation you ever paid someone to build?"
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "decide whether to buy an automation project",
        "trigger_event": "",
        "pain_statement": "trying to gauge the ROI",
        "failure_mode": "trying to gauge the ROI",
        "current_workaround": "custom scripts",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "roi_or_vendor_shopping_prompt" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_advice_seeking_without_stakes():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/advice",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "How do you handle supplier product data and enrichment?",
        "body_excerpt": (
            "I'm looking for advice on optimizing our product data workflow. "
            "We receive spreadsheets from suppliers and I manually restructure them."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "organize supplier product data",
        "trigger_event": "when supplier files arrive",
        "pain_statement": "product onboarding is extremely manual",
        "failure_mode": "manually restructure supplier files",
        "current_workaround": "spreadsheets, manual work, copy/paste",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "advice_seeking_without_actionable_stakes" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_career_guidance_thread():
    finding_data = {
        "source": "reddit-problem/accounting",
        "source_url": "https://reddit.com/r/accounting/comments/resume",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Feeling lost with career, please roast my resume.",
        "body_excerpt": (
            "Currently working at a local firm and feeling stagnant. "
            "Please dissect my resume and tell me how to transition to a larger firm."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "improve my resume",
        "trigger_event": "trying to transition to a larger firm",
        "pain_statement": "resume feels wordy",
        "failure_mode": "resume is not strong enough",
        "current_workaround": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "career_guidance_thread" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_instructional_tutorial_post():
    finding_data = {
        "source": "reddit-problem/EtsySellers",
        "source_url": "https://reddit.com/r/EtsySellers/comments/tutorial",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "How to ship via USPS 1st class letter rate with Etsy shipping labels",
        "body_excerpt": (
            "I thought I would put this information all together and return the favor. "
            "Here are the steps I use to ship these orders."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "ship via Etsy labels",
        "trigger_event": "",
        "pain_statement": "sharing how I do it",
        "failure_mode": "manual shipping setup",
        "current_workaround": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "tutorial_or_instructional_post" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_help_choosing_vendor_prompt():
    finding_data = {
        "source": "reddit-problem/smallbusiness",
        "source_url": "https://reddit.com/r/smallbusiness/comments/vendor-prompt",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Help choosing an inventory management system",
        "body_excerpt": (
            "I run a frozen food manufacturing business and need help choosing an inventory management system. "
            "We currently track inventory manually and want recommendations."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "choose an inventory system",
        "trigger_event": "",
        "pain_statement": "manual tracking is painful",
        "failure_mode": "manual inventory tracking",
        "current_workaround": "weekly physical counts",
        "urgency_clues": "",
        "frequency_clues": "weekly",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "generic_request_or_vendor_shopping" in screening["negative_signals"]
    assert "too_generic_after_review" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_virtual_card_expense_shopping_prompt():
    finding_data = {
        "source": "reddit-problem/smallbusiness",
        "source_url": "https://reddit.com/r/smallbusiness/comments/cards",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Growing team looking for the best virtual credit card for expense automation",
        "body_excerpt": (
            "We run a 14-person team and everyone puts subscriptions, ad spend, and office costs on personal cards. "
            "Now we manually chase receipts for reimbursements and want the best virtual card solution."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "user_role": "finance lead",
        "job_to_be_done": "Automate expense tracking and reimbursement for team spending",
        "trigger_event": "monthly close",
        "pain_statement": "The process is slow and error-prone because receipts are collected manually.",
        "failure_mode": "manual chasing receipts for reimbursements",
        "current_workaround": "collect receipts in spreadsheets and email",
        "urgency_clues": "",
        "frequency_clues": "monthly",
        "cost_consequence_clues": "slow and error-prone",
        "why_now_clues": "growing team",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "finance_tool_shopping_without_specific_failure" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_finance_acquisition_chatter_without_specific_failure():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/melio",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Xero just bought Melio. Will this finally fix vendor payments for small ecommerce brands?",
        "body_excerpt": (
            "The whole vendor payments stack has always felt like a bolt-on. "
            "Net terms are buried in inboxes and approvals are scattered, but the real question is whether this acquisition finally fixes it."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "user_role": "small ecommerce brand owner",
        "job_to_be_done": "Streamline and automate the entire vendor payment process",
        "trigger_event": "weekly vendor payments",
        "pain_statement": "The stack feels bolted on and messy.",
        "failure_mode": "vendor payments are messy and spread across inboxes",
        "current_workaround": "email threads and spreadsheets",
        "urgency_clues": "",
        "frequency_clues": "weekly",
        "cost_consequence_clues": "time lost",
        "why_now_clues": "after the acquisition",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "finance_acquisition_or_vendor_chatter" in screening["negative_signals"]


def test_qualify_problem_signal_keeps_specific_stripe_quickbooks_reconciliation_failure():
    finding_data = {
        "source": "reddit-problem/accounting",
        "source_url": "https://reddit.com/r/accounting/comments/reconcile",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Stripe CSV dates do not match QuickBooks bank feed imports",
        "body_excerpt": (
            "Every week we export Stripe payouts and rework the CSV because QuickBooks imports the dates incorrectly. "
            "Refunded payments and fees do not reconcile with the bank feed unless we manually split rows."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "user_role": "accountant",
        "job_to_be_done": "Import Stripe payout data into QuickBooks without reconciliation drift",
        "trigger_event": "weekly close",
        "pain_statement": "QuickBooks imports Stripe CSVs with wrong dates and mismatched fees.",
        "failure_mode": "Stripe CSV dates and fees do not reconcile with QuickBooks bank feeds",
        "current_workaround": "manually split rows and correct dates in Excel",
        "urgency_clues": "",
        "frequency_clues": "weekly",
        "cost_consequence_clues": "hours lost each close",
        "why_now_clues": "refund volume increased",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is True
    assert "finance_tool_shopping_without_specific_failure" not in screening["negative_signals"]
    assert "finance_acquisition_or_vendor_chatter" not in screening["negative_signals"]


def test_qualify_problem_signal_keeps_multistep_workflow_gap_question():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/workflow",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": 'How are u guys managing fulfilment between "order received" and "label printed"?',
        "body_excerpt": (
            "Our workflow looks like Received -> Picking -> Processing -> Packed -> Shipped, "
            "and the team ends up coordinating through WhatsApp and spreadsheets."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "keep fulfillment handoffs in sync without manual cleanup",
        "trigger_event": "Received -> Picking -> Processing -> Packed -> Shipped",
        "pain_statement": "the team coordinates through WhatsApp and spreadsheets",
        "failure_mode": "manual coordination between order received and label printed",
        "current_workaround": "spreadsheets, manual work",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "now",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is True
    assert "advice_seeking_without_actionable_stakes" not in screening["negative_signals"]


def test_qualify_problem_signal_rejects_vendor_vent_without_transferable_workflow():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/vent",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "I am done with Avalara",
        "body_excerpt": (
            "FYI I am not looking for product recommendations, I just want vent my frustrations. "
            "Set it and forget it, more like set it and spend the rest of your days doing manual reconciliations."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal"},
    }
    atom_payload = {
        "job_to_be_done": "use Avalara",
        "trigger_event": "fyi I am not looking for product recommendations",
        "pain_statement": "I just want vent my frustrations",
        "failure_mode": "manual reconciliations",
        "current_workaround": "manual work",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "consequence",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "venting_without_transferable_workflow_problem" in screening["negative_signals"]


def test_classify_source_signal_rejects_github_internal_maintenance_issue():
    finding_data = {
        "source": "github-issue/example/repo",
        "source_url": "https://github.com/example/repo/issues/1",
        "finding_kind": "problem_signal",
    }
    signal_payload = {
        "title": "[nightly] Missing unit tests for all 7 CSV importers",
        "body_excerpt": "Backlog cleanup item to improve coverage and add regression files for nightly QA.",
        "source_type": "github_issue",
        "metadata_json": {"evidence": {}},
    }
    atom_payload = {
        "user_role": "engineer",
        "segment": "engineering team",
        "job_to_be_done": "increase test coverage",
        "trigger_event": "nightly QA",
        "failure_mode": "missing unit tests",
        "current_workaround": "manual regression checklist",
        "cost_consequence_clues": "",
        "frequency_clues": "nightly",
    }

    classification = classify_source_signal(finding_data, signal_payload, atom_payload)

    assert classification["source_class"] == "low_signal_summary"
    assert "github_product_specific_issue" in classification["reasons"]


def test_classify_source_signal_rejects_finance_acquisition_chatter():
    finding_data = {
        "source": "reddit-problem/ecommerce",
        "source_url": "https://reddit.com/r/ecommerce/comments/melio",
        "finding_kind": "problem_signal",
    }
    signal_payload = {
        "title": "Xero just bought Melio. Will this finally fix vendor payments?",
        "body_excerpt": (
            "This is a big deal for finance teams, but it mostly sounds like another acquisition and "
            "the same old duct-taped vendor payments story."
        ),
        "source_type": "forum",
        "metadata_json": {"evidence": {}},
    }
    atom_payload = {
        "user_role": "finance lead",
        "segment": "small ecommerce operators",
        "job_to_be_done": "streamline vendor payments",
        "trigger_event": "after the acquisition",
        "failure_mode": "vendor payment stack feels bolted on",
        "current_workaround": "using inboxes and spreadsheets",
        "cost_consequence_clues": "time loss",
        "frequency_clues": "weekly",
    }

    classification = classify_source_signal(finding_data, signal_payload, atom_payload)

    assert classification["source_class"] == "competition_signal"
    assert "finance_acquisition_or_vendor_chatter" in classification["reasons"]


def test_classify_source_signal_rejects_broad_buying_prompt_even_with_recommendation_comments():
    finding_data = {
        "source": "reddit-problem/smallbusiness",
        "source_url": "https://reddit.com/r/smallbusiness/comments/cards",
        "finding_kind": "problem_signal",
    }
    signal_payload = {
        "title": "Growing team looking for the best virtual credit card for expense automation",
        "body_excerpt": (
            "We are drowning in receipts, evaluating Ramp and Brex, and want a corporate card solution "
            "for subscriptions, ad spend, and reimbursements."
        ),
        "source_type": "forum",
        "metadata_json": {
            "evidence": {
                "comments": [
                    "Ramp solved this for us.",
                    "Brex or Airbase will clean up the receipt mess fast.",
                    "Wise is better if your issue is international declines.",
                ]
            }
        },
    }
    atom_payload = {
        "user_role": "finance lead",
        "segment": "small business operations",
        "job_to_be_done": "automate expense tracking and reimbursement",
        "trigger_event": "during month-end reimbursements",
        "pain_statement": "receipt chasing is slow and messy",
        "failure_mode": "manual receipt collection for reimbursements",
        "current_workaround": "personal cards and reimbursement spreadsheets",
        "cost_consequence_clues": "time loss",
        "frequency_clues": "monthly",
    }

    classification = classify_source_signal(finding_data, signal_payload, atom_payload)

    assert classification["source_class"] == "low_signal_summary"
    assert "finance_tool_shopping_without_specific_failure" in classification["reasons"]


def test_qualify_problem_signal_keeps_github_transferable_workflow_failure():
    finding_data = {
        "source": "github-issue/example/repo",
        "source_url": "https://github.com/example/repo/issues/2",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "CSV import duplicates invoices during reconciliation",
        "body_excerpt": (
            "Ops admins importing Stripe payout CSVs into QuickBooks get duplicate invoices, "
            "then fall back to spreadsheet cleanup before month-end close."
        ),
        "source_type": "github_issue",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "import payout csvs into accounting without duplicate cleanup",
        "trigger_event": "during month-end reconciliation",
        "pain_statement": "duplicate invoices break reconciliation",
        "failure_mode": "csv import creates duplicate invoices",
        "current_workaround": "spreadsheet cleanup and manual matching",
        "urgency_clues": "month-end close",
        "frequency_clues": "every week",
        "cost_consequence_clues": "hours lost and reconciliation risk",
        "why_now_clues": "after import",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is True
    assert "github_without_external_workflow_context" not in screening["negative_signals"]


def test_qualify_problem_signal_rejects_finance_tool_shopping_prompt():
    finding_data = {
        "source": "reddit-problem/smallbusiness",
        "source_url": "https://reddit.com/r/smallbusiness/comments/cards",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Growing team looking for the best virtual credit card for expense automation",
        "body_excerpt": (
            "We are drowning in receipts, evaluating Ramp and Brex, and want a corporate card solution "
            "for subscriptions, ad spend, and reimbursements."
        ),
        "source_type": "forum",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "automate expense tracking and reimbursement",
        "trigger_event": "during month-end reimbursements",
        "pain_statement": "receipt chasing is slow and messy",
        "failure_mode": "manual receipt collection for reimbursements",
        "current_workaround": "personal cards and reimbursement spreadsheets",
        "urgency_clues": "",
        "frequency_clues": "monthly",
        "cost_consequence_clues": "time loss",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "finance_tool_shopping_without_specific_failure" in screening["negative_signals"]
    assert "broad_buying_prompt_without_wedge_slice" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_broad_finance_visibility_prompt():
    finding_data = {
        "source": "github-issue/example/repo",
        "source_url": "https://github.com/example/repo/issues/3",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "SMB owners can't see unified cash position across multiple payment channels",
        "body_excerpt": (
            "I want a single up-to-date view of what has been received versus what is still outstanding "
            "across bank transfers, Stripe, Venmo, cash, and checks."
        ),
        "source_type": "github_issue",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "see a unified cash position across payment channels",
        "trigger_event": "when clients pay through many channels",
        "pain_statement": "cash visibility is fragmented",
        "failure_mode": "revenue is spread across channels with no shared ledger",
        "current_workaround": "checking channels separately and manual reconciliation",
        "urgency_clues": "",
        "frequency_clues": "every week",
        "cost_consequence_clues": "guessing cash position",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "broad_finance_visibility_without_specific_failure" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_review_without_transferable_workflow():
    finding_data = {
        "source": "shopify-review/parcel-intelligence",
        "source_url": "https://apps.shopify.com/parcel-intelligence/reviews/1",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Support ignored me and pricing is awful",
        "body_excerpt": "The app is expensive, support never replied, and I regret paying for it.",
        "source_type": "review",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "use the app",
        "trigger_event": "",
        "pain_statement": "support ignored me and pricing is awful",
        "failure_mode": "the app is bad",
        "current_workaround": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "review_without_transferable_workflow" in screening["negative_signals"]


def test_qualify_problem_signal_keeps_review_with_transferable_workflow_failure():
    finding_data = {
        "source": "wordpress-review/updraftplus",
        "source_url": "https://wordpress.org/support/topic/backup-restore-fails/",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Backup restore fails and we fall back to manual file recovery",
        "body_excerpt": (
            "After updates, restore jobs fail, customer sites stay down, and the team manually "
            "downloads files to recover content."
        ),
        "source_type": "review",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "restore customer sites reliably after updates",
        "trigger_event": "after updates",
        "pain_statement": "restore jobs fail and sites stay down",
        "failure_mode": "backup restore fails",
        "current_workaround": "manual file recovery",
        "urgency_clues": "sites stay down",
        "frequency_clues": "every update",
        "cost_consequence_clues": "downtime risk and support load",
        "why_now_clues": "after updates",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is True
    assert "review_without_transferable_workflow" not in screening["negative_signals"]


def test_qualify_problem_signal_rejects_youtube_roundup_without_concrete_comments():
    finding_data = {
        "source": "youtube",
        "source_url": "https://www.youtube.com/watch?v=abc123",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "Best Shopify Apps for 2026",
        "body_excerpt": "My top tools and app recommendations for every store.",
        "source_type": "youtube",
        "metadata_json": {
            "source_class": "pain_signal",
            "evidence": {
                "comments": [
                    {"text": "Great list"},
                    {"text": "Thanks for the recommendation"},
                ]
            },
        },
    }
    atom_payload = {
        "job_to_be_done": "pick a better app",
        "trigger_event": "",
        "pain_statement": "looking for recommendations",
        "failure_mode": "shopping for a better app",
        "current_workaround": "",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "youtube_roundup_or_recommendation_chatter" in screening["negative_signals"]


def test_qualify_problem_signal_rejects_stackoverflow_local_implementation_issue():
    finding_data = {
        "source": "stackoverflow",
        "source_url": "https://stackoverflow.com/questions/1",
        "finding_kind": "problem_signal",
        "source_class": "pain_signal",
    }
    signal_payload = {
        "title": "React component unit test mock failing after refactor",
        "body_excerpt": "Need help fixing a mock setup in Jest after refactoring component props.",
        "source_type": "stackoverflow",
        "metadata_json": {"source_class": "pain_signal", "evidence": {}},
    }
    atom_payload = {
        "job_to_be_done": "fix react tests",
        "trigger_event": "after refactor",
        "pain_statement": "mock setup broke",
        "failure_mode": "unit test mock failing",
        "current_workaround": "manual mocking",
        "urgency_clues": "",
        "frequency_clues": "",
        "cost_consequence_clues": "",
        "why_now_clues": "",
    }

    screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)

    assert screening["accepted"] is False
    assert "stackoverflow_local_implementation_issue" in screening["negative_signals"]


def test_cluster_label_prefers_specific_trigger_or_failure():
    atoms = [
        ProblemAtom(
            signal_id=1,
            finding_id=1,
            cluster_key="ops|emails|confirmation",
            segment="commerce operators",
            user_role="operations lead",
            job_to_be_done="keep order confirmations and automated emails reliable",
            trigger_event="after Etsy stops sending confirmation emails",
            pain_statement="Customers are not getting confirmation emails.",
            failure_mode="confirmation emails fail after purchase",
            current_workaround="manual email replies",
            current_tools="Etsy, Gmail",
            urgency_clues="need",
            frequency_clues="every day",
            emotional_intensity=0.7,
            cost_consequence_clues="time",
            why_now_clues="platform change",
            confidence=0.8,
            atom_json="{}",
        )
    ]
    signals = [
        RawSignal(
            finding_id=1,
            source_name="reddit-problem",
            source_type="forum",
            source_url="https://reddit.com/test",
            title="Trouble with Etsy Automated Emails?",
            body_excerpt="Customers are not getting confirmation emails after purchase.",
            quote_text="Customers are not getting confirmation emails after purchase.",
            published_at=None,
            role_hint="operations lead",
            timestamp_hint="",
            content_hash="abc",
            metadata={},
            id=1,
        )
    ]

    summary = build_cluster_summary(atoms, signals)

    assert "operations lead" in summary["label"].lower()
    assert "confirmation emails" in summary["label"].lower()
    assert "bottlenecks" not in summary["label"].lower()


def test_validation_experiment_hypothesis_uses_structured_fields():
    atom = ProblemAtom(
        signal_id=1,
        finding_id=1,
        cluster_key="ops|payouts|exceptions",
        segment="finance ops",
        user_role="finance lead",
        job_to_be_done="reconcile payouts and exceptions without manual cleanup",
        trigger_event="after payout exceptions hit",
        pain_statement="Teams still fix payout exceptions by hand.",
        failure_mode="the current workflow breaks on payout exceptions",
        current_workaround="spreadsheets, copy paste",
        current_tools="Excel",
        urgency_clues="urgent",
        frequency_clues="every week",
        emotional_intensity=0.7,
        cost_consequence_clues="time, consequence",
        why_now_clues="pricing change",
        confidence=0.8,
        atom_json="{}",
    )

    plan = plan_validation_experiment(
        atom,
        {"label": "finance lead - reconcile payouts", "summary_json": {"cluster_context": "after payout exceptions hit"}, "atom_count": 3, "evidence_quality": 0.8},
        {"frequency_score": 0.7, "evidence_quality": 0.8, "reachability": 0.7, "cost_of_inaction": 0.7, "segment_concentration": 0.7},
        {"market_gap": "underserved_edge_case"},
    )

    assert "finance lead teams experiencing after payout exceptions hit" in plan["hypothesis"].lower()
    assert "spreadsheets" in plan["hypothesis"].lower()
    assert "missed work" in plan["hypothesis"].lower() or "operational risk" in plan["hypothesis"].lower()


def test_score_opportunity_lifts_ops_and_compliance_value_support():
    atom = ProblemAtom(
        signal_id=1,
        finding_id=1,
        cluster_key="compliance|audit|exports",
        segment="small business compliance",
        user_role="compliance lead",
        job_to_be_done="keep multi-framework compliance evidence monitoring reliable",
        trigger_event="after manual audit exports pile up",
        pain_statement="Teams still merge audit exports manually.",
        failure_mode="manual audit exports break reporting",
        current_workaround="manual work, custom scripts",
        current_tools="M365, spreadsheets",
        urgency_clues="urgent",
        frequency_clues="every week",
        emotional_intensity=0.6,
        cost_consequence_clues="time, audit risk",
        why_now_clues="soc 2 deadline",
        confidence=0.8,
        atom_json="{}",
    )
    signal = RawSignal(
        finding_id=1,
        source_name="reddit-problem",
        source_type="forum",
        source_url="https://reddit.com/r/sysadmin/comments/1",
        title="Manual audit exports are breaking compliance reporting",
        body_excerpt="Manual audit exports still require custom scripts and spreadsheet cleanup.",
        quote_text="Manual audit exports still require custom scripts and spreadsheet cleanup.",
        published_at=None,
        role_hint="compliance lead",
        timestamp_hint="",
        content_hash="value-lift",
        metadata={},
        id=1,
    )
    cluster_summary = {"atom_count": 2, "evidence_quality": 0.7}
    validation_evidence = {
        "scores": {
            "problem_score": 0.42,
            "feasibility_score": 0.7,
            "value_score": 0.4,
        },
        "evidence": {
            "recurrence_query_coverage": 0.4,
            "recurrence_doc_count": 3,
            "recurrence_domain_count": 2,
            "recurrence_results_by_source": {"reddit": 3, "github": 0, "web": 0},
        },
        "corroboration": {
            "corroboration_score": 0.41,
            "evidence_sufficiency": 0.34,
            "cross_source_match_score": 0.22,
            "source_family_diversity": 1,
            "core_source_family_diversity": 1,
            "generalizability_score": 0.62,
            "generalizability_penalty": 0.0,
        },
        "market_enrichment": {
            "operational_buyer_score": 0.35,
            "compliance_burden_score": 0.54,
            "cost_pressure_score": 0.54,
            "buyer_intent_score": 0.56,
            "demand_score": 0.45,
            "competition_score": 0.0,
            "trend_score": 0.38,
            "review_signal_score": 0.0,
            "willingness_to_pay_signal": 0.0,
            "wedge_value_lift": 0.0,
            "multi_source_value_lift": 0.23,
        },
    }
    market_gap = {"market_gap": "underserved_edge_case", "why_now_strength": 0.35}

    scorecard = score_opportunity(atom, signal, cluster_summary, validation_evidence, market_gap)

    # Note: operational_value_lift and value_support were reduced after v3 bug fix
    # that removed triple-counted variables (operational_buyer, compliance_burden, cost_pressure)
    # from these calculations. The new values reflect corrected scoring without inflation.
    assert scorecard["operational_value_lift"] > 0.05  # Reduced from 0.25 due to bug fix
    assert scorecard["value_support"] > 0.30  # Reduced from 0.5 due to bug fix
