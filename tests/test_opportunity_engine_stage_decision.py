"""Focused tests for stage_decision near-miss handling."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opportunity_engine import diagnose_stage_decision, stage_decision


def test_stage_decision_parks_sharp_but_thin_evidence_candidate():
    decision = stage_decision(
        {
            "decision_score": 0.121,
            "problem_truth_score": 0.13,
            "revenue_readiness_score": 0.2,
            "frequency_score": 0.28,
            "evidence_quality": 0.44,
            "value_support": 0.41,
            "cost_of_inaction": 0.47,
            "workaround_density": 0.39,
            "buildability": 0.58,
            "boring_money_fit": 0.48,
            "incumbent_gap_score": 0.38,
            "recurring_workflow_score": 0.39,
        },
        {"market_gap": "needs_more_recurrence_evidence"},
        [],
        promotion_threshold=0.18,
        park_threshold=0.15,
    )
    assert decision["recommendation"] == "park"
    assert decision["decision_reason"] == "sharp_but_thin_evidence"


def test_stage_decision_does_not_rescue_weak_candidate():
    decision = stage_decision(
        {
            "decision_score": 0.122,
            "problem_truth_score": 0.12,
            "revenue_readiness_score": 0.19,
            "frequency_score": 0.27,
            "evidence_quality": 0.39,
            "value_support": 0.33,
            "cost_of_inaction": 0.35,
            "workaround_density": 0.28,
            "buildability": 0.47,
            "boring_money_fit": 0.2,
            "incumbent_gap_score": 0.16,
            "recurring_workflow_score": 0.22,
        },
        {"market_gap": "needs_more_recurrence_evidence"},
        [],
        promotion_threshold=0.18,
        park_threshold=0.15,
    )
    assert decision["recommendation"] == "kill"
    assert decision["decision_reason"] == "unlikely_or_economically_weak"


def test_diagnose_stage_decision_reports_sharp_near_miss_flag():
    diagnosis = diagnose_stage_decision(
        {
            "decision_score": 0.121,
            "problem_truth_score": 0.13,
            "revenue_readiness_score": 0.2,
            "frequency_score": 0.28,
            "evidence_quality": 0.44,
            "value_support": 0.41,
            "cost_of_inaction": 0.47,
            "workaround_density": 0.39,
            "buildability": 0.58,
            "boring_money_fit": 0.48,
            "incumbent_gap_score": 0.38,
            "recurring_workflow_score": 0.39,
            "composite_score": 0.12,
            "problem_plausibility": 0.5,
            "evidence_sufficiency": 0.32,
        },
        {"market_gap": "needs_more_recurrence_evidence"},
        [],
        promotion_threshold=0.18,
        park_threshold=0.15,
    )
    assert diagnosis["sharp_research_candidate"] is True
    assert diagnosis["decision"]["decision_reason"] == "sharp_but_thin_evidence"


def test_stage_decision_does_not_promote_non_boring_money_candidate():
    decision = stage_decision(
        {
            "decision_score": 0.46,
            "problem_truth_score": 0.21,
            "revenue_readiness_score": 0.31,
            "frequency_score": 0.55,
            "evidence_quality": 0.73,
            "value_support": 0.58,
            "cost_of_inaction": 0.49,
            "workaround_density": 0.19,
            "buildability": 0.62,
            "boring_money_fit": 0.23,
            "incumbent_gap_score": 0.18,
            "recurring_workflow_score": 0.27,
        },
        {"market_gap": "underserved_edge_case"},
        [],
        promotion_threshold=0.42,
        park_threshold=0.15,
    )
    assert decision["recommendation"] == "park"
    assert decision["decision_reason"] == "park_recurrence"


def test_stage_decision_promotes_strong_evidence_boring_money_candidate():
    decision = stage_decision(
        {
            "decision_score": 0.3858,
            "problem_truth_score": 0.4011,
            "revenue_readiness_score": 0.3671,
            "frequency_score": 0.5292,
            "evidence_quality": 0.7297,
            "value_support": 0.2849,
            "cost_of_inaction": 0.5008,
            "workaround_density": 0.5,
            "buildability": 0.59,
            "boring_money_fit": 0.5301,
            "incumbent_gap_score": 0.6042,
            "recurring_workflow_score": 0.5689,
        },
        {"market_gap": "needs_more_recurrence_evidence"},
        [],
        promotion_threshold=0.42,
        park_threshold=0.15,
    )
    assert decision["recommendation"] == "promote"
    assert decision["decision_reason"] == "strong_evidence_override"
