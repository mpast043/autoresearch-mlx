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
        },
        {"market_gap": "needs_more_recurrence_evidence"},
        [],
        promotion_threshold=0.18,
        park_threshold=0.15,
    )
    assert decision["recommendation"] == "kill"
    assert decision["decision_reason"] == "below_threshold"


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
