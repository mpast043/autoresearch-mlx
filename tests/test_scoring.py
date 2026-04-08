"""Tests for the scoring module."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research.scoring import (
    score_market_fit,
    score_technical_fit,
    score_distribution_fit,
    compute_composite_score,
    make_decision,
    score_recurrence,
    score_corroboration,
)


class TestScoreMarketFit:
    """Tests for the score_market_fit function."""

    def test_empty_evidence_returns_zero(self):
        assert score_market_fit({}) == 0.0

    def test_revenue_mention(self):
        score = score_market_fit({"mentions_revenue": True})
        assert score == pytest.approx(0.3)

    def test_cost_mention(self):
        score = score_market_fit({"mentions_cost": True})
        assert score == pytest.approx(0.2)

    def test_revenue_and_cost(self):
        score = score_market_fit({"mentions_revenue": True, "mentions_cost": True})
        assert score == pytest.approx(0.5)

    def test_user_count_tiers(self):
        # user_count > 10 adds 0.1, > 100 adds 0.2, > 1000 adds 0.3
        assert score_market_fit({"user_count": 5}) == pytest.approx(0.0)   # <= 10
        assert score_market_fit({"user_count": 50}) == pytest.approx(0.1)  # > 10, <= 100
        assert score_market_fit({"user_count": 101}) == pytest.approx(0.2)
        assert score_market_fit({"user_count": 1001}) == pytest.approx(0.3)

    def test_frequent_usage(self):
        score = score_market_fit({"frequent_usage": True})
        assert score == pytest.approx(0.2)

    def test_all_evidence_max_capped_at_one(self):
        score = score_market_fit({
            "mentions_revenue": True,
            "mentions_cost": True,
            "user_count": 5000,
            "frequent_usage": True,
        })
        # 0.3 + 0.2 + 0.3 + 0.2 = 1.0, capped at 1.0
        assert score == pytest.approx(1.0)

    def test_combined_evidence_partial(self):
        score = score_market_fit({
            "mentions_revenue": True,
            "user_count": 50,
        })
        # 0.3 (revenue) + 0.1 (user_count > 10) = 0.4
        assert score == pytest.approx(0.4)

    def test_default_user_count_is_zero(self):
        # user_count defaults to 0 when not provided
        score = score_market_fit({"mentions_revenue": True})
        assert score == pytest.approx(0.3)

    def test_user_count_boundary_values(self):
        # Boundaries are strict: > 10, > 100, > 1000
        assert score_market_fit({"user_count": 10}) == pytest.approx(0.0)   # not > 10
        assert score_market_fit({"user_count": 100}) == pytest.approx(0.1)  # > 10 but not > 100
        assert score_market_fit({"user_count": 1000}) == pytest.approx(0.2) # > 100 but not > 1000

    def test_false_flags_dont_contribute(self):
        # False boolean flags should not add to score
        score = score_market_fit({
            "mentions_revenue": False,
            "mentions_cost": False,
            "frequent_usage": False,
        })
        assert score == pytest.approx(0.0)


class TestScoreTechnicalFit:
    """Tests for the score_technical_fit function."""

    def test_empty_evidence_returns_zero(self):
        assert score_technical_fit({}) == 0.0

    def test_specific_problem(self):
        score = score_technical_fit({"specific_problem": True})
        assert score == pytest.approx(0.4)

    def test_technical_complexity(self):
        score = score_technical_fit({"technical_complexity": True})
        assert score == pytest.approx(0.3)

    def test_has_current_solution(self):
        score = score_technical_fit({"has_current_solution": True})
        assert score == pytest.approx(0.2)

    def test_willingness_to_pay(self):
        score = score_technical_fit({"willingness_to_pay": True})
        assert score == pytest.approx(0.1)

    def test_all_technical_evidence(self):
        score = score_technical_fit({
            "specific_problem": True,
            "technical_complexity": True,
            "has_current_solution": True,
            "willingness_to_pay": True,
        })
        # 0.4 + 0.3 + 0.2 + 0.1 = 1.0
        assert score == pytest.approx(1.0)

    def test_partial_technical_evidence(self):
        score = score_technical_fit({
            "specific_problem": True,
            "willingness_to_pay": True,
        })
        assert score == pytest.approx(0.5)

    def test_false_flags_dont_contribute(self):
        score = score_technical_fit({
            "specific_problem": False,
            "technical_complexity": False,
        })
        assert score == pytest.approx(0.0)


class TestScoreDistributionFit:
    """Tests for the score_distribution_fit function."""

    def test_empty_evidence_returns_zero(self):
        assert score_distribution_fit({}) == 0.0

    def test_has_distribution_channel(self):
        score = score_distribution_fit({"has_distribution_channel": True})
        assert score == pytest.approx(0.4)

    def test_accessible_segment(self):
        score = score_distribution_fit({"accessible_segment": True})
        assert score == pytest.approx(0.3)

    def test_community_size_tiers(self):
        assert score_distribution_fit({"community_size": 500}) == pytest.approx(0.0)
        assert score_distribution_fit({"community_size": 1500}) == pytest.approx(0.2)
        assert score_distribution_fit({"community_size": 15000}) == pytest.approx(0.3)

    def test_all_distribution_evidence_capped(self):
        score = score_distribution_fit({
            "has_distribution_channel": True,
            "community_size": 50000,
            "accessible_segment": True,
        })
        # 0.4 + 0.3 + 0.3 = 1.0
        assert score == pytest.approx(1.0)

    def test_community_size_boundary_values(self):
        assert score_distribution_fit({"community_size": 1000}) == pytest.approx(0.0)
        assert score_distribution_fit({"community_size": 10000}) == pytest.approx(0.2)


class TestComputeCompositeScore:
    """Tests for the compute_composite_score function."""

    def test_default_weights(self):
        # Default weights: market=0.4, technical=0.35, distribution=0.25
        score = compute_composite_score(0.5, 0.5, 0.5)
        expected = 0.5 * 0.4 + 0.5 * 0.35 + 0.5 * 0.25
        assert score == pytest.approx(expected)

    def test_custom_weights(self):
        weights = {"market": 0.5, "technical": 0.3, "distribution": 0.2}
        score = compute_composite_score(1.0, 0.0, 0.0, weights=weights)
        assert score == pytest.approx(0.5)

    def test_zero_scores(self):
        score = compute_composite_score(0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)

    def test_perfect_scores(self):
        score = compute_composite_score(1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_mixed_scores(self):
        score = compute_composite_score(1.0, 0.5, 0.0)
        expected = 1.0 * 0.4 + 0.5 * 0.35 + 0.0 * 0.25
        assert score == pytest.approx(expected)

    def test_missing_weight_uses_default(self):
        weights = {"market": 0.5}
        score = compute_composite_score(1.0, 1.0, 1.0, weights=weights)
        # market=0.5, technical defaults to 0.35, distribution defaults to 0.25
        expected = 1.0 * 0.5 + 1.0 * 0.35 + 1.0 * 0.25
        assert score == pytest.approx(expected)


class TestMakeDecision:
    """Tests for the make_decision function."""

    def test_high_score_promotes(self):
        assert make_decision(0.8) == "promote"

    def test_low_score_kills(self):
        assert make_decision(0.2) == "kill"

    def test_mid_score_parks(self):
        assert make_decision(0.5) == "park"

    def test_default_promotion_threshold(self):
        # At exactly 0.65 with default promotion_threshold=0.65 => promote
        assert make_decision(0.65) == "promote"

    def test_default_park_threshold(self):
        # At exactly 0.35 with default park_threshold=0.35 => kill
        assert make_decision(0.35) == "kill"

    def test_just_above_park_threshold(self):
        # 0.36 is above kill threshold (0.35) and below promotion (0.65) => park
        assert make_decision(0.36) == "park"

    def test_just_below_promotion_threshold(self):
        # 0.64 is below promotion (0.65) and above kill (0.35) => park
        assert make_decision(0.64) == "park"

    def test_custom_thresholds(self):
        assert make_decision(0.5, promotion_threshold=0.5, park_threshold=0.2) == "promote"
        assert make_decision(0.3, promotion_threshold=0.5, park_threshold=0.2) == "park"
        assert make_decision(0.1, promotion_threshold=0.5, park_threshold=0.2) == "kill"

    def test_score_one_promotes(self):
        assert make_decision(1.0) == "promote"

    def test_score_zero_kills(self):
        assert make_decision(0.0) == "kill"

    def test_boundary_exactly_at_promotion(self):
        assert make_decision(0.65, promotion_threshold=0.65) == "promote"

    def test_boundary_exactly_at_park_kill(self):
        assert make_decision(0.35, park_threshold=0.35) == "kill"


class TestScoreRecurrence:
    """Tests for the score_recurrence function."""

    def test_empty_documents(self):
        assert score_recurrence([]) == 0.0

    def test_no_recurrence_keywords(self):
        docs = [
            {"title": "hello world", "snippet": "a simple greeting"},
        ]
        assert score_recurrence(docs) == pytest.approx(0.0)

    def test_single_recurrence_keyword(self):
        docs = [
            {"title": "I keep having this issue", "snippet": "problem persists"},
        ]
        # One match: 1/3 ≈ 0.333
        assert score_recurrence(docs) == pytest.approx(1 / 3)

    def test_multiple_recurrence_keywords(self):
        docs = [
            {"title": "keep having problems", "snippet": "every time it fails"},
            {"title": "this happens repeatedly", "snippet": "over and over again"},
        ]
        # First doc: "keep having" + "every time" = 1 match (per doc)
        # Second doc: "repeatedly" = 1 match (but "over and over" might not be in the list)
        # Recurrence keywords in score_recurrence are: "keep having", "every time", "over and over", "repeatedly"
        # Both docs have at least one keyword each, so recurrence_count = 2
        # 2/3 ≈ 0.667
        assert score_recurrence(docs) == pytest.approx(2 / 3)

    def test_max_recurrence_capped_at_one(self):
        docs = [
            {"title": "keep having", "snippet": ""},
            {"title": "every time", "snippet": ""},
            {"title": "over and over", "snippet": ""},
            {"title": "repeatedly", "snippet": ""},
        ]
        # 4 docs with keywords => 4/3 > 1.0, capped at 1.0
        assert score_recurrence(docs) == pytest.approx(1.0)

    def test_doc_missing_fields(self):
        docs = [
            {},  # no title or snippet
        ]
        # Empty string won't match any keyword
        assert score_recurrence(docs) == pytest.approx(0.0)

    def test_snippet_also_searched(self):
        docs = [
            {"title": "normal title", "snippet": "keep having this problem"},
        ]
        assert score_recurrence(docs) == pytest.approx(1 / 3)


class TestScoreCorroboration:
    """Tests for the score_corroboration function."""

    def test_empty_matches(self):
        assert score_corroboration([]) == 0.0

    def test_single_match_no_source(self):
        # 1 match, no source => count_score = min(0.7, 1/2*0.7) = 0.35, diversity_bonus = 0
        score = score_corroboration([{"title": "test"}])
        assert score == pytest.approx(0.35)

    def test_single_match_with_source(self):
        # 1 match, 1 unique source => count_score = 0.35, diversity_bonus = 0.1
        score = score_corroboration([{"title": "test", "source": "reddit"}])
        assert score == pytest.approx(0.45)

    def test_multiple_matches_same_source(self):
        # 2 matches, 1 source => count_score = min(0.7, 2/2*0.7) = 0.7, diversity_bonus = 0.1
        matches = [
            {"source": "reddit"},
            {"source": "reddit"},
        ]
        score = score_corroboration(matches, required=2)
        assert score == pytest.approx(0.8)

    def test_multiple_matches_diverse_sources(self):
        # 3 matches, 3 sources => count_score = min(0.7, 3/2*0.7) = 0.7, diversity_bonus = min(0.3, 3*0.1) = 0.3
        matches = [
            {"source": "reddit"},
            {"source": "github"},
            {"source": "web"},
        ]
        score = score_corroboration(matches, required=2)
        assert score == pytest.approx(1.0)

    def test_custom_required_count(self):
        # 1 match, required=5 => count_score = min(0.7, 1/5*0.7) = 0.14
        score = score_corroboration([{"source": "reddit"}], required=5)
        # diversity_bonus = 0.1
        assert score == pytest.approx(0.24)

    def test_diversity_bonus_capped(self):
        # 5 sources, diversity_bonus = min(0.3, 5*0.1) = 0.3
        matches = [
            {"source": "a"},
            {"source": "b"},
            {"source": "c"},
            {"source": "d"},
            {"source": "e"},
        ]
        score = score_corroboration(matches, required=1)
        # count_score = min(0.7, 5/1*0.7) = 0.7
        # diversity_bonus = min(0.3, 5*0.1) = 0.3
        # total = 1.0, capped at 1.0
        assert score == pytest.approx(1.0)

    def test_matches_without_source_key(self):
        # Empty source strings are excluded from diversity count
        matches = [
            {"source": ""},
            {"source": "reddit"},
        ]
        # Only 1 unique non-empty source => diversity_bonus = 0.1
        # count_score = min(0.7, 2/2*0.7) = 0.7
        score = score_corroboration(matches, required=2)
        assert score == pytest.approx(0.8)