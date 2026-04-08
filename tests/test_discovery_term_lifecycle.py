"""Tests for discovery term lifecycle management."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database
from src.discovery_term_lifecycle import (
    TermLifecycleManager,
    TermMetrics,
    calculate_quality_score,
    calculate_specificity_score,
    calculate_consequence_score,
    calculate_platform_native_score,
    calculate_plugin_fit_score,
    calculate_wedge_quality_score,
    recompute_wedge_quality_score,
    is_vague_bucket,
    get_platform_from_term,
    compute_next_state,
    DEFAULT_CONFIG,
    LifecycleConfig,
)


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


class TestTermMetrics:
    """Tests for TermMetrics and quality scoring."""

    def test_calculate_quality_score_with_success(self):
        """High quality when producing candidates."""
        metrics = TermMetrics(
            times_searched=3,
            findings_emitted=5,
            validations=2,
            passes=1,
            prototype_candidates=1,
            build_briefs=0,
            screened_out=2,
        )
        score = calculate_quality_score(metrics)
        assert score > 0.5

    def test_calculate_quality_score_with_failure(self):
        """Low quality when producing no findings."""
        metrics = TermMetrics(
            times_searched=5,
            findings_emitted=0,
            validations=0,
            passes=0,
            prototype_candidates=0,
            build_briefs=0,
            screened_out=10,
            low_yield_count=3,
        )
        score = calculate_quality_score(metrics)
        assert score < 0.3

    def test_calculate_quality_score_new_term(self):
        """New terms have neutral quality."""
        metrics = TermMetrics(times_searched=0)
        score = calculate_quality_score(metrics)
        assert score == 0.0


class TestStateTransitions:
    """Tests for state transition logic."""

    def test_new_to_active_on_first_use(self):
        """New term becomes active on first use."""
        metrics = TermMetrics(times_searched=1, findings_emitted=1)
        new_state, reason = compute_next_state("new", metrics)
        assert new_state == "active"
        assert "first use" in reason

    def test_active_to_high_performing_on_success(self):
        """Active term promotes to high_performing with candidates."""
        metrics = TermMetrics(
            times_searched=3,
            findings_emitted=5,
            validations=2,
            prototype_candidates=1,
            passes=1,
        )
        new_state, reason = compute_next_state("active", metrics)
        assert new_state == "high_performing"

    def test_active_to_weak_on_repeated_failure(self):
        """Active term demotes to weak on repeated low yield."""
        config = LifecycleConfig(max_low_yield_before_demotion=2, min_runs_for_assessment=2)
        metrics = TermMetrics(
            times_searched=3,
            findings_emitted=0,
            validations=0,
            low_yield_count=2,
        )
        new_state, reason = compute_next_state("active", metrics, config)
        assert new_state == "weak"

    def test_weak_to_paused_on_continued_failure(self):
        """Weak term goes to paused after more failures."""
        config = LifecycleConfig(max_low_yield_before_pause=2, min_runs_for_assessment=2)
        metrics = TermMetrics(
            times_searched=5,
            findings_emitted=0,
            low_yield_count=2,
        )
        new_state, reason = compute_next_state("weak", metrics, config)
        assert new_state == "paused"

    def test_weak_to_exhausted_on_too_many_failures(self):
        """Weak term exhausts after too many failures."""
        config = LifecycleConfig(max_weak_runs_before_exhausted=3, min_runs_for_assessment=2)
        metrics = TermMetrics(
            times_searched=6,
            low_yield_count=3,
        )
        new_state, reason = compute_next_state("weak", metrics, config)
        assert new_state == "exhausted"

    def test_banned_stays_banned(self):
        """Banned terms cannot be auto-transitioned."""
        metrics = TermMetrics(times_searched=10, prototype_candidates=5)
        new_state, reason = compute_next_state("banned", metrics)
        assert new_state == "banned"
        assert "manually banned" in reason

    def test_high_performing_to_completed_with_build_brief(self):
        """High performing term completes if it produces a build brief."""
        metrics = TermMetrics(
            times_searched=5,
            build_briefs=1,
            prototype_candidates=2,
        )
        new_state, reason = compute_next_state("high_performing", metrics)
        assert new_state == "completed"

    def test_active_to_completed_with_build_brief(self):
        """Active term completes if it produces a build brief."""
        metrics = TermMetrics(
            times_searched=3,
            build_briefs=1,
        )
        new_state, reason = compute_next_state("active", metrics)
        assert new_state == "completed"


class TestTermLifecycleManager:
    """Tests for TermLifecycleManager."""

    def test_ensure_term_exists_creates_new_term(self, temp_db):
        """Ensure term creates a new term if it doesn't exist."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "test keyword")

        term = temp_db.get_search_term("keyword", "test keyword")
        assert term is not None
        assert term["term_value"] == "test keyword"
        assert term["state"] == "new"

    def test_record_search_run_with_success(self, temp_db):
        """Recording a successful search run promotes the term."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "successful keyword")

        result = manager.record_search_run(
            "keyword",
            "successful keyword",
            findings_emitted=3,
            validations=1,
            prototype_candidates=1,
        )

        assert result["old_state"] == "new"
        assert result["new_state"] == "active"

        # After more runs with success, should promote
        result2 = manager.record_search_run(
            "keyword",
            "successful keyword",
            findings_emitted=2,
            validations=1,
            prototype_candidates=1,
        )
        assert result2["new_state"] == "high_performing"

    def test_record_search_run_with_failure(self, temp_db):
        """Recording failed search runs demotes the term."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "failing keyword")

        # First run - no findings
        result = manager.record_search_run(
            "keyword",
            "failing keyword",
            findings_emitted=0,
            validations=0,
            low_yield=True,
        )
        assert result["new_state"] == "active"  # First run still active

        # Second run - still no findings
        result2 = manager.record_search_run(
            "keyword",
            "failing keyword",
            findings_emitted=0,
            validations=0,
            low_yield=True,
        )
        # Should demote to weak
        assert result2["new_state"] == "weak"

    def test_ban_term(self, temp_db):
        """Manually banning a term sets state to banned."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "bad keyword")

        manager.ban_term("keyword", "bad keyword", reason="test ban")

        term = temp_db.get_search_term("keyword", "bad keyword")
        assert term["state"] == "banned"
        assert "test ban" in term["notes"]

    def test_reactivate_term(self, temp_db):
        """Reactivating a paused term sets state to active."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "paused keyword")

        # Manually set to paused
        temp_db.update_search_term_state("keyword", "paused keyword", "paused", notes="test pause")

        # Reactivate
        manager.reactivate_term("keyword", "paused keyword")

        term = temp_db.get_search_term("keyword", "paused keyword")
        assert term["state"] == "active"

    def test_reset_term(self, temp_db):
        """Resetting a term clears all metrics."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "reset keyword")

        # Add some metrics
        temp_db.update_search_term_metrics(
            "keyword",
            "reset keyword",
            times_searched=5,
            findings_emitted=10,
            validations=3,
        )

        # Reset
        manager.reset_term("keyword", "reset keyword")

        term = temp_db.get_search_term("keyword", "reset keyword")
        assert term["state"] == "new"
        assert term["times_searched"] == 0
        assert term["findings_emitted"] == 0

    def test_get_available_terms(self, temp_db):
        """Getting available terms excludes exhausted/banned."""
        manager = TermLifecycleManager(temp_db)

        # Add terms in different states
        manager.ensure_term_exists("keyword", "new term")
        manager.ensure_term_exists("keyword", "active term")
        temp_db.update_search_term_state("keyword", "active term", "active")

        manager.ensure_term_exists("keyword", "exhausted term")
        temp_db.update_search_term_state("keyword", "exhausted term", "exhausted")

        manager.ensure_term_exists("keyword", "banned term")
        temp_db.update_search_term_state("keyword", "banned term", "banned")

        manager.ensure_term_exists("keyword", "paused term")
        temp_db.update_search_term_state("keyword", "paused term", "paused")

        available = manager.get_available_terms("keyword")
        available_values = [t["term_value"] for t in available]

        assert "new term" in available_values
        assert "active term" in available_values
        assert "exhausted term" not in available_values
        assert "banned term" not in available_values
        assert "paused term" not in available_values

    def test_filter_terms_for_discovery(self, temp_db):
        """Filtering terms excludes specified states."""
        manager = TermLifecycleManager(temp_db)

        terms = [
            {"term_value": "new", "state": "new"},
            {"term_value": "active", "state": "active"},
            {"term_value": "exhausted", "state": "exhausted"},
            {"term_value": "paused", "state": "paused"},
            {"term_value": "banned", "state": "banned"},
        ]

        filtered = manager.filter_terms_for_discovery(terms)

        assert len(filtered) == 2
        assert any(t["term_value"] == "new" for t in filtered)
        assert any(t["term_value"] == "active" for t in filtered)


class TestDatabaseSearchTerms:
    """Tests for database search term operations."""

    def test_insert_and_get_search_term(self, temp_db):
        """Can insert and retrieve a search term."""
        term_id = temp_db.insert_search_term("keyword", "test keyword", state="new")
        assert term_id > 0

        term = temp_db.get_search_term("keyword", "test keyword")
        assert term is not None
        assert term["term_type"] == "keyword"
        assert term["term_value"] == "test keyword"
        assert term["state"] == "new"

    def test_update_search_term_state(self, temp_db):
        """Can update term state."""
        temp_db.insert_search_term("keyword", "test keyword")
        temp_db.update_search_term_state("keyword", "test keyword", "active", notes="testing")

        term = temp_db.get_search_term("keyword", "test keyword")
        assert term["state"] == "active"
        assert term["notes"] == "testing"

    def test_update_search_term_metrics(self, temp_db):
        """Can update term metrics."""
        temp_db.insert_search_term("keyword", "test keyword")
        temp_db.update_search_term_metrics(
            "keyword",
            "test keyword",
            times_searched=5,
            findings_emitted=10,
        )

        term = temp_db.get_search_term("keyword", "test keyword")
        assert term["times_searched"] == 5
        assert term["findings_emitted"] == 10

    def test_list_search_terms(self, temp_db):
        """Can list search terms with filters."""
        temp_db.insert_search_term("keyword", "kw1", state="active")
        temp_db.insert_search_term("keyword", "kw2", state="exhausted")
        temp_db.insert_search_term("subreddit", "sub1", state="active")

        # Filter by type
        kw_terms = temp_db.list_search_terms(term_type="keyword")
        assert len(kw_terms) == 2

        # Filter by state
        active_terms = temp_db.list_search_terms(state="active")
        assert len(active_terms) == 2

    def test_bulk_upsert_search_terms(self, temp_db):
        """Can bulk insert multiple terms."""
        count = temp_db.bulk_upsert_search_terms(
            "keyword",
            ["bulk kw1", "bulk kw2", "bulk kw3"],
        )
        assert count == 3

        terms = temp_db.list_search_terms(term_type="keyword")
        assert len(terms) == 3

    def test_search_term_exists(self, temp_db):
        """Can check if term exists."""
        assert not temp_db.search_term_exists("keyword", "test")

        temp_db.insert_search_term("keyword", "test")

        assert temp_db.search_term_exists("keyword", "test")


class TestNicheQualityScoring:
    """Tests for niche quality scoring functions."""

    def test_specificity_score_vague_term(self):
        """Vague terms get low specificity scores."""
        score = calculate_specificity_score("manual work workflow")
        assert score < 0.5

    def test_specificity_score_specific_term(self):
        """Specific terms get high specificity scores."""
        score = calculate_specificity_score("invoice error duplicate payment")
        assert score > 0.5

    def test_consequence_score(self):
        """Consequence-heavy terms score higher."""
        low_score = calculate_consequence_score("manual work is annoying")
        high_score = calculate_consequence_score("invoice error caused late fee penalty")
        assert high_score > low_score

    def test_platform_native_score_shopify(self):
        """Shopify terms get high platform-native score."""
        score = calculate_platform_native_score("shopify app inventory sync")
        assert score >= 0.4

    def test_platform_native_score_generic(self):
        """Generic terms get low platform-native score."""
        score = calculate_platform_native_score("manual data entry workflow")
        assert score < 0.3

    def test_plugin_fit_score(self):
        """Plugin-fit terms score higher."""
        score = calculate_plugin_fit_score("chrome extension for invoice tracking")
        assert score >= 0.4

    def test_wedge_quality_score_combines_factors(self):
        """Wedge quality combines all factors."""
        # Good term: specific + consequence + platform + plugin fit
        good_score = calculate_wedge_quality_score(
            "shopify invoice error tracking app",
            specificity=0.8,
            consequence=0.7,
            platform_native=0.8,
            plugin_fit=0.8,
        )
        # Bad term: vague + no consequence
        bad_score = calculate_wedge_quality_score(
            "manual workflow sync",
            specificity=0.2,
            consequence=0.2,
            platform_native=0.1,
            plugin_fit=0.1,
        )
        assert good_score > bad_score

    def test_is_vague_bucket(self):
        """Detects vague bucket patterns."""
        assert is_vague_bucket("manual work")
        assert is_vague_bucket("keep in sync")
        assert is_vague_bucket("spreadsheet hell")
        assert not is_vague_bucket("invoice error payment")

    def test_get_platform_from_term(self):
        """Extracts platform from term."""
        assert get_platform_from_term("shopify app for inventory") == "shopify"
        assert get_platform_from_term("google docs add-on") == "google docs"
        assert get_platform_from_term("manual spreadsheet workflow") is None

    def test_term_lifecycle_manager_wedge_quality_ranking(self, temp_db):
        """Terms can be ranked by wedge quality."""
        manager = TermLifecycleManager(temp_db)

        # Add terms with different quality profiles
        manager.ensure_term_exists("keyword", "shopify inventory sync")
        manager.record_search_run("keyword", "shopify inventory sync", prototype_candidates=1, is_platform_native=True, is_buildable_wedge=True)

        manager.ensure_term_exists("keyword", "manual data entry")
        manager.record_search_run("keyword", "manual data entry", findings_emitted=5, is_vague=True)

        # Get by wedge quality
        wedge_terms = manager.get_terms_for_expansion_by_wedge_quality("keyword")
        assert len(wedge_terms) >= 2
        # Shop term should rank higher
        shop_idx = next(i for i, t in enumerate(wedge_terms) if "shopify" in t["term_value"])
        manual_idx = next(i for i, t in enumerate(wedge_terms) if "manual" in t["term_value"])
        assert shop_idx < manual_idx


class TestRecomputeWedgeQualityScore:
    """Tests for the Bayesian-adjacent wedge quality score blending."""

    def test_no_outcomes_returns_heuristic(self):
        """With zero total outcomes, return heuristic unchanged."""
        h = 0.7
        assert recompute_wedge_quality_score(h) == h

    def test_below_min_samples_returns_heuristic(self):
        """With total_opportunities < min_samples, return heuristic unchanged."""
        h = 0.7
        assert recompute_wedge_quality_score(h, buildable_opportunity_count=1, total_opportunities=1, min_samples=2) == h

    def test_with_buildable_boosts_score(self):
        """At min_samples=2 with 2 buildable, score should increase."""
        h = 0.5
        result = recompute_wedge_quality_score(
            h,
            buildable_opportunity_count=2,
            total_opportunities=2,
            min_samples=2,
        )
        # Weight = min(2/6, 0.5) ≈ 0.33
        # outcome_rate = 2/2 = 1.0
        # blended = 0.67 * 0.5 + 0.33 * 1.0 ≈ 0.67
        assert result > h
        assert result <= 1.0

    def test_with_vague_penalizes_score(self):
        """Vague outcomes should pull the score down."""
        h = 0.6
        result_no_vague = recompute_wedge_quality_score(
            h,
            buildable_opportunity_count=3,
            vague_bucket_count=0,
            total_opportunities=3,
            min_samples=2,
        )
        result_with_vague = recompute_wedge_quality_score(
            h,
            buildable_opportunity_count=3,
            vague_bucket_count=2,
            total_opportunities=5,
            min_samples=2,
        )
        assert result_with_vague < result_no_vague

    def test_large_sample_weight_caps_at_half(self):
        """At 6+ outcomes, weight approaches 0.5 but never exceeds it."""
        result = recompute_wedge_quality_score(
            heuristic_score=0.3,
            buildable_opportunity_count=5,
            vague_bucket_count=1,
            total_opportunities=6,
            min_samples=2,
        )
        # Weight = min(6/6, 0.5) = 0.5
        # outcome_rate = 5/6 * (1 - 0.3 * 1/6) ≈ 0.792
        # blended = 0.5 * 0.3 + 0.5 * 0.792 ≈ 0.546
        assert result > 0.3  # Better than heuristic
        assert result < 0.8  # But not overly optimistic

    def test_all_vague_reduces_score(self):
        """If all outcomes are vague, score should drop below heuristic."""
        h = 0.6
        result = recompute_wedge_quality_score(
            h,
            buildable_opportunity_count=0,
            vague_bucket_count=3,
            total_opportunities=3,
            min_samples=2,
        )
        # outcome_rate = 0/3 = 0, vague_penalty still applied
        assert result < h

    def test_clamped_to_zero_one(self):
        """Result is always clamped to [0, 1]."""
        result = recompute_wedge_quality_score(
            heuristic_score=1.0,
            buildable_opportunity_count=100,
            total_opportunities=100,
            min_samples=2,
        )
        assert 0.0 <= result <= 1.0


class TestRecordWedgeFeedback:
    """Tests for TermLifecycleManager.record_wedge_feedback()."""

    def test_buildable_increments_count(self, temp_db):
        """Recording buildable feedback increments buildable_opportunity_count."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "shopify inventory sync")
        temp_db.update_search_term_metrics("keyword", "shopify inventory sync", wedge_quality_score=0.5)

        manager.record_wedge_feedback(
            "shopify inventory sync",
            is_buildable_wedge=True,
            verdict="build_now",
        )

        term = temp_db.get_search_term("keyword", "shopify inventory sync")
        assert term["buildable_opportunity_count"] == 1

    def test_too_broad_increments_vague(self, temp_db):
        """Recording too_broad feedback increments vague_bucket_count."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "spreadsheet workflow")
        temp_db.update_search_term_metrics("keyword", "spreadsheet workflow", wedge_quality_score=0.5)

        manager.record_wedge_feedback(
            "spreadsheet workflow",
            is_too_broad=True,
            verdict="reject",
        )

        term = temp_db.get_search_term("keyword", "spreadsheet workflow")
        assert term["vague_bucket_count"] == 1

    def test_unknown_term_logs_warning(self, temp_db):
        """Unknown term should not crash, just log a warning."""
        manager = TermLifecycleManager(temp_db)
        # Should not raise
        manager.record_wedge_feedback(
            "nonexistent query term",
            is_buildable_wedge=True,
            verdict="build_now",
        )

    def test_recalculated_score(self, temp_db):
        """Wedge quality score is recalculated after feedback."""
        manager = TermLifecycleManager(temp_db)
        manager.ensure_term_exists("keyword", "shopify inventory sync")
        temp_db.update_search_term_metrics(
            "keyword", "shopify inventory sync",
            wedge_quality_score=0.3,
            buildable_opportunity_count=0,
            vague_bucket_count=0,
        )

        # Record enough feedback to exceed min_samples
        manager.record_wedge_feedback("shopify inventory sync", is_buildable_wedge=True, verdict="build_now")
        manager.record_wedge_feedback("shopify inventory sync", is_buildable_wedge=True, verdict="build_now")

        term = temp_db.get_search_term("keyword", "shopify inventory sync")
        # Score should be updated (not necessarily higher since we're blending with 0.3 heuristic)
        assert term["buildable_opportunity_count"] == 2
        # The exact value depends on the formula, but it should have changed
        assert term["wedge_quality_score"] is not None


class TestBuildableOpportunityCountTransitions:
    """Tests for state transitions influenced by buildable_opportunity_count."""

    def test_should_promote_with_buildable_count(self):
        """buildable_opportunity_count >= 1 should promote to high_performing."""
        metrics = TermMetrics(
            times_searched=1,
            buildable_opportunity_count=1,
        )
        from src.discovery_term_lifecycle import should_promote_to_high_performing
        assert should_promote_to_high_performing(metrics) is True

    def test_should_mark_completed_unchanged(self):
        """buildable_opportunity_count should NOT trigger completed."""
        metrics = TermMetrics(
            times_searched=3,
            buildable_opportunity_count=1,
            build_briefs=0,
        )
        from src.discovery_term_lifecycle import should_mark_completed
        assert should_mark_completed(metrics) is False

    def test_completed_still_requires_build_briefs(self):
        """Completion still requires build_briefs, not just buildable count."""
        metrics = TermMetrics(
            times_searched=3,
            buildable_opportunity_count=5,
            build_briefs=0,
        )
        from src.discovery_term_lifecycle import should_mark_completed
        assert should_mark_completed(metrics) is False

        metrics2 = TermMetrics(
            times_searched=3,
            buildable_opportunity_count=0,
            build_briefs=1,
        )
        assert should_mark_completed(metrics2) is True