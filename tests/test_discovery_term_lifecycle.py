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