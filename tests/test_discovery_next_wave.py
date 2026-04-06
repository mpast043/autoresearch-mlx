"""Tests for discovery_next_wave hybrid selector."""

import os
import sys
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database
from src.discovery_next_wave import (
    calculate_hybrid_score,
    generate_next_wave,
    load_locked_terms,
    normalize_term,
    REGRESSION_BLOCKLIST,
    LOCKED_KEYWORDS,
    LOCKED_SUBREDDITS,
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


class TestHybridScoring:
    """Tests for hybrid scoring formula."""

    def test_hybrid_score_combines_quality_and_output(self, temp_db):
        """Hybrid score combines quality (60%) and output (40%)."""
        # Insert a term with high output
        temp_db.insert_search_term("keyword", "manual reconciliation", state="completed")
        temp_db.update_search_term_metrics(
            "keyword", "manual reconciliation",
            times_searched=5, findings_emitted=10,
            validations=3, prototype_candidates=2, build_briefs=1
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.3,
                platform_native_score = 0.0,
                plugin_fit_score = 0.1
            WHERE term_value = 'manual reconciliation'
        ''')
        conn.commit()

        result = calculate_hybrid_score("manual reconciliation", "keyword", conn)

        assert result is not None
        assert result['hybrid_score'] > 0
        assert result['quality_component'] > 0
        assert result['output_component'] > 0

    def test_regression_blocklist_penalizes_generic_terms(self, temp_db):
        """Terms on regression blocklist get heavily penalized."""
        temp_db.insert_search_term("keyword", "keep sync and data handoff workflows reliable", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "keep sync and data handoff workflows reliable",
            times_searched=10, findings_emitted=5
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.5
            WHERE term_value = 'keep sync and data handoff workflows reliable'
        ''')
        conn.commit()

        result = calculate_hybrid_score("keep sync and data handoff workflows reliable", "keyword", conn)

        # Should be heavily penalized
        assert result['hybrid_score'] < 0.2
        assert result['quality_component'] < 0.2

    def test_vague_bucket_penalty(self, temp_db):
        """Vague bucket terms get penalty."""
        temp_db.insert_search_term("keyword", "spreadsheet hell", state="active")
        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.3,
                vague_bucket_count = 1
            WHERE term_value = 'spreadsheet hell'
        ''')
        conn.commit()

        result = calculate_hybrid_score("spreadsheet hell", "keyword", conn)

        # Quality component should be penalized
        assert result['quality_component'] < 0.1

    def test_high_output_boosts_score(self, temp_db):
        """Terms with high prototype candidates get boosted."""
        temp_db.insert_search_term("keyword", "good term", state="high_performing")
        temp_db.update_search_term_metrics(
            "keyword", "good term",
            findings_emitted=20, validations=5,
            prototype_candidates=5, build_briefs=2
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.2
            WHERE term_value = 'good term'
        ''')
        conn.commit()

        result = calculate_hybrid_score("good term", "keyword", conn)

        # Output component should be significant
        assert result['output_component'] > result['quality_component']


class TestAliasDeduplication:
    """Tests for alias normalization."""

    def test_normalize_removes_prefixes(self):
        """Prefixes are removed during normalization."""
        assert normalize_term("operator - keep sync") == "keep sync"
        assert normalize_term("developer - manual work") == "manual work"
        assert normalize_term("finance - spreadsheet pain") == "spreadsheet pain"

    def test_normalize_lowercases(self):
        """Normalization lowercases terms."""
        assert normalize_term("MANUAL RECONCILIATION") == "manual reconciliation"
        assert normalize_term("Shopify App") == "shopify app"

    def test_normalize_collapses_whitespace(self):
        """Normalization collapses whitespace."""
        assert normalize_term("manual   reconciliation") == "manual reconciliation"
        assert normalize_term("  spreadsheet  ") == "spreadsheet"


class TestLockedTermRetention:
    """Tests for locked term retention."""

    def test_locked_terms_are_retained(self, temp_db):
        """Locked terms are retained in next wave."""
        # Add locked keywords
        for kw in LOCKED_KEYWORDS:
            temp_db.insert_search_term("keyword", kw, state="active")
            temp_db.update_search_term_metrics(
                "keyword", kw,
                findings_emitted=10, prototype_candidates=1
            )

        # Add some challenger terms with lower scores
        temp_db.insert_search_term("keyword", "new challenger", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "new challenger",
            findings_emitted=1
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.2
            WHERE term_value LIKE '%challenger%'
        ''')
        conn.commit()

        result = generate_next_wave(
            temp_db,
            max_keywords=5,
            use_locked_as_seed=True,
            allow_replacement=False
        )

        # All locked terms should be retained
        result_kw = [kw['term_value'] for kw in result['keywords']]
        for locked_kw in LOCKED_KEYWORDS[:5]:
            # At least some should be in results
            assert any(locked_kw in r for r in result_kw)


class TestChallengerReplacement:
    """Tests for challenger replacement logic."""

    def test_challenger_can_replace_locked_with_margin(self, temp_db):
        """Challengers can replace locked terms if they beat by margin."""
        # Add locked term with low score
        temp_db.insert_search_term("keyword", LOCKED_KEYWORDS[0], state="active")
        temp_db.update_search_term_metrics(
            "keyword", LOCKED_KEYWORDS[0],
            findings_emitted=1
        )

        # Add challenger with high score
        temp_db.insert_search_term("keyword", "superior challenger", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "superior challenger",
            findings_emitted=50, prototype_candidates=10, build_briefs=5
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.4
            WHERE term_value = 'superior challenger'
        ''')
        conn.commit()

        result = generate_next_wave(
            temp_db,
            max_keywords=5,
            use_locked_as_seed=True,
            allow_replacement=True,
            challenger_margin=0.2
        )

        # Challenger should be selected
        result_kw = [kw['term_value'] for kw in result['keywords']]
        assert "superior challenger" in result_kw


class TestLifecycleExclusions:
    """Tests for lifecycle-based exclusions."""

    def test_exhausted_terms_excluded(self, temp_db):
        """Exhausted terms are excluded from next wave."""
        temp_db.insert_search_term("keyword", "exhausted term", state="exhausted")
        temp_db.insert_search_term("keyword", "active term", state="active")

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.5, findings_emitted = 10
            WHERE term_value IN ('exhausted term', 'active term')
        ''')
        conn.commit()

        result = generate_next_wave(temp_db, max_keywords=5)

        result_kw = [kw['term_value'] for kw in result['keywords']]
        assert "active term" in result_kw
        assert "exhausted term" not in result_kw

    def test_paused_terms_excluded(self, temp_db):
        """Paused terms are excluded."""
        temp_db.insert_search_term("keyword", "paused term", state="paused")
        temp_db.insert_search_term("keyword", "active term", state="active")

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.5, findings_emitted = 10
            WHERE term_value IN ('paused term', 'active term')
        ''')
        conn.commit()

        result = generate_next_wave(temp_db, max_keywords=5)

        result_kw = [kw['term_value'] for kw in result['keywords']]
        assert "active term" in result_kw
        assert "paused term" not in result_kw

    def test_banned_terms_excluded(self, temp_db):
        """Banned terms are excluded."""
        temp_db.insert_search_term("keyword", "banned term", state="banned")
        temp_db.insert_search_term("keyword", "active term", state="active")

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.5, findings_emitted = 10
            WHERE term_value IN ('banned term', 'active term')
        ''')
        conn.commit()

        result = generate_next_wave(temp_db, max_keywords=5)

        result_kw = [kw['term_value'] for kw in result['keywords']]
        assert "active term" in result_kw
        assert "banned term" not in result_kw


class TestRegressionBlocking:
    """Tests for regression guardrails."""

    def test_regression_blocklist_contains_known_bad_terms(self):
        """Blocklist contains known bad terms."""
        assert any('keep sync' in t for t in REGRESSION_BLOCKLIST)
        assert any('copy paste' in t for t in REGRESSION_BLOCKLIST)
        assert any('spreadsheet hell' in t for t in REGRESSION_BLOCKLIST)

    def test_copy_paste_workflow_blocked(self, temp_db):
        """copy paste workflow is blocked."""
        temp_db.insert_search_term("keyword", "copy paste workflow", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "copy paste workflow",
            findings_emitted=5
        )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.4
            WHERE term_value = 'copy paste workflow'
        ''')
        conn.commit()

        result = calculate_hybrid_score("copy paste workflow", "keyword", conn)

        # Should be heavily penalized
        assert result['hybrid_score'] < 0.1


class TestNextWaveGeneration:
    """Tests for full next-wave generation."""

    def test_generates_correct_count(self, temp_db):
        """Next wave generates correct number of terms."""
        # Add enough terms
        for i in range(10):
            temp_db.insert_search_term("keyword", f"keyword_{i}", state="active")
            temp_db.update_search_term_metrics(
                "keyword", f"keyword_{i}",
                findings_emitted=i+1
            )

        for i in range(10):
            temp_db.insert_search_term("subreddit", f"sub_{i}", state="active")
            temp_db.update_search_term_metrics(
                "subreddit", f"sub_{i}",
                findings_emitted=i+1
            )

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.3
        ''')
        conn.commit()

        result = generate_next_wave(temp_db, max_keywords=5, max_subreddits=5)

        assert len(result['keywords']) == 5
        assert len(result['subreddits']) == 5

    def test_output_includes_observability(self, temp_db):
        """Output includes observability data."""
        # Add terms
        temp_db.insert_search_term("keyword", "test keyword", state="active")
        temp_db.insert_search_term("subreddit", "testsub", state="active")

        conn = temp_db._get_connection()
        conn.execute('''
            UPDATE discovery_search_terms
            SET wedge_quality_score = 0.3, findings_emitted = 5
        ''')
        conn.commit()

        result = generate_next_wave(temp_db, max_keywords=3, max_subreddits=3)

        assert 'observability' in result
        assert 'locked_keywords' in result['observability']
        assert 'keyword_retained' in result['observability']


class TestDefaultConfigs:
    """Tests for default configurations."""

    def test_locked_keywords_exist(self):
        """Default locked keywords are defined."""
        assert len(LOCKED_KEYWORDS) == 5
        assert isinstance(LOCKED_KEYWORDS, list)

    def test_locked_subreddits_exist(self):
        """Default locked subreddits are defined."""
        assert len(LOCKED_SUBREDDITS) == 5
        assert isinstance(LOCKED_SUBREDDITS, list)

    def test_load_locked_terms_returns_defaults(self, tmp_path):
        """Loading locked terms returns defaults when no file exists."""
        result = load_locked_terms(tmp_path / "nonexistent.json")
        assert 'keywords' in result
        assert 'subreddits' in result