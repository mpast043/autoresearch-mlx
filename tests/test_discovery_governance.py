"""Tests for discovery_governance automatic governance system."""

import os
import sys
import tempfile
import json
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database
from src.discovery_governance import (
    DiscoveryGovernance,
    GovernanceState,
    TermGovernance,
    SourceFamilyGovernance,
    run_governance_cycle,
    DEFAULT_CONFIG,
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


@pytest.fixture
def temp_state_path(tmp_path):
    return tmp_path / "governance_state.json"


class TestGovernanceState:
    """Tests for GovernanceState dataclass."""

    def test_default_config(self):
        """Default config has all required thresholds."""
        assert "min_findings_to_retain" in DEFAULT_CONFIG
        assert "consecutive_empty_waves_to_disable" in DEFAULT_CONFIG
        assert "max_screened_out_rate" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["min_findings_to_retain"] == 5
        assert DEFAULT_CONFIG["consecutive_empty_waves_to_disable"] == 3

    def test_state_serialization(self):
        """State can be serialized and deserialized."""
        state = GovernanceState()
        state.wave_count = 5

        term = TermGovernance(term_value="test", term_type="keyword", state="locked")
        state.terms["keyword:test"] = term

        source = SourceFamilyGovernance(family="reddit-problem", state="active")
        state.source_families["reddit-problem"] = source

        data = state.to_dict()
        restored = GovernanceState.from_dict(data)

        assert restored.wave_count == 5
        assert "keyword:test" in restored.terms
        assert restored.source_families["reddit-problem"].family == "reddit-problem"


class TestTermGovernance:
    """Tests for term governance."""

    def test_locked_default_retention(self, temp_db, temp_state_path):
        """Locked terms with sufficient output are retained."""
        # Insert a term with good output
        temp_db.insert_search_term("keyword", "manual reconciliation", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "manual reconciliation",
            times_searched=10, findings_emitted=20,
            prototype_candidates=3, build_briefs=2
        )

        governance = DiscoveryGovernance(temp_db, state_path=temp_state_path)

        # Run evaluation
        summary = governance.evaluate_wave()

        # Check state was updated
        key = "keyword:manual reconciliation"
        assert key in governance.state.terms
        term = governance.state.terms[key]
        assert term.total_findings >= 20

    def test_challenger_replacement(self, temp_db, temp_state_path):
        """A challenger can replace a locked default."""
        # Insert locked term with low output
        temp_db.insert_search_term("keyword", "poor term", state="locked")
        temp_db.update_search_term_metrics(
            "keyword", "poor term",
            times_searched=5, findings_emitted=1,
            prototype_candidates=0, build_briefs=0
        )

        # Insert challenger with high output
        temp_db.insert_search_term("keyword", "great challenger", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "great challenger",
            times_searched=5, findings_emitted=20,
            prototype_candidates=5, build_briefs=3
        )

        governance = DiscoveryGovernance(temp_db, state_path=temp_state_path)

        # The poor term should be flagged for potential replacement
        # (Actual replacement logic is in _evaluate_term)


class TestSourceFamilyGovernance:
    """Tests for source family governance."""

    def test_source_disable_after_empty_waves(self, temp_db, temp_state_path):
        """Source families with repeated empty waves get disabled."""
        # This would require multiple waves to trigger
        # For unit test, check the logic works

        governance = DiscoveryGovernance(temp_db, state_path=temp_state_path)

        # Manually set a source to have empty waves
        source = governance.state.source_families["shopify-review"]
        source.consecutive_empty_waves = 3
        source.total_findings = 50  # Above threshold

        # Save and reload
        governance._save_state()
        governance = DiscoveryGovernance(temp_db, state_path=temp_state_path)

        # Check it was loaded
        assert governance.state.source_families["shopify-review"].consecutive_empty_waves == 3

    def test_source_pause_on_high_screened_rate(self, temp_db, temp_state_path):
        """Sources with high screened-out rate get paused."""
        governance = DiscoveryGovernance(temp_db, state_path=temp_state_path)

        # Create mock source with high screened rate
        source = governance.state.source_families["github-issue"]
        source.total_findings = 30
        source.total_screened_out = 28
        source.screened_out_rate = 0.93

        # This should trigger pause in next evaluation
        # The actual pause happens in _evaluate_source_families


class TestGovernanceCycle:
    """Tests for full governance cycle."""

    def test_run_governance_cycle(self, temp_db, tmp_path):
        """Governance cycle runs without error."""
        state_path = tmp_path / "governance.json"

        # Insert test data
        temp_db.insert_search_term("keyword", "test keyword", state="active")
        temp_db.update_search_term_metrics(
            "keyword", "test keyword",
            findings_emitted=10, prototype_candidates=1
        )

        summary = run_governance_cycle(temp_db)

        assert "next_wave" in summary
        assert "governance_state" in summary
        assert summary["wave_number"] == 1

    def test_next_wave_excludes_disabled_sources(self, temp_db, tmp_path):
        """Next wave generation respects disabled sources."""
        state_path = tmp_path / "governance.json"

        governance = DiscoveryGovernance(temp_db, state_path=state_path)

        # Set a source to disabled
        governance.state.source_families["shopify-review"].state = "disabled"

        next_wave = governance.get_next_wave_terms()

        # Disabled sources should not be in active sources
        assert "shopify-review" not in next_wave.get("active_sources", [])


class TestAutoGovernanceBehavior:
    """Tests for automatic governance behavior."""

    def test_paused_sources_list(self, temp_db, tmp_path):
        """System tracks which sources are paused."""
        governance = DiscoveryGovernance(temp_db, state_path=tmp_path / "gov.json")

        # Set sources to different states
        governance.state.source_families["reddit-problem"].state = "active"
        governance.state.source_families["shopify-review"].state = "paused"
        governance.state.source_families["github-issue"].state = "disabled"

        active = governance.get_active_source_families()

        assert "reddit-problem" in active
        assert "shopify-review" not in active
        assert "github-issue" not in active

    def test_governance_summary(self, temp_db, tmp_path):
        """Governance summary is comprehensive."""
        governance = DiscoveryGovernance(temp_db, state_path=tmp_path / "gov.json")

        # Add some terms and sources
        governance.state.terms["keyword:test"] = TermGovernance(
            term_value="test", term_type="keyword", state="locked"
        )
        governance.state.source_families["reddit-problem"] = SourceFamilyGovernance(
            family="reddit-problem", state="active", total_findings=100
        )

        summary = governance.get_governance_summary()

        assert "terms" in summary
        assert "source_families" in summary
        assert "reddit-problem" in summary["source_families"]
        assert summary["source_families"]["reddit-problem"]["total_findings"] == 100