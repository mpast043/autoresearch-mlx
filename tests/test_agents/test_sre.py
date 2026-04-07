"""Tests for SREAgent."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.sre import (
    SREAgent,
    SREReport,
    WedgeHealth,
    Regression,
    HealthStatus,
    RegressionType,
)


class MockOpportunity:
    """Mock opportunity for testing."""

    def __init__(
        self,
        id=1,
        wedge_id=1,
        title="Test Opportunity",
        composite_score=0.8,
        validation_status="passed",
        last_validated_at="2026-04-07T10:00:00",
        selection_status="prototype_candidate",
    ):
        self.id = id
        self.wedge_id = wedge_id
        self.title = title
        self.composite_score = composite_score
        self.validation_status = validation_status
        self.last_validated_at = last_validated_at
        self.selection_status = selection_status


class MockAtom:
    """Mock problem atom for testing."""

    def __init__(self, id=1, wedge_id=1):
        self.id = id
        self.wedge_id = wedge_id


class MockDatabase:
    """Mock database for testing."""

    def __init__(self, opportunities=None, atoms=None):
        self._opportunities = opportunities or []
        self._atoms = atoms or []

    def get_opportunities(self, limit=100):
        return self._opportunities[:limit]

    def get_all_opportunities(self):
        return self._opportunities

    def get_problem_atoms(self, limit=100):
        return self._atoms[:limit]

    def _get_connection(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (0.5,)
        return conn


class TestSREAgent:
    """Test suite for SREAgent."""

    def test_check_wedge_health_healthy(self):
        """Test healthy wedge detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(wedge_id=1, composite_score=0.8)],
            atoms=[MockAtom(wedge_id=1), MockAtom(wedge_id=1)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        health = agent.check_wedge_health(1)

        assert health.wedge_id == 1
        assert health.status == HealthStatus.HEALTHY
        assert health.score == 0.8

    def test_check_wedge_health_degraded(self):
        """Test degraded wedge detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(wedge_id=1, composite_score=0.5)],
            atoms=[MockAtom(wedge_id=1)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        health = agent.check_wedge_health(1)

        assert health.status in (HealthStatus.DEGRADED, HealthStatus.CRITICAL)

    def test_check_wedge_health_unknown(self):
        """Test unknown wedge (no opportunities)."""
        db = MockDatabase(opportunities=[], atoms=[])
        config = {}
        agent = SREAgent(db, config)

        health = agent.check_wedge_health(999)

        assert health.status == HealthStatus.UNKNOWN
        assert "No opportunities" in health.issues[0]

    def test_detect_regression_score_decay(self):
        """Test score decay regression detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(wedge_id=1, composite_score=0.3)],
            atoms=[MockAtom(wedge_id=1)],
        )
        config = {"score_drop_threshold": 0.15}
        agent = SREAgent(db, config)

        # Mock previous score
        with patch.object(agent, "_get_previous_score", return_value=0.8):
            regressions = agent.detect_regressions(1)

        score_reg = [r for r in regressions if r.regression_type == RegressionType.SCORE_DECAY]
        assert len(score_reg) > 0

    def test_detect_regression_corroboration_loss(self):
        """Test corroboration loss detection."""
        db = MockDatabase(opportunities=[MockOpportunity(wedge_id=1)], atoms=[])
        config = {"corroboration_min": 0.8}  # Higher than mock's 0.5
        agent = SREAgent(db, config)

        regressions = agent.detect_regressions(1)

        corr_reg = [r for r in regressions if r.regression_type == RegressionType.CORROBORATION_LOSS]
        assert len(corr_reg) > 0

    def test_generate_report(self):
        """Test comprehensive report generation."""
        db = MockDatabase(
            opportunities=[
                MockOpportunity(wedge_id=1, composite_score=0.8),
                MockOpportunity(wedge_id=2, composite_score=0.4),
            ],
            atoms=[MockAtom(wedge_id=1)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        report = agent.generate_report()

        assert report.wedges_checked == 2
        assert report.healthy_count + report.degraded_count + report.critical_count == 2

    def test_format_wedge_status(self):
        """Test wedge status formatting."""
        db = MockDatabase(
            opportunities=[MockOpportunity(wedge_id=1, title="Test Wedge", composite_score=0.7)],
            atoms=[MockAtom(wedge_id=1)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        status = agent.format_wedge_status(1)

        assert "Wedge 1" in status
        assert "Test Wedge" in status
        assert "score" in status.lower()

    def test_format_summary(self):
        """Test report summary formatting."""
        db = MockDatabase(
            opportunities=[
                MockOpportunity(wedge_id=1, composite_score=0.8),
            ],
            atoms=[],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        report = agent.generate_report()

        assert "Wedge Health Report" in report.summary
        assert "Healthy:" in report.summary


class TestWedgeHealth:
    """Test WedgeHealth dataclass."""

    def test_wedge_health_creation(self):
        """Test creating a wedge health record."""
        health = WedgeHealth(
            wedge_id=1,
            wedge_title="Test Wedge",
            status=HealthStatus.HEALTHY,
            score=0.8,
            previous_score=0.7,
            validation_count=5,
            last_validation="2026-04-07",
            atom_count=10,
            corroboration_depth=0.6,
            opportunity_count=3,
            build_ready_count=1,
            issues=[],
            recommendations=["Keep it up"],
        )

        assert health.wedge_id == 1
        assert health.status == HealthStatus.HEALTHY
        assert health.score == 0.8
        assert "Keep it up" in health.recommendations


class TestRegression:
    """Test Regression dataclass."""

    def test_regression_creation(self):
        """Test creating a regression record."""
        reg = Regression(
            wedge_id=1,
            wedge_title="Test",
            regression_type=RegressionType.SCORE_DECAY,
            severity="warning",
            current_value=0.3,
            previous_value=0.8,
            change_percent=-62.5,
            detected_at="2026-04-07T10:00:00",
            cause="Score dropped significantly",
            recommended_action="Re-run validation",
        )

        assert reg.wedge_id == 1
        assert reg.regression_type == RegressionType.SCORE_DECAY
        assert reg.change_percent == -62.5