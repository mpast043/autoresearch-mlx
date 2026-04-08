"""Tests for SREAgent."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.sre import (
    SREAgent,
    SREReport,
    OpportunityHealth,
    Regression,
    HealthStatus,
    RegressionType,
)


class MockOpportunity:
    """Mock opportunity for testing."""

    def __init__(
        self,
        id=1,
        cluster_id=1,
        title="Test Opportunity",
        composite_score=0.8,
        selection_status="prototype_candidate",
        evaluated_at="2026-04-07T10:00:00",
        last_rescored_at="2026-04-07T10:00:00",
    ):
        self.id = id
        self.cluster_id = cluster_id
        self.title = title
        self.composite_score = composite_score
        self.selection_status = selection_status
        self.evaluated_at = evaluated_at
        self.last_rescored_at = last_rescored_at


class MockAtom:
    """Mock problem atom for testing."""

    def __init__(self, id=1, cluster_key="1"):
        self.id = id
        self.cluster_key = cluster_key


class MockDatabase:
    """Mock database for testing."""

    def __init__(self, opportunities=None, atoms=None, corroborations=None):
        self._opportunities = opportunities or []
        self._atoms = atoms or []
        self._corroborations = corroborations or []

    def get_opportunities(self, limit=100, status=None, status_filter=None):
        return self._opportunities[:limit]

    def get_opportunity(self, opportunity_id):
        for opp in self._opportunities:
            if opp.id == opportunity_id:
                return opp
        return None

    def get_problem_atoms(self, limit=100):
        return self._atoms[:limit]

    def get_corroborations(self, cluster_id=None, limit=100):
        if cluster_id is not None:
            return [c for c in self._corroborations if getattr(c, "cluster_id", None) == cluster_id][:limit]
        return self._corroborations[:limit]


class TestSREAgent:
    """Test suite for SREAgent."""

    def test_check_opportunity_health_healthy(self):
        """Test healthy opportunity detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(cluster_id=1, composite_score=0.8)],
            atoms=[MockAtom(cluster_key="1"), MockAtom(cluster_key="1")],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.5)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        health = agent.check_opportunity_health(1)

        assert health.opportunity_id == 1
        assert health.status == HealthStatus.HEALTHY
        assert health.score == 0.8

    def test_check_opportunity_health_degraded(self):
        """Test degraded opportunity detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(cluster_id=1, composite_score=0.5)],
            atoms=[MockAtom(cluster_key="1")],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.3)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        health = agent.check_opportunity_health(1)

        assert health.status in (HealthStatus.DEGRADED, HealthStatus.CRITICAL)

    def test_check_opportunity_health_unknown(self):
        """Test unknown opportunity (no matching record)."""
        db = MockDatabase(opportunities=[], atoms=[])
        config = {}
        agent = SREAgent(db, config)

        health = agent.check_opportunity_health(999)

        assert health.status == HealthStatus.UNKNOWN
        assert "Opportunity not found" in health.issues[0]

    def test_detect_regression_score_decay(self):
        """Test score decay regression detection."""
        db = MockDatabase(
            opportunities=[MockOpportunity(cluster_id=1, composite_score=0.3)],
            atoms=[MockAtom(cluster_key="1")],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.5)],
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
        db = MockDatabase(
            opportunities=[MockOpportunity(cluster_id=1)],
            atoms=[],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.1)],
        )
        config = {"corroboration_min": 0.8}  # Higher than mock's 0.1
        agent = SREAgent(db, config)

        regressions = agent.detect_regressions(1)

        corr_reg = [r for r in regressions if r.regression_type == RegressionType.CORROBORATION_LOSS]
        assert len(corr_reg) > 0

    def test_generate_report(self):
        """Test comprehensive report generation."""
        db = MockDatabase(
            opportunities=[
                MockOpportunity(cluster_id=1, composite_score=0.8),
                MockOpportunity(id=2, cluster_id=2, composite_score=0.4),
            ],
            atoms=[MockAtom(cluster_key="1")],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.5)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        report = agent.generate_report()

        assert report.opportunities_checked == 2
        assert report.healthy_count + report.degraded_count + report.critical_count <= 2

    def test_format_opportunity_status(self):
        """Test opportunity status formatting."""
        db = MockDatabase(
            opportunities=[MockOpportunity(cluster_id=1, title="Test Opp", composite_score=0.7)],
            atoms=[MockAtom(cluster_key="1")],
            corroborations=[MagicMock(cluster_id=1, corroboration_depth=0.5)],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        status = agent.format_opportunity_status(1)

        assert "Opportunity 1" in status
        assert "Test Opp" in status
        assert "score" in status.lower()

    def test_format_summary(self):
        """Test report summary formatting."""
        db = MockDatabase(
            opportunities=[
                MockOpportunity(cluster_id=1, composite_score=0.8),
            ],
            atoms=[],
            corroborations=[],
        )
        config = {"corroboration_min": 0.3}
        agent = SREAgent(db, config)

        report = agent.generate_report()

        assert "Opportunity Health Report" in report.summary


class TestOpportunityHealth:
    """Test OpportunityHealth dataclass."""

    def test_opportunity_health_creation(self):
        """Test creating an opportunity health record."""
        health = OpportunityHealth(
            opportunity_id=1,
            cluster_id=1,
            title="Test Opp",
            status=HealthStatus.HEALTHY,
            score=0.8,
            previous_score=0.7,
            validation_count=5,
            last_validation="2026-04-07",
            atom_count=10,
            corroboration_depth=0.6,
            issues=[],
            recommendations=["Keep it up"],
        )

        assert health.opportunity_id == 1
        assert health.status == HealthStatus.HEALTHY
        assert health.score == 0.8
        assert "Keep it up" in health.recommendations


class TestRegression:
    """Test Regression dataclass."""

    def test_regression_creation(self):
        """Test creating a regression record."""
        reg = Regression(
            opportunity_id=1,
            title="Test",
            regression_type=RegressionType.SCORE_DECAY,
            severity="warning",
            current_value=0.3,
            previous_value=0.8,
            change_percent=-62.5,
            detected_at="2026-04-07T10:00:00",
            cause="Score dropped significantly",
            recommended_action="Re-run validation",
        )

        assert reg.opportunity_id == 1
        assert reg.regression_type == RegressionType.SCORE_DECAY
        assert reg.change_percent == -62.5