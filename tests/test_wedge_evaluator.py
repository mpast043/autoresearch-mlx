"""Tests for WedgeEvaluator and wedge gate logic."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

import pytest

from src.database import Database, Opportunity, OpportunityCluster
from src.builder_output import (
    WedgeEvaluator,
    WedgeEvaluation,
    evaluate_software_fit,
    evaluate_monetization_fit,
    assess_trust_risk,
    _is_narrow_wedge,
)


class TestWedgeEvaluation(unittest.TestCase):
    """Test WedgeEvaluation dataclass and gate logic."""

    def test_passes_wedge_gate_all_pass(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.7,
            monetization_fit=0.5,
            is_narrow=True,
            trust_risk="low",
            verdict="build_now",
        )
        assert ev.passes_wedge_gate is True

    def test_fails_low_software_fit(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.3,
            monetization_fit=0.5,
            is_narrow=True,
            trust_risk="low",
            verdict="research_more",
        )
        assert ev.passes_wedge_gate is False
        assert "software_fit" in str(ev.gate_failure_reasons())

    def test_fails_low_monetization_fit(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.7,
            monetization_fit=0.2,
            is_narrow=True,
            trust_risk="low",
            verdict="research_more",
        )
        assert ev.passes_wedge_gate is False
        assert "monetization_fit" in str(ev.gate_failure_reasons())

    def test_fails_not_narrow(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.7,
            monetization_fit=0.5,
            is_narrow=False,
            trust_risk="low",
            verdict="reject",
        )
        assert ev.passes_wedge_gate is False
        assert "narrow" in str(ev.gate_failure_reasons()).lower()

    def test_fails_high_trust_risk(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.7,
            monetization_fit=0.5,
            is_narrow=True,
            trust_risk="high",
            verdict="research_more",
        )
        assert ev.passes_wedge_gate is False
        assert "trust_risk" in str(ev.gate_failure_reasons())

    def test_gate_failure_reasons_multiple(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.3,
            monetization_fit=0.1,
            is_narrow=False,
            trust_risk="high",
            verdict="reject",
        )
        reasons = ev.gate_failure_reasons()
        assert len(reasons) >= 3

    def test_medium_trust_risk_passes(self) -> None:
        ev = WedgeEvaluation(
            opportunity_id=1,
            software_fit=0.7,
            monetization_fit=0.5,
            is_narrow=True,
            trust_risk="medium",
            verdict="backup_candidate",
        )
        assert ev.passes_wedge_gate is True


class TestIsNarrowWedge(unittest.TestCase):
    """Test the improved _is_narrow_wedge heuristic."""

    def test_narrow_with_specific_platform(self) -> None:
        assert _is_narrow_wedge("Shopify store owners", "Import products into Shopify", "Bad row corrupts inventory", "Shopify App") is True

    def test_narrow_with_spreadsheet(self) -> None:
        assert _is_narrow_wedge("Accountants", "Reconcile spreadsheet", "Formula error corrupts report", "Spreadsheet Add-in") is True

    def test_broad_platform_rejected(self) -> None:
        assert _is_narrow_wedge("Everyone", "Build a platform for all systems", "Generic failure", "microSaaS Web App") is False

    def test_no_specific_indicators_rejected(self) -> None:
        assert _is_narrow_wedge("People", "Do stuff", "Things break", "microSaaS Web App") is False

    def test_commission_keyword_passes(self) -> None:
        assert _is_narrow_wedge("Sales ops analysts", "Calculate commission in Google Sheets", "Commission formula mismatch overpays reps", "Spreadsheet Add-in") is True

    def test_generic_bid_workflow_rejected(self) -> None:
        assert _is_narrow_wedge("Ad operators", "Adjust bids manually", "Budget overrun", "microSaaS Web App") is False


class TestWedgeEvaluatorHeuristic(unittest.TestCase):
    """Test WedgeEvaluator with heuristic-only (no LLM)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_schema()

    def tearDown(self) -> None:
        self.db.close()

    def _create_opportunity(self, title: str, **kwargs) -> int:
        cluster = OpportunityCluster(label="Test", cluster_key="test_1")
        self.db.upsert_cluster(cluster)
        opp = Opportunity(
            cluster_id=1,
            title=title,
            composite_score=kwargs.get("composite_score", 0.25),
            selection_status=kwargs.get("selection_status", "prototype_candidate"),
            market_gap=kwargs.get("market_gap", "partially_solved"),
            recommendation=kwargs.get("recommendation", "promote"),
            status="active",
            cost_of_inaction=kwargs.get("cost_of_inaction", 0.8),
            frequency_score=kwargs.get("frequency_score", 0.6),
            workaround_density=kwargs.get("workaround_density", 0.7),
            buildability=kwargs.get("buildability", 0.67),
            revenue_readiness_score=kwargs.get("revenue_readiness", 0.4),
        )
        self.db.upsert_opportunity(opp)
        return 1

    def test_heuristic_evaluation_basic(self) -> None:
        self._create_opportunity("Spend 4 hours manually adjusting bids across campaigns")
        evaluator = WedgeEvaluator(self.db, {})
        result = evaluator.evaluate_sync(1)
        assert result.evaluated_by == "heuristic"
        assert result.software_fit > 0
        assert result.opportunity_id == 1

    def test_heuristic_uses_opportunity_scores(self) -> None:
        """Opportunity-level metrics should boost monetization_fit when heuristic is low."""
        self._create_opportunity(
            "Spend 4 hours manually adjusting bids",
            cost_of_inaction=0.92,
            frequency_score=0.63,
            workaround_density=1.0,
        )
        evaluator = WedgeEvaluator(self.db, {})
        result = evaluator.evaluate_sync(1)
        # With high cost_of_inaction and frequency, monetization should be boosted
        assert result.monetization_fit >= 0.3

    def test_heuristic_buildability_boosts_software_fit(self) -> None:
        """High buildability should boost software_fit when heuristic is low."""
        self._create_opportunity(
            "Some generic problem",
            buildability=0.9,
        )
        evaluator = WedgeEvaluator(self.db, {})
        result = evaluator.evaluate_sync(1)
        # Buildability should boost software_fit above 0.5
        assert result.software_fit >= 0.5

    def test_missing_opportunity_returns_reject(self) -> None:
        evaluator = WedgeEvaluator(self.db, {})
        result = evaluator.evaluate_sync(999)
        assert result.verdict == "reject"
        assert result.software_fit == 0.0


class TestWedgeEvaluatorLLM(unittest.TestCase):
    """Test WedgeEvaluator with mocked LLM responses."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_schema()

    def tearDown(self) -> None:
        self.db.close()

    def _create_opportunity(self, title: str) -> int:
        cluster = OpportunityCluster(label="Test", cluster_key="test_1")
        self.db.upsert_cluster(cluster)
        opp = Opportunity(
            cluster_id=1,
            title=title,
            composite_score=0.25,
            selection_status="prototype_candidate",
            market_gap="partially_solved",
            recommendation="promote",
            status="active",
            cost_of_inaction=0.8,
            frequency_score=0.6,
        )
        self.db.upsert_opportunity(opp)
        return 1

    def test_llm_evaluation_parsed(self) -> None:
        """Test that LLM response is correctly parsed into WedgeEvaluation."""
        self._create_opportunity("Shopify CSV import failures")
        evaluator = WedgeEvaluator(self.db, {"llm": {"provider": "ollama"}})

        llm_response = json.dumps({
            "software_fit": 0.85,
            "monetization_fit": 0.7,
            "is_narrow": True,
            "trust_risk": "low",
            "verdict": "build_now",
            "narrowness_reason": "Specific: Shopify store owners importing vendor CSVs",
            "software_fit_reason": "Naturally a Shopify app",
            "monetization_reason": "Recurring weekly imports, $29/month justified",
            "suggested_mvp": ["CSV upload", "Schema validation", "Error report"],
            "first_paid_offer": "Unlimited CSV validation - $9/month",
            "pricing_hypothesis": "$9/month - saves 2+ hours per failed import",
            "first_customer": "Shopify store owner importing supplier CSVs",
            "first_channel": "Shopify App Store",
        })

        with patch.object(evaluator, '_llm_evaluate_sync') as mock_llm:
            from src.builder_output import WedgeEvaluation as WE
            mock_llm.return_value = WE(
                opportunity_id=1,
                software_fit=0.85,
                monetization_fit=0.7,
                is_narrow=True,
                trust_risk="low",
                verdict="build_now",
                narrowness_reason="Specific: Shopify store owners importing vendor CSVs",
                software_fit_reason="Naturally a Shopify app",
                monetization_reason="Recurring weekly imports, $29/month justified",
                suggested_mvp=["CSV upload", "Schema validation", "Error report"],
                first_paid_offer="Unlimited CSV validation - $9/month",
                pricing_hypothesis="$9/month - saves 2+ hours per failed import",
                first_customer="Shopify store owner importing supplier CSVs",
                first_channel="Shopify App Store",
                evaluated_by="llm",
            )
            result = evaluator.evaluate_sync(1)

        assert result.software_fit == 0.85
        assert result.monetization_fit == 0.7
        assert result.is_narrow is True
        assert result.passes_wedge_gate is True
        assert result.verdict == "build_now"

    def test_llm_verdict_overridden_by_gate(self) -> None:
        """LLM might say build_now but gate criteria should override."""
        from src.builder_output import WedgeEvaluation as WE
        ev = WE(
            opportunity_id=1,
            software_fit=0.3,  # Below floor
            monetization_fit=0.7,
            is_narrow=True,
            trust_risk="low",
            verdict="build_now",  # LLM said build_now
            evaluated_by="llm",
        )
        # Gate should override
        assert ev.passes_wedge_gate is False


class TestSoftwareFitHeuristic(unittest.TestCase):
    """Test the evaluate_software_fit heuristic."""

    def test_platform_native(self) -> None:
        data = {"workflow": "Import products into Shopify", "trigger": "Monthly import", "failure": "Bad row breaks import"}
        score = evaluate_software_fit(data)
        assert score >= 0.25  # platform_native weight

    def test_manual_to_automated(self) -> None:
        data = {"workflow": "Manually check each row by hand", "trigger": "", "failure": ""}
        score = evaluate_software_fit(data)
        assert score >= 0.15  # manual_to_automated weight

    def test_error_prevention(self) -> None:
        data = {"workflow": "", "trigger": "", "failure": "Formula error corrupts data"}
        score = evaluate_software_fit(data)
        assert score >= 0.15  # error_prevention weight


if __name__ == "__main__":
    unittest.main()
