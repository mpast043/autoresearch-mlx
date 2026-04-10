"""Tests for build-prep selection and build brief helpers."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_prep import (
    build_brief_payload,
    determine_selection_state,
    determine_narrow_output_type,
    evaluate_build_ready_sharpness,
    is_allowed_selection_transition,
    PlatformFit,
    VAGUE_PATTERNS,
    configure_build_prep,
    get_platform_classification_config,
    _parse_platform_fit_response,
    _classify_via_ollama,
    _classify_via_anthropic,
    _determine_product_via_keyword,
)


class DummyFinding:
    source = "reddit-problem/test"
    source_url = "https://example.com/thread"
    finding_kind = "problem_signal"
    source_class = "pain_signal"
    evidence = {
        "screening": {"score": 4},
        "source_policy": {"reasons": ["specific_forum_pain_evidence"]},
        "source_classification": {"source_class": "pain_signal", "negative_signals": []},
    }


class DummyAtom:
    pain_statement = "backup restores fail and leave systems unreachable"
    failure_mode = "restored environments stay unreachable"
    current_workaround = "manual rollback scripts"
    current_tools = "backup console, shell scripts"
    trigger_event = "restore jobs fail after recovery"


class TestBuildPrepHelpers(unittest.TestCase):
    def setUp(self):
        configure_build_prep(
            {
                "build_prep": {
                    "platform_classification": {
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:9",
                        "model": "gemma4:latest",
                    }
                }
            }
        )

    def tearDown(self):
        configure_build_prep({})

    @staticmethod
    def _consumer_posture(brief):
        prototype_gate = brief["prototype_gate"]
        posture = brief["prototype_spec_posture"]
        return {
            "confidence_label": posture["confidence_label"],
            "scope_rule": posture["build_scope_rule"],
            "messaging_rule": posture["messaging_rule"],
            "allow_market_confirmed_language": prototype_gate["prototype_gate_mode"] == "strict_validated",
        }

    def test_selection_gate_requires_real_support(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.7,
                "value_support": 0.63,
                "composite_score": 0.58,
            },
            corroboration={
                "corroboration_score": 0.72,
                "source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
            },
            market_enrichment={"wedge_active": True},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "validated_selection_gate")
        self.assertTrue(gate["eligible"])

    def test_selection_gate_blocks_weak_items(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.45,
                "value_support": 0.32,
                "composite_score": 0.29,
            },
            corroboration={
                "corroboration_score": 0.41,
                "source_family_diversity": 1,
                "generalizability_class": "product_specific_issue",
                "recurrence_state": "timeout",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "research_more")
        self.assertEqual(reason, "selection_gate_not_met")
        self.assertIn("single_family_support", gate["blocked_by"])
        self.assertIn("recurrence_timeout", gate["blocked_by"])

    def test_selection_gate_preserves_promote_decisions_under_v4_scoring(self):
        status, reason, gate = determine_selection_state(
            decision="promote",
            scorecard={
                "decision_score": 0.19,
                "problem_truth_score": 0.12,
                "revenue_readiness_score": 0.23,
                "frequency_score": 0.30,
                "evidence_quality": 0.60,
                "value_support": 0.45,
                "composite_score": 0.20,
            },
            corroboration={
                "corroboration_score": 0.70,
                "source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "supported",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "validated_selection_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("validation_recommended_promote", gate["reasons"])

    def test_selection_gate_allows_multifamily_near_miss_prototype_candidate(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.4511,
                "value_support": 0.6015,
                "composite_score": 0.3468,
            },
            corroboration={
                "corroboration_score": 0.4215,
                "source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "thin",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["gate_version"], "prototype_candidate_v1")
        self.assertIn("prototype_candidate_multifamily_near_miss", gate["reasons"])

    def test_selection_gate_allows_timeout_multifamily_checkpoint_candidate(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.53,
                "value_support": 0.64,
                "composite_score": 0.43,
            },
            corroboration={
                "corroboration_score": 0.48,
                "source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "timeout",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("prototype_candidate_multifamily_near_miss", gate["reasons"])

    def test_selection_gate_allows_sharp_multifamily_timeout_candidate(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.43,
                "value_support": 0.49,
                "composite_score": 0.32,
                "frequency_score": 0.28,
                "workaround_density": 0.41,
                "cost_of_inaction": 0.48,
                "buildability": 0.61,
            },
            corroboration={
                "corroboration_score": 0.24,
                "cross_source_match_score": 0.21,
                "generalizability_score": 0.67,
                "source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "timeout",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("prototype_candidate_sharp_checkpoint", gate["reasons"])

    def test_selection_gate_allows_exceptional_single_family_prototype_candidate(self):
        # With source_family_diversity=1, single_family_explore path requires stricter
        # recurrence (must be supported) and lower thresholds vs multi_family path.
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.5586,
                "value_support": 0.5245,
                "composite_score": 0.3939,
            },
            corroboration={
                "corroboration_score": 0.534,
                "source_family_diversity": 1,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "supported",
            },
            market_enrichment={"wedge_active": False},
        )
        # single_family_explore fires with these values -> prototype_candidate
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("prototype_candidate_single_family_exception", gate["reasons"])

    def test_selection_gate_allows_supported_single_family_ops_checkpoint_case(self):
        # single_family_explore path: source_family_diversity=1, supported recurrence,
        # below validated gate thresholds but above exploratory thresholds.
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.494,
                "value_support": 0.5164,
                "composite_score": 0.4102,
            },
            corroboration={
                "corroboration_score": 0.4569,
                "source_family_diversity": 1,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "supported",
            },
            market_enrichment={"wedge_active": False},
        )
        # single_family_explore fires -> prototype_candidate
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])

    def test_transition_rules_block_skips(self):
        self.assertTrue(is_allowed_selection_transition("prototype_candidate", "prototype_ready"))
        self.assertFalse(is_allowed_selection_transition("prototype_candidate", "build_ready"))

    def test_build_brief_payload_contains_required_traceability(self):
        payload = build_brief_payload(
            run_id="run-1",
            opportunity_id=2,
            validation_id=3,
            cluster_id=4,
            linked_finding_ids=[7, 8],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "keep backup restore and recovery reliable",
                "user_role": "operator",
                "summary": {"human_summary": "operators need reliable restore flows"},
            },
            anchor_atom=DummyAtom(),
            corroboration={
                "source_families": ["github", "reddit"],
                "source_family_match_counts": {"github": 2, "reddit": 1},
                "core_source_families": ["github", "reddit"],
                "core_source_family_diversity": 2,
                "source_family_diversity": 2,
                "cross_source_match_score": 0.61,
                "corroboration_score": 0.88,
                "generalizability_class": "reusable_workflow_pain",
                "generalizability_score": 0.82,
                "query_set_hash": "abc",
                "results_by_source": {"github": 6, "reddit": 4},
            },
            market_enrichment={
                "wedge_name": "backup_restore_reliability",
                "wedge_active": True,
                "wedge_fit_score": 0.76,
                "demand_score": 0.41,
                "buyer_intent_score": 0.52,
                "willingness_to_pay_signal": 0.63,
                "multi_source_value_lift": 0.58,
            },
            evidence_payload={
                "summary": {"problem_statement": "restore jobs keep failing"},
                "evidence_assessment": {"value_support": 0.72, "problem_plausibility": 0.68},
                "queries_executed": ['"backup restore" operator'],
                "recurrence_budget_profile": {"query_limit": 3, "subreddit_limit": 2},
                "recurrence_gap_reason": "single_source_confirmation_only",
                "recurrence_failure_class": "single_source_only",
                "recurrence_probe_summary": {"probe_hit_count": 1, "branched_after_probe": False},
                "counterevidence": [{"status": "supported", "summary": "Need clearer buyer ownership"}],
            },
            experiment_hypothesis="Ops teams will engage with a restore workflow diagnostic.",
            selection_status="prototype_candidate",
            selection_reason="validated_selection_gate",
            selection_gate={"eligible": True, "reasons": ["multi_family_support"], "blocked_by": []},
        )
        self.assertEqual(payload["opportunity_id"], 2)
        self.assertEqual(payload["validation_id"], 3)
        self.assertEqual(payload["linked_finding_ids"], [7, 8])
        self.assertEqual(payload["source_family_corroboration"]["source_family_diversity"], 2)
        self.assertEqual(payload["source_family_corroboration"]["recurrence_gap_reason"], "single_source_confirmation_only")
        self.assertEqual(payload["source_family_corroboration"]["recurrence_failure_class"], "single_source_only")
        self.assertEqual(payload["wedge_profitability_relevance"]["wedge_name"], "backup_restore_reliability")
        self.assertEqual(payload["evidence_provenance"]["queries_executed"], ['"backup restore" operator'])
        self.assertEqual(payload["evidence_provenance"]["recurrence_failure_class"], "single_source_only")
        self.assertIn("single_source_confirmation_only", payload["open_questions_risks"])
        self.assertTrue(payload["launch_artifact_plan"])
        self.assertEqual(payload["prototype_gate"]["prototype_gate_mode"], "strict_validated")
        self.assertEqual(payload["prototype_gate"]["market_confidence_level"], "market_confirmed")
        self.assertEqual(payload["prototype_gate"]["validation_certainty"], "validated_selection_gate_met")

    def test_build_brief_payload_marks_prototype_candidate_exception_uncertainty(self):
        payload = build_brief_payload(
            run_id="run-2",
            opportunity_id=3,
            validation_id=4,
            cluster_id=5,
            linked_finding_ids=[3],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "keep operations data sync without manual spreadsheet breakage",
                "user_role": "operations lead",
                "summary": {"human_summary": "ops teams are held together by duct tape and spreadsheets"},
            },
            anchor_atom=type(
                "OpsAtom",
                (),
                {
                    "pain_statement": "business operations are held together by duct tape and Excel",
                    "failure_mode": "handoff data gets lost between spreadsheets",
                    "current_workaround": "copy and paste between spreadsheets and email",
                    "current_tools": "excel, email",
                    "trigger_event": "status updates and approvals arrive out of order",
                },
            )(),
            corroboration={
                "source_families": ["reddit"],
                "source_family_match_counts": {"reddit": 2},
                "core_source_families": ["reddit"],
                "core_source_family_diversity": 1,
                "cross_source_match_score": 0.31,
                "corroboration_score": 0.4569,
                "generalizability_class": "reusable_workflow_pain",
                "generalizability_score": 0.78,
                "query_set_hash": "ops-123",
                "results_by_source": {"reddit": 5},
                "recurrence_state": "supported",
            },
            market_enrichment={
                "wedge_name": "",
                "wedge_active": False,
                "wedge_fit_score": 0.12,
                "demand_score": 0.38,
                "buyer_intent_score": 0.41,
                "willingness_to_pay_signal": 0.47,
                "multi_source_value_lift": 0.24,
            },
            evidence_payload={
                "summary": {"problem_statement": "ops teams are stuck in spreadsheet handoff chaos"},
                "evidence_assessment": {
                    "value_support": 0.5164,
                    "problem_plausibility": 0.61,
                    "composite_score": 0.4102,
                },
                "queries_executed": ['"spreadsheet workaround" "small business"'],
                "recurrence_budget_profile": {"query_limit": 3, "subreddit_limit": 2},
                "counterevidence": [{"status": "supported", "summary": "Need broader buyer proof"}],
            },
            experiment_hypothesis="Ops teams will try a workflow reliability assistant that replaces spreadsheet handoffs.",
            selection_status="prototype_candidate",
            selection_reason="prototype_candidate_gate",
            selection_gate={
                "eligible": True,
                "gate_version": "prototype_candidate_v1",
                "reasons": [
                    "generalizable_workflow_pain",
                    "recurrence_state_supported",
                    "prototype_candidate_single_family_exception",
                ],
                "blocked_by": [],
            },
        )
        self.assertEqual(payload["recommended_narrow_output_type"], "sync_handoff_assistant")
        self.assertEqual(payload["prototype_gate"]["prototype_gate_mode"], "prototype_candidate_exception")
        self.assertEqual(payload["prototype_gate"]["prototype_gate_basis"], "supported_single_family_workflow_pain")
        self.assertEqual(payload["prototype_gate"]["prototype_gate_family_count"], 1)
        self.assertEqual(payload["prototype_gate"]["market_confidence_level"], "prototype_checkpoint")
        self.assertEqual(
            payload["prototype_gate"]["validation_certainty"],
            "credible_prototype_candidate_not_market_confirmed",
        )
        self.assertIn("do not claim full market validation", payload["prototype_gate"]["overclaim_guardrail"])
        self.assertEqual(payload["prototype_spec_posture"]["confidence_label"], "prototype_checkpoint")
        self.assertIn("narrow and diagnostic", payload["prototype_spec_posture"]["build_scope_rule"])
        self.assertIn("do not claim full market validation", payload["prototype_spec_posture"]["messaging_rule"])

    def test_downstream_consumer_preserves_certainty_boundary(self):
        validated_payload = build_brief_payload(
            run_id="run-validated",
            opportunity_id=10,
            validation_id=11,
            cluster_id=12,
            linked_finding_ids=[1, 2],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "keep backup restore and recovery reliable",
                "user_role": "operator",
                "summary": {"human_summary": "operators need reliable restore flows"},
            },
            anchor_atom=DummyAtom(),
            corroboration={
                "source_families": ["github", "reddit"],
                "source_family_match_counts": {"github": 2, "reddit": 1},
                "core_source_families": ["github", "reddit"],
                "core_source_family_diversity": 2,
                "cross_source_match_score": 0.61,
                "corroboration_score": 0.88,
                "generalizability_class": "reusable_workflow_pain",
                "generalizability_score": 0.82,
                "query_set_hash": "validated-abc",
                "results_by_source": {"github": 6, "reddit": 4},
                "recurrence_state": "supported",
            },
            market_enrichment={
                "wedge_name": "backup_restore_reliability",
                "wedge_active": True,
                "wedge_fit_score": 0.76,
                "demand_score": 0.41,
                "buyer_intent_score": 0.52,
                "willingness_to_pay_signal": 0.63,
                "multi_source_value_lift": 0.58,
            },
            evidence_payload={
                "summary": {"problem_statement": "restore jobs keep failing"},
                "evidence_assessment": {
                    "value_support": 0.72,
                    "problem_plausibility": 0.68,
                    "composite_score": 0.58,
                },
                "queries_executed": ['"backup restore" operator'],
                "recurrence_budget_profile": {"query_limit": 3, "subreddit_limit": 2},
                "counterevidence": [],
            },
            experiment_hypothesis="Ops teams will engage with a restore workflow diagnostic.",
            selection_status="prototype_candidate",
            selection_reason="validated_selection_gate",
            selection_gate={"eligible": True, "reasons": ["multi_family_support"], "blocked_by": []},
        )
        checkpoint_payload = build_brief_payload(
            run_id="run-checkpoint",
            opportunity_id=20,
            validation_id=21,
            cluster_id=22,
            linked_finding_ids=[3],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "keep operations data sync without manual spreadsheet breakage",
                "user_role": "operations lead",
                "summary": {"human_summary": "ops teams are held together by duct tape and spreadsheets"},
            },
            anchor_atom=type(
                "OpsAtom",
                (),
                {
                    "pain_statement": "business operations are held together by duct tape and Excel",
                    "failure_mode": "handoff data gets lost between spreadsheets",
                    "current_workaround": "copy and paste between spreadsheets and email",
                    "current_tools": "excel, email",
                    "trigger_event": "status updates and approvals arrive out of order",
                },
            )(),
            corroboration={
                "source_families": ["reddit"],
                "source_family_match_counts": {"reddit": 2},
                "core_source_families": ["reddit"],
                "core_source_family_diversity": 1,
                "cross_source_match_score": 0.31,
                "corroboration_score": 0.4569,
                "generalizability_class": "reusable_workflow_pain",
                "generalizability_score": 0.78,
                "query_set_hash": "ops-123",
                "results_by_source": {"reddit": 5},
                "recurrence_state": "supported",
            },
            market_enrichment={
                "wedge_name": "",
                "wedge_active": False,
                "wedge_fit_score": 0.12,
                "demand_score": 0.38,
                "buyer_intent_score": 0.41,
                "willingness_to_pay_signal": 0.47,
                "multi_source_value_lift": 0.24,
            },
            evidence_payload={
                "summary": {"problem_statement": "ops teams are stuck in spreadsheet handoff chaos"},
                "evidence_assessment": {
                    "value_support": 0.5164,
                    "problem_plausibility": 0.61,
                    "composite_score": 0.4102,
                },
                "queries_executed": ['"spreadsheet workaround" "small business"'],
                "recurrence_budget_profile": {"query_limit": 3, "subreddit_limit": 2},
                "counterevidence": [{"status": "supported", "summary": "Need broader buyer proof"}],
            },
            experiment_hypothesis="Ops teams will try a workflow reliability assistant that replaces spreadsheet handoffs.",
            selection_status="prototype_candidate",
            selection_reason="prototype_candidate_gate",
            selection_gate={
                "eligible": True,
                "gate_version": "prototype_candidate_v1",
                "reasons": [
                    "generalizable_workflow_pain",
                    "recurrence_state_supported",
                    "prototype_candidate_single_family_exception",
                ],
                "blocked_by": [],
            },
        )

        validated_posture = self._consumer_posture(validated_payload)
        checkpoint_posture = self._consumer_posture(checkpoint_payload)

        self.assertEqual(validated_posture["confidence_label"], "market_confirmed")
        self.assertTrue(validated_posture["allow_market_confirmed_language"])
        self.assertNotIn("do not claim full market validation", validated_posture["messaging_rule"])

        self.assertEqual(checkpoint_posture["confidence_label"], "prototype_checkpoint")
        self.assertFalse(checkpoint_posture["allow_market_confirmed_language"])
        self.assertIn("narrow and diagnostic", checkpoint_posture["scope_rule"])
        self.assertIn("do not claim full market validation", checkpoint_posture["messaging_rule"])


class TestPlatformFit(unittest.TestCase):
    def setUp(self):
        configure_build_prep(
            {
                "build_prep": {
                    "platform_classification": {
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:9",
                        "model": "gemma4:latest",
                    }
                }
            }
        )

    def tearDown(self):
        configure_build_prep({})

    def test_platform_fit_returns_structured_object(self):
        """Test that determine_narrow_output_type returns PlatformFit."""
        result = determine_narrow_output_type(
            wedge_name="test",
            job_to_be_done="contract errors",
            failure_mode="wrong company name",
            user_role="freelancer",
            trigger_event="sending contract",  # Required for LLM trigger
            pain_statement="almost lost 50k client",  # Required for LLM trigger
            cluster_summary="template error",  # Required for LLM trigger
        )
        self.assertIsInstance(result, PlatformFit)
        self.assertIsNotNone(result.product_name)
        self.assertIsNotNone(result.host_platform)
        self.assertIsNotNone(result.product_format)

    def test_platform_fit_includes_fallback_metadata(self):
        """Test that fallback includes proper metadata."""
        result = determine_narrow_output_type(
            wedge_name="test",
            job_to_be_done="keep operations data sync",  # Triggers sync keyword
            failure_mode="handoff data gets lost",
            user_role="operator",
            trigger_event="status updates arrive out of order",  # Required
            pain_statement="copy paste between spreadsheets",  # Required
            cluster_summary="spreadsheet breakage",  # Required
        )
        self.assertIsInstance(result, PlatformFit)
        self.assertFalse(result.llm_used)
        self.assertTrue(result.fallback_used)

    def test_platform_fit_detects_vague_output(self):
        """Test that vague outputs are flagged."""
        vague = PlatformFit(
            product_name="workflow_reliability_assistant",
            host_platform="Unknown",
            product_format="lightweight microSaaS",
        )
        self.assertTrue(vague.is_vague)

        sync_vague = PlatformFit(
            product_name="sync_handoff_assistant",
            host_platform="Internal workflow",
            product_format="internal workflow tool",
        )
        self.assertTrue(sync_vague.is_vague)

        specific = PlatformFit(
            product_name="Contract Guard",
            host_platform="Google Docs",
            product_format="Google Docs add-on",
        )
        self.assertFalse(specific.is_vague)

    def test_platform_fit_detects_platform_native(self):
        """Test that platform-native formats are detected."""
        add_on = PlatformFit(
            product_name="Test",
            host_platform="Google Docs",
            product_format="Google Docs add-on",
        )
        self.assertTrue(add_on.is_platform_native)

        extension = PlatformFit(
            product_name="Test",
            host_platform="Browser",
            product_format="Chrome extension",
        )
        self.assertTrue(extension.is_platform_native)

        vague = PlatformFit(
            product_name="Test",
            host_platform="Unknown",
            product_format="lightweight microSaaS",
        )
        self.assertFalse(vague.is_platform_native)

    def test_platform_fit_csv_import_fallback_is_specific(self):
        result = determine_narrow_output_type(
            wedge_name="",
            job_to_be_done="Import supplier pricing and inventory data from CSV files",
            failure_mode="A single bad value in a CSV file causes import failure or downstream corruption",
            user_role="operations lead",
            trigger_event="before weekly vendor import",
            current_workaround="manual spreadsheet cleanup before import",
            pain_statement="one malformed row breaks the import",
            cluster_summary="csv import failures",
        )
        self.assertEqual(result.host_platform, "CSV import workflow")
        self.assertEqual(result.product_name, "csv_import_guard")
        self.assertEqual(result.product_format, "web-based CSV validator")

    def test_platform_fit_normalizes_incompatible_llm_surface_for_accounting_imports(self):
        llm_result = PlatformFit(
            host_platform="Google Docs",
            product_format="Google Docs add-on",
            product_name="QuickBooks CSV Importer",
            one_sentence_product="Automates Stripe and bank-feed CSV reconciliation before month-end close",
            why_this_format="Generated by LLM",
            llm_used=True,
            fallback_used=False,
        )
        with mock.patch("build_prep._determine_product_via_llm", return_value=llm_result):
            result = determine_narrow_output_type(
                wedge_name="",
                job_to_be_done="Import Stripe and bank-feed CSVs into QuickBooks without reconciliation drift",
                failure_mode="QuickBooks reconciliation drifts after importing Stripe payout and bank feed CSVs",
                user_role="bookkeeper",
                trigger_event="Uploading Stripe payout CSVs during month-end close",
                current_workaround="manual spreadsheet reconciliation before posting journal entries",
                pain_statement="Accountants spend hours tracing mismatched imports that should have reconciled automatically",
                cluster_summary="stripe quickbooks reconciliation drift",
            )

        self.assertEqual(result.host_platform, "QuickBooks")
        self.assertEqual(result.product_format, "QuickBooks App")
        self.assertEqual(result.product_name, "QuickBooks CSV Importer")
        self.assertTrue(result.llm_used)
        self.assertTrue(result.fallback_used)


class TestBuildReadySharpness(unittest.TestCase):
    def test_sharpness_gate_accepts_specific_platform_fit(self):
        gate = evaluate_build_ready_sharpness(
            {
                "platform_fit": {
                    "host_platform": "Google Docs",
                    "product_format": "Google Docs add-on",
                    "product_name": "Contract Guard",
                },
                "job_to_be_done": "Review Google Docs contracts before sending them to clients",
                "pain_workaround": {
                    "failure_mode": "Template reuse leaves the wrong legal entity on client contracts",
                    "trigger_event": "Right before sending a renewal contract for signature",
                },
                "source_family_corroboration": {
                    "source_family_diversity": 2,
                },
            }
        )
        self.assertTrue(gate["passes"])
        self.assertEqual(gate["reasons"], [])

    def test_sharpness_gate_rejects_unknown_generic_output(self):
        gate = evaluate_build_ready_sharpness(
            {
                "platform_fit": {
                    "host_platform": "Unknown",
                    "product_format": "lightweight microSaaS",
                    "product_name": "workflow_diagnostic_prototype",
                },
                "job_to_be_done": "daily operation",
                "pain_workaround": {
                    "failure_mode": "manual execution of tasks",
                    "trigger_event": "daily operation",
                },
                "source_family_corroboration": {
                    "source_family_diversity": 1,
                },
            }
        )
        self.assertFalse(gate["passes"])
        self.assertIn("unknown_host_platform", gate["reasons"])
        self.assertIn("vague_product_name", gate["reasons"])
        self.assertIn("insufficient_source_family_diversity", gate["reasons"])

    def test_sharpness_gate_rejects_generic_internal_sync_fallback(self):
        gate = evaluate_build_ready_sharpness(
            {
                "platform_fit": {
                    "host_platform": "Internal workflow",
                    "product_format": "internal workflow tool",
                    "product_name": "sync_handoff_assistant",
                },
                "job_to_be_done": "Managing and importing various business data using CSV files",
                "pain_workaround": {
                    "failure_mode": "A single bad value in a CSV file causing import failure or data corruption.",
                    "trigger_event": "When importing data from CSV files",
                },
                "source_family_corroboration": {
                    "source_family_diversity": 2,
                },
            }
        )
        self.assertFalse(gate["passes"])
        self.assertIn("vague_product_name", gate["reasons"])
        self.assertIn("generic_job_to_be_done", gate["reasons"])

    def test_build_brief_payload_includes_platform_fit(self):
        """Test that build_brief_payload includes platform_fit."""
        payload = build_brief_payload(
            run_id="test-run",
            opportunity_id=1,
            validation_id=1,
            cluster_id=1,
            linked_finding_ids=[1],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "contract errors",
                "user_role": "freelancer",
                "summary": {"human_summary": "contract template mistake"},
            },
            anchor_atom=type(
                "TestAtom",
                (),
                {
                    "pain_statement": "wrong company name in contract",
                    "failure_mode": "template reuse error",
                    "current_workaround": "manual check",
                    "trigger_event": "sending contract",
                },
            )(),
            corroboration={
                "source_families": ["reddit"],
                "source_family_match_counts": {"reddit": 3},
                "core_source_families": ["reddit"],
                "core_source_family_diversity": 1,
                "cross_source_match_score": 0.35,
                "corroboration_score": 0.42,
                "recurrence_state": "supported",
            },
            market_enrichment={
                "wedge_name": "contract_template_error",
                "wedge_active": True,
                "wedge_fit_score": 0.65,
            },
            evidence_payload={
                "recurrence_gap_reason": "",
                "evidence_assessment": {
                    "problem_plausibility": 0.6,
                    "value_support": 0.5,
                    "composite_score": 0.35,
                },
            },
            experiment_hypothesis="test hypothesis",
            selection_status="prototype_candidate",
            selection_reason="validated_selection_gate",
            selection_gate={"eligible": True, "reasons": [], "blocked_by": []},
        )

        self.assertIn("platform_fit", payload)
        self.assertIsInstance(payload["platform_fit"], dict)
        self.assertIn("host_platform", payload["platform_fit"])
        self.assertIn("product_format", payload["platform_fit"])
        self.assertIn("product_name", payload["platform_fit"])
        self.assertIn("llm_used", payload["platform_fit"])
        self.assertIn("fallback_used", payload["platform_fit"])

    def test_build_brief_payload_preserves_specific_user_and_import_scope(self):
        payload = build_brief_payload(
            run_id="test-run",
            opportunity_id=1,
            validation_id=1,
            cluster_id=1,
            linked_finding_ids=[1],
            finding=DummyFinding(),
            cluster={
                "job_to_be_done": "Managing and importing various business data (pricing, inventory, vendors, reporting) using CSV files",
                "user_role": "small business owner",
                "summary": {"human_summary": "How do you prevent spreadsheet or CSV errors from breaking your operations?"},
            },
            anchor_atom=type(
                "ImportAtom",
                (),
                {
                    "pain_statement": "A single bad value in a CSV file can cause an import failure or lead to incorrect downstream data.",
                    "failure_mode": "A single bad value in a CSV file causing import failure or data corruption.",
                    "current_workaround": "spreadsheets, csv exports, manual work",
                    "current_tools": "Excel",
                    "trigger_event": "When importing data from CSV files",
                },
            )(),
            corroboration={
                "source_families": ["reddit", "web"],
                "source_family_match_counts": {"web": 1},
                "core_source_families": ["reddit"],
                "core_source_family_diversity": 1,
                "source_family_diversity": 2,
                "cross_source_match_score": 0.36,
                "corroboration_score": 0.35,
                "recurrence_state": "thin",
                "generalizability_class": "reusable_workflow_pain",
                "generalizability_score": 0.58,
            },
            market_enrichment={"wedge_name": "", "wedge_active": False, "wedge_fit_score": 0.12},
            evidence_payload={"summary": {"problem_statement": "broad csv question"}, "evidence_assessment": {}},
            experiment_hypothesis="test hypothesis",
            selection_status="prototype_candidate",
            selection_reason="validated_selection_gate",
            selection_gate={"eligible": True, "reasons": [], "blocked_by": []},
        )

        self.assertEqual(payload["user_role"], "small business owner")
        self.assertEqual(
            payload["problem_summary"],
            "A single bad value in a CSV file can cause an import failure or lead to incorrect downstream data.",
        )
        self.assertEqual(
            payload["job_to_be_done"],
            "Validate CSV imports before they corrupt downstream business data",
        )


class TestProviderClassification(unittest.TestCase):
    def setUp(self):
        configure_build_prep({})

    def tearDown(self):
        configure_build_prep({})

    def test_get_platform_classification_config_defaults(self):
        """Test default config values."""
        # Clear any existing env vars
        original_env = os.environ.copy()
        try:
            # Remove test env vars
            for key in ["PLATFORM_FIT_LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
                os.environ.pop(key, None)

            config = get_platform_classification_config()
            self.assertEqual(config["provider"], "auto")
            self.assertEqual(config["ollama_base_url"], "http://127.0.0.1:11434")
            self.assertEqual(config["ollama_model"], "gemma4:latest")
            self.assertEqual(config["timeout_seconds"], 60.0)
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_get_platform_classification_config_env_override(self):
        """Test env vars override defaults."""
        original_env = os.environ.copy()
        try:
            os.environ["PLATFORM_FIT_LLM_PROVIDER"] = "ollama"
            os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
            os.environ["OLLAMA_MODEL"] = "llama3:8b"

            config = get_platform_classification_config()
            self.assertEqual(config["provider"], "ollama")
            self.assertEqual(config["ollama_base_url"], "http://localhost:11434")
            self.assertEqual(config["ollama_model"], "llama3:8b")
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_get_platform_classification_config_runtime_override(self):
        configure_build_prep(
            {
                "llm": {
                    "provider": "ollama",
                    "model": "gemma4:latest",
                    "base_url": "http://localhost:11434",
                },
                "build_prep": {
                    "platform_classification": {
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "ollama",
                        "model": "gemma4:latest",
                        "timeout_seconds": 90,
                    }
                },
            }
        )

        config = get_platform_classification_config()
        self.assertEqual(config["provider"], "ollama")
        self.assertEqual(config["ollama_base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(config["ollama_api_key"], "ollama")
        self.assertEqual(config["ollama_model"], "gemma4:latest")
        self.assertEqual(config["timeout_seconds"], 90.0)

    def test_keyword_fallback_handles_spreadsheet_revision_workflows(self):
        result = _determine_product_via_keyword(
            wedge_name="Local permitting document revision tracking",
            job_to_be_done="Track spreadsheet-driven submittal revisions and milestone billing changes",
            failure_mode="Excel-based revision sheets drift and contract milestone billing goes wrong",
            user_role="project coordinator",
        )

        self.assertEqual(result.host_platform, "Spreadsheet workflow")
        self.assertEqual(result.product_name, "spreadsheet_workflow_guard")

    def test_parse_platform_fit_response_valid_json(self):
        """Test parsing valid JSON response."""
        raw = '{"host_platform": "Google Docs", "product_format": "Google Docs add-on", "product_name": "Contract Guard", "one_sentence_product": "Prevents wrong company names in contracts", "why_this_format": "Attaches to Google Docs"}'
        result = _parse_platform_fit_response(raw, "test")
        self.assertIsNotNone(result)
        self.assertEqual(result.product_name, "Contract Guard")
        self.assertEqual(result.host_platform, "Google Docs")
        self.assertEqual(result.product_format, "Google Docs add-on")
        self.assertTrue(result.llm_used)
        self.assertFalse(result.fallback_used)

    def test_parse_platform_fit_response_with_markdown(self):
        """Test parsing JSON with markdown fences."""
        raw = '```json\n{"host_platform": "Gmail", "product_format": "Gmail add-on", "product_name": "Attachment Guard", "one_sentence_product": "Checks attachments before send", "why_this_format": "Integrated in Gmail"}\n```'
        result = _parse_platform_fit_response(raw, "test")
        self.assertIsNotNone(result)
        self.assertEqual(result.product_name, "Attachment Guard")

    def test_parse_platform_fit_response_empty(self):
        """Test parsing empty response returns None."""
        result = _parse_platform_fit_response("", "test")
        self.assertIsNone(result)

    def test_parse_platform_fit_response_no_product_name(self):
        """Test parsing response without product_name returns None."""
        raw = '{"host_platform": "Google Docs", "product_format": "Google Docs add-on"}'
        result = _parse_platform_fit_response(raw, "test")
        self.assertIsNone(result)

    def test_ollama_unavailable_returns_none(self):
        """Test Ollama returns None when unavailable."""
        configure_build_prep(
            {
                "build_prep": {
                    "platform_classification": {
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:9",
                        "model": "gemma4:latest",
                    }
                }
            }
        )
        result = _classify_via_ollama(
            job_to_be_done="contract errors",
            failure_mode="wrong company name",
            trigger_event="sending contract",
            current_workaround="manual check",
            pain_statement="lost client",
            user_role="freelancer",
            cluster_summary="template error",
        )
        # Should return None when Ollama is not running
        self.assertIsNone(result)

    def test_ollama_openai_compat_endpoint_uses_chat_completions(self):
        configure_build_prep(
            {
                "build_prep": {
                    "platform_classification": {
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "ollama",
                        "model": "gemma4:latest",
                    }
                }
            }
        )

        captured: dict[str, object] = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{\\"host_platform\\":\\"QuickBooks\\",'
                    b'\\"product_format\\":\\"QuickBooks add-on\\",'
                    b'\\"product_name\\":\\"Reconciliation Guard\\",'
                    b'\\"one_sentence_product\\":\\"Flags Stripe/QBO CSV reconciliation drift\\",'
                    b'\\"why_this_format\\":\\"Lives inside the accounting workflow\\"}"}}]}'
                )

        def _fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            return _FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = _classify_via_ollama(
                job_to_be_done="Import Stripe and bank-feed CSVs into QuickBooks without reconciliation drift",
                failure_mode="QuickBooks reconciliation drift after CSV import",
                trigger_event="Uploading Stripe payout and bank feed CSVs",
                current_workaround="manual reconciliation in spreadsheets",
                pain_statement="Accountants spend hours reconciling imports that should have matched automatically",
                user_role="bookkeeper",
                cluster_summary="stripe quickbooks reconciliation",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.product_name, "Reconciliation Guard")
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer ollama")

    def test_anthropic_no_key_returns_none(self):
        """Test Anthropic returns None when no API key."""
        # Temporarily clear API key
        original_env = os.environ.copy()
        try:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("CLAUDE_API_KEY", None)

            result = _classify_via_anthropic(
                job_to_be_done="contract errors",
                failure_mode="wrong company name",
                trigger_event="sending contract",
                current_workaround="manual check",
                pain_statement="lost client",
                user_role="freelancer",
                cluster_summary="template error",
            )
            self.assertIsNone(result)
        finally:
            os.environ.clear()
            os.environ.update(original_env)
