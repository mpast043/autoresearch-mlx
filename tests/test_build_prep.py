"""Tests for build-prep selection and build brief helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_prep import (
    build_brief_payload,
    determine_selection_state,
    is_allowed_selection_transition,
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
                "core_source_family_diversity": 2,
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
                "core_source_family_diversity": 1,
                "generalizability_class": "product_specific_issue",
                "recurrence_state": "timeout",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "research_more")
        self.assertEqual(reason, "selection_gate_not_met")
        self.assertIn("single_family_support", gate["blocked_by"])
        self.assertIn("recurrence_timeout", gate["blocked_by"])

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
                "core_source_family_diversity": 2,
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
                "core_source_family_diversity": 2,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "timeout",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("prototype_candidate_multifamily_near_miss", gate["reasons"])

    def test_selection_gate_allows_exceptional_single_family_prototype_candidate(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.5586,
                "value_support": 0.5245,
                "composite_score": 0.3939,
            },
            corroboration={
                "corroboration_score": 0.534,
                "core_source_family_diversity": 1,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "supported",
            },
            market_enrichment={"wedge_active": False},
        )
        self.assertEqual(status, "prototype_candidate")
        self.assertEqual(reason, "prototype_candidate_gate")
        self.assertTrue(gate["eligible"])
        self.assertIn("prototype_candidate_single_family_exception", gate["reasons"])

    def test_selection_gate_allows_supported_single_family_ops_checkpoint_case(self):
        status, reason, gate = determine_selection_state(
            decision="park",
            scorecard={
                "evidence_quality": 0.494,
                "value_support": 0.5164,
                "composite_score": 0.4102,
            },
            corroboration={
                "corroboration_score": 0.4569,
                "core_source_family_diversity": 1,
                "generalizability_class": "reusable_workflow_pain",
                "recurrence_state": "supported",
            },
            market_enrichment={"wedge_active": False},
        )
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
        self.assertEqual(payload["source_family_corroboration"]["core_source_family_diversity"], 2)
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
        self.assertEqual(payload["recommended_narrow_output_type"], "workflow_reliability_assistant")
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
