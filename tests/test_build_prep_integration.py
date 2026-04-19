"""Deterministic integration test for validation -> build-prep artifact flow."""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.agents.build_prep import ExperimentDesignAgent, SolutionFramingAgent, SpecGenerationAgent
from src.builder_output import WedgeEvaluation
from src.agents.validation import ValidationAgent
import src.agents.validation as validation_module
from src.database import BuildBrief, BuildPrepOutput, Database, Finding, Opportunity, OpportunityCluster, Validation
from src.messaging import MessageType, create_message


def test_validation_to_build_prep_artifacts_end_to_end():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/test",
                source_url="https://example.com/thread",
                entrepreneur="Operations lead",
                product_built="Spreadsheet handoff failures in operations workflows",
                outcome_summary=(
                    "Ops teams repeatedly lose state across spreadsheet handoffs and use manual "
                    "copy/paste workarounds to recover."
                ),
                finding_kind="problem_signal",
                source_class="pain_signal",
                status="qualified",
            )
        )

        validation_agent = ValidationAgent(db)

        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.76,
                "value_score": 0.74,
                "feasibility_score": 0.71,
                "solution_gap_score": 0.68,
                "saturation_score": 0.61,
                "evidence": {
                    "recurrence_state": "supported",
                    "recurrence_timeout": False,
                    "competitor_timeout": False,
                    "recurrence_gap_reason": "",
                    "queries_executed": ['"spreadsheet handoff failures" operations'],
                    "queries_considered": ['"spreadsheet handoff failures" operations'],
                },
            }

        validation_agent.toolkit.validate_problem = fake_validate_problem

        original_determine_selection_state = validation_module.determine_selection_state
        validation_module.determine_selection_state = lambda **_kwargs: (
            "prototype_candidate",
            "validated_selection_gate",
            {"eligible": True, "gate_version": "build_prep_v1", "reasons": ["integration_test"], "blocked_by": []},
        )
        try:
            validation_result = asyncio.run(
                validation_agent.process(
                    create_message(
                        from_agent="evidence",
                        to_agent="validation",
                        msg_type=MessageType.EVIDENCE,
                        payload={
                            "finding_id": finding_id,
                            "source_class": "pain_signal",
                            "corroboration": {
                                "recurrence_state": "supported",
                                "corroboration_score": 0.72,
                                "core_source_family_diversity": 2,
                                "generalizability_class": "reusable_workflow_pain",
                                "source_families": ["reddit", "web"],
                                "core_source_families": ["reddit", "web"],
                                "source_family_match_counts": {"reddit": 3, "web": 2},
                            },
                            "market_enrichment": {
                                "wedge_active": False,
                                "demand_score": 0.6,
                                "buyer_intent_score": 0.62,
                                "value_signal_score": 0.58,
                            },
                        },
                    )
                )
            )
        finally:
            validation_module.determine_selection_state = original_determine_selection_state

        assert validation_result["success"] is True
        assert validation_result["build_brief_id"] > 0

        build_brief_id = int(validation_result["build_brief_id"])
        opportunity_id = int(validation_result["opportunity_id"])
        validation_id = int(validation_result["validation_id"])

        solution_agent = SolutionFramingAgent(db)
        experiment_agent = ExperimentDesignAgent(db)
        spec_agent = SpecGenerationAgent(db)

        sf_result = asyncio.run(
            solution_agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="solution_framing",
                    msg_type=MessageType.BUILD_BRIEF,
                    payload={
                        "build_brief_id": build_brief_id,
                        "opportunity_id": opportunity_id,
                        "validation_id": validation_id,
                    },
                )
            )
        )
        assert sf_result["success"] is True

        ed_result = asyncio.run(
            experiment_agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="experiment_design",
                    msg_type=MessageType.BUILD_PREP,
                    payload={
                        "build_brief_id": build_brief_id,
                        "opportunity_id": opportunity_id,
                        "validation_id": validation_id,
                        "next_agent": "experiment_design",
                    },
                )
            )
        )
        assert ed_result["success"] is True

        with patch(
            "src.builder_output.WedgeEvaluator.evaluate_sync",
            return_value=WedgeEvaluation(
                opportunity_id=opportunity_id,
                software_fit=0.82,
                monetization_fit=0.58,
                is_narrow=True,
                trust_risk="low",
                verdict="build_now",
            ),
        ), patch(
            "src.agents.build_prep.evaluate_build_ready_sharpness",
            return_value={"passes": True, "reasons": []},
        ):
            sg_result = asyncio.run(
                spec_agent.process(
                    create_message(
                        from_agent="orchestrator",
                        to_agent="spec_generation",
                        msg_type=MessageType.BUILD_PREP,
                        payload={
                            "build_brief_id": build_brief_id,
                            "opportunity_id": opportunity_id,
                            "validation_id": validation_id,
                            "next_agent": "spec_generation",
                        },
                    )
                )
            )
        assert sg_result["success"] is True

        outputs = db.list_build_prep_outputs(run_id=db.active_run_id, build_brief_id=build_brief_id, limit=10)
        assert len(outputs) == 3
        assert {item.agent_name for item in outputs} == {"solution_framing", "experiment_design", "spec_generation"}

        brief = db.get_build_brief(build_brief_id)
        assert brief is not None
        assert brief.status == "build_ready"
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_spec_generation_sharpness_failure_blocks_output_and_demotes_lifecycle():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/test",
                source_url="https://example.com/broad-thread",
                product_built="Generic operations dashboard",
                outcome_summary="The thread is too broad to define a build-ready wedge.",
                finding_kind="problem_signal",
                source_class="pain_signal",
                status="promoted",
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                label="broad operations dashboard",
                cluster_key="test-sharpness-failure",
                signal_count=1,
                atom_count=1,
            )
        )
        validation_id = db.insert_validation(
            Validation(
                finding_id=finding_id,
                run_id=db.active_run_id,
                passed=True,
                evidence={"recommendation": "promote"},
            )
        )
        opportunity_id = db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="Broad operations dashboard",
                market_gap="Operators want something better, but the wedge is underspecified.",
                recommendation="promote",
                status="promoted",
                selection_status="prototype_ready",
                selection_reason="solution_framing_complete",
                notes={},
            )
        )
        build_brief_id = db.upsert_build_brief(
            BuildBrief(
                run_id=db.active_run_id,
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                cluster_id=cluster_id,
                status="prototype_ready",
                recommended_output_type="Generic dashboard",
                brief_json=json.dumps(
                    {
                        "problem_summary": "A broad workflow is painful.",
                        "recommended_narrow_output_type": "Generic dashboard",
                        "platform_fit": {
                            "product_name": "Generic dashboard",
                            "host_platform": "Unknown",
                            "product_format": "dashboard",
                        },
                        "pain_workaround": {"failure_mode": "manual work is annoying"},
                    }
                ),
            )
        )
        db.upsert_build_prep_output(
            BuildPrepOutput(
                run_id=db.active_run_id,
                build_brief_id=build_brief_id,
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                agent_name="solution_framing",
                prep_stage="solution_framing",
                status="ready",
                output_json=json.dumps({"readiness_score": 0.8}),
            )
        )
        db.upsert_build_prep_output(
            BuildPrepOutput(
                run_id=db.active_run_id,
                build_brief_id=build_brief_id,
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                agent_name="experiment_design",
                prep_stage="experiment_design",
                status="ready",
                output_json=json.dumps({"acceptance_gate": "operator confirms the wedge"}),
            )
        )

        spec_agent = SpecGenerationAgent(db)
        with patch(
            "src.agents.build_prep.evaluate_build_ready_sharpness",
            return_value={"passes": False, "reasons": ["unknown_host_platform", "vague_product_name"]},
        ):
            sg_result = asyncio.run(
                spec_agent.process(
                    create_message(
                        from_agent="orchestrator",
                        to_agent="spec_generation",
                        msg_type=MessageType.BUILD_PREP,
                        payload={
                            "build_brief_id": build_brief_id,
                            "opportunity_id": opportunity_id,
                            "validation_id": validation_id,
                            "next_agent": "spec_generation",
                        },
                    )
                )
            )

        assert sg_result["success"] is True
        assert sg_result["gate"] == "sharpness_failed"

        outputs = db.list_build_prep_outputs(run_id=db.active_run_id, build_brief_id=build_brief_id, limit=10)
        assert len(outputs) == 3
        assert {output.agent_name for output in outputs} == {
            "solution_framing",
            "experiment_design",
            "spec_generation",
        }
        assert {output.status for output in outputs} == {"blocked"}
        spec_output = next(output for output in outputs if output.agent_name == "spec_generation")
        assert spec_output.output["sharpness_gate"]["passes"] is False

        brief = db.get_build_brief(build_brief_id)
        assert brief is not None
        assert brief.status == "archive"

        opportunity = db.get_opportunity(opportunity_id)
        assert opportunity is not None
        assert opportunity.selection_status == "research_more"
        assert opportunity.selection_reason == "sharpness_gate_failed:unknown_host_platform, vague_product_name"
        assert (opportunity.notes or {})["build_ready_sharpness_gate"]["passes"] is False
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_promoted_research_more_candidate_starts_spec_draft_without_build_ready():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/test",
                source_url="https://example.com/excel-billing-macro",
                entrepreneur="Finance operations lead",
                product_built="Legacy Excel macro causes billing reconciliation mismatches",
                outcome_summary=(
                    "Billing teams rely on an old Excel VBA macro with hardcoded rates, "
                    "no audit trail, and manual Friday handoff."
                ),
                finding_kind="problem_signal",
                source_class="pain_signal",
                status="qualified",
            )
        )

        validation_agent = ValidationAgent(db)

        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.62,
                "value_score": 0.56,
                "feasibility_score": 0.68,
                "solution_gap_score": 0.58,
                "saturation_score": 0.52,
                "evidence": {
                    "recurrence_state": "thin",
                    "recurrence_timeout": False,
                    "competitor_timeout": False,
                    "recurrence_gap_reason": "single_source_confirmation_only",
                    "queries_executed": ['"billing reconciliation" "excel macro"'],
                    "queries_considered": ['"billing reconciliation" "excel macro"'],
                },
            }

        validation_agent.toolkit.validate_problem = fake_validate_problem

        original_determine_selection_state = validation_module.determine_selection_state
        original_stage_decision = validation_module.stage_decision
        validation_module.determine_selection_state = lambda **_kwargs: (
            "research_more",
            "selection_gate_not_met",
            {
                "eligible": False,
                "gate_version": "build_prep_v1",
                "reasons": ["generalizable_workflow_pain"],
                "blocked_by": ["single_family_support"],
            },
        )
        validation_module.stage_decision = lambda *_args, **_kwargs: {
            "recommendation": "promote",
            "status": "promoted",
            "reason": "validated_selection_gate",
            "decision_reason": "validated_selection_gate",
            "park_subreason": "",
        }
        try:
            validation_result = asyncio.run(
                validation_agent.process(
                    create_message(
                        from_agent="evidence",
                        to_agent="validation",
                        msg_type=MessageType.EVIDENCE,
                        payload={
                            "finding_id": finding_id,
                            "source_class": "pain_signal",
                            "corroboration": {
                                "recurrence_state": "thin",
                                "corroboration_score": 0.42,
                                "source_family_diversity": 1,
                                "generalizability_class": "reusable_workflow_pain",
                                "source_families": ["reddit"],
                            },
                            "market_enrichment": {
                                "demand_score": 0.42,
                                "buyer_intent_score": 0.45,
                                "value_signal_score": 0.4,
                            },
                        },
                    )
                )
            )
        finally:
            validation_module.determine_selection_state = original_determine_selection_state
            validation_module.stage_decision = original_stage_decision

        assert validation_result["success"] is True
        assert validation_result["decision"] == "promote"
        assert validation_result["selection_status"] == "research_more"
        assert validation_result["build_brief_id"] > 0

        build_brief_id = int(validation_result["build_brief_id"])
        opportunity_id = int(validation_result["opportunity_id"])
        validation_id = int(validation_result["validation_id"])
        brief = db.get_build_brief(build_brief_id)
        assert brief is not None
        assert brief.status == "spec_draft"
        assert brief.brief["spec_mode"] == "product_spec_draft"

        solution_agent = SolutionFramingAgent(db)
        experiment_agent = ExperimentDesignAgent(db)
        spec_agent = SpecGenerationAgent(db)

        sf_result = asyncio.run(
            solution_agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="solution_framing",
                    msg_type=MessageType.BUILD_BRIEF,
                    payload={
                        "build_brief_id": build_brief_id,
                        "opportunity_id": opportunity_id,
                        "validation_id": validation_id,
                    },
                )
            )
        )
        assert sf_result["success"] is True

        ed_result = asyncio.run(
            experiment_agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="experiment_design",
                    msg_type=MessageType.BUILD_PREP,
                    payload={
                        "build_brief_id": build_brief_id,
                        "opportunity_id": opportunity_id,
                        "validation_id": validation_id,
                        "next_agent": "experiment_design",
                    },
                )
            )
        )
        assert ed_result["success"] is True

        with patch(
            "src.agents.build_prep.evaluate_build_ready_sharpness",
            return_value={"passes": False, "reasons": ["single_family_support"]},
        ):
            sg_result = asyncio.run(
                spec_agent.process(
                    create_message(
                        from_agent="orchestrator",
                        to_agent="spec_generation",
                        msg_type=MessageType.BUILD_PREP,
                        payload={
                            "build_brief_id": build_brief_id,
                            "opportunity_id": opportunity_id,
                            "validation_id": validation_id,
                            "next_agent": "spec_generation",
                        },
                    )
                )
            )

        assert sg_result["success"] is True
        assert sg_result["gate"] == "spec_draft"

        outputs = db.list_build_prep_outputs(run_id=db.active_run_id, build_brief_id=build_brief_id, limit=10)
        assert len(outputs) == 3
        spec_output = next(output for output in outputs if output.agent_name == "spec_generation")
        assert spec_output.status == "draft"
        assert spec_output.output["spec_mode"] == "product_spec_draft"
        assert spec_output.output["sharpness_gate"]["passes"] is False
        assert spec_output.output["product_name"]
        assert spec_output.output["target_user"]
        assert spec_output.output["core_workflow"]
        product_spec = spec_output.output["product_spec"]
        assert product_spec["problem_statement"]
        assert product_spec["target_user"]
        assert product_spec["workflow"]
        assert product_spec["functional_requirements"]
        assert product_spec["research_gates_before_build"] == ["single_family_support"]
        assert product_spec["mvp_boundaries"]["out_of_scope"]

        brief = db.get_build_brief(build_brief_id)
        assert brief is not None
        assert brief.status == "spec_draft"
        opportunity = db.get_opportunity(opportunity_id)
        assert opportunity is not None
        assert opportunity.selection_status == "research_more"
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)
