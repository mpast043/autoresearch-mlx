"""Deterministic integration test for validation -> build-prep artifact flow."""

import asyncio
import os
import sys
import tempfile


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.agents.build_prep import ExperimentDesignAgent, SolutionFramingAgent, SpecGenerationAgent
from src.agents.validation import ValidationAgent
import src.agents.validation as validation_module
from src.database import Database, Finding
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
