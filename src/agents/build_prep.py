"""Build-prep agents for solution framing, experiment design, and spec generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.build_prep import is_allowed_selection_transition
from src.database import BuildPrepOutput, Database
from src.messaging import MessageQueue, MessageType


class _BuildPrepAgent(BaseAgent):
    def __init__(self, name: str, db: Database, message_queue: Optional[MessageQueue] = None):
        super().__init__(name, message_queue)
        self.db = db

    def _load_context(self, payload: Dict[str, Any]) -> tuple[Any, dict[str, Any], int]:
        build_brief_id = int(payload["build_brief_id"])
        brief = self.db.get_build_brief(build_brief_id)
        if brief is None:
            raise ValueError(f"build brief {build_brief_id} not found")
        brief_payload = brief.brief
        return brief, brief_payload, build_brief_id


class SolutionFramingAgent(_BuildPrepAgent):
    """Turn a selected build brief into a narrow solution frame."""

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None):
        super().__init__("solution_framing", db, message_queue)

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type != MessageType.BUILD_BRIEF:
            return {"ignored": True}

        brief, brief_payload, build_brief_id = self._load_context(message.payload)

        # Get platform_fit for structured output
        platform_fit = brief_payload.get("platform_fit", {})
        host_platform = platform_fit.get("host_platform", "Unknown")
        product_format = platform_fit.get("product_format", "lightweight microSaaS")
        product_name = platform_fit.get("product_name", brief_payload.get("recommended_narrow_output_type", ""))
        one_sentence_product = platform_fit.get("one_sentence_product", "")
        why_this_format = platform_fit.get("why_this_format", "")

        framing = {
            "problem_frame": brief_payload.get("problem_summary", ""),
            "target_user": brief_payload.get("user_role", brief_payload.get("job_to_be_done", "")),
            "narrow_solution_bet": product_name,
            "host_platform": host_platform,
            "product_format": product_format,
            "one_sentence_product": one_sentence_product,
            "why_this_format": why_this_format,
            "excluded_scope": [
                "full product build",
                "broad multi-segment workflow suite",
                "unverified expansion workflows",
            ],
            "value_claim": one_sentence_product or "Reduce the recurring workflow failure before replacing the entire system.",
            "open_questions": brief_payload.get("open_questions_risks", [])[:5],
            "readiness_score": 0.82,
            "traceability": {
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
                "linked_finding_ids": brief_payload.get("linked_finding_ids", []),
            },
        }
        output_id = self.db.upsert_build_prep_output(
            BuildPrepOutput(
                build_brief_id=build_brief_id,
                opportunity_id=brief.opportunity_id,
                validation_id=brief.validation_id,
                agent_name=self.name,
                prep_stage="solution_framing",
                status="ready",
                output_json=json.dumps(framing),
                run_id=brief.run_id,
            )
        )

        if is_allowed_selection_transition(brief.status, "prototype_ready"):
            self.db.update_build_brief_status(build_brief_id, "prototype_ready")
        opportunity = self.db.get_opportunity(brief.opportunity_id)
        if opportunity and is_allowed_selection_transition(opportunity.selection_status, "prototype_ready"):
            self.db.update_opportunity_selection(
                brief.opportunity_id,
                selection_status="prototype_ready",
                selection_reason="solution_framing_complete",
            )

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.BUILD_PREP,
            payload={
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
                "agent_name": self.name,
                "prep_stage": "solution_framing",
                "output_id": output_id,
                "next_agent": "experiment_design",
            },
            priority=2,
        )
        return {"success": True, "output_id": output_id}


class ExperimentDesignAgent(_BuildPrepAgent):
    """Turn the framed opportunity into a concrete prototype experiment plan."""

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None):
        super().__init__("experiment_design", db, message_queue)

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type != MessageType.BUILD_PREP:
            return {"ignored": True}
        if message.payload.get("next_agent") != self.name:
            return {"ignored": True}

        brief, brief_payload, build_brief_id = self._load_context(message.payload)
        experiment = {
            "test_sequence": [
                "confirm the triggering workflow step",
                "walk through the current workaround",
                "show the narrow output concept",
                "ask for replacement threshold and adoption blockers",
            ],
            "acceptance_gate": "At least 3 of 5 target users prefer the narrow output over the manual fallback for the specific workflow slice.",
            "measurement_plan": {
                "primary_signal": brief_payload.get("first_experiment_hypothesis", ""),
                "secondary_signals": [
                    "time saved in the failing workflow step",
                    "confidence in replacing the workaround",
                    "willingness to pilot in the next 14 days",
                ],
            },
            "traceability": {
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
            },
        }
        output_id = self.db.upsert_build_prep_output(
            BuildPrepOutput(
                build_brief_id=build_brief_id,
                opportunity_id=brief.opportunity_id,
                validation_id=brief.validation_id,
                agent_name=self.name,
                prep_stage="experiment_design",
                status="ready",
                output_json=json.dumps(experiment),
                run_id=brief.run_id,
            )
        )
        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.BUILD_PREP,
            payload={
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
                "agent_name": self.name,
                "prep_stage": "experiment_design",
                "output_id": output_id,
                "next_agent": "spec_generation",
            },
            priority=2,
        )
        return {"success": True, "output_id": output_id}


class SpecGenerationAgent(_BuildPrepAgent):
    """Convert the brief plus prep outputs into a build-ready spec slice."""

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None):
        super().__init__("spec_generation", db, message_queue)

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type != MessageType.BUILD_PREP:
            return {"ignored": True}
        if message.payload.get("next_agent") != self.name:
            return {"ignored": True}

        brief, brief_payload, build_brief_id = self._load_context(message.payload)
        prior_outputs = self.db.list_build_prep_outputs(build_brief_id=build_brief_id, run_id=brief.run_id)
        output_map = {item.agent_name: item.output for item in prior_outputs}

        # Get platform_fit for structured output
        platform_fit = brief_payload.get("platform_fit", {})
        product_name = platform_fit.get("product_name", brief_payload.get("recommended_narrow_output_type", ""))
        host_platform = platform_fit.get("host_platform", "Unknown")
        product_format = platform_fit.get("product_format", "lightweight microSaaS")

        spec = {
            "scope": {
                "narrow_output_type": product_name,
                "host_platform": host_platform,
                "product_format": product_format,
                "must_solve": brief_payload.get("pain_workaround", {}).get("failure_mode", ""),
                "non_goals": output_map.get("solution_framing", {}).get("excluded_scope", []),
            },
            "artifact_checklist": brief_payload.get("launch_artifact_plan", []),
            "acceptance_criteria": output_map.get("experiment_design", {}).get("acceptance_gate", ""),
            "handoff_notes": {
                "linked_finding_ids": brief_payload.get("linked_finding_ids", []),
                "source_families": brief_payload.get("source_family_corroboration", {}).get("source_families", []),
                "open_questions": brief_payload.get("open_questions_risks", []),
            },
            "readiness_score": 0.84,
            "traceability": {
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
            },
        }
        output_id = self.db.upsert_build_prep_output(
            BuildPrepOutput(
                build_brief_id=build_brief_id,
                opportunity_id=brief.opportunity_id,
                validation_id=brief.validation_id,
                agent_name=self.name,
                prep_stage="spec_generation",
                status="ready",
                output_json=json.dumps(spec),
                run_id=brief.run_id,
            )
        )

        if is_allowed_selection_transition(brief.status, "build_ready"):
            self.db.update_build_brief_status(build_brief_id, "build_ready")
        opportunity = self.db.get_opportunity(brief.opportunity_id)
        if opportunity and is_allowed_selection_transition(opportunity.selection_status, "build_ready"):
            self.db.update_opportunity_selection(
                brief.opportunity_id,
                selection_status="build_ready",
                selection_reason="spec_generation_complete",
            )

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.BUILD_PREP,
            payload={
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
                "agent_name": self.name,
                "prep_stage": "spec_generation",
                "output_id": output_id,
                "next_agent": "",
            },
            priority=2,
        )
        return {"success": True, "output_id": output_id}

