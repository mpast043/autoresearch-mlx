"""Build-prep agents for solution framing, experiment design, and spec generation."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.build_prep import evaluate_build_ready_sharpness, is_allowed_selection_transition
from src.database import BuildPrepOutput, Database
from src.messaging import MessageQueue, MessageType

logger = logging.getLogger(__name__)


class _BuildPrepAgent(BaseAgent):
    def __init__(self, name: str, db: Database, message_queue: Optional[MessageQueue] = None, config: dict[str, Any] | None = None):
        super().__init__(name, message_queue)
        self.db = db
        self.config = config or {}

    def _load_context(self, payload: Dict[str, Any]) -> tuple[Any, dict[str, Any], int]:
        build_brief_id = int(payload["build_brief_id"])
        brief = self.db.get_build_brief(build_brief_id)
        if brief is None:
            raise ValueError(f"build brief {build_brief_id} not found")
        brief_payload = brief.brief
        return brief, brief_payload, build_brief_id

    def _wedge_feedback_enabled(self) -> bool:
        wedge_cfg = self.config.get("build_prep", {}).get("wedge_evaluation", {}) or {}
        return bool(wedge_cfg.get("feedback_to_discovery_terms", False))


class SolutionFramingAgent(_BuildPrepAgent):
    """Turn a selected build brief into a narrow solution frame."""

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None, config: dict[str, Any] | None = None):
        super().__init__("solution_framing", db, message_queue, config)

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
            "target_user": (
                brief_payload.get("user_role")
                or brief_payload.get("segment")
                or brief_payload.get("job_to_be_done", "")
            ),
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

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None, config: dict[str, Any] | None = None):
        super().__init__("experiment_design", db, message_queue, config)

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

    def __init__(self, db: Database, message_queue: Optional[MessageQueue] = None, config: dict[str, Any] | None = None):
        super().__init__("spec_generation", db, message_queue, config)

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
        solution_output = output_map.get("solution_framing", {}) or {}
        experiment_output = output_map.get("experiment_design", {}) or {}
        pain_workaround = brief_payload.get("pain_workaround", {}) or {}
        target_user = (
            solution_output.get("target_user")
            or brief_payload.get("user_role")
            or brief_payload.get("segment")
            or "target operator"
        )
        problem_statement = (
            solution_output.get("problem_frame")
            or brief_payload.get("problem_summary")
            or pain_workaround.get("pain_statement", "")
            or pain_workaround.get("failure_mode", "")
        )
        core_workflow = (
            pain_workaround.get("pain_statement", "")
            or brief_payload.get("job_to_be_done")
            or problem_statement
        )
        trigger_event = pain_workaround.get("trigger_event", "")
        current_workaround = pain_workaround.get("current_workaround", "")
        current_tools = pain_workaround.get("current_tools", "")
        must_solve = pain_workaround.get("failure_mode", "") or problem_statement
        if not product_name or product_name == "workflow_diagnostic_prototype":
            context_text = " ".join(
                str(item or "")
                for item in [
                    problem_statement,
                    core_workflow,
                    must_solve,
                    current_workaround,
                    current_tools,
                    target_user,
                ]
            ).lower()
            if "billing" in context_text and "reconcil" in context_text:
                product_name = "Billing Reconciliation Diagnostic"
            elif "inventory" in context_text:
                product_name = "Inventory Validation Gate"
            elif "journal" in context_text or "receipt" in context_text:
                product_name = "Receipt-to-Journal Entry Automator"
            elif "rfp" in context_text or "proposal" in context_text:
                product_name = "Proposal Review Checklist"
            elif "spreadsheet" in context_text or "excel" in context_text:
                product_name = "Spreadsheet Workflow Diagnostic"
            else:
                product_name = "Workflow Failure Diagnostic"
        value_claim = solution_output.get("value_claim") or platform_fit.get("one_sentence_product", "")
        non_goals = solution_output.get("excluded_scope", [])
        selection_blockers = (brief_payload.get("selection_gate", {}) or {}).get("blocked_by", [])
        acceptance_gate = experiment_output.get("acceptance_gate", "")
        measurement_plan = experiment_output.get("measurement_plan", {}) or {}
        launch_artifacts = brief_payload.get("launch_artifact_plan", [])
        functional_requirements = [
            {
                "id": "FR-1",
                "requirement": f"Capture the exact failing workflow context for {target_user}.",
                "source": "target_user/problem_frame",
            },
            {
                "id": "FR-2",
                "requirement": f"Accept the current artifacts or tools involved: {current_tools or 'the operator-provided workflow inputs'}.",
                "source": "pain_workaround.current_tools",
            },
            {
                "id": "FR-3",
                "requirement": f"Detect or expose the failure mode: {must_solve}.",
                "source": "pain_workaround.failure_mode",
            },
            {
                "id": "FR-4",
                "requirement": f"Produce a narrow {product_name or product_format} output that helps the user decide the next action.",
                "source": "recommended_narrow_output_type",
            },
            {
                "id": "FR-5",
                "requirement": "Preserve traceability back to the source finding, validation, and evidence blockers.",
                "source": "traceability/evidence_status",
            },
        ]
        workflow_steps = [
            {
                "step": 1,
                "name": "Intake current workflow evidence",
                "description": current_workaround or "Collect the artifact, export, message, or screen the operator currently uses.",
            },
            {
                "step": 2,
                "name": "Normalize the failing handoff",
                "description": core_workflow,
            },
            {
                "step": 3,
                "name": "Surface exceptions and gaps",
                "description": must_solve,
            },
            {
                "step": 4,
                "name": "Generate the narrow output",
                "description": value_claim or f"Create the first {product_format} slice for the specific workflow.",
            },
            {
                "step": 5,
                "name": "Review and decide",
                "description": acceptance_gate or "Confirm whether the draft output is better than the existing manual fallback.",
            },
        ]
        product_spec = {
            "product_name": product_name,
            "target_user": target_user,
            "one_sentence_product": solution_output.get("one_sentence_product") or value_claim,
            "problem_statement": problem_statement,
            "core_workflow": core_workflow,
            "trigger_event": trigger_event,
            "current_workaround": current_workaround,
            "current_tools": current_tools,
            "mvp_boundaries": {
                "in_scope": [
                    must_solve,
                    f"single narrow {product_format} workflow slice",
                    "evidence-linked product-spec draft",
                ],
                "out_of_scope": non_goals,
            },
            "workflow": workflow_steps,
            "functional_requirements": functional_requirements,
            "non_functional_requirements": [
                "Human-reviewable output before any automated action",
                "Traceable inputs, assumptions, and evidence blockers",
                "No broad workflow suite or unsupported segment expansion",
            ],
            "data_inputs": [
                item
                for item in [
                    current_tools,
                    trigger_event,
                    brief_payload.get("evidence_provenance", {}).get("origin_url", ""),
                ]
                if item
            ],
            "primary_outputs": [
                product_name or product_format,
                "exception summary",
                "next-action recommendation",
            ],
            "success_metrics": [
                metric
                for metric in [
                    acceptance_gate,
                    measurement_plan.get("primary_signal", ""),
                    "3 of 5 target users would replace the current workaround for this slice",
                ]
                if metric
            ],
            "research_gates_before_build": selection_blockers,
            "open_questions": brief_payload.get("open_questions_risks", []),
            "launch_artifacts": launch_artifacts,
        }

        spec = {
            "spec_mode": brief_payload.get("spec_mode", "build_ready_spec"),
            "product_name": product_name,
            "target_user": target_user,
            "problem_statement": problem_statement,
            "core_workflow": core_workflow,
            "product_spec": product_spec,
            "scope": {
                "narrow_output_type": product_name,
                "host_platform": host_platform,
                "product_format": product_format,
                "must_solve": must_solve,
                "non_goals": non_goals,
            },
            "artifact_checklist": launch_artifacts,
            "acceptance_criteria": acceptance_gate,
            "handoff_notes": {
                "linked_finding_ids": brief_payload.get("linked_finding_ids", []),
                "source_families": brief_payload.get("source_family_corroboration", {}).get("source_families", []),
                "open_questions": brief_payload.get("open_questions_risks", []),
            },
            "evidence_status": {
                "selection_status": brief_payload.get("selection_status", ""),
                "selection_reason": brief_payload.get("selection_reason", ""),
                "selection_blockers": selection_blockers,
                "recurrence_state": brief_payload.get("source_family_corroboration", {}).get("recurrence_state", ""),
                "corroboration_score": brief_payload.get("source_family_corroboration", {}).get("corroboration_score", 0.0),
                "source_family_diversity": brief_payload.get("source_family_corroboration", {}).get("source_family_diversity", 0),
            },
            "readiness_score": 0.84,
            "traceability": {
                "build_brief_id": build_brief_id,
                "opportunity_id": brief.opportunity_id,
                "validation_id": brief.validation_id,
            },
        }
        # Deterministic sharpness gate runs before LLM judgment so broad
        # or placeholder briefs never reach build_ready just because the
        # evaluator can invent a plausible commercialization story.
        from src.builder_output import WedgeEvaluator, WedgeEvaluation

        opportunity = self.db.get_opportunity(brief.opportunity_id)
        sharpness_gate = evaluate_build_ready_sharpness(brief_payload)
        spec_draft_mode = brief.status == "spec_draft" or brief_payload.get("spec_mode") == "product_spec_draft"
        output_status = "draft" if spec_draft_mode else ("ready" if sharpness_gate["passes"] else "blocked")
        spec["sharpness_gate"] = sharpness_gate
        output_id = self.db.upsert_build_prep_output(
            BuildPrepOutput(
                build_brief_id=build_brief_id,
                opportunity_id=brief.opportunity_id,
                validation_id=brief.validation_id,
                agent_name=self.name,
                prep_stage="spec_generation",
                status=output_status,
                output_json=json.dumps(spec),
                run_id=brief.run_id,
            )
        )

        if opportunity:
            notes = json.loads(opportunity.notes_json) if hasattr(opportunity, 'notes_json') and opportunity.notes_json else {}
            notes["build_ready_sharpness_gate"] = sharpness_gate
            if spec_draft_mode:
                notes["product_spec_draft"] = {
                    "build_brief_id": build_brief_id,
                    "output_id": output_id,
                    "status": "draft",
                    "blocked_by": sharpness_gate.get("reasons", []),
                }
                self.db.update_opportunity_notes(brief.opportunity_id, json.dumps(notes))
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
                return {"success": True, "output_id": output_id, "gate": "spec_draft"}

            if not sharpness_gate["passes"]:
                self.db.update_opportunity_notes(brief.opportunity_id, json.dumps(notes))
                if is_allowed_selection_transition(brief.status, "archive"):
                    self.db.update_build_brief_status(build_brief_id, "archive")
                    self.db.update_build_prep_outputs_status(build_brief_id, "blocked")
                if is_allowed_selection_transition(opportunity.selection_status, "research_more"):
                    failure_reasons = ", ".join(sharpness_gate["reasons"])
                    self.db.update_opportunity_selection(
                        brief.opportunity_id,
                        selection_status="research_more",
                        selection_reason=f"sharpness_gate_failed:{failure_reasons}",
                    )
                logger.info(
                    "Opp #%s failed build-ready sharpness gate: %s",
                    brief.opportunity_id,
                    ", ".join(sharpness_gate["reasons"]),
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
                return {"success": True, "output_id": output_id, "gate": "sharpness_failed"}

            wedge_evaluator = WedgeEvaluator(self.db, self.config if hasattr(self, 'config') else {})
            wedge_eval = wedge_evaluator.evaluate_sync(brief.opportunity_id)

            # Store evaluation on opportunity notes
            notes["wedge_evaluation"] = {
                "software_fit": wedge_eval.software_fit,
                "monetization_fit": wedge_eval.monetization_fit,
                "is_narrow": wedge_eval.is_narrow,
                "trust_risk": wedge_eval.trust_risk,
                "verdict": wedge_eval.verdict,
                "narrowness_reason": wedge_eval.narrowness_reason,
                "software_fit_reason": wedge_eval.software_fit_reason,
                "monetization_reason": wedge_eval.monetization_reason,
                "suggested_mvp": wedge_eval.suggested_mvp,
                "first_paid_offer": wedge_eval.first_paid_offer,
                "pricing_hypothesis": wedge_eval.pricing_hypothesis,
                "first_customer": wedge_eval.first_customer,
                "first_channel": wedge_eval.first_channel,
                "evaluated_by": wedge_eval.evaluated_by,
            }
            self.db.update_opportunity_notes(brief.opportunity_id, json.dumps(notes))

            if wedge_eval.passes_wedge_gate and is_allowed_selection_transition(opportunity.selection_status, "build_ready"):
                self.db.update_opportunity_selection(
                    brief.opportunity_id,
                    selection_status="build_ready",
                    selection_reason=f"wedge_gate_passed:{wedge_eval.verdict}",
                )
                if is_allowed_selection_transition(brief.status, "build_ready"):
                    self.db.update_build_brief_status(build_brief_id, "build_ready")
            elif is_allowed_selection_transition(opportunity.selection_status, "research_more"):
                failure_reasons = ", ".join(wedge_eval.gate_failure_reasons())
                self.db.update_opportunity_selection(
                    brief.opportunity_id,
                    selection_status="research_more",
                    selection_reason=f"wedge_gate_failed:{failure_reasons}",
                )
                logger.info(
                    f"Opp #{brief.opportunity_id} failed wedge gate: {failure_reasons}"
                )

            # Propagate wedge evaluation results back to source search terms
            # only when explicitly enabled. This feedback loop is powerful and
            # should not reshape discovery from a single soft judgment by default.
            if self._wedge_feedback_enabled():
                self._feedback_wedge_result(brief.opportunity_id, wedge_eval, notes)

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

    def _feedback_wedge_result(
        self,
        opportunity_id: int,
        wedge_eval: Any,
        notes: dict[str, Any],
    ) -> None:
        """Propagate wedge evaluation results back to the source search terms.

        Maps gate outcomes to term lifecycle counters:
        - passes_wedge_gate -> buildable_opportunity_count++
        - not is_narrow -> vague_bucket_count++

        Uses a hash-based dedup guard to avoid double-counting on re-runs
        while allowing re-evaluation when the model or prompt changes.
        """
        try:
            from src.discovery_term_lifecycle import TermLifecycleManager
        except ImportError:
            logger.debug("TermLifecycleManager not available, skipping wedge feedback")
            return

        # Dedup guard: hash of opportunity_id + verdict + evaluated_by
        # This allows re-evaluation when model/prompt changes (different hash)
        dedup_id = hashlib.md5(
            f"{opportunity_id}:{wedge_eval.verdict}:{wedge_eval.evaluated_by}".encode()
        ).hexdigest()
        if notes.get("wedge_feedback_id") == dedup_id:
            logger.debug(f"Wedge feedback already sent for opp #{opportunity_id}, skipping")
            return

        try:
            queries = self.db.get_search_terms_for_opportunity(opportunity_id)
        except Exception:
            logger.warning(f"Wedge feedback: could not resolve search terms for opp #{opportunity_id}")
            return

        if not queries:
            logger.debug(f"Wedge feedback: no source queries found for opp #{opportunity_id}")
            return

        lifecycle = TermLifecycleManager(self.db)

        is_buildable = wedge_eval.passes_wedge_gate
        is_too_broad = not wedge_eval.is_narrow

        for query in queries:
            try:
                lifecycle.record_wedge_feedback(
                    query,
                    is_buildable_wedge=is_buildable,
                    is_too_broad=is_too_broad,
                    verdict=wedge_eval.verdict,
                )
            except Exception as e:
                logger.debug(f"Wedge feedback failed for term '{query}': {e}")

        # Mark feedback as sent
        notes["wedge_feedback_id"] = dedup_id
        self.db.update_opportunity_notes(opportunity_id, json.dumps(notes))
        logger.info(
            f"Wedge feedback sent for opp #{opportunity_id}: "
            f"{len(queries)} queries, buildable={is_buildable}, too_broad={is_too_broad}"
        )
