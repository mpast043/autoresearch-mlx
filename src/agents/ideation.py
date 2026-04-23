"""Idea generation agent."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.database import Database, Idea
from src.messaging import MessageQueue, MessageType
from src.opportunity_evaluation import canonical_scorecard_snapshot
from src.opportunity_spec import build_research_spec
from src.research_tools import slugify


class IdeationAgent(BaseAgent):
    """Turns promoted opportunities into research briefs, not default product builds."""

    def __init__(
        self,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("ideation", message_queue)
        self.db = db
        self.config = config or {}
        self._llm_client = None
        try:
            from src.llm_discovery_expander import LLMClient
            if config and config.get("llm", {}).get("reasoning_model"):
                self._llm_client = LLMClient(config)
        except Exception:
            pass

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type == MessageType.VALIDATION:
            return await self._generate_idea(message.payload)
        return {"processed": True}

    async def _generate_idea(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        passed_flag = bool(payload.get("passed"))
        selection_status = str(payload.get("selection_status") or "")
        build_brief_id = int(payload.get("build_brief_id") or 0)
        if not passed_flag and not (selection_status == "prototype_candidate" and build_brief_id > 0):
            return {"success": False, "reason": "validation did not pass or not prototype_candidate with brief"}

        validation = self.db.get_validation(payload["validation_id"])
        finding = self.db.get_finding(payload["finding_id"])
        if validation is None or finding is None:
            return {"success": False, "reason": "missing validation context"}

        evidence = validation.evidence_dict
        opportunity_evaluation = evidence.get("opportunity_evaluation")
        if not isinstance(opportunity_evaluation, dict):
            opportunity_evaluation = {}
        evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
        cluster = evidence.get("cluster", {})
        scorecard = evidence.get("opportunity_scorecard") or canonical_scorecard_snapshot(
            opportunity_evaluation
        )
        experiment_id = evidence.get("experiment_id")
        experiment_rows = []
        if experiment_id and hasattr(self.db, "get_experiments"):
            experiment_rows = self.db.get_experiments(opportunity_id=payload.get("opportunity_id"))
        experiment = experiment_rows[0] if experiment_rows else None
        validation_plan = (
            experiment.plan
            if experiment
            else dict(evaluation_evidence.get("validation_plan") or evidence.get("validation_plan", {}) or {})
        )

        title, description, product_type, audience, monetization, features = await self._generate_idea_fields(
            finding=finding,
            cluster=cluster,
            scorecard=scorecard,
            market_gap_state=str(
                evaluation_evidence.get("market_gap_state")
                or evidence.get("market_gap_state", "unknown")
                or "unknown"
            ),
            experiment=experiment,
            evidence=evidence,
        )

        idea_slug = slugify(title)
        existing = self.db.get_idea_by_slug(idea_slug)

        confidence = round(
            min(
                0.95,
                validation.overall_score * 0.75
                + scorecard.get("evidence_multiplier", 0.0) * 0.15
                + cluster.get("evidence_quality", 0.0) * 0.1,
            ),
            4,
        )

        build_ready = False

        spec = build_research_spec(
            slug=idea_slug,
            product_type=product_type,
            problem_statement=finding.product_built or "",
            value_hypothesis=description,
            core_features=features,
            audience=audience,
            monetization_strategy=monetization,
            source_finding_kind=finding.finding_kind,
            validation=validation,
            evidence={**evidence, "selection_status": evidence.get("selection_status") or payload.get("selection_status", "")},
            validation_plan=validation_plan,
            build_ready=build_ready,
        )
        pattern_ids = [finding.id] if finding.id is not None else []

        if existing is not None:
            merged_pattern_ids = sorted(set(existing.pattern_id_list + pattern_ids))
            merged_spec = existing.spec
            merged_spec.update(spec)
            updated_confidence = max(existing.confidence_score, confidence)
            status = (
                "research_backlog"
                if existing.status in {"proposed", "approved", "research_backlog"}
                else existing.status
            )
            self.db.update_idea(
                existing.id,
                description=description,
                pattern_ids=json.dumps(merged_pattern_ids),
                estimated_market_size=self._market_size_label(scorecard),
                technical_complexity=self._complexity_label(evidence.get("scores", {})),
                status=status,
                audience=audience,
                monetization_strategy=monetization,
                confidence_score=updated_confidence,
                product_type=product_type,
                spec_json=json.dumps(merged_spec),
            )
            idea_id = existing.id
        else:
            idea = Idea(
                slug=idea_slug,
                title=title,
                description=description,
                pattern_ids=json.dumps(pattern_ids),
                estimated_market_size=self._market_size_label(scorecard),
                technical_complexity=self._complexity_label(evidence.get("scores", {})),
                status="research_backlog",
                audience=audience,
                monetization_strategy=monetization,
                confidence_score=confidence,
                product_type=product_type,
                spec_json=json.dumps(spec),
            )
            idea_id = self.db.insert_idea(idea)

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.IDEA,
            payload={
                "idea_id": idea_id,
                "title": title,
                "confidence_score": confidence,
                "product_type": product_type,
                "refined": existing is not None,
                "build_ready": build_ready,
            },
            priority=2,
        )
        return {"success": True, "idea_id": idea_id, "title": title, "refined": existing is not None}

    async def _generate_idea_fields(
        self,
        *,
        finding,
        cluster: Dict[str, Any],
        scorecard: Dict[str, Any],
        market_gap_state: str,
        experiment,
        evidence: Dict[str, Any],
    ) -> tuple[str, str, str, str, str, list[str]]:
        """Try LLM idea generation first, fall back to template."""
        if self._llm_client:
            llm_result = await self._llm_generate_idea(
                finding=finding,
                cluster=cluster,
                scorecard=scorecard,
                market_gap_state=market_gap_state,
                experiment=experiment,
                evidence=evidence,
            )
            if llm_result:
                return llm_result
        opportunity_evaluation = evidence.get("opportunity_evaluation")
        if not isinstance(opportunity_evaluation, dict):
            opportunity_evaluation = {}
        evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
        return self._idea_from_validation(
            finding=finding,
            cluster=cluster,
            scorecard=scorecard,
            market_gap_state=market_gap_state,
            experiment=experiment,
            validation_plan=dict(
                evaluation_evidence.get("validation_plan")
                or evidence.get("validation_plan", {})
                or {}
            ),
        )

    async def _llm_generate_idea(
        self,
        *,
        finding,
        cluster: Dict[str, Any],
        scorecard: Dict[str, Any],
        market_gap_state: str,
        experiment,
        evidence: Dict[str, Any],
    ) -> tuple[str, str, str, str, str, list[str]] | None:
        """Generate idea via reasoning model. Returns None on failure."""
        if not self._llm_client:
            return None

        cluster_label = cluster.get("label") or finding.product_built or "Opportunity"
        scores = evidence.get("scores", {})
        validation_plan = experiment.plan if experiment else {}
        opportunity_evaluation = evidence.get("opportunity_evaluation")
        if not isinstance(opportunity_evaluation, dict):
            opportunity_evaluation = {}
        evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
        if not validation_plan:
            validation_plan = dict(evaluation_evidence.get("validation_plan") or evidence.get("validation_plan", {}) or {})
        counterevidence = list(
            evaluation_evidence.get("counterevidence")
            or evidence.get("counterevidence", [])
            or []
        )

        system_prompt = (
            "You are a product ideation analyst. Given evidence about a validated problem, "
            "generate a concrete product idea with a real product name, clear value proposition, "
            "target audience, monetization strategy, and key features.\n\n"
            "Return a JSON object with keys: product_title (string, max 60 chars), "
            "description (string, 2-3 sentences), product_type (string), audience (string), "
            "monetization (string), features (array of 3-5 strings)."
        )

        user_prompt = (
            f"PROBLEM: {finding.product_built or cluster_label}\n"
            f"OUTCOME: {finding.outcome_summary or ''}\n"
            f"CLUSTER: {cluster_label}\n"
            f"SCORE: {scorecard.get('total_score', 0):.2f}\n"
            f"DECISION: {scorecard.get('decision', 'park')}\n"
            f"MARKET GAP: {market_gap_state}\n"
            f"VALIDATION TEST: {validation_plan.get('test_type', 'problem_interviews')}\n"
            f"SMALLEST TEST: {validation_plan.get('smallest_test', '')}\n"
            f"COUNTEREVIDENCE: {counterevidence[:3] if counterevidence else 'none'}\n"
            f"FEASIBILITY: {scores.get('feasibility_score', 0):.2f}\n"
            f"PROBLEM SCORE: {scores.get('problem_score', 0):.2f}\n"
            f"VALUE SCORE: {scores.get('value_score', 0):.2f}\n"
        )

        try:
            raw = await self._llm_client.reasoning_agenerate(system_prompt, user_prompt)
            if not raw:
                return None
            from src.llm_discovery_expander import _extract_json
            parsed = _extract_json(raw)
            if not parsed or not isinstance(parsed, dict):
                return None
            title = str(parsed.get("product_title", ""))[:60]
            description = str(parsed.get("description", ""))
            product_type = str(parsed.get("product_type", "research-brief"))
            audience = str(parsed.get("audience", ""))
            monetization = str(parsed.get("monetization", ""))
            features = parsed.get("features", [])
            if not isinstance(features, list):
                features = [str(features)]
            features = [str(f) for f in features[:5]]
            if not title:
                return None
            return title, description, product_type, audience, monetization, features
        except Exception:
            return None

    def _idea_from_validation(
        self,
        *,
        finding,
        cluster: Dict[str, Any],
        scorecard: Dict[str, Any],
        market_gap_state: str,
        experiment,
        validation_plan: Dict[str, Any] | None = None,
    ) -> tuple[str, str, str, str, str, list[str]]:
        cluster_label = cluster.get("label") or finding.product_built or "Opportunity"
        product_title = f"{cluster_label[:44].strip()} Brief"
        audience = cluster.get("summary", {}).get("segment", "") or "operators dealing with repeated workflow pain"
        validation_plan = dict(validation_plan or (experiment.plan if experiment else {}) or {})
        description = (
            f"Research brief for the opportunity around '{finding.product_built or cluster_label}'. "
            f"Market state: {market_gap_state}. Next falsifiable test: "
            f"{validation_plan.get('test_type', 'problem_interviews').replace('_', ' ')}."
        )
        product_type = "research-brief"
        monetization = "No product build yet. Validate behavior and willingness to change workflow first."
        features = [
            f"Score: {scorecard.get('total_score', 0):.2f}",
            f"Decision: {scorecard.get('decision', 'park')}",
            f"Primary test: {validation_plan.get('smallest_test', 'Run operator interviews')}",
            "Counterevidence checklist included in spec_json",
        ]
        return product_title, description, product_type, audience, monetization, features

    def _market_size_label(self, scores: Dict[str, Any]) -> str:
        value = scores.get("total_score", 0) or (
            scores.get("problem_score", 0) + scores.get("value_score", 0)
        )
        if value >= 0.75:
            return "large"
        if value >= 0.55:
            return "medium"
        return "small"

    def _complexity_label(self, scores: Dict[str, Any]) -> str:
        feasibility = scores.get("feasibility_score", 0)
        if feasibility >= 0.75:
            return "low"
        if feasibility >= 0.55:
            return "medium"
        return "high"
