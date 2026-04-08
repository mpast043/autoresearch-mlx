"""Orchestrator for routing evidence through validation and optional downstream flows."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.messaging import MessageBus, MessageType, create_message

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates agents and routes messages through the pipeline."""

    def __init__(
        self,
        db,
        status_tracker=None,
        auto_build: bool = False,
        auto_ideate: bool = False,
        shutdown_event: Optional[asyncio.Event] = None,
        stop_on_hit_config: Optional[Dict[str, Any]] = None,
    ):
        self._db = db
        self._agents: Dict[str, BaseAgent] = {}
        self._message_queue = MessageBus()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._status_tracker = status_tracker
        self._auto_build = auto_build
        self._auto_ideate = auto_ideate
        self._shutdown_event = shutdown_event
        self._stop_on_hit_config: Dict[str, Any] = dict(stop_on_hit_config or {})
        self._started_agents: set[str] = set()

    @staticmethod
    def stop_on_hit_matches(stop_cfg: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        """True if validation payload satisfies orchestration.stop_on_hit rules."""
        if not stop_cfg.get("enabled"):
            return False
        sel = str(payload.get("selection_status") or "")
        dec = str(payload.get("decision") or "")
        statuses = stop_cfg.get("selection_status_any", None)
        decisions = stop_cfg.get("decision_any", None)
        if statuses is None:
            statuses = ["prototype_candidate"]
        elif not isinstance(statuses, list):
            statuses = ["prototype_candidate"]
        if decisions is None:
            decisions = ["promote"]
        elif not isinstance(decisions, list):
            decisions = ["promote"]
        if statuses and sel in statuses:
            return True
        if decisions and dec in decisions:
            return True
        return False

    def _request_shutdown_on_hit(self, payload: Dict[str, Any]) -> None:
        if not self.stop_on_hit_matches(self._stop_on_hit_config, payload):
            return
        if self._stop_on_hit_config.get("exit_on_hit") is False:
            logger.info(
                "stop_on_hit matched selection_status=%s decision=%s — exit_on_hit=false, continuing",
                payload.get("selection_status"),
                payload.get("decision"),
            )
            return
        if self._shutdown_event is None:
            return
        logger.info(
            "stop_on_hit triggered selection_status=%s decision=%s — shutting down pipeline",
            payload.get("selection_status"),
            payload.get("decision"),
        )
        self._shutdown_event.set()

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
        agent._message_queue = self._message_queue

    async def start(self, skip_agents: Optional[set[str]] = None) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._started_agents = set()
        skipped = skip_agents or set()
        for agent in self._agents.values():
            if agent.name in skipped:
                continue
            await agent.start()
            self._started_agents.add(agent.name)

    async def consume_until_quiet(self, max_messages: int = 50, timeout: float = 5.0) -> int:
        """
        Consume messages from the queue until it's quiet or max_messages reached.
        Returns number of messages processed.
        """
        processed = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while processed < max_messages:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                msg = await asyncio.wait_for(
                    self._message_queue.receive("orchestrator"),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            await self._handle_orchestrator_message(msg)
            processed += 1
        return processed

    async def stop(self) -> None:
        self._running = False
        for name, agent in self._agents.items():
            if self._started_agents and name not in self._started_agents:
                continue
            await agent.stop()
        self._started_agents.clear()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while self._running:
            try:
                message = await self._message_queue.receive("orchestrator")
                await self._handle_orchestrator_message(message)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("orchestrator loop error: %s", exc)
                if self._status_tracker:
                    self._status_tracker.fail(str(exc))

    async def _handle_orchestrator_message(self, message) -> None:
        if message.msg_type == MessageType.FINDING:
            if self._status_tracker:
                self._status_tracker.set_stage("evidence")
                self._status_tracker.update(
                    discoveries=len(self._db.get_findings(limit=200)),
                    rawSignals=len(self._db.get_raw_signals(limit=500)) if hasattr(self._db, "get_raw_signals") else 0,
                    problemAtoms=len(self._db.get_problem_atoms(limit=500)) if hasattr(self._db, "get_problem_atoms") else 0,
                    opportunities=[
                        {
                            "name": finding.product_built,
                            "revenue": finding.monetization_method,
                            "kind": finding.finding_kind,
                        }
                        for finding in self._db.get_findings(limit=20)
                    ],
                )
                self._status_tracker.log(f"routing finding {message.payload.get('finding_id')} to evidence")

            await self.send_message(
                to_agent="evidence",
                msg_type=MessageType.FINDING,
                payload=message.payload,
                priority=2,
            )
            return

        if message.msg_type == MessageType.EVIDENCE:
            if self._status_tracker:
                self._status_tracker.set_stage("validation")
                self._status_tracker.log(
                    f"routing evidence for finding {message.payload.get('finding_id')} to validation"
                )

            await self.send_message(
                to_agent="validation",
                msg_type=MessageType.EVIDENCE,
                payload=message.payload,
                priority=2,
            )
            return

        elif message.msg_type == MessageType.FINDING_UNSEEDED:
            # Unseeded loop: skip evidence stage → route directly to validation
            finding_id = message.payload.get("finding_id")
            if self._status_tracker:
                self._status_tracker.set_stage("validation")
                self._status_tracker.log(f"[unseeded] routing finding {finding_id} directly to validation")
            # Send via message bus for proper routing
            await self.send_message(
                to_agent="validation",
                msg_type=MessageType.VALIDATION,
                payload={"finding_id": finding_id},
                priority=2,
            )
            return

        if message.msg_type == MessageType.VALIDATION:
            validations = self._db.get_recent_validations(limit=20)
            opportunities = self._db.get_opportunities(limit=20) if hasattr(self._db, "get_opportunities") else []
            build_brief_id = message.payload.get("build_brief_id")
            selection_status = message.payload.get("selection_status", "")

            # Fire-and-forget stop-on-hit check after validation routing
            self._request_shutdown_on_hit(message.payload)

            if self._status_tracker:
                self._status_tracker.set_stage("opportunity")
                self._status_tracker.update(
                    validated=len([row for row in validations if row["passed"]]),
                    clusters=len(self._db.get_clusters(limit=500)) if hasattr(self._db, "get_clusters") else 0,
                    experiments=len(self._db.get_experiments(limit=500)) if hasattr(self._db, "get_experiments") else 0,
                    ledgerEntries=len(self._db.list_ledger_entries(limit=500))
                    if hasattr(self._db, "list_ledger_entries")
                    else 0,
                    decisionBreakdown=self._validation_decision_breakdown(validations),
                    recentValidationReview=self._db.get_validation_review(limit=5)
                    if hasattr(self._db, "get_validation_review")
                    else [],
                    validatedIdeas=[
                        {
                            "name": row["product_built"],
                            "validation_score": row["overall_score"],
                            "decision": row.get("decision", "park"),
                        }
                        for row in validations
                    ][:10],
                    opportunities=[
                        {
                            "name": opportunity.title,
                            "status": opportunity.status,
                            "score": opportunity.composite_score,
                        }
                        for opportunity in opportunities[:10]
                    ],
                )
                self._status_tracker.log(
                    f"validation result for {message.payload.get('finding_id')}: decision={message.payload.get('decision', 'park')}"
                )
                if build_brief_id:
                    self._status_tracker.log(
                        f"selection ready for opportunity {message.payload.get('opportunity_id')}: {selection_status}"
                    )

            if build_brief_id and selection_status == "prototype_candidate":
                await self.send_message(
                    to_agent="solution_framing",
                    msg_type=MessageType.BUILD_BRIEF,
                    payload=message.payload,
                    priority=2,
                )

            # Ideation triggers on promote OR on prototype_candidate with build brief
            should_route_to_ideation = (
                self._auto_ideate
                and (
                    (message.payload.get("passed") and message.payload.get("decision") == "promote")
                    or (build_brief_id and selection_status == "prototype_candidate")
                )
            )
            if should_route_to_ideation:
                await self.send_message(
                    to_agent="ideation",
                    msg_type=MessageType.VALIDATION,
                    payload=message.payload,
                    priority=2,
                )
            return

        if message.msg_type == MessageType.BUILD_PREP:
            next_agent = message.payload.get("next_agent")
            if self._status_tracker:
                self._status_tracker.set_stage("build_prep")
                self._status_tracker.log(
                    f"build prep {message.payload.get('prep_stage')} complete for brief {message.payload.get('build_brief_id')}"
                )

            if next_agent:
                await self.send_message(
                    to_agent=next_agent,
                    msg_type=MessageType.BUILD_PREP,
                    payload=message.payload,
                    priority=2,
                )
            elif (
                self._auto_build
                and message.payload.get("prep_stage") == "spec_generation"
            ):
                await self.send_message(
                    to_agent="builder",
                    msg_type=MessageType.BUILD_REQUEST,
                    payload={
                        "build_brief_id": message.payload.get("build_brief_id"),
                        "opportunity_id": message.payload.get("opportunity_id"),
                        "validation_id": message.payload.get("validation_id"),
                    },
                    priority=2,
                )
            return

        if message.msg_type == MessageType.IDEA:
            ideas = self._db.get_ideas(limit=20)
            if self._status_tracker:
                self._status_tracker.set_stage("planning")
                self._status_tracker.update(
                    ideas=len(ideas),
                    generatedIdeas=[
                        {
                            "title": idea.title,
                            "description": idea.description,
                            "monetization_strategy": idea.monetization_strategy,
                        }
                        for idea in ideas[:10]
                    ],
                )
                self._status_tracker.log(f"idea ready: {message.payload.get('idea_id')}")

            if self._auto_build and message.payload.get("build_ready"):
                await self.send_message(
                    to_agent="builder",
                    msg_type=MessageType.BUILD_REQUEST,
                    payload={"idea_id": message.payload.get("idea_id")},
                    priority=2,
                )
            return

        if message.msg_type == MessageType.RESULT:
            products = self._db.get_products(limit=20)
            if self._status_tracker:
                self._status_tracker.update(
                    built=len(products),
                    mvps=[product["location"] for product in products[:10]],
                )
                self._status_tracker.complete()
            return

        logger.info("ignored orchestrator message: %s", message.msg_type)

    def _validation_decision_breakdown(self, validations: list[dict[str, Any]]) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for row in validations:
            decision = row.get("decision")
            if not decision:
                evidence = row.get("evidence")
                if isinstance(evidence, str):
                    try:
                        evidence = json.loads(evidence)
                    except Exception:
                        evidence = {}
                if isinstance(evidence, dict):
                    decision = evidence.get("decision")
            key = decision or "unknown"
            breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown

    async def send_message(
        self,
        to_agent: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 3,
    ) -> None:
        message = create_message(
            from_agent="orchestrator",
            to_agent=to_agent,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        await self._message_queue.put(message)
