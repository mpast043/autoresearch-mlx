"""Validation agent with evidence-first scoring, selection gating, and build-brief generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from agents.base import BaseAgent
from build_prep import build_brief_payload, determine_selection_state
from database import (
    BuildBrief,
    Database,
    EvidenceLedgerEntry,
    Opportunity,
    OpportunityCluster,
    ProblemAtom,
    RawSignal,
    Validation,
    ValidationExperiment,
)
from messaging import MessageQueue, MessageType
from opportunity_engine import (
    assess_market_gap,
    build_cluster_summary,
    build_counterevidence,
    build_problem_atom,
    build_raw_signal_payload,
    plan_validation_experiment,
    score_opportunity,
    stage_decision,
)
from research_tools import ResearchToolkit
from source_policy import atom_generation_allowed
from validation_thresholds import resolve_promotion_park_thresholds


class ValidationAgent(BaseAgent):
    """Validate qualified findings and promote only well-supported opportunities."""

    def __init__(
        self,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("validation", message_queue)
        self.db = db
        self.config = config or {}
        self.toolkit = ResearchToolkit(self.config)

        validation_config = self.config.get("validation", {})
        weights = validation_config.get("weights", {})
        self.market_weight = weights.get("market", 0.4)
        self.technical_weight = weights.get("technical", 0.35)
        self.distribution_weight = weights.get("distribution", 0.25)

        thresholds = validation_config.get("thresholds", {})
        self.gate_threshold = thresholds.get("gate", 0.6)
        self.overall_threshold = thresholds.get("overall", 0.7)

        self.promotion_threshold, self.park_threshold = resolve_promotion_park_thresholds(self.config)

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type in (MessageType.FINDING, MessageType.EVIDENCE, MessageType.VALIDATION):
            return await self._validate_finding(message.payload)

        payload = message.payload
        command = payload.get("command")
        if command == "set_weights":
            return await self._set_weights(payload)
        if command == "get_thresholds":
            return {
                "gate_threshold": self.gate_threshold,
                "overall_threshold": self.overall_threshold,
            }
        return {"processed": True, "unknown_command": command}

    def reddit_runtime_summary(self) -> dict[str, Any]:
        return self.toolkit.get_reddit_runtime_metrics()

    async def _validate_finding(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        finding_id = finding_data.get("finding_id")
        if not finding_id:
            return {"success": False, "error": "No finding_id provided"}

        finding = self.db.get_finding(finding_id)
        if finding is None:
            return {"success": False, "error": f"finding {finding_id} not found"}

        finding_data = {
            **finding_data,
            "source_class": finding_data.get("source_class") or finding.source_class,
            "evidence": finding_data.get("evidence") or (finding.evidence or {}),
        }

        title = finding.product_built or "Untitled"
        summary = finding.outcome_summary or ""
        audience_hint = finding.entrepreneur or ""

        atoms = self._ensure_problem_atoms(finding_id, finding_data, finding.finding_kind)
        if not atoms:
            return {"success": False, "error": f"finding {finding_id} is not eligible for problem atom generation"}

        evidence_scores = finding_data.get("evidence_scores")
        if not evidence_scores:
            evidence_scores = await self.toolkit.validate_problem(
                title=title,
                summary=summary,
                finding_kind=finding.finding_kind,
                audience_hint=audience_hint,
            )

        market_score = round((float(evidence_scores["problem_score"]) + float(evidence_scores["value_score"])) / 2.0, 4)
        technical_score = round(float(evidence_scores["feasibility_score"]), 4)
        distribution_score = round(
            (float(evidence_scores["solution_gap_score"]) + float(evidence_scores["saturation_score"])) / 2.0,
            4,
        )

        anchor_atom = atoms[0]
        anchor_signal = self.db.get_raw_signal(anchor_atom.signal_id)
        if anchor_signal is None:
            return {"success": False, "error": f"signal {anchor_atom.signal_id} not found"}

        cluster_id, cluster, cluster_atoms = self._cluster_atoms(anchor_atom)
        review_feedback = self.db.get_review_feedback_summary(finding_id=finding_id, cluster_id=cluster_id)
        validation_payload = {
            "scores": evidence_scores,
            "evidence": evidence_scores["evidence"],
            "corroboration": finding_data.get("corroboration") or {},
            "market_enrichment": finding_data.get("market_enrichment") or {},
            "review_feedback": review_feedback,
        }
        cluster_context = self._cluster_context(cluster)
        market_gap = assess_market_gap(cluster_context, validation_payload)
        scorecard = score_opportunity(
            anchor_atom,
            anchor_signal,
            cluster_context,
            validation_payload,
            market_gap,
            review_feedback=review_feedback,
        )
        counterevidence = build_counterevidence(scorecard, market_gap)
        decision = stage_decision(
            scorecard,
            market_gap,
            counterevidence,
            promotion_threshold=self.promotion_threshold,
            park_threshold=self.park_threshold,
            review_feedback=review_feedback,
        )

        opportunity = Opportunity(
            cluster_id=cluster_id,
            title=cluster.label,
            market_gap=market_gap["market_gap"],
            recommendation=decision["recommendation"],
            status=decision["status"],
            pain_severity=scorecard["pain_severity"],
            frequency_score=scorecard["frequency_score"],
            cost_of_inaction=scorecard["cost_of_inaction"],
            workaround_density=scorecard["workaround_density"],
            urgency_score=scorecard["urgency_score"],
            segment_concentration=scorecard["segment_concentration"],
            reachability=scorecard["reachability"],
            timing_shift=scorecard["timing_shift"],
            buildability=scorecard["buildability"],
            expansion_potential=scorecard["expansion_potential"],
            education_burden=scorecard["education_burden"],
            dependency_risk=scorecard["dependency_risk"],
            adoption_friction=scorecard["adoption_friction"],
            evidence_quality=scorecard["evidence_quality"],
            composite_score=scorecard["composite_score"],
            confidence=scorecard["confidence"],
            selection_status="research_more",
            selection_reason="pending_selection",
            notes={
                "scorecard": scorecard,
                "market_gap": market_gap,
                "counterevidence": counterevidence,
                "cluster_summary": cluster.summary,
            },
        )

        selection_status, selection_reason, selection_gate = determine_selection_state(
            decision=decision["recommendation"],
            scorecard=scorecard,
            corroboration=validation_payload.get("corroboration", {}),
            market_enrichment=validation_payload.get("market_enrichment", {}),
        )
        opportunity.selection_status = selection_status
        opportunity.selection_reason = selection_reason
        opportunity_id = self.db.upsert_opportunity(opportunity)

        experiment_plan = plan_validation_experiment(anchor_atom, cluster_context, scorecard, market_gap)
        experiment = ValidationExperiment(
            opportunity_id=opportunity_id,
            cluster_id=cluster_id,
            test_type=experiment_plan["test_type"],
            hypothesis=experiment_plan["hypothesis"],
            falsifier=experiment_plan["falsifier"],
            smallest_test=experiment_plan["smallest_test"],
            success_signal=experiment_plan["success_signal"],
            failure_signal=experiment_plan["failure_signal"],
            status="proposed",
            result=experiment_plan,
            run_id=self.db.active_run_id,
        )
        experiment_id = self.db.insert_experiment(experiment)

        passed = decision["recommendation"] == "promote"
        overall_score = scorecard["composite_score"]
        evidence = self._build_evidence_payload(
            finding_id=finding_id,
            finding_kind=finding.finding_kind,
            title=title,
            summary=summary,
            evidence_scores=evidence_scores,
            corroboration=validation_payload.get("corroboration", {}),
            market_enrichment=validation_payload.get("market_enrichment", {}),
            review_feedback=review_feedback,
            market_score=market_score,
            technical_score=technical_score,
            distribution_score=distribution_score,
            overall_score=overall_score,
            cluster_id=cluster_id,
            cluster=cluster,
            market_gap=market_gap,
            scorecard=scorecard,
            opportunity_id=opportunity_id,
            experiment_id=experiment_id,
            counterevidence=counterevidence,
            decision=decision,
            selection_status=selection_status,
            selection_reason=selection_reason,
            selection_gate=selection_gate,
        )

        validation = Validation(
            finding_id=finding_id,
            market_score=market_score,
            technical_score=technical_score,
            distribution_score=distribution_score,
            overall_score=overall_score,
            passed=passed,
            evidence=evidence,
            run_id=self.db.active_run_id,
        )
        validation_id = self.db.upsert_validation(validation)

        build_brief_id = 0
        if selection_status == "prototype_candidate":
            linked_finding_ids = sorted({atom.finding_id for atom in cluster_atoms if getattr(atom, "finding_id", None) is not None})
            brief_payload = build_brief_payload(
                run_id=self.db.active_run_id,
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                cluster_id=cluster_id,
                linked_finding_ids=linked_finding_ids,
                finding=finding,
                cluster={
                    "label": cluster.label,
                    "job_to_be_done": cluster.job_to_be_done,
                    "user_role": cluster.user_role,
                    "summary": cluster.summary,
                },
                anchor_atom=anchor_atom,
                corroboration=validation_payload.get("corroboration", {}),
                market_enrichment=validation_payload.get("market_enrichment", {}),
                evidence_payload=evidence,
                experiment_hypothesis=experiment_plan["hypothesis"],
                selection_status=selection_status,
                selection_reason=selection_reason,
                selection_gate=selection_gate,
            )
            brief_json = json.dumps(brief_payload)
            build_brief = BuildBrief(
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                cluster_id=cluster_id,
                status="prototype_candidate",
                recommended_output_type=brief_payload.get("recommended_narrow_output_type", ""),
                brief_hash=hashlib.sha1(brief_json.encode("utf-8")).hexdigest(),
                brief_json=brief_json,
                run_id=self.db.active_run_id,
            )
            build_brief_id = self.db.upsert_build_brief(build_brief)

        finding_status = {
            "promote": "promoted",
            "park": "parked",
            "kill": "killed",
        }.get(decision["recommendation"], "reviewed")
        self.db.update_finding_status(finding_id, finding_status)

        discovery_meta = finding.evidence or {}
        discovery_query = discovery_meta.get("discovery_query")
        source_plan = discovery_meta.get("source_plan")
        if source_plan and discovery_query:
            self.db.record_validation_feedback(
                source_plan,
                discovery_query,
                passed=passed,
                overall_score=overall_score,
                selection_status=selection_status,
                build_brief_created=bool(build_brief_id),
            )

        self._record_ledger_entries(opportunity_id, cluster_id, evidence, decision["recommendation"], experiment_id)

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.VALIDATION,
            payload={
                "validation_id": validation_id,
                "finding_id": finding_id,
                "passed": passed,
                "overall_score": overall_score,
                "title": title,
                "finding_kind": finding.finding_kind,
                "decision": decision["recommendation"],
                "cluster_id": cluster_id,
                "opportunity_id": opportunity_id,
                "market_gap_state": market_gap["market_gap"],
                "experiment_id": experiment_id,
                "selection_status": selection_status,
                "selection_reason": selection_reason,
                "build_brief_id": build_brief_id,
            },
            priority=2 if passed else 4,
        )

        return {
            "success": True,
            "validation_id": validation_id,
            "finding_id": finding_id,
            "passed": passed,
            "overall_score": overall_score,
            "market_score": market_score,
            "technical_score": technical_score,
            "distribution_score": distribution_score,
            "value_score": evidence_scores["value_score"],
            "decision": decision["recommendation"],
            "cluster_id": cluster_id,
            "opportunity_id": opportunity_id,
            "experiment_id": experiment_id,
            "market_gap_state": market_gap["market_gap"],
            "selection_status": selection_status,
            "selection_reason": selection_reason,
            "build_brief_id": build_brief_id,
            "evidence": evidence,
        }

    def _using_mocked_score_methods(self) -> bool:
        return any(
            [
                type(self)._check_market_proof is not ValidationAgent._check_market_proof,
                type(self)._check_technical_feasibility is not ValidationAgent._check_technical_feasibility,
                type(self)._check_distribution is not ValidationAgent._check_distribution,
            ]
        )

    def _ensure_problem_atoms(
        self,
        finding_id: int,
        finding_data: Dict[str, Any],
        finding_kind: str,
    ) -> list[ProblemAtom]:
        atoms = self.db.get_problem_atoms_by_finding(finding_id)
        if atoms:
            return atoms

        source_class = finding_data.get("source_class")
        if not atom_generation_allowed(source_class, finding_kind):
            return []

        signals = self.db.get_raw_signals_by_finding(finding_id)
        if signals:
            signal = signals[0]
        else:
            signal_payload = build_raw_signal_payload(finding_data)
            signal = RawSignal(
                finding_id=finding_id,
                source_name=signal_payload["source_name"],
                source_type=signal_payload["source_type"],
                source_url=signal_payload["source_url"],
                title=signal_payload["title"],
                body_excerpt=signal_payload["body_excerpt"],
                quote_text=signal_payload["quote_text"],
                role_hint=signal_payload["role_hint"],
                published_at=signal_payload["published_at"],
                timestamp_hint=signal_payload["timestamp_hint"],
                content_hash=finding_data.get("content_hash", ""),
                metadata=signal_payload["metadata_json"],
                source_class=source_class or "pain_signal",
            )
            signal.id = self.db.insert_raw_signal(signal)

        atom_payload = build_problem_atom(
            {
                "source_name": signal.source_name,
                "source_type": signal.source_type,
                "source_url": signal.source_url,
                "title": signal.title,
                "body_excerpt": signal.body_excerpt,
                "quote_text": signal.quote_text,
                "role_hint": signal.role_hint,
                "published_at": signal.published_at,
                "timestamp_hint": signal.timestamp_hint,
                "metadata_json": signal.metadata or {},
            },
            {"finding_kind": finding_kind, "tool_used": "", "evidence": (signal.metadata or {}).get("evidence", {})},
        )
        atom = ProblemAtom(
            signal_id=signal.id or 0,
            finding_id=finding_id,
            cluster_key=atom_payload["cluster_key"],
            segment=atom_payload["segment"],
            user_role=atom_payload["user_role"],
            job_to_be_done=atom_payload["job_to_be_done"],
            trigger_event=atom_payload["trigger_event"],
            pain_statement=atom_payload["pain_statement"],
            failure_mode=atom_payload["failure_mode"],
            current_workaround=atom_payload["current_workaround"],
            current_tools=atom_payload["current_tools"],
            urgency_clues=atom_payload["urgency_clues"],
            frequency_clues=atom_payload["frequency_clues"],
            emotional_intensity=atom_payload["emotional_intensity"],
            cost_consequence_clues=atom_payload["cost_consequence_clues"],
            why_now_clues=atom_payload["why_now_clues"],
            confidence=atom_payload["confidence"],
            atom_json=json.dumps(atom_payload["atom_json"]),
        )
        atom.id = self.db.insert_problem_atom(atom)
        return [atom]

    def _cluster_atoms(self, anchor_atom: ProblemAtom) -> tuple[int, OpportunityCluster, list[ProblemAtom]]:
        cluster_atoms = self.db.get_problem_atoms_by_cluster_key(anchor_atom.cluster_key)
        if not cluster_atoms:
            cluster_atoms = [anchor_atom]
        signals = [self.db.get_raw_signal(atom.signal_id) for atom in cluster_atoms]
        resolved_signals = [signal for signal in signals if signal is not None]
        cluster_payload = build_cluster_summary(cluster_atoms, resolved_signals)
        cluster = OpportunityCluster(
            cluster_key=anchor_atom.cluster_key,
            label=cluster_payload["label"],
            segment=cluster_payload["segment"],
            user_role=cluster_payload["user_role"],
            job_to_be_done=cluster_payload["job_to_be_done"],
            trigger_summary=cluster_payload["trigger_summary"],
            signal_count=cluster_payload["signal_count"],
            atom_count=cluster_payload["atom_count"],
            evidence_quality=cluster_payload["evidence_quality"],
            status="candidate",
            summary=cluster_payload["summary_json"],
        )
        cluster_id = self.db.upsert_cluster(cluster)
        stored_cluster = self.db.get_cluster_record(cluster_id)
        return cluster_id, stored_cluster or cluster, cluster_atoms

    def _cluster_context(self, cluster: OpportunityCluster) -> dict[str, Any]:
        summary = cluster.summary or {}
        return {
            "label": cluster.label,
            "user_role": cluster.user_role,
            "job_to_be_done": cluster.job_to_be_done,
            "summary": summary,
            "human_summary": summary.get("human_summary") or cluster.label,
            "trigger_summary": summary.get("trigger_summary") or cluster.trigger_summary,
            "atom_count": summary.get("atom_count", cluster.atom_count),
            "signal_count": summary.get("signal_count", cluster.signal_count),
            "evidence_quality": summary.get("evidence_quality", cluster.evidence_quality),
            "sample_pains": summary.get("sample_pains", []),
            "sample_failures": summary.get("sample_failures", []),
            "sample_workarounds": summary.get("sample_workarounds", []),
        }

    def _build_evidence_payload(
        self,
        *,
        finding_id: int,
        finding_kind: str,
        title: str,
        summary: str,
        evidence_scores: Dict[str, Any],
        corroboration: dict[str, Any],
        market_enrichment: dict[str, Any],
        review_feedback: dict[str, Any],
        market_score: float,
        technical_score: float,
        distribution_score: float,
        overall_score: float,
        cluster_id: int,
        cluster: OpportunityCluster,
        market_gap: dict[str, Any],
        scorecard: dict[str, Any],
        opportunity_id: int,
        experiment_id: int,
        counterevidence: list[dict[str, Any]],
        decision: dict[str, Any],
        selection_status: str,
        selection_reason: str,
        selection_gate: dict[str, Any],
    ) -> dict[str, Any]:
        stage_gates = {
            "signal_gate": bool(cluster.evidence_quality >= 0.35 and cluster.atom_count >= 1),
            "cluster_gate": bool(cluster.atom_count >= 1),
            "opportunity_gate": bool(scorecard["composite_score"] >= self.promotion_threshold),
        }
        return {
            "finding_id": finding_id,
            "finding_kind": finding_kind,
            "decision": decision["recommendation"],
            "decision_reason": decision.get("decision_reason", decision.get("reason", "")),
            "park_subreason": decision.get("park_subreason", ""),
            "recurrence_timeout": evidence_scores["evidence"].get("recurrence_timeout", False),
            "competitor_timeout": evidence_scores["evidence"].get("competitor_timeout", False),
            "recurrence_state": evidence_scores["evidence"].get("recurrence_state"),
            "recurrence_gap_reason": evidence_scores["evidence"].get("recurrence_gap_reason", ""),
            "recurrence_failure_class": evidence_scores["evidence"].get("recurrence_failure_class", ""),
            "queries_considered": evidence_scores["evidence"].get("queries_considered", []),
            "queries_executed": evidence_scores["evidence"].get("queries_executed", []),
            "recurrence_budget_profile": evidence_scores["evidence"].get("recurrence_budget_profile", {}),
            "candidate_meaningful": evidence_scores["evidence"].get("candidate_meaningful", {}),
            "recurrence_probe_summary": evidence_scores["evidence"].get("recurrence_probe_summary", {}),
            "recurrence_source_branch": evidence_scores["evidence"].get("recurrence_source_branch", {}),
            "last_action": evidence_scores["evidence"].get("last_action", ""),
            "last_transition_reason": evidence_scores["evidence"].get("last_transition_reason", ""),
            "chosen_family": evidence_scores["evidence"].get("chosen_family", ""),
            "expected_gain_class": evidence_scores["evidence"].get("expected_gain_class", ""),
            "source_attempts_snapshot": evidence_scores["evidence"].get("source_attempts_snapshot", {}),
            "skipped_families": evidence_scores["evidence"].get("skipped_families", {}),
            "controller_actions": evidence_scores["evidence"].get("controller_actions", []),
            "budget_snapshot": evidence_scores["evidence"].get("budget_snapshot", {}),
            "fallback_strategy_used": evidence_scores["evidence"].get("fallback_strategy_used", ""),
            "decomposed_atom_queries": evidence_scores["evidence"].get("decomposed_atom_queries", []),
            "routing_override_reason": evidence_scores["evidence"].get("routing_override_reason", ""),
            "cohort_query_pack_used": evidence_scores["evidence"].get("cohort_query_pack_used", False),
            "cohort_query_pack_name": evidence_scores["evidence"].get("cohort_query_pack_name", ""),
            "web_query_strategy_path": evidence_scores["evidence"].get("web_query_strategy_path", []),
            "specialized_surface_targeting_used": evidence_scores["evidence"].get("specialized_surface_targeting_used", False),
            "promotion_gap_class": evidence_scores["evidence"].get("promotion_gap_class", ""),
            "near_miss_enrichment_action": evidence_scores["evidence"].get("near_miss_enrichment_action", ""),
            "sufficiency_priority_reason": evidence_scores["evidence"].get("sufficiency_priority_reason", ""),
            "value_enrichment_used": evidence_scores["evidence"].get("value_enrichment_used", False),
            "value_enrichment_queries": evidence_scores["evidence"].get("value_enrichment_queries", []),
            "value_enrichment_docs": evidence_scores["evidence"].get("value_enrichment_docs", []),
            "matched_results_by_source": evidence_scores["evidence"].get("matched_results_by_source", {}),
            "partial_results_by_source": evidence_scores["evidence"].get("partial_results_by_source", {}),
            "matched_docs_by_source": evidence_scores["evidence"].get("matched_docs_by_source", {}),
            "partial_docs_by_source": evidence_scores["evidence"].get("partial_docs_by_source", {}),
            "family_confirmation_count": evidence_scores["evidence"].get("family_confirmation_count", 0),
            "source_yield": evidence_scores["evidence"].get("source_yield", {}),
            "reshaped_query_history": evidence_scores["evidence"].get("reshaped_query_history", []),
            "scores": {
                "problem_score": evidence_scores["problem_score"],
                "solution_gap_score": evidence_scores["solution_gap_score"],
                "saturation_score": evidence_scores["saturation_score"],
                "feasibility_score": evidence_scores["feasibility_score"],
                "value_score": evidence_scores["value_score"],
                "market_score": market_score,
                "technical_score": technical_score,
                "distribution_score": distribution_score,
                "overall_score": overall_score,
            },
            "gates": {
                "recurring_problem": evidence_scores["problem_score"] >= self.gate_threshold,
                "solution_gap": evidence_scores["solution_gap_score"] >= 0.4,
                "saturation": evidence_scores["saturation_score"] >= 0.35,
                "feasibility": evidence_scores["feasibility_score"] >= self.gate_threshold,
                "value": evidence_scores["value_score"] >= 0.45,
                **stage_gates,
            },
            "summary": {
                "problem_statement": title,
                "value_signal": summary[:300],
            },
            "cluster": {
                "cluster_id": cluster_id,
                "cluster_key": cluster.cluster_key,
                "label": cluster.label,
                "summary": cluster.summary,
                "atom_count": cluster.atom_count,
                "signal_count": cluster.signal_count,
                "evidence_quality": cluster.evidence_quality,
            },
            "market_gap_state": market_gap["market_gap"],
            "market_gap": market_gap,
            "opportunity_scorecard": scorecard,
            "evidence_assessment": {
                "problem_plausibility": scorecard.get("problem_plausibility", 0.0),
                "evidence_sufficiency": scorecard.get("evidence_sufficiency", 0.0),
                "value_support": scorecard.get("value_support", 0.0),
                "composite_score": scorecard.get("composite_score", 0.0),
            },
            "corroboration": corroboration,
            "market_enrichment": market_enrichment,
            "review_feedback": {
                "review_feedback_count": review_feedback.get("count", 0),
                "review_feedback_labels": review_feedback.get("labels", []),
                "review_feedback_strongest_label": review_feedback.get("strongest_label", ""),
                "review_feedback_strongest_count": review_feedback.get("strongest_count", 0),
                "review_feedback_consistency": review_feedback.get("consistency", 0.0),
                "review_feedback_park_bias": review_feedback.get("park_bias", 0.0),
                "review_feedback_kill_bias": review_feedback.get("kill_bias", 0.0),
            },
            "counterevidence": counterevidence,
            "opportunity_id": opportunity_id,
            "experiment_id": experiment_id,
            "selection_status": selection_status,
            "selection_reason": selection_reason,
            "selection_gate": selection_gate,
        }

    def _record_ledger_entries(
        self,
        opportunity_id: int,
        cluster_id: int,
        evidence: dict[str, Any],
        decision: str,
        experiment_id: int,
    ) -> None:
        run_id = self.db.active_run_id
        self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="opportunity",
                entity_id=opportunity_id,
                entry_kind="decision",
                summary=f"Validation decision: {decision}",
                entry_json=evidence,
                run_id=run_id,
            )
        )
        self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="cluster",
                entity_id=cluster_id,
                entry_kind="validation_summary",
                summary="Cluster validation summary updated.",
                entry_json=evidence.get("cluster", {}),
                run_id=run_id,
            )
        )
        self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="experiment",
                entity_id=experiment_id,
                entry_kind="experiment_plan",
                summary="Validation experiment proposed.",
                entry_json={"experiment_id": experiment_id, "opportunity_id": opportunity_id},
                run_id=run_id,
            )
        )

    async def _set_weights(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.market_weight = float(payload.get("market_weight", self.market_weight))
        self.technical_weight = float(payload.get("technical_weight", self.technical_weight))
        self.distribution_weight = float(payload.get("distribution_weight", self.distribution_weight))
        return {
            "market_weight": self.market_weight,
            "technical_weight": self.technical_weight,
            "distribution_weight": self.distribution_weight,
        }

    async def _check_market_proof(self, evidence_scores: Dict[str, Any]) -> float:
        return (float(evidence_scores.get("problem_score", 0.0)) + float(evidence_scores.get("value_score", 0.0))) / 2.0

    async def _check_technical_feasibility(self, evidence_scores: Dict[str, Any]) -> float:
        return float(evidence_scores.get("feasibility_score", 0.0))

    async def _check_distribution(self, evidence_scores: Dict[str, Any]) -> float:
        return (float(evidence_scores.get("solution_gap_score", 0.0)) + float(evidence_scores.get("saturation_score", 0.0))) / 2.0

    async def _run_legacy_validation(self, finding: Any, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        title = finding.product_built or "Untitled"
        summary = finding.outcome_summary or ""
        evidence_scores = await self.toolkit.validate_problem(
            title=title,
            summary=summary,
            finding_kind=finding.finding_kind,
            audience_hint=finding.entrepreneur or "",
        )
        market_score = round(await self._check_market_proof(evidence_scores), 4)
        technical_score = round(await self._check_technical_feasibility(evidence_scores), 4)
        distribution_score = round(await self._check_distribution(evidence_scores), 4)
        return {
            "success": True,
            "finding_id": finding.id,
            "passed": market_score >= self.gate_threshold and technical_score >= self.gate_threshold,
            "overall_score": round((market_score + technical_score + distribution_score) / 3.0, 4),
            "market_score": market_score,
            "technical_score": technical_score,
            "distribution_score": distribution_score,
            "value_score": evidence_scores.get("value_score", 0.0),
            "evidence": {"scores": evidence_scores},
        }
