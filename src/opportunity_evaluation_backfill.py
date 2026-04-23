"""Backfill canonical opportunity evaluations onto legacy validation rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.opportunity_evaluation import (
    OPPORTUNITY_EVALUATION_SCHEMA_VERSION,
    build_opportunity_evaluation,
)
from src.validation_thresholds import resolve_promotion_park_thresholds


OPPORTUNITY_EVALUATION_BACKFILL_VERSION = "opportunity_evaluation_backfill_v1"
SOURCE_OF_TRUTH_PRECEDENCE = [
    "validation.evidence.opportunity_evaluation",
    "validations.evidence",
    "corroborations",
    "market_enrichments",
    "opportunities",
    "artifact_payloads",
]
CRITICAL_GROUPS = {
    "source_finding_kind",
    "validation_inputs",
    "measures",
    "policy.decision",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _record_evidence(record: Any) -> dict[str, Any]:
    return _coerce_dict(getattr(record, "evidence", {}) or {})


def _build_review_feedback_inputs(evidence: dict[str, Any], db: Any, *, finding_id: int, cluster_id: int) -> tuple[dict[str, Any], str]:
    review_feedback = _coerce_dict(evidence.get("review_feedback"))
    if review_feedback:
        return (
            {
                "count": int(review_feedback.get("review_feedback_count", review_feedback.get("count", 0)) or 0),
                "labels": review_feedback.get("review_feedback_labels", review_feedback.get("labels", [])) or [],
                "strongest_label": str(
                    review_feedback.get("review_feedback_strongest_label", review_feedback.get("strongest_label", "")) or ""
                ),
                "strongest_count": int(
                    review_feedback.get("review_feedback_strongest_count", review_feedback.get("strongest_count", 0)) or 0
                ),
                "consistency": float(
                    review_feedback.get("review_feedback_consistency", review_feedback.get("consistency", 0.0)) or 0.0
                ),
                "park_bias": float(
                    review_feedback.get("review_feedback_park_bias", review_feedback.get("park_bias", 0.0)) or 0.0
                ),
                "kill_bias": float(
                    review_feedback.get("review_feedback_kill_bias", review_feedback.get("kill_bias", 0.0)) or 0.0
                ),
            },
            "validations.evidence.review_feedback",
        )
    summary = db.get_review_feedback_summary(finding_id=finding_id, cluster_id=cluster_id or None)
    return (
        {
            "count": int(summary.get("count", 0) or 0),
            "labels": summary.get("labels", []) or [],
            "strongest_label": str(summary.get("strongest_label", "") or ""),
            "strongest_count": int(summary.get("strongest_count", 0) or 0),
            "consistency": float(summary.get("consistency", 0.0) or 0.0),
            "park_bias": float(summary.get("park_bias", 0.0) or 0.0),
            "kill_bias": float(summary.get("kill_bias", 0.0) or 0.0),
        },
        "review_feedback",
    )


def _build_market_enrichment_inputs(evidence: dict[str, Any], db: Any, *, finding_id: int) -> tuple[dict[str, Any], str]:
    payload = _coerce_dict(evidence.get("market_enrichment"))
    source = "validations.evidence.market_enrichment"
    if not payload:
        record = db.get_latest_market_enrichment(finding_id)
        payload = _record_evidence(record)
        if record is not None:
            payload.setdefault("demand_score", float(record.demand_score or 0.0))
            payload.setdefault("buyer_intent_score", float(record.buyer_intent_score or 0.0))
            payload.setdefault("competition_score", float(record.competition_score or 0.0))
            payload.setdefault("trend_score", float(record.trend_score or 0.0))
            payload.setdefault("review_signal_score", float(record.review_signal_score or 0.0))
            payload.setdefault("value_signal_score", float(record.value_signal_score or 0.0))
            source = "market_enrichments"
    return payload, source


def _build_corroboration_inputs(evidence: dict[str, Any], db: Any, *, finding_id: int) -> tuple[dict[str, Any], str]:
    payload = _coerce_dict(evidence.get("corroboration"))
    source = "validations.evidence.corroboration"
    if not payload:
        record = db.get_latest_corroboration(finding_id)
        payload = _record_evidence(record)
        if record is not None:
            payload.setdefault("recurrence_state", str(record.recurrence_state or ""))
            payload.setdefault("corroboration_score", float(record.corroboration_score or 0.0))
            payload.setdefault("evidence_sufficiency", float(record.evidence_sufficiency or 0.0))
            payload.setdefault("query_coverage", float(record.query_coverage or 0.0))
            payload.setdefault("family_confirmation_count", int(record.independent_confirmations or 0))
            payload.setdefault("source_family_diversity", int(record.source_diversity or 0))
            payload.setdefault("query_set_hash", str(record.query_set_hash or ""))
            source = "corroborations"
    return payload, source


def _scorecard_from_opportunity(opportunity: Any, evidence: dict[str, Any], cluster_record: Any) -> dict[str, Any]:
    notes = _coerce_dict(getattr(opportunity, "notes", {}))
    notes_scorecard = _coerce_dict(notes.get("scorecard"))
    cluster = _coerce_dict(evidence.get("cluster"))
    scorecard = {
        **notes_scorecard,
        "decision_score": float(getattr(opportunity, "decision_score", 0.0) or 0.0),
        "problem_truth_score": float(getattr(opportunity, "problem_truth_score", 0.0) or 0.0),
        "revenue_readiness_score": float(getattr(opportunity, "revenue_readiness_score", 0.0) or 0.0),
        "composite_score": float(getattr(opportunity, "composite_score", 0.0) or 0.0),
        "problem_plausibility": float(getattr(opportunity, "problem_plausibility", 0.0) or 0.0),
        "value_support": float(getattr(opportunity, "value_support", 0.0) or 0.0),
        "corroboration_strength": float(getattr(opportunity, "corroboration_strength", 0.0) or 0.0),
        "evidence_sufficiency": float(getattr(opportunity, "evidence_sufficiency", 0.0) or 0.0),
        "willingness_to_pay_proxy": float(getattr(opportunity, "willingness_to_pay_proxy", 0.0) or 0.0),
        "pain_severity": float(getattr(opportunity, "pain_severity", 0.0) or 0.0),
        "frequency_score": float(getattr(opportunity, "frequency_score", 0.0) or 0.0),
        "cost_of_inaction": float(getattr(opportunity, "cost_of_inaction", 0.0) or 0.0),
        "workaround_density": float(getattr(opportunity, "workaround_density", 0.0) or 0.0),
        "urgency_score": float(getattr(opportunity, "urgency_score", 0.0) or 0.0),
        "segment_concentration": float(getattr(opportunity, "segment_concentration", 0.0) or 0.0),
        "reachability": float(getattr(opportunity, "reachability", 0.0) or 0.0),
        "timing_shift": float(getattr(opportunity, "timing_shift", 0.0) or 0.0),
        "buildability": float(getattr(opportunity, "buildability", 0.0) or 0.0),
        "expansion_potential": float(getattr(opportunity, "expansion_potential", 0.0) or 0.0),
        "education_burden": float(getattr(opportunity, "education_burden", 0.0) or 0.0),
        "dependency_risk": float(getattr(opportunity, "dependency_risk", 0.0) or 0.0),
        "adoption_friction": float(getattr(opportunity, "adoption_friction", 0.0) or 0.0),
        "evidence_quality": float(getattr(opportunity, "evidence_quality", 0.0) or 0.0),
        "confidence": float(getattr(opportunity, "confidence", 0.0) or 0.0),
        "cluster_signal_count": int(cluster.get("signal_count", getattr(cluster_record, "signal_count", 0)) or 0),
        "cluster_atom_count": int(cluster.get("atom_count", getattr(cluster_record, "atom_count", 0)) or 0),
    }
    return scorecard


def _build_validation_plan(evidence: dict[str, Any], db: Any, *, opportunity_id: int) -> tuple[dict[str, Any], str]:
    validation_plan = _coerce_dict(evidence.get("validation_plan"))
    if validation_plan:
        return validation_plan, "validations.evidence.validation_plan"
    if opportunity_id:
        experiments = db.get_experiments(opportunity_id=opportunity_id, limit=1)
        if experiments:
            experiment = experiments[0]
            plan = _coerce_dict(getattr(experiment, "result", {}))
            if not plan:
                plan = {
                    "test_type": getattr(experiment, "test_type", ""),
                    "hypothesis": getattr(experiment, "hypothesis", ""),
                    "falsifier": getattr(experiment, "falsifier", ""),
                    "smallest_test": getattr(experiment, "smallest_test", ""),
                    "success_signal": getattr(experiment, "success_signal", ""),
                    "failure_signal": getattr(experiment, "failure_signal", ""),
                }
            return plan, "validation_experiments"
    return {}, ""


def _build_atom_summary(evidence: dict[str, Any], db: Any, *, finding_id: int, cluster_record: Any) -> tuple[dict[str, Any], str]:
    atoms = db.get_problem_atoms_by_finding(finding_id)
    if atoms:
        atom = atoms[0]
        return (
            {
                "segment": getattr(atom, "segment", ""),
                "user_role": getattr(atom, "user_role", ""),
                "job_to_be_done": getattr(atom, "job_to_be_done", ""),
                "trigger_event": getattr(atom, "trigger_event", ""),
                "failure_mode": getattr(atom, "failure_mode", ""),
                "current_workaround": getattr(atom, "current_workaround", ""),
            },
            "problem_atoms",
        )
    cluster = _coerce_dict(evidence.get("cluster"))
    cluster_summary = _coerce_dict(cluster.get("summary"))
    return (
        {
            "segment": cluster_summary.get("segment", getattr(cluster_record, "segment", "")),
            "user_role": cluster_summary.get("user_role", cluster.get("user_role", getattr(cluster_record, "user_role", ""))),
            "job_to_be_done": cluster_summary.get("job_to_be_done", cluster.get("job_to_be_done", getattr(cluster_record, "job_to_be_done", ""))),
            "trigger_event": "",
            "failure_mode": "",
            "current_workaround": "",
        },
        "cluster_summary",
    )


def build_backfilled_opportunity_evaluation(
    db: Any,
    *,
    validation: Any,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconstruct the canonical evaluation from existing stored sources."""
    evidence = validation.evidence_dict
    existing = _coerce_dict(evidence.get("opportunity_evaluation"))
    if existing.get("schema_version") == OPPORTUNITY_EVALUATION_SCHEMA_VERSION:
        return {
            "status": "already_canonical",
            "written": False,
            "validation_id": int(validation.id or 0),
            "finding_id": int(validation.finding_id or 0),
            "run_id": str(validation.run_id or ""),
            "sources_used": {"evaluation": "validation.evidence.opportunity_evaluation"},
            "missing_fields": [],
            "evaluation": existing,
            "backfill_metadata": {},
        }

    finding = db.get_finding(validation.finding_id)
    cluster_payload = _coerce_dict(evidence.get("cluster"))
    cluster_id = int(cluster_payload.get("cluster_id", 0) or 0)
    opportunity_id = int(evidence.get("opportunity_id", 0) or 0)
    opportunity = db.get_opportunity(opportunity_id) if opportunity_id else None
    if opportunity is None and cluster_id:
        opportunity = db.get_opportunity_by_cluster_id(cluster_id)
    if opportunity is not None and not opportunity_id:
        opportunity_id = int(opportunity.id or 0)
    if opportunity is not None and not cluster_id:
        cluster_id = int(opportunity.cluster_id or 0)
    cluster_record = db.get_cluster(cluster_id) if cluster_id else None

    sources_used: dict[str, str] = {}
    missing_fields: list[str] = []

    source_finding_kind = str(evidence.get("finding_kind") or getattr(finding, "finding_kind", "") or "")
    if source_finding_kind:
        sources_used["source_finding_kind"] = "validations.evidence" if evidence.get("finding_kind") else "findings"
    else:
        missing_fields.append("source_finding_kind")

    atom_summary, atom_source = _build_atom_summary(evidence, db, finding_id=validation.finding_id, cluster_record=cluster_record)
    sources_used["atom_summary"] = atom_source
    if not any(str(value or "").strip() for value in atom_summary.values()):
        missing_fields.append("inputs.atom")

    validation_scores = _coerce_dict(evidence.get("scores"))
    validation_inputs = {
        "problem_score": float(validation_scores.get("problem_score", 0.0) or 0.0),
        "solution_gap_score": float(validation_scores.get("solution_gap_score", 0.0) or 0.0),
        "saturation_score": float(validation_scores.get("saturation_score", 0.0) or 0.0),
        "feasibility_score": float(validation_scores.get("feasibility_score", 0.0) or 0.0),
        "value_score": float(validation_scores.get("value_score", 0.0) or 0.0),
        "market_score": float(validation.market_score or 0.0),
        "technical_score": float(validation.technical_score or 0.0),
        "distribution_score": float(validation.distribution_score or 0.0),
        "overall_score": float(validation.overall_score or 0.0),
        "cluster_signal_count": int(cluster_payload.get("signal_count", getattr(cluster_record, "signal_count", 0)) or 0),
        "cluster_atom_count": int(cluster_payload.get("atom_count", getattr(cluster_record, "atom_count", 0)) or 0),
        "recurrence_timeout": bool(evidence.get("recurrence_timeout", False)),
        "competitor_timeout": bool(evidence.get("competitor_timeout", False)),
    }
    if any(value for key, value in validation_inputs.items() if key not in {"recurrence_timeout", "competitor_timeout"}):
        sources_used["validation_inputs"] = "validations.evidence"
    else:
        missing_fields.append("validation_inputs")

    corroboration_inputs, corroboration_source = _build_corroboration_inputs(evidence, db, finding_id=validation.finding_id)
    if corroboration_inputs:
        sources_used["corroboration_inputs"] = corroboration_source

    market_enrichment_inputs, market_source = _build_market_enrichment_inputs(evidence, db, finding_id=validation.finding_id)
    if market_enrichment_inputs:
        sources_used["market_enrichment_inputs"] = market_source

    review_feedback_inputs, review_source = _build_review_feedback_inputs(
        evidence,
        db,
        finding_id=validation.finding_id,
        cluster_id=cluster_id,
    )
    if review_feedback_inputs:
        sources_used["review_feedback_inputs"] = review_source

    measures = _coerce_dict(evidence.get("opportunity_scorecard"))
    if measures:
        sources_used["measures"] = "validations.evidence.opportunity_scorecard"
    elif opportunity is not None:
        measures = _scorecard_from_opportunity(opportunity, evidence, cluster_record)
        sources_used["measures"] = "opportunities"
    else:
        missing_fields.append("measures")

    market_gap = _coerce_dict(evidence.get("market_gap"))
    if not market_gap and opportunity is not None:
        market_gap = _coerce_dict(getattr(opportunity, "notes", {}) or {}).get("market_gap", {})
    market_gap_state = str(evidence.get("market_gap_state") or market_gap.get("market_gap") or "")
    if market_gap_state:
        sources_used["evidence.market_gap_state"] = "validations.evidence" if evidence.get("market_gap_state") else "opportunities"

    counterevidence = _coerce_list(evidence.get("counterevidence"))
    if not counterevidence and opportunity is not None:
        counterevidence = _coerce_list(_coerce_dict(getattr(opportunity, "notes", {}) or {}).get("counterevidence"))
        if counterevidence:
            sources_used["evidence.counterevidence"] = "opportunities"
    elif counterevidence:
        sources_used["evidence.counterevidence"] = "validations.evidence"

    validation_plan, validation_plan_source = _build_validation_plan(evidence, db, opportunity_id=opportunity_id)
    if validation_plan:
        sources_used["evidence.validation_plan"] = validation_plan_source

    family_confirmation_count = int(
        evidence.get("family_confirmation_count", corroboration_inputs.get("family_confirmation_count", 0)) or 0
    )
    recurrence_state = str(evidence.get("recurrence_state") or corroboration_inputs.get("recurrence_state", "") or "")
    evidence_block = {
        "market_gap_state": market_gap_state or "unknown",
        "recurrence_state": recurrence_state,
        "family_confirmation_count": family_confirmation_count,
        "counterevidence": counterevidence,
        "validation_plan": validation_plan,
    }

    decision = str(
        evidence.get("decision")
        or getattr(opportunity, "recommendation", "")
        or ""
    )
    if decision:
        sources_used["policy.decision"] = "validations.evidence" if evidence.get("decision") else "opportunities"
    else:
        missing_fields.append("policy.decision")

    decision_reason = str(
        evidence.get("decision_reason")
        or evidence.get("reason")
        or ""
    )
    if decision_reason:
        sources_used["policy.decision_reason"] = "validations.evidence"

    selection_status = str(
        evidence.get("selection_status")
        or getattr(opportunity, "selection_status", "")
        or ""
    )
    if selection_status:
        sources_used["selection.selection_status"] = "validations.evidence" if evidence.get("selection_status") else "opportunities"
    else:
        missing_fields.append("selection.selection_status")

    selection_reason = str(
        evidence.get("selection_reason")
        or getattr(opportunity, "selection_reason", "")
        or ""
    )
    if selection_reason:
        sources_used["selection.selection_reason"] = "validations.evidence" if evidence.get("selection_reason") else "opportunities"

    selection_checks = _coerce_dict(evidence.get("selection_gate"))
    if selection_checks:
        sources_used["selection.selection_checks"] = "validations.evidence"
    elif opportunity_id:
        brief = db.get_build_brief_for_opportunity(opportunity_id, run_id=validation.run_id or None)
        brief_payload = _coerce_dict(getattr(brief, "brief", {}) if brief is not None else {})
        selection_checks = _coerce_dict(brief_payload.get("selection_gate"))
        if selection_checks:
            sources_used["selection.selection_checks"] = "artifact_payloads"

    promote_thresh, park_thresh = resolve_promotion_park_thresholds(config or {})

    critical_missing = [field for field in missing_fields if field in CRITICAL_GROUPS]
    if critical_missing:
        status = "unreconstructable"
        evaluation = {}
    else:
        evaluation = build_opportunity_evaluation(
            run_id=str(validation.run_id or ""),
            finding_id=int(validation.finding_id or 0),
            cluster_id=cluster_id,
            opportunity_id=opportunity_id,
            validation_id=int(validation.id or 0),
            source_finding_kind=source_finding_kind,
            atom_summary=atom_summary,
            validation_inputs=validation_inputs,
            corroboration_inputs=corroboration_inputs,
            market_enrichment_inputs=market_enrichment_inputs,
            review_feedback_inputs=review_feedback_inputs,
            measures=measures,
            evidence=evidence_block,
            decision=decision,
            decision_reason=decision_reason,
            promotion_threshold=promote_thresh,
            park_threshold=park_thresh,
            selection_status=selection_status,
            selection_reason=selection_reason,
            selection_checks=selection_checks,
        )
        status = "partially_reconstructable" if missing_fields else "reconstructable"

    backfill_metadata = {
        "version": OPPORTUNITY_EVALUATION_BACKFILL_VERSION,
        "status": status,
        "backfilled_at": _now_iso(),
        "sources_used": sources_used,
        "missing_fields": missing_fields,
        "source_of_truth_precedence": SOURCE_OF_TRUTH_PRECEDENCE,
    }
    return {
        "status": status,
        "written": False,
        "validation_id": int(validation.id or 0),
        "finding_id": int(validation.finding_id or 0),
        "run_id": str(validation.run_id or ""),
        "sources_used": sources_used,
        "missing_fields": missing_fields,
        "evaluation": evaluation,
        "backfill_metadata": backfill_metadata,
    }


def backfill_opportunity_evaluations(
    db: Any,
    *,
    config: dict[str, Any] | None,
    run_id: str | None = None,
    limit: int = 250,
    apply: bool = False,
) -> dict[str, Any]:
    """Backfill canonical evaluation snapshots for legacy validation rows."""
    rows = db.list_validation_evidence_payloads(
        run_id=(run_id or "").strip() or None,
        limit=limit,
        all_runs=not bool((run_id or "").strip()),
    )
    counts = {
        "already_canonical": 0,
        "reconstructable": 0,
        "partially_reconstructable": 0,
        "unreconstructable": 0,
        "written": 0,
    }
    previews: list[dict[str, Any]] = []

    with db.batch():
        for row in rows:
            validation = db.get_validation(int(row["validation_id"]))
            if validation is None:
                continue
            result = build_backfilled_opportunity_evaluation(db, validation=validation, config=config)
            counts[result["status"]] = counts.get(result["status"], 0) + 1

            can_write = result["status"] in {"reconstructable", "partially_reconstructable"}
            if apply and can_write:
                evidence = validation.evidence_dict
                evidence["opportunity_evaluation"] = result["evaluation"]
                evidence["opportunity_evaluation_backfill"] = result["backfill_metadata"]
                db.update_validation_evidence(validation.id or 0, evidence)
                result["written"] = True
                counts["written"] += 1

            if len(previews) < 12:
                previews.append(
                    {
                        "validation_id": result["validation_id"],
                        "finding_id": result["finding_id"],
                        "run_id": result["run_id"],
                        "status": result["status"],
                        "written": result["written"],
                        "missing_fields": result["missing_fields"],
                        "sources_used": result["sources_used"],
                    }
                )

    return {
        "scope": {
            "mode": "run" if run_id else "all_runs",
            "run_id": run_id or None,
            "limit": limit,
            "apply": apply,
        },
        "counts": counts,
        "previews": previews,
    }
