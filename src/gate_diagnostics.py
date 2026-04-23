"""Operator-facing gate diagnostics (why promote/park/kill and selection_status)."""

from __future__ import annotations

import json
from typing import Any

from src.build_prep import explain_selection_gate_detail
from src.opportunity_engine import diagnose_stage_decision
from src.validation_thresholds import resolve_promotion_park_thresholds


def _review_feedback_for_engine(evidence: dict[str, Any]) -> dict[str, Any]:
    """Map validation evidence.review_feedback into stage_decision review_feedback shape."""
    evaluation = evidence.get("opportunity_evaluation")
    canonical_feedback = {}
    if isinstance(evaluation, dict):
        canonical_feedback = ((evaluation.get("inputs", {}) or {}).get("review_feedback", {}) or {})
    rf = canonical_feedback or evidence.get("review_feedback") or {}
    return {
        "park_bias": float(rf.get("review_feedback_park_bias", rf.get("park_bias", 0.0)) or 0.0),
        "kill_bias": float(rf.get("review_feedback_kill_bias", rf.get("kill_bias", 0.0)) or 0.0),
    }


def _canonical_snapshot(evidence: dict[str, Any]) -> dict[str, Any]:
    evaluation = evidence.get("opportunity_evaluation")
    if not isinstance(evaluation, dict):
        return {"available": False}

    inputs = evaluation.get("inputs", {}) or {}
    corroboration_inputs = inputs.get("corroboration", {}) or {}
    market_inputs = inputs.get("market_enrichment", {}) or {}
    measures = evaluation.get("measures", {}) or {}
    scores = measures.get("scores", {}) or {}
    dimensions = measures.get("dimensions", {}) or {}
    transition = measures.get("transition", {}) or {}
    policy = evaluation.get("policy", {}) or {}
    selection = evaluation.get("selection", {}) or {}
    shadow = evaluation.get("shadow", {}) or {}

    return {
        "available": True,
        "decision": str(policy.get("decision", "") or ""),
        "decision_reason": str(policy.get("decision_reason", "") or ""),
        "policy_version": str(policy.get("policy_version", "") or ""),
        "policy_checks": dict(policy.get("policy_checks", {}) or {}),
        "selection_status": str(selection.get("selection_status", "") or ""),
        "selection_reason": str(selection.get("selection_reason", "") or ""),
        "selection_checks": dict(selection.get("selection_checks", {}) or {}),
        "build_prep_eligible": bool(selection.get("build_prep_eligible", False)),
        "build_prep_route": str(selection.get("build_prep_route", "") or ""),
        "decision_score": float(scores.get("decision_score", 0.0) or 0.0),
        "problem_truth_score": float(scores.get("problem_truth_score", 0.0) or 0.0),
        "revenue_readiness_score": float(scores.get("revenue_readiness_score", 0.0) or 0.0),
        "frequency_score": float(dimensions.get("frequency_score", 0.0) or 0.0),
        "evidence_quality": float(dimensions.get("evidence_quality", 0.0) or 0.0),
        "value_support": float(dimensions.get("value_support", 0.0) or 0.0),
        "composite_score": float(transition.get("composite_score", 0.0) or 0.0),
        "market_gap_state": str((evaluation.get("evidence", {}) or {}).get("market_gap_state", "") or ""),
        "shadow_score_v2_lite": shadow.get("shadow_score_v2_lite"),
        "corroboration_inputs": corroboration_inputs,
        "market_enrichment_inputs": market_inputs,
        "evaluation": evaluation,
    }


def _scorecard_from_canonical(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_score": snapshot.get("decision_score", 0.0),
        "problem_truth_score": snapshot.get("problem_truth_score", 0.0),
        "revenue_readiness_score": snapshot.get("revenue_readiness_score", 0.0),
        "frequency_score": snapshot.get("frequency_score", 0.0),
        "evidence_quality": snapshot.get("evidence_quality", 0.0),
        "value_support": snapshot.get("value_support", 0.0),
        "composite_score": snapshot.get("composite_score", 0.0),
    }


def explain_validation_evidence(evidence: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    """Full gate breakdown for one persisted validation ``evidence`` JSON blob."""
    cfg = config or {}
    promote, park = resolve_promotion_park_thresholds(cfg)
    canonical = _canonical_snapshot(evidence)
    canonical_evidence = (
        ((canonical.get("evaluation") or {}).get("evidence", {}) or {})
        if canonical.get("available")
        else {}
    )

    scorecard = (
        _scorecard_from_canonical(canonical)
        if canonical["available"]
        else (evidence.get("opportunity_scorecard") or {})
    )
    market_gap = canonical_evidence.get("market_gap") or evidence.get("market_gap") or {}
    counterevidence = canonical_evidence.get("counterevidence") or evidence.get("counterevidence") or []
    review_fb = _review_feedback_for_engine(evidence)

    stage_diag = diagnose_stage_decision(
        scorecard,
        market_gap,
        counterevidence,
        promotion_threshold=promote,
        park_threshold=park,
        review_feedback=review_fb,
    )

    raw_decision = str(canonical.get("decision") or evidence.get("decision") or "park")
    if raw_decision not in {"promote", "park", "kill"}:
        raw_decision = "park"

    corroboration = canonical.get("corroboration_inputs") or evidence.get("corroboration") or {}
    market_enrichment = canonical.get("market_enrichment_inputs") or evidence.get("market_enrichment") or {}
    selection_diag = explain_selection_gate_detail(
        decision=raw_decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        opportunity_evaluation=canonical.get("evaluation") if canonical.get("available") else None,
    )

    vc = cfg.get("validation", {}) or {}
    thresholds_block = vc.get("thresholds", {}) or {}

    return {
        "effective_thresholds": {
            "promotion_threshold": promote,
            "park_threshold": park,
            "evidence_gate_threshold": float(thresholds_block.get("gate", 0.6)),
            "overall_threshold": float(thresholds_block.get("overall", 0.7)),
            "resolution_order": [
                "validation.decisions.promote_score / park_score",
                "orchestration.promotion_threshold / park_threshold",
                "validation.promotion_threshold / park_threshold",
                "defaults 0.62 / 0.48",
            ],
        },
        "canonical_evaluation": canonical,
        "stage_decision": stage_diag,
        "selection_gate": selection_diag,
    }


def build_gate_diagnostics_report(
    db: Any,
    *,
    config: dict[str, Any] | None,
    run_id: str | None = None,
    limit: int = 25,
    finding_id: int | None = None,
) -> dict[str, Any]:
    """Aggregate diagnostics for the current or given run (optionally one finding)."""
    cfg = config or {}
    rid = (run_id or "").strip() or getattr(db, "get_active_run_id", lambda: "")() or ""
    if not rid:
        rid = db.get_latest_run_id()
    promote, park = resolve_promotion_park_thresholds(cfg)

    rows = db.list_validation_evidence_payloads(run_id=rid, limit=500)
    if finding_id is not None:
        rows = [row for row in rows if int(row.get("finding_id") or 0) == int(finding_id)]

    items: list[dict[str, Any]] = []
    canonical_rows = 0
    for row in rows[:limit]:
        ev = row.get("evidence") or {}
        detail = explain_validation_evidence(ev, cfg)
        canonical = detail.get("canonical_evaluation", {}) or {}
        if canonical.get("available"):
            canonical_rows += 1
        items.append(
            {
                "validation_id": row.get("validation_id"),
                "finding_id": row.get("finding_id"),
                "title": (ev.get("summary") or {}).get("problem_statement")
                or ev.get("finding_kind", ""),
                "decision": canonical.get("decision") or ev.get("decision"),
                "selection_status": canonical.get("selection_status") or ev.get("selection_status"),
                "selection_reason": canonical.get("selection_reason") or ev.get("selection_reason"),
                "diagnostics": detail,
            }
        )

    return {
        "run_id": rid,
        "resolved_promotion_park": {"promotion_threshold": promote, "park_threshold": park},
        "count": len(items),
        "canonical_evaluation_rows": canonical_rows,
        "validations": items,
    }


def format_gate_diagnostics_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
