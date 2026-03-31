"""Operator-facing gate diagnostics (why promote/park/kill and selection_status)."""

from __future__ import annotations

import json
from typing import Any

from src.build_prep import explain_selection_gate_detail
from src.opportunity_engine import diagnose_stage_decision
from src.validation_thresholds import resolve_promotion_park_thresholds


def _review_feedback_for_engine(evidence: dict[str, Any]) -> dict[str, Any]:
    """Map validation evidence.review_feedback into stage_decision review_feedback shape."""
    rf = evidence.get("review_feedback") or {}
    return {
        "park_bias": float(rf.get("review_feedback_park_bias", 0.0) or 0.0),
        "kill_bias": float(rf.get("review_feedback_kill_bias", 0.0) or 0.0),
    }


def explain_validation_evidence(evidence: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    """Full gate breakdown for one persisted validation ``evidence`` JSON blob."""
    cfg = config or {}
    promote, park = resolve_promotion_park_thresholds(cfg)

    scorecard = evidence.get("opportunity_scorecard") or {}
    market_gap = evidence.get("market_gap") or {}
    counterevidence = evidence.get("counterevidence") or []
    review_fb = _review_feedback_for_engine(evidence)

    stage_diag = diagnose_stage_decision(
        scorecard,
        market_gap,
        counterevidence,
        promotion_threshold=promote,
        park_threshold=park,
        review_feedback=review_fb,
    )

    raw_decision = str(evidence.get("decision") or "park")
    if raw_decision not in {"promote", "park", "kill"}:
        raw_decision = "park"

    corroboration = evidence.get("corroboration") or {}
    market_enrichment = evidence.get("market_enrichment") or {}
    selection_diag = explain_selection_gate_detail(
        decision=raw_decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
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
    for row in rows[:limit]:
        ev = row.get("evidence") or {}
        detail = explain_validation_evidence(ev, cfg)
        items.append(
            {
                "validation_id": row.get("validation_id"),
                "finding_id": row.get("finding_id"),
                "title": (ev.get("summary") or {}).get("problem_statement")
                or ev.get("finding_kind", ""),
                "decision": ev.get("decision"),
                "selection_status": ev.get("selection_status"),
                "selection_reason": ev.get("selection_reason"),
                "diagnostics": detail,
            }
        )

    return {
        "run_id": rid,
        "resolved_promotion_park": {"promotion_threshold": promote, "park_threshold": park},
        "count": len(items),
        "validations": items,
    }


def format_gate_diagnostics_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
