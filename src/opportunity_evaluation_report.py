"""Operator-facing comparison harness for canonical evaluation vs shadow scoring."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from src.opportunity_evaluation import v2_lite_shadow_score


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    bounded = max(0.0, min(1.0, quantile))
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * bounded))))
    return round(sorted(values)[index], 4)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.median(values)), 4)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_id": row["validation_id"],
        "finding_id": row["finding_id"],
        "run_id": row["run_id"],
        "title": row["title"],
        "decision": row["decision"],
        "selection_status": row["selection_status"],
        "selection_reason": row["selection_reason"],
        "decision_score": row["decision_score"],
        "shadow_score_v2_lite": row["shadow_score_v2_lite"],
        "shadow_origin": row["shadow_origin"],
        "query_overlap_ratio": row.get("query_overlap_ratio", 0.0),
        "matched_url_overlap_ratio": row.get("matched_url_overlap_ratio", 0.0),
        "comparison_sibling_finding_id": row.get("comparison_sibling_finding_id", 0),
        "comparison_scope": row.get("comparison_scope", ""),
        "query_origin_counts": row.get("query_origin_counts", {}),
        "attribution_scope_counts": row.get("attribution_scope_counts", {}),
    }


def _extract_atom_summary(evidence: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluation_inputs = evaluation.get("inputs", {}) or {}
    atom_summary = evaluation_inputs.get("atom", {}) or {}
    cluster = evidence.get("cluster", {}) or {}
    cluster_summary = cluster.get("summary", {}) or {}
    return {
        "segment": atom_summary.get("segment") or cluster_summary.get("segment", ""),
        "user_role": atom_summary.get("user_role") or cluster_summary.get("user_role", "") or cluster.get("user_role", ""),
        "job_to_be_done": atom_summary.get("job_to_be_done") or cluster_summary.get("job_to_be_done", "") or cluster.get("job_to_be_done", ""),
        "trigger_event": atom_summary.get("trigger_event", ""),
        "failure_mode": atom_summary.get("failure_mode", ""),
        "current_workaround": atom_summary.get("current_workaround", ""),
    }


def _extract_shadow_row(row: dict[str, Any]) -> dict[str, Any] | None:
    evidence = row.get("evidence") or {}
    evaluation = evidence.get("opportunity_evaluation") if isinstance(evidence.get("opportunity_evaluation"), dict) else {}
    policy = evaluation.get("policy", {}) or {}
    selection = evaluation.get("selection", {}) or {}
    evaluation_evidence = evaluation.get("evidence", {}) or {}
    measures = evaluation.get("measures", {}) or {}
    scores = measures.get("scores", {}) or {}
    legacy_scorecard = evidence.get("opportunity_scorecard", {}) or {}

    decision = str(policy.get("decision") or evidence.get("decision") or "park")
    if decision not in {"promote", "park", "kill"}:
        decision = "park"
    selection_status = str(selection.get("selection_status") or evidence.get("selection_status") or "")
    selection_reason = str(selection.get("selection_reason") or evidence.get("selection_reason") or "")

    title = (
        (evidence.get("summary") or {}).get("problem_statement")
        or (evidence.get("cluster") or {}).get("label")
        or f"validation-{row.get('validation_id')}"
    )

    shadow_block = evaluation.get("shadow", {}) or {}
    if shadow_block.get("shadow_score_v2_lite") is not None:
        shadow_score = float(shadow_block.get("shadow_score_v2_lite") or 0.0)
        shadow_diagnostics = dict(shadow_block.get("comparison_diagnostics", {}) or {})
        shadow_origin = "canonical_snapshot"
    else:
        evaluation_inputs = evaluation.get("inputs", {}) or {}
        corroboration = evidence.get("corroboration", {}) or evaluation_inputs.get("corroboration", {}) or {}
        market_enrichment = evidence.get("market_enrichment", {}) or evaluation_inputs.get("market_enrichment", {}) or {}
        has_inputs = bool(legacy_scorecard or corroboration or market_enrichment)
        if not has_inputs:
            return None
        shadow_score, shadow_diagnostics = v2_lite_shadow_score(
            atom_summary=_extract_atom_summary(evidence, evaluation),
            measures=legacy_scorecard,
            corroboration_inputs=corroboration,
            market_enrichment_inputs=market_enrichment,
        )
        shadow_origin = "recomputed_from_validation_evidence"

    decision_score = float(
        scores.get("decision_score", legacy_scorecard.get("decision_score", 0.0)) or 0.0
    )
    return {
        "validation_id": int(row.get("validation_id") or 0),
        "finding_id": int(row.get("finding_id") or 0),
        "run_id": str(row.get("run_id") or ""),
        "title": str(title),
        "decision": decision,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "decision_score": round(decision_score, 4),
        "shadow_score_v2_lite": round(float(shadow_score), 4),
        "shadow_origin": shadow_origin,
        "query_origin_counts": dict(evaluation_evidence.get("query_origin_counts", evidence.get("query_origin_counts", {}) or {})),
        "attribution_scope_counts": dict(evaluation_evidence.get("attribution_scope_counts", evidence.get("attribution_scope_counts", {}) or {})),
        "comparison_sibling_finding_id": int(evaluation_evidence.get("comparison_sibling_finding_id", evidence.get("comparison_sibling_finding_id", 0)) or 0),
        "comparison_scope": str(evaluation_evidence.get("comparison_scope", evidence.get("comparison_scope", "")) or ""),
        "query_overlap_ratio": float(evaluation_evidence.get("query_overlap_ratio", evidence.get("query_overlap_ratio", 0.0)) or 0.0),
        "matched_url_overlap_ratio": float(evaluation_evidence.get("matched_url_overlap_ratio", evidence.get("matched_url_overlap_ratio", 0.0)) or 0.0),
        "shadow_diagnostics": shadow_diagnostics,
    }


def _bucket_stats(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    bucketed: dict[str, list[float]] = {}
    for row in rows:
        bucket = str(row.get(key) or "unknown")
        bucketed.setdefault(bucket, []).append(float(row["shadow_score_v2_lite"]))
    return [
        {
            key: bucket,
            "count": len(values),
            "avg_shadow_score": _mean(values),
            "median_shadow_score": _median(values),
        }
        for bucket, values in sorted(bucketed.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def build_shadow_scoring_report(
    db: Any,
    *,
    run_id: str | None = None,
    limit: int = 250,
) -> dict[str, Any]:
    """Compare current stored decisions against shadow-only v2_lite scores."""
    scoped_run_id = (run_id or "").strip() or None
    rows = db.list_validation_evidence_payloads(
        run_id=scoped_run_id,
        limit=limit,
        all_runs=scoped_run_id is None,
    )

    extracted_rows: list[dict[str, Any]] = []
    unreconstructable = 0
    canonical_count = 0
    recomputed_count = 0
    for row in rows:
        extracted = _extract_shadow_row(row)
        if extracted is None:
            unreconstructable += 1
            continue
        extracted_rows.append(extracted)
        if extracted["shadow_origin"] == "canonical_snapshot":
            canonical_count += 1
        else:
            recomputed_count += 1

    scores = [float(row["shadow_score_v2_lite"]) for row in extracted_rows]
    high_cutoff = _percentile(scores, 0.8)
    low_cutoff = _percentile(scores, 0.2)
    sorted_desc = sorted(extracted_rows, key=lambda item: (-item["shadow_score_v2_lite"], item["validation_id"]))
    sorted_asc = sorted(extracted_rows, key=lambda item: (item["shadow_score_v2_lite"], item["validation_id"]))

    top_band = [row for row in extracted_rows if row["shadow_score_v2_lite"] >= high_cutoff]
    bottom_band = [row for row in extracted_rows if row["shadow_score_v2_lite"] <= low_cutoff]

    decision_mix = Counter(row["decision"] for row in extracted_rows)
    selection_mix = Counter(row["selection_status"] or "unknown" for row in extracted_rows)

    return {
        "scope": {
            "mode": "run" if scoped_run_id else "all_runs",
            "run_id": scoped_run_id,
            "limit": limit,
        },
        "coverage": {
            "rows_examined": len(rows),
            "rows_compared": len(extracted_rows),
            "canonical_snapshot_rows": canonical_count,
            "recomputed_shadow_rows": recomputed_count,
            "unreconstructable_rows": unreconstructable,
        },
        "current_state_mix": {
            "decision": dict(sorted(decision_mix.items(), key=lambda item: (-item[1], item[0]))),
            "selection_status": dict(sorted(selection_mix.items(), key=lambda item: (-item[1], item[0]))),
        },
        "shadow_score_summary": {
            "min": round(min(scores), 4) if scores else 0.0,
            "p20": low_cutoff,
            "median": _median(scores),
            "p80": high_cutoff,
            "p90": _percentile(scores, 0.9),
            "max": round(max(scores), 4) if scores else 0.0,
        },
        "shadow_by_decision": _bucket_stats(extracted_rows, "decision"),
        "shadow_by_selection_status": _bucket_stats(extracted_rows, "selection_status"),
        "agreement_signals": {
            "top_quintile_promote_rate": round(
                sum(1 for row in top_band if row["decision"] == "promote") / len(top_band),
                4,
            ) if top_band else 0.0,
            "top_quintile_prototype_candidate_rate": round(
                sum(1 for row in top_band if row["selection_status"] == "prototype_candidate") / len(top_band),
                4,
            ) if top_band else 0.0,
            "bottom_quintile_promote_rate": round(
                sum(1 for row in bottom_band if row["decision"] == "promote") / len(bottom_band),
                4,
            ) if bottom_band else 0.0,
            "bottom_quintile_prototype_candidate_rate": round(
                sum(1 for row in bottom_band if row["selection_status"] == "prototype_candidate") / len(bottom_band),
                4,
            ) if bottom_band else 0.0,
        },
        "top_shadow_rows": [_preview(row) for row in sorted_desc[: min(10, len(sorted_desc))]],
        "notable_disagreements": {
            "high_shadow_not_promoted": [
                _preview(row)
                for row in sorted_desc
                if row["shadow_score_v2_lite"] >= high_cutoff and row["decision"] != "promote"
            ][:5],
            "low_shadow_promoted": [
                _preview(row)
                for row in sorted_asc
                if row["shadow_score_v2_lite"] <= low_cutoff and row["decision"] == "promote"
            ][:5],
            "high_shadow_archived": [
                _preview(row)
                for row in sorted_desc
                if row["shadow_score_v2_lite"] >= high_cutoff and row["selection_status"] == "archive"
            ][:5],
            "low_shadow_prototype_candidate": [
                _preview(row)
                for row in sorted_asc
                if row["shadow_score_v2_lite"] <= low_cutoff and row["selection_status"] == "prototype_candidate"
            ][:5],
        },
    }
