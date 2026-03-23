"""Read-model helpers for validation review, workbench, and digest projections."""

from __future__ import annotations

from typing import Any


def sorted_recurrence_match_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    normalized: list[dict[str, Any]] = [record for record in records if isinstance(record, dict)]
    return sorted(
        normalized,
        key=lambda record: (
            str(record.get("normalized_url", "") or ""),
            str(record.get("title", "") or ""),
            str(record.get("query_text", "") or ""),
            str(record.get("source", "") or ""),
        ),
    )


def reviewable_recurrence_matches_by_source(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    strong_by_source = evidence.get("matched_docs_by_source", {}) or {}
    partial_by_source = evidence.get("partial_docs_by_source", {}) or {}
    labels = sorted({*strong_by_source.keys(), *partial_by_source.keys()})
    reviewable: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        strong_records = sorted_recurrence_match_records(strong_by_source.get(label, []))
        partial_records = sorted_recurrence_match_records(partial_by_source.get(label, []))
        if strong_records or partial_records:
            reviewable[label] = [*strong_records, *partial_records]
    return reviewable


def build_recent_validation_row(row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    matched_docs_by_source = {
        source: sorted_recurrence_match_records(records)
        for source, records in (evidence.get("matched_docs_by_source", {}) or {}).items()
    }
    partial_docs_by_source = {
        source: sorted_recurrence_match_records(records)
        for source, records in (evidence.get("partial_docs_by_source", {}) or {}).items()
    }
    return {
        **row,
        "decision": evidence.get("decision", "park"),
        "passed": bool(row["passed"]),
        "matched_docs_by_source": matched_docs_by_source,
        "partial_docs_by_source": partial_docs_by_source,
        "reviewable_recurrence_matches_by_source": reviewable_recurrence_matches_by_source(
            {
                "matched_docs_by_source": matched_docs_by_source,
                "partial_docs_by_source": partial_docs_by_source,
            }
        ),
        "matched_results_by_source": evidence.get("matched_results_by_source", {}),
        "partial_results_by_source": evidence.get("partial_results_by_source", {}),
        "source_yield": evidence.get("source_yield", {}),
    }


def build_validation_review_row(
    row: dict[str, Any],
    *,
    evidence: dict[str, Any],
    finding_evidence: dict[str, Any],
    corroboration_evidence: dict[str, Any] | None = None,
    corroboration_row: dict[str, Any] | None = None,
    market_evidence: dict[str, Any] | None = None,
    market_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matched_docs_by_source = {
        source: sorted_recurrence_match_records(records)
        for source, records in (evidence.get("matched_docs_by_source", {}) or {}).items()
    }
    partial_docs_by_source = {
        source: sorted_recurrence_match_records(records)
        for source, records in (evidence.get("partial_docs_by_source", {}) or {}).items()
    }
    result = {
        "validation_id": row["id"],
        "run_id": row["run_id"] or "",
        "finding_id": row["finding_id"],
        "title": finding_evidence.get("title") or row["outcome_summary"] or row["product_built"] or row["source_url"],
        "decision": evidence.get("decision"),
        "decision_reason": evidence.get("decision_reason"),
        "selection_status": evidence.get("selection_status", "research_more"),
        "selection_reason": evidence.get("selection_reason", ""),
        "park_subreason": evidence.get("park_subreason"),
        "composite_score": evidence.get("composite_score", row["overall_score"] or 0.0),
        "overall_score": row["overall_score"] or 0.0,
        "market_score": row["market_score"] or 0.0,
        "technical_score": row["technical_score"] or 0.0,
        "distribution_score": row["distribution_score"] or 0.0,
        "recurrence_state": evidence.get("recurrence_state"),
        "corroboration_score": 0.0,
        "value_support": evidence.get(
            "value_support",
            evidence.get("evidence_assessment", {}).get(
                "value_support",
                evidence.get("opportunity_scorecard", {}).get("value_support", 0.0),
            ),
        ),
        "review_feedback_count": evidence.get("review_feedback_count", 0),
        "source": row["source"],
        "source_url": row["source_url"],
        "source_class": row["source_class"],
        "validated_at": row["validated_at"],
        "recurrence_timeout": evidence.get("recurrence_timeout", False),
        "competitor_timeout": evidence.get("competitor_timeout", False),
        "recurrence_gap_reason": evidence.get("recurrence_gap_reason", ""),
        "recurrence_failure_class": evidence.get("recurrence_failure_class", ""),
        "queries_executed": evidence.get("queries_executed", []),
        "recurrence_budget_profile": evidence.get("recurrence_budget_profile", {}),
        "candidate_meaningful": evidence.get("candidate_meaningful", {}),
        "recurrence_probe_summary": evidence.get("recurrence_probe_summary", {}),
        "recurrence_source_branch": evidence.get("recurrence_source_branch", {}),
        "last_action": evidence.get("last_action", ""),
        "last_transition_reason": evidence.get("last_transition_reason", ""),
        "chosen_family": evidence.get("chosen_family", ""),
        "expected_gain_class": evidence.get("expected_gain_class", ""),
        "source_attempts_snapshot": evidence.get("source_attempts_snapshot", {}),
        "skipped_families": evidence.get("skipped_families", {}),
        "controller_actions": evidence.get("controller_actions", []),
        "budget_snapshot": evidence.get("budget_snapshot", {}),
        "fallback_strategy_used": evidence.get("fallback_strategy_used", ""),
        "decomposed_atom_queries": evidence.get("decomposed_atom_queries", []),
        "routing_override_reason": evidence.get("routing_override_reason", ""),
        "cohort_query_pack_used": evidence.get("cohort_query_pack_used", False),
        "cohort_query_pack_name": evidence.get("cohort_query_pack_name", ""),
        "web_query_strategy_path": evidence.get("web_query_strategy_path", []),
        "specialized_surface_targeting_used": evidence.get("specialized_surface_targeting_used", False),
        "promotion_gap_class": evidence.get("promotion_gap_class", ""),
        "near_miss_enrichment_action": evidence.get("near_miss_enrichment_action", ""),
        "sufficiency_priority_reason": evidence.get("sufficiency_priority_reason", ""),
        "value_enrichment_used": evidence.get("value_enrichment_used", False),
        "value_enrichment_queries": evidence.get("value_enrichment_queries", []),
        "matched_results_by_source": evidence.get("matched_results_by_source", {}),
        "partial_results_by_source": evidence.get("partial_results_by_source", {}),
        "matched_docs_by_source": matched_docs_by_source,
        "partial_docs_by_source": partial_docs_by_source,
        "reviewable_recurrence_matches_by_source": reviewable_recurrence_matches_by_source(
            {
                "matched_docs_by_source": matched_docs_by_source,
                "partial_docs_by_source": partial_docs_by_source,
            }
        ),
        "family_confirmation_count": evidence.get("family_confirmation_count", 0),
        "source_yield": evidence.get("source_yield", {}),
        "reshaped_query_history": evidence.get("reshaped_query_history", []),
    }
    if corroboration_row:
        corr_evidence = corroboration_evidence or {}
        result["recurrence_state"] = corroboration_row["recurrence_state"]
        result["corroboration_score"] = corroboration_row["corroboration_score"] or 0.0
        result.update(
            {
                "cross_source_match_score": corr_evidence.get("cross_source_match_score", 0.0),
                "source_families": corr_evidence.get("source_families", []),
                "source_family_diversity": corr_evidence.get("source_family_diversity", 0),
                "core_source_family_diversity": corr_evidence.get("core_source_family_diversity", 0),
                "generalizability_class": corr_evidence.get("generalizability_class", ""),
            }
        )
    if market_row:
        mkt_evidence = market_evidence or {}
        result.update(
            {
                "demand_score": market_row["demand_score"] or 0.0,
                "buyer_intent_score": market_row["buyer_intent_score"] or 0.0,
                "competition_score": market_row["competition_score"] or 0.0,
                "trend_score": market_row["trend_score"] or 0.0,
                "review_signal_score": market_row["review_signal_score"] or 0.0,
                "value_signal_score": market_row["value_signal_score"] or 0.0,
                "wedge_name": mkt_evidence.get("wedge_name", ""),
                "wedge_active": mkt_evidence.get("wedge_active", False),
                "wedge_fit_score": mkt_evidence.get("wedge_fit_score", 0.0),
            }
        )
    return result


def best_surfaced_evidence(row: dict[str, Any]) -> dict[str, Any]:
    reviewable = row.get("reviewable_recurrence_matches_by_source", {}) or {}
    candidates: list[dict[str, Any]] = []
    for source, records in reviewable.items():
        for record in records:
            candidates.append(
                {
                    "source_family": record.get("source_family") or source,
                    "source": record.get("source") or source,
                    "query_text": record.get("query_text", ""),
                    "normalized_url": record.get("normalized_url", ""),
                    "title": record.get("title", ""),
                    "snippet": record.get("snippet", ""),
                    "match_class": record.get("match_class", ""),
                }
            )
    if candidates:
        return candidates[0]
    return {
        "source_family": row.get("source_class") or row.get("source", ""),
        "source": row.get("source", ""),
        "query_text": "",
        "normalized_url": row.get("source_url", ""),
        "title": row.get("title", ""),
        "snippet": "",
        "match_class": "origin",
    }


def confidence_posture(row: dict[str, Any], brief_payload: dict[str, Any]) -> str:
    prototype_gate = brief_payload.get("prototype_gate", {}) or {}
    confidence = str(prototype_gate.get("market_confidence_level", "") or "")
    if confidence:
        return confidence
    selection_reason = str(row.get("selection_reason", "") or "")
    if selection_reason == "validated_selection_gate":
        return "market_confirmed"
    if selection_reason == "prototype_candidate_gate":
        return "prototype_checkpoint"
    if row.get("decision") == "kill" or row.get("decision_reason") == "unlikely_or_economically_weak":
        return "discarded"
    return "evidence_incomplete"


def repeatability_posture(row: dict[str, Any]) -> str:
    recurrence_state = str(row.get("recurrence_state", "") or "")
    failure_class = str(row.get("recurrence_failure_class", "") or "")
    family_confirmation_count = int(row.get("family_confirmation_count", 0) or 0)
    matched_by_source = row.get("matched_results_by_source", {}) or {}
    partial_by_source = row.get("partial_results_by_source", {}) or {}
    web_contributed = bool(
        int(matched_by_source.get("web", 0) or 0) > 0
        or int(partial_by_source.get("web", 0) or 0) > 0
    )
    if family_confirmation_count >= 2 and recurrence_state in {"supported", "strong"}:
        return "multi_family_repeatable"
    if family_confirmation_count >= 2 or web_contributed:
        return "cross_family_signal"
    if failure_class == "budget_exhausted" or recurrence_state == "timeout":
        return "budget_limited"
    if failure_class == "single_source_only" or family_confirmation_count <= 1:
        return "single_family_volatile"
    if recurrence_state in {"thin", "weak"}:
        return "thin_repeatability"
    return "unclear"


def next_recommended_action(row: dict[str, Any], build_brief_present: bool) -> str:
    selection_status = str(row.get("selection_status", "") or "")
    decision = str(row.get("decision", "") or "")
    decision_reason = str(row.get("decision_reason", "") or "")
    recurrence_state = str(row.get("recurrence_state", "") or "")
    failure_class = str(row.get("recurrence_failure_class", "") or "")
    if selection_status in {"prototype_candidate", "prototype_ready", "build_ready"} or build_brief_present:
        return "prototype_now"
    if selection_status == "archive" or decision == "kill" or decision_reason == "unlikely_or_economically_weak":
        return "archive"
    if recurrence_state == "timeout" or failure_class == "budget_exhausted":
        return "watchlist"
    return "gather_more_evidence"


def build_candidate_workbench_item(row: dict[str, Any], brief: Any | None) -> dict[str, Any]:
    brief_payload = brief.brief if brief else {}
    build_brief_present = brief is not None
    return {
        "validation_id": row.get("validation_id"),
        "finding_id": row.get("finding_id"),
        "title": row.get("title", ""),
        "state": row.get("selection_status") or row.get("decision") or "research_more",
        "decision": row.get("decision", ""),
        "decision_reason": row.get("decision_reason", ""),
        "selection_status": row.get("selection_status", "research_more"),
        "selection_reason": row.get("selection_reason", ""),
        "confidence_posture": confidence_posture(row, brief_payload),
        "repeatability_posture": repeatability_posture(row),
        "family_confirmation_count": int(row.get("family_confirmation_count", 0) or 0),
        "best_surfaced_evidence": best_surfaced_evidence(row),
        "build_brief_present": build_brief_present,
        "build_brief_id": brief.id if brief else 0,
        "recommended_output_type": (
            brief.recommended_output_type
            if brief
            else (brief_payload.get("recommended_narrow_output_type", "") or "")
        ),
        "next_recommended_action": next_recommended_action(row, build_brief_present),
    }


def build_validation_corroboration_digest(row: dict[str, Any]) -> dict[str, Any]:
    reviewable_web_matches = list((row.get("reviewable_recurrence_matches_by_source", {}) or {}).get("web", []))
    web_yield = dict((row.get("source_yield", {}) or {}).get("web", {}) or {})
    return {
        "validation_id": row.get("validation_id"),
        "finding_id": row.get("finding_id"),
        "title": row.get("title", ""),
        "decision": row.get("decision", ""),
        "decision_reason": row.get("decision_reason", ""),
        "recurrence_state": row.get("recurrence_state"),
        "family_confirmation_count": row.get("family_confirmation_count", 0),
        "web_contributed": bool(
            int((row.get("matched_results_by_source", {}) or {}).get("web", 0) or 0) > 0
            or int((row.get("partial_results_by_source", {}) or {}).get("web", 0) or 0) > 0
            or int(web_yield.get("docs_retrieved", 0) or 0) > 0
        ),
        "web_source_yield": web_yield,
        "web_matched_count": int((row.get("matched_results_by_source", {}) or {}).get("web", 0) or 0),
        "web_partial_count": int((row.get("partial_results_by_source", {}) or {}).get("web", 0) or 0),
        "web_matches": reviewable_web_matches,
    }
