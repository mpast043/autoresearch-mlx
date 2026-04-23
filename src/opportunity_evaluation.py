"""Canonical post-validation opportunity evaluation contract."""

from __future__ import annotations

from typing import Any


OPPORTUNITY_EVALUATION_SCHEMA_VERSION = "opportunity_evaluation_v1"
OPPORTUNITY_POLICY_VERSION = "stage_decision_v1"
V2_LITE_SHADOW_VERSION = "v2_lite_shadow_v1"


def _build_prep_route(*, decision: str, selection_status: str) -> tuple[bool, str]:
    if selection_status == "prototype_candidate":
        return True, "prototype_candidate"
    if decision == "promote" and selection_status == "research_more":
        return True, "spec_draft"
    return False, "none"


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _average_available(*values: Any) -> float:
    usable = []
    for value in values:
        if value is None:
            continue
        try:
            usable.append(float(value))
        except (TypeError, ValueError):
            continue
    if not usable:
        return 0.0
    return sum(usable) / len(usable)


def _normalized_count(value: Any, *, cap: float = 3.0) -> float:
    if cap <= 0:
        return 0.0
    return _clamp01((float(value or 0.0)) / cap)


def v2_lite_shadow_score(
    *,
    atom_summary: dict[str, Any],
    measures: dict[str, Any],
    corroboration_inputs: dict[str, Any],
    market_enrichment_inputs: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Compute the shadow-only v2_lite score from existing stored fields.

    The scorer deliberately stays narrow: it uses only current scorecard fields
    and safe proxies so we can compare ranking quality before any policy cutover.
    """
    pain_severity = _clamp01(measures.get("pain_severity", 0.0))
    frequency_score = _clamp01(measures.get("frequency_score", 0.0))
    urgency_score = _clamp01(measures.get("urgency_score", 0.0))
    reachability = _clamp01(measures.get("reachability", 0.0))
    buildability = _clamp01(measures.get("buildability", 0.0))
    expansion_potential = _clamp01(measures.get("expansion_potential", 0.0))
    evidence_quality = _clamp01(measures.get("evidence_quality", 0.0))
    segment_concentration = _clamp01(measures.get("segment_concentration", 0.0))
    dependency_risk = _clamp01(measures.get("dependency_risk", 0.0))
    adoption_friction = _clamp01(measures.get("adoption_friction", 0.0))
    willingness_to_pay_proxy = _clamp01(measures.get("willingness_to_pay_proxy", 0.0))
    corroboration_strength = _clamp01(
        measures.get("corroboration_strength", corroboration_inputs.get("corroboration_score", 0.0))
    )

    buyer_intent_score = _clamp01(market_enrichment_inputs.get("buyer_intent_score", 0.0))
    operational_buyer_score = _clamp01(market_enrichment_inputs.get("operational_buyer_score", 0.0))
    competition_score = _clamp01(market_enrichment_inputs.get("competition_score", 0.0))

    family_confirmation_score = _normalized_count(
        corroboration_inputs.get("family_confirmation_count", 0),
        cap=3.0,
    )
    source_family_diversity = _normalized_count(
        corroboration_inputs.get("source_family_diversity", 0),
        cap=3.0,
    )

    buyer_value_proxy = _clamp01(
        _average_available(willingness_to_pay_proxy, buyer_intent_score)
    )
    corroboration_proxy = _clamp01(
        _average_available(family_confirmation_score, corroboration_strength)
    )
    buyer_clarity_proxy = _clamp01(
        max(1.0 if str(atom_summary.get("user_role", "") or "").strip() else 0.0, operational_buyer_score)
    )
    target_clarity_proxy = segment_concentration
    competition_headroom = _clamp01(1.0 - competition_score)

    components = {
        "pain_severity": pain_severity,
        "frequency_score": frequency_score,
        "buyer_value_proxy": buyer_value_proxy,
        "urgency_score": urgency_score,
        "reachability": reachability,
        "buildability": buildability,
        "expansion_potential": expansion_potential,
        "competition_headroom": competition_headroom,
        "evidence_quality": evidence_quality,
        "corroboration_proxy": corroboration_proxy,
        "source_family_diversity": source_family_diversity,
        "buyer_clarity_proxy": buyer_clarity_proxy,
        "target_clarity_proxy": target_clarity_proxy,
    }
    weights = {
        "pain_severity": 0.15,
        "frequency_score": 0.12,
        "buyer_value_proxy": 0.12,
        "urgency_score": 0.08,
        "reachability": 0.08,
        "buildability": 0.08,
        "expansion_potential": 0.06,
        "competition_headroom": 0.08,
        "evidence_quality": 0.12,
        "corroboration_proxy": 0.07,
        "source_family_diversity": 0.02,
        "buyer_clarity_proxy": 0.01,
        "target_clarity_proxy": 0.01,
    }
    penalties = {
        "dependency_risk": dependency_risk * 0.05,
        "adoption_friction": adoption_friction * 0.05,
    }

    base_score = sum(components[name] * weight for name, weight in weights.items())
    penalty_total = sum(penalties.values())
    shadow_score = round(_clamp01(base_score - penalty_total), 4)

    diagnostics = {
        "formula_version": V2_LITE_SHADOW_VERSION,
        "components": {name: round(value, 4) for name, value in components.items()},
        "weights": weights,
        "penalties": {name: round(value, 4) for name, value in penalties.items()},
        "base_score": round(base_score, 4),
        "penalty_total": round(penalty_total, 4),
        "net_score": shadow_score,
    }
    return shadow_score, diagnostics


def build_opportunity_evaluation(
    *,
    run_id: str,
    finding_id: int,
    cluster_id: int,
    opportunity_id: int | None,
    validation_id: int | None,
    source_finding_kind: str,
    atom_summary: dict[str, Any],
    validation_inputs: dict[str, Any],
    corroboration_inputs: dict[str, Any],
    market_enrichment_inputs: dict[str, Any],
    review_feedback_inputs: dict[str, Any],
    measures: dict[str, Any],
    evidence: dict[str, Any],
    decision: str,
    decision_reason: str,
    promotion_threshold: float,
    park_threshold: float,
    selection_status: str,
    selection_reason: str,
    selection_checks: dict[str, Any],
    shadow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical evaluation snapshot from current validation outputs.

    This contract is intentionally strict about ownership boundaries so the repo
    can converge on one evaluation shape instead of accumulating more blobs.
    """
    build_prep_eligible, build_prep_route = _build_prep_route(
        decision=decision,
        selection_status=selection_status,
    )
    supported_counterevidence = sum(
        1 for item in (evidence.get("counterevidence", []) or []) if item.get("status") == "supported"
    )
    shadow_score_v2_lite, comparison_diagnostics = v2_lite_shadow_score(
        atom_summary=atom_summary,
        measures=measures,
        corroboration_inputs=corroboration_inputs,
        market_enrichment_inputs=market_enrichment_inputs,
    )
    if shadow:
        if shadow.get("shadow_score_v2_lite") is not None:
            shadow_score_v2_lite = float(shadow.get("shadow_score_v2_lite") or 0.0)
        if shadow.get("comparison_diagnostics"):
            comparison_diagnostics = dict(shadow.get("comparison_diagnostics") or {})
    return {
        "schema_version": OPPORTUNITY_EVALUATION_SCHEMA_VERSION,
        "inputs": {
            "ids": {
                "run_id": run_id,
                "finding_id": finding_id,
                "cluster_id": cluster_id,
                "opportunity_id": opportunity_id,
                "validation_id": validation_id,
            },
            "source": {
                "finding_kind": source_finding_kind,
            },
            "atom": {
                "segment": atom_summary.get("segment", ""),
                "user_role": atom_summary.get("user_role", ""),
                "job_to_be_done": atom_summary.get("job_to_be_done", ""),
                "trigger_event": atom_summary.get("trigger_event", ""),
                "failure_mode": atom_summary.get("failure_mode", ""),
                "current_workaround": atom_summary.get("current_workaround", ""),
            },
            "validation": validation_inputs,
            "corroboration": corroboration_inputs,
            "market_enrichment": market_enrichment_inputs,
            "review_feedback": review_feedback_inputs,
        },
        "measures": {
            "scores": {
                "decision_score": float(measures.get("decision_score", 0.0) or 0.0),
                "problem_truth_score": float(measures.get("problem_truth_score", 0.0) or 0.0),
                "revenue_readiness_score": float(measures.get("revenue_readiness_score", 0.0) or 0.0),
            },
            "dimensions": {
                "pain_severity": float(measures.get("pain_severity", 0.0) or 0.0),
                "frequency_score": float(measures.get("frequency_score", 0.0) or 0.0),
                "urgency_score": float(measures.get("urgency_score", 0.0) or 0.0),
                "cost_of_inaction": float(measures.get("cost_of_inaction", 0.0) or 0.0),
                "workaround_density": float(measures.get("workaround_density", 0.0) or 0.0),
                "reachability": float(measures.get("reachability", 0.0) or 0.0),
                "buildability": float(measures.get("buildability", 0.0) or 0.0),
                "expansion_potential": float(measures.get("expansion_potential", 0.0) or 0.0),
                "segment_concentration": float(measures.get("segment_concentration", 0.0) or 0.0),
                "dependency_risk": float(measures.get("dependency_risk", 0.0) or 0.0),
                "adoption_friction": float(measures.get("adoption_friction", 0.0) or 0.0),
                "value_support": float(measures.get("value_support", 0.0) or 0.0),
                "willingness_to_pay_proxy": float(measures.get("willingness_to_pay_proxy", 0.0) or 0.0),
                "evidence_quality": float(measures.get("evidence_quality", 0.0) or 0.0),
                "corroboration_strength": float(measures.get("corroboration_strength", 0.0) or 0.0),
            },
            "transition": {
                "composite_score": float(measures.get("composite_score", 0.0) or 0.0),
                "problem_plausibility": float(measures.get("problem_plausibility", 0.0) or 0.0),
                "evidence_sufficiency": float(measures.get("evidence_sufficiency", 0.0) or 0.0),
            },
        },
        "evidence": {
            "market_gap_state": evidence.get("market_gap_state", "unknown"),
            "recurrence_state": evidence.get("recurrence_state", ""),
            "family_confirmation_count": int(evidence.get("family_confirmation_count", 0) or 0),
            "source_family_diversity": int(corroboration_inputs.get("source_family_diversity", 0) or 0),
            "cross_source_match_score": float(corroboration_inputs.get("cross_source_match_score", 0.0) or 0.0),
            "generalizability_class": str(corroboration_inputs.get("generalizability_class", "") or ""),
            "generalizability_score": float(corroboration_inputs.get("generalizability_score", 0.0) or 0.0),
            "counterevidence": list(evidence.get("counterevidence", []) or []),
            "validation_plan": dict(evidence.get("validation_plan", {}) or {}),
        },
        "policy": {
            "decision": decision,
            "decision_reason": decision_reason,
            "policy_version": OPPORTUNITY_POLICY_VERSION,
            "policy_checks": {
                "promotion_threshold": float(promotion_threshold),
                "park_threshold": float(park_threshold),
                "decision_score": float(measures.get("decision_score", 0.0) or 0.0),
                "problem_truth_score": float(measures.get("problem_truth_score", 0.0) or 0.0),
                "revenue_readiness_score": float(measures.get("revenue_readiness_score", 0.0) or 0.0),
                "frequency_score": float(measures.get("frequency_score", 0.0) or 0.0),
                "supported_counterevidence_count": supported_counterevidence,
                "market_gap_state": evidence.get("market_gap_state", "unknown"),
            },
        },
        "selection": {
            "selection_status": selection_status,
            "selection_reason": selection_reason,
            "selection_checks": dict(selection_checks or {}),
            "build_prep_eligible": build_prep_eligible,
            "build_prep_route": build_prep_route,
        },
        "shadow": {
            "shadow_score_v2_lite": shadow_score_v2_lite,
            "comparison_diagnostics": comparison_diagnostics,
        },
    }
