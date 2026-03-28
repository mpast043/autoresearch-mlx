"""Selection-state and build-prep helpers for post-validation workflows."""

from __future__ import annotations

from typing import Any

BUILD_BRIEF_SCHEMA_VERSION = "build_brief_v1"
BUILD_PREP_RULE_VERSION = "build_prep_v1"
PROTOTYPE_CANDIDATE_RULE_VERSION = "prototype_candidate_v1"

SELECTION_STATES = {
    "research_more",
    "prototype_candidate",
    "prototype_ready",
    "build_ready",
    "launched",
    "iterate",
    "expand",
    "archive",
}

ALLOWED_SELECTION_TRANSITIONS: dict[str, set[str]] = {
    "research_more": {"prototype_candidate", "archive"},
    "prototype_candidate": {"prototype_ready", "research_more", "archive"},
    "prototype_ready": {"build_ready", "research_more", "archive"},
    "build_ready": {"launched", "iterate", "archive"},
    "launched": {"iterate", "expand", "archive"},
    "iterate": {"build_ready", "expand", "archive"},
    "expand": {"iterate", "archive"},
    "archive": set(),
}


def is_allowed_selection_transition(current: str, target: str) -> bool:
    if current == target:
        return True
    return target in ALLOWED_SELECTION_TRANSITIONS.get(current, set())


def determine_selection_state(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Map validation output into the first build-prep lifecycle state."""
    if decision == "kill":
        return (
            "archive",
            "validation_kill",
            {
                "eligible": False,
                "gate_version": BUILD_PREP_RULE_VERSION,
                "reasons": ["validation_recommended_kill"],
                "blocked_by": ["decision_kill"],
            },
        )

    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    core_family_diversity = int(corroboration.get("core_source_family_diversity", 0) or 0)
    generalizability_class = str(corroboration.get("generalizability_class", "") or "")
    recurrence_state = str(corroboration.get("recurrence_state", "") or "")
    evidence_quality = float(scorecard.get("evidence_quality", 0.0) or 0.0)
    value_support = float(scorecard.get("value_support", 0.0) or 0.0)
    composite_score = float(scorecard.get("composite_score", 0.0) or 0.0)
    wedge_active = bool(market_enrichment.get("wedge_active"))

    reasons: list[str] = []
    blocked_by: list[str] = []

    if generalizability_class == "reusable_workflow_pain":
        reasons.append("generalizable_workflow_pain")
    else:
        blocked_by.append("not_generalizable_enough")

    if core_family_diversity >= 2:
        reasons.append("multi_family_support")
    else:
        blocked_by.append("single_family_support")

    if recurrence_state == "timeout":
        blocked_by.append("recurrence_timeout")
    elif recurrence_state in {"supported", "strong"}:
        reasons.append("recurrence_state_supported")

    if corroboration_score >= 0.6:
        reasons.append("corroboration_threshold_met")
    else:
        blocked_by.append("corroboration_below_threshold")

    if value_support >= 0.55:
        reasons.append("value_support_threshold_met")
    else:
        blocked_by.append("value_support_below_threshold")

    if evidence_quality >= 0.6:
        reasons.append("evidence_quality_threshold_met")
    else:
        blocked_by.append("evidence_quality_below_threshold")

    if composite_score >= 0.5:
        reasons.append("composite_threshold_met")
    else:
        blocked_by.append("composite_below_threshold")

    if wedge_active:
        reasons.append("wedge_active")

    eligible = (
        generalizability_class == "reusable_workflow_pain"
        and core_family_diversity >= 2
        and corroboration_score >= 0.6
        and value_support >= 0.55
        and evidence_quality >= 0.6
        and composite_score >= 0.5
    )

    # Timeout can happen in constrained environments (e.g. bridge/auth failures or strict budgets),
    # but we only treat timeout as exploratory-eligible under stronger evidence conditions.
    exploratory_recurrence_ok = recurrence_state in {"thin", "supported", "strong"}
    timeout_checkpoint_candidate = (
        recurrence_state == "timeout"
        and core_family_diversity >= 2
        and corroboration_score >= 0.45
        and value_support >= 0.6
        and evidence_quality >= 0.5
        and composite_score >= 0.4
    )

    exploratory_candidate = (
        generalizability_class == "reusable_workflow_pain"
        and (
            (
                core_family_diversity >= 2
                and (exploratory_recurrence_ok or timeout_checkpoint_candidate)
                and corroboration_score >= 0.25
                and value_support >= 0.55
                and evidence_quality >= 0.45
                and composite_score >= 0.34
            )
            or (
                core_family_diversity == 1
                and exploratory_recurrence_ok
                and corroboration_score >= 0.3
                and value_support >= 0.5
                and evidence_quality >= 0.49
                and composite_score >= 0.39
            )
        )
    )

    if eligible:
        return (
            "prototype_candidate",
            "validated_selection_gate",
            {
                "eligible": True,
                "gate_version": BUILD_PREP_RULE_VERSION,
                "reasons": reasons,
                "blocked_by": blocked_by,
            },
        )

    if exploratory_candidate:
        exploratory_reasons = list(dict.fromkeys(reasons))
        if core_family_diversity >= 2:
            exploratory_reasons.append("prototype_candidate_multifamily_near_miss")
        else:
            exploratory_reasons.append("prototype_candidate_single_family_exception")
        return (
            "prototype_candidate",
            "prototype_candidate_gate",
            {
                "eligible": True,
                "gate_version": PROTOTYPE_CANDIDATE_RULE_VERSION,
                "reasons": exploratory_reasons,
                "blocked_by": [],
            },
        )

    return (
        "research_more",
        "selection_gate_not_met",
        {
            "eligible": False,
            "gate_version": BUILD_PREP_RULE_VERSION,
            "reasons": reasons,
            "blocked_by": blocked_by,
        },
    )


def explain_selection_gate_detail(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
) -> dict[str, Any]:
    """Structured mirror of ``determine_selection_state`` thresholds for operator debugging."""
    status, reason, gate = determine_selection_state(
        decision=decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
    )

    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    core_family_diversity = int(corroboration.get("core_source_family_diversity", 0) or 0)
    generalizability_class = str(corroboration.get("generalizability_class", "") or "")
    recurrence_state = str(corroboration.get("recurrence_state", "") or "")
    evidence_quality = float(scorecard.get("evidence_quality", 0.0) or 0.0)
    value_support = float(scorecard.get("value_support", 0.0) or 0.0)
    composite_score = float(scorecard.get("composite_score", 0.0) or 0.0)
    wedge_active = bool(market_enrichment.get("wedge_active"))

    exploratory_recurrence_ok = recurrence_state in {"thin", "supported", "strong"}
    timeout_checkpoint_candidate = (
        recurrence_state == "timeout"
        and core_family_diversity >= 2
        and corroboration_score >= 0.45
        and value_support >= 0.6
        and evidence_quality >= 0.5
        and composite_score >= 0.4
    )

    strict_checks: list[dict[str, Any]] = [
        {
            "id": "generalizability_class",
            "pass": generalizability_class == "reusable_workflow_pain",
            "actual": generalizability_class,
            "need": "reusable_workflow_pain",
        },
        {
            "id": "core_source_family_diversity",
            "pass": core_family_diversity >= 2,
            "actual": core_family_diversity,
            "need": ">= 2",
        },
        {
            "id": "corroboration_score",
            "pass": corroboration_score >= 0.6,
            "actual": round(corroboration_score, 4),
            "need": ">= 0.6",
        },
        {
            "id": "value_support",
            "pass": value_support >= 0.55,
            "actual": round(value_support, 4),
            "need": ">= 0.55",
        },
        {
            "id": "evidence_quality",
            "pass": evidence_quality >= 0.6,
            "actual": round(evidence_quality, 4),
            "need": ">= 0.6",
        },
        {
            "id": "composite_score",
            "pass": composite_score >= 0.5,
            "actual": round(composite_score, 4),
            "need": ">= 0.5",
        },
    ]

    multifamily_explore = (
        core_family_diversity >= 2
        and (exploratory_recurrence_ok or timeout_checkpoint_candidate)
        and corroboration_score >= 0.25
        and value_support >= 0.55
        and evidence_quality >= 0.45
        and composite_score >= 0.34
    )
    single_family_explore = (
        core_family_diversity == 1
        and exploratory_recurrence_ok
        and corroboration_score >= 0.3
        and value_support >= 0.5
        and evidence_quality >= 0.49
        and composite_score >= 0.39
    )

    exploratory_checks: list[dict[str, Any]] = [
        {
            "id": "exploratory_base_generalizability",
            "pass": generalizability_class == "reusable_workflow_pain",
            "actual": generalizability_class,
            "need": "reusable_workflow_pain",
        },
        {
            "id": "exploratory_multifamily_branch",
            "pass": multifamily_explore,
            "detail": {
                "recurrence_ok_or_timeout_checkpoint": exploratory_recurrence_ok or timeout_checkpoint_candidate,
                "timeout_checkpoint_candidate": timeout_checkpoint_candidate,
                "corroboration_floor": 0.25,
                "value_support_floor": 0.55,
                "evidence_quality_floor": 0.45,
                "composite_floor": 0.34,
            },
        },
        {
            "id": "exploratory_single_family_branch",
            "pass": single_family_explore,
            "detail": {
                "requires_recurrence_ok_not_timeout_only": exploratory_recurrence_ok,
                "corroboration_floor": 0.3,
                "value_support_floor": 0.5,
                "evidence_quality_floor": 0.49,
                "composite_floor": 0.39,
            },
        },
    ]

    return {
        "resolved_selection_status": status,
        "resolved_selection_reason": reason,
        "selection_gate": gate,
        "recurrence_state": recurrence_state,
        "strict_path_checks": strict_checks,
        "exploratory_path_checks": exploratory_checks,
        "wedge_active": wedge_active,
        "hints": [
            "If decision is park/kill, prototype_candidate is never selected regardless of corroboration.",
            "timeout + multi-family can still reach prototype_candidate via timeout_checkpoint_candidate floors.",
            "Strict numeric gates ignore recurrence_state; timeout still adds blocked_by labels in the gate payload.",
        ],
    }


def determine_narrow_output_type(
    *,
    wedge_name: str,
    job_to_be_done: str,
    failure_mode: str,
    user_role: str,
) -> str:
    text = " ".join([wedge_name, job_to_be_done, failure_mode, user_role]).lower()
    if "backup" in text or "restore" in text or "recovery" in text:
        return "workflow_reliability_console"
    if "sync" in text or "handoff" in text or "import" in text:
        return "workflow_reliability_assistant"
    if "compliance" in text or "evidence" in text or "monitoring" in text:
        return "operator_evidence_workspace"
    if "shipping" in text or "postage" in text or "listing" in text:
        return "operator_workflow_patch"
    return "workflow_diagnostic_prototype"


def build_launch_artifact_plan(output_type: str) -> list[dict[str, str]]:
    return [
        {"artifact": "brief", "purpose": "one-page positioning and target user summary"},
        {"artifact": "prototype_spec", "purpose": f"narrow {output_type} scope and non-goals"},
        {"artifact": "experiment_script", "purpose": "five-user workflow walkthrough plan"},
        {"artifact": "landing_copy", "purpose": "problem-first smoke-test copy for the first launch surface"},
    ]


def _prototype_gate_metadata(
    *,
    selection_reason: str,
    selection_gate: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    gate_mode = "strict_validated" if selection_reason == "validated_selection_gate" else "prototype_candidate_exception"
    gate_reasons = list(selection_gate.get("reasons", []) or [])
    basis = "full_market_proof"
    if "prototype_candidate_single_family_exception" in gate_reasons:
        basis = "supported_single_family_workflow_pain"
    elif "prototype_candidate_multifamily_near_miss" in gate_reasons:
        basis = "multifamily_near_miss"

    evidence_strength = {
        "recurrence_state": str(corroboration.get("recurrence_state", "") or ""),
        "corroboration_score": float(corroboration.get("corroboration_score", 0.0) or 0.0),
        "family_count": int(corroboration.get("core_source_family_diversity", 0) or 0),
        "value_support": float((evidence_payload.get("evidence_assessment", {}) or {}).get("value_support", 0.0) or 0.0),
        "problem_plausibility": float((evidence_payload.get("evidence_assessment", {}) or {}).get("problem_plausibility", 0.0) or 0.0),
        "composite_score": float((evidence_payload.get("evidence_assessment", {}) or {}).get("composite_score", 0.0) or 0.0),
        "wedge_active": bool(market_enrichment.get("wedge_active")),
    }

    if gate_mode == "strict_validated":
        confidence_level = "market_confirmed"
        validation_certainty = "validated_selection_gate_met"
        overclaim_guardrail = "Treat as evidence-backed market validation."
    else:
        confidence_level = "prototype_checkpoint"
        validation_certainty = "credible_prototype_candidate_not_market_confirmed"
        overclaim_guardrail = "Treat as a first prototype candidate only; do not claim full market validation or broad demand proof."

    return {
        "prototype_gate_mode": gate_mode,
        "prototype_gate_basis": basis,
        "prototype_gate_family_count": evidence_strength["family_count"],
        "prototype_gate_evidence_strength": evidence_strength,
        "market_confidence_level": confidence_level,
        "validation_certainty": validation_certainty,
        "overclaim_guardrail": overclaim_guardrail,
    }


def build_brief_payload(
    *,
    run_id: str,
    opportunity_id: int,
    validation_id: int,
    cluster_id: int,
    linked_finding_ids: list[int],
    finding: Any,
    cluster: dict[str, Any],
    anchor_atom: Any,
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    evidence_payload: dict[str, Any],
    experiment_hypothesis: str,
    selection_status: str,
    selection_reason: str,
    selection_gate: dict[str, Any],
) -> dict[str, Any]:
    screening = (getattr(finding, "evidence", {}) or {})
    source_policy = screening.get("source_policy", {})
    source_classification = screening.get("source_classification", {})
    counterevidence = evidence_payload.get("counterevidence", []) or []
    open_questions = [
        item.get("summary", "")
        for item in counterevidence
        if item.get("status") == "supported" and item.get("summary")
    ]
    open_questions.extend(market_enrichment.get("wedge_block_reasons", []) or [])

    wedge_name = str(market_enrichment.get("wedge_name", "") or "")
    recommended_output_type = determine_narrow_output_type(
        wedge_name=wedge_name,
        job_to_be_done=cluster.get("job_to_be_done", ""),
        failure_mode=getattr(anchor_atom, "failure_mode", ""),
        user_role=cluster.get("user_role", ""),
    )
    prototype_gate = _prototype_gate_metadata(
        selection_reason=selection_reason,
        selection_gate=selection_gate,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        evidence_payload=evidence_payload,
    )

    return {
        "schema_version": BUILD_BRIEF_SCHEMA_VERSION,
        "rule_version": BUILD_PREP_RULE_VERSION,
        "run_id": run_id,
        "opportunity_id": opportunity_id,
        "validation_id": validation_id,
        "cluster_id": cluster_id,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "selection_gate": selection_gate,
        "prototype_gate": prototype_gate,
        "linked_finding_ids": linked_finding_ids,
        "problem_summary": cluster.get("summary", {}).get("human_summary")
        or evidence_payload.get("summary", {}).get("problem_statement", ""),
        "job_to_be_done": cluster.get("job_to_be_done", ""),
        "pain_workaround": {
            "pain_statement": getattr(anchor_atom, "pain_statement", ""),
            "failure_mode": getattr(anchor_atom, "failure_mode", ""),
            "current_workaround": getattr(anchor_atom, "current_workaround", ""),
            "current_tools": getattr(anchor_atom, "current_tools", ""),
            "trigger_event": getattr(anchor_atom, "trigger_event", ""),
        },
        "evidence_provenance": {
            "origin_source": getattr(finding, "source", ""),
            "origin_url": getattr(finding, "source_url", ""),
            "validation_id": validation_id,
            "finding_kind": getattr(finding, "finding_kind", ""),
            "recurrence_query_hash": corroboration.get("query_set_hash", ""),
            "recurrence_results_by_source": corroboration.get("results_by_source", {}),
            "queries_executed": evidence_payload.get("queries_executed", []),
            "recurrence_budget_profile": evidence_payload.get("recurrence_budget_profile", {}),
            "candidate_meaningful": evidence_payload.get("candidate_meaningful", {}),
            "recurrence_failure_class": evidence_payload.get("recurrence_failure_class", ""),
            "recurrence_probe_summary": evidence_payload.get("recurrence_probe_summary", {}),
            "recurrence_source_branch": evidence_payload.get("recurrence_source_branch", {}),
            "last_action": evidence_payload.get("last_action", ""),
            "last_transition_reason": evidence_payload.get("last_transition_reason", ""),
            "chosen_family": evidence_payload.get("chosen_family", ""),
            "expected_gain_class": evidence_payload.get("expected_gain_class", ""),
            "source_attempts_snapshot": evidence_payload.get("source_attempts_snapshot", {}),
            "skipped_families": evidence_payload.get("skipped_families", {}),
            "controller_actions": evidence_payload.get("controller_actions", []),
            "budget_snapshot": evidence_payload.get("budget_snapshot", {}),
            "fallback_strategy_used": evidence_payload.get("fallback_strategy_used", ""),
            "decomposed_atom_queries": evidence_payload.get("decomposed_atom_queries", []),
            "routing_override_reason": evidence_payload.get("routing_override_reason", ""),
            "cohort_query_pack_used": evidence_payload.get("cohort_query_pack_used", False),
            "cohort_query_pack_name": evidence_payload.get("cohort_query_pack_name", ""),
            "web_query_strategy_path": evidence_payload.get("web_query_strategy_path", []),
            "specialized_surface_targeting_used": evidence_payload.get("specialized_surface_targeting_used", False),
            "promotion_gap_class": evidence_payload.get("promotion_gap_class", ""),
            "near_miss_enrichment_action": evidence_payload.get("near_miss_enrichment_action", ""),
            "sufficiency_priority_reason": evidence_payload.get("sufficiency_priority_reason", ""),
            "value_enrichment_used": evidence_payload.get("value_enrichment_used", False),
            "value_enrichment_queries": evidence_payload.get("value_enrichment_queries", []),
            "matched_results_by_source": evidence_payload.get("matched_results_by_source", {}),
            "partial_results_by_source": evidence_payload.get("partial_results_by_source", {}),
            "family_confirmation_count": evidence_payload.get("family_confirmation_count", 0),
            "source_yield": evidence_payload.get("source_yield", {}),
            "reshaped_query_history": evidence_payload.get("reshaped_query_history", []),
        },
        "source_family_corroboration": {
            "recurrence_state": corroboration.get("recurrence_state", ""),
            "recurrence_gap_reason": evidence_payload.get("recurrence_gap_reason", ""),
            "recurrence_failure_class": evidence_payload.get("recurrence_failure_class", ""),
            "source_families": corroboration.get("source_families", []),
            "source_family_match_counts": corroboration.get("source_family_match_counts", {}),
            "core_source_families": corroboration.get("core_source_families", []),
            "core_source_family_diversity": corroboration.get("core_source_family_diversity", 0),
            "cross_source_match_score": corroboration.get("cross_source_match_score", 0.0),
            "corroboration_score": corroboration.get("corroboration_score", 0.0),
            "generalizability_class": corroboration.get("generalizability_class", ""),
            "generalizability_score": corroboration.get("generalizability_score", 0.0),
        },
        "screening_summary": {
            "source_class": source_classification.get("source_class", getattr(finding, "source_class", "")),
            "policy_reasons": source_policy.get("reasons", []),
            "negative_signals": source_classification.get("negative_signals", []),
            "screening_score": screening.get("screening", {}).get("score", 0.0),
        },
        "wedge_profitability_relevance": {
            "wedge_name": wedge_name,
            "wedge_active": market_enrichment.get("wedge_active", False),
            "wedge_fit_score": market_enrichment.get("wedge_fit_score", 0.0),
            "value_support": evidence_payload.get("evidence_assessment", {}).get("value_support", 0.0),
            "demand_score": market_enrichment.get("demand_score", 0.0),
            "buyer_intent_score": market_enrichment.get("buyer_intent_score", 0.0),
            "willingness_to_pay_signal": market_enrichment.get("willingness_to_pay_signal", 0.0),
            "multi_source_value_lift": market_enrichment.get("multi_source_value_lift", 0.0),
            "relevance_score": evidence_payload.get("evidence_assessment", {}).get("problem_plausibility", 0.0),
        },
        "recommended_narrow_output_type": recommended_output_type,
        "first_experiment_hypothesis": experiment_hypothesis,
        "launch_artifact_plan": build_launch_artifact_plan(recommended_output_type),
        "prototype_spec_posture": {
            "recommended_output_type": recommended_output_type,
            "confidence_label": prototype_gate["market_confidence_level"],
            "build_scope_rule": (
                "Keep the first prototype narrow and diagnostic."
                if prototype_gate["prototype_gate_mode"] == "prototype_candidate_exception"
                else "Keep the first prototype narrow, but it may lean harder into validated demand."
            ),
            "messaging_rule": prototype_gate["overclaim_guardrail"],
        },
        "open_questions_risks": [
            *([evidence_payload.get("recurrence_gap_reason", "")] if evidence_payload.get("recurrence_gap_reason") else []),
            *[item for item in open_questions if item],
        ],
    }
