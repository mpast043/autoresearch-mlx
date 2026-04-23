"""Canonical opportunity/research spec construction.

This module is intentionally small: it defines the contract that downstream
renderers and build-prep paths should consume instead of each stage inventing
its own partial payload.
"""

from __future__ import annotations

from typing import Any

from src.opportunity_evaluation import canonical_evidence_assessment, canonical_scorecard_snapshot


RESEARCH_SPEC_SCHEMA_VERSION = "research_spec_v1"
RESEARCH_SPEC_ARTIFACT_TYPE = "research_spec"


def build_research_spec(
    *,
    slug: str,
    product_type: str,
    problem_statement: str,
    value_hypothesis: str,
    core_features: list[str],
    audience: str,
    monetization_strategy: str,
    source_finding_kind: str,
    validation: Any,
    evidence: dict[str, Any],
    validation_plan: dict[str, Any],
    build_ready: bool = False,
) -> dict[str, Any]:
    """Build the canonical post-validation research artifact.

    The contract keeps computed validation facts beside product framing so
    markdown exporters, idea rows, and build-prep handoffs do not drift apart.
    """
    opportunity_evaluation = evidence.get("opportunity_evaluation")
    if not isinstance(opportunity_evaluation, dict):
        opportunity_evaluation = {}

    evaluation_inputs = opportunity_evaluation.get("inputs", {}) or {}
    evaluation_validation = evaluation_inputs.get("validation", {}) or {}
    evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
    evaluation_policy = opportunity_evaluation.get("policy", {}) or {}
    evaluation_selection = opportunity_evaluation.get("selection", {}) or {}

    scorecard = evidence.get("opportunity_scorecard") or canonical_scorecard_snapshot(opportunity_evaluation)
    evidence_assessment = evidence.get("evidence_assessment") or canonical_evidence_assessment(opportunity_evaluation)
    selection_status = str(
        evaluation_selection.get("selection_status")
        or evidence.get("selection_status", "")
        or ""
    )
    source_validation = {
        "decision": str(
            evaluation_policy.get("decision")
            or evidence.get("decision", "")
            or ""
        ),
        "decision_reason": str(
            evaluation_policy.get("decision_reason")
            or evidence.get("decision_reason", "")
            or ""
        ),
        "passed": bool(getattr(validation, "passed", False)),
        "overall_score": float(
            evaluation_validation.get("overall_score", getattr(validation, "overall_score", 0.0))
            or 0.0
        ),
    }

    return {
        "schema_version": RESEARCH_SPEC_SCHEMA_VERSION,
        "artifact_type": RESEARCH_SPEC_ARTIFACT_TYPE,
        "slug": slug,
        "product_type": product_type,
        "problem_statement": problem_statement,
        "value_hypothesis": value_hypothesis,
        "core_features": core_features,
        "audience": audience,
        "monetization_strategy": monetization_strategy,
        "source_finding_kind": source_finding_kind,
        "evidence_refresh_from_validation_id": getattr(validation, "id", None),
        "market_gap_state": str(
            evaluation_evidence.get("market_gap_state")
            or evidence.get("market_gap_state", "unknown")
            or "unknown"
        ),
        "market_gap": dict(evaluation_evidence.get("market_gap") or evidence.get("market_gap", {}) or {}),
        "validation_plan": dict(evaluation_evidence.get("validation_plan", validation_plan) or {}),
        "opportunity_scorecard": scorecard,
        "evidence_assessment": evidence_assessment,
        "selection_status": selection_status,
        "selection_reason": str(
            evaluation_selection.get("selection_reason")
            or evidence.get("selection_reason", "")
            or ""
        ),
        "selection_gate": dict(
            evaluation_selection.get("selection_checks")
            or evidence.get("selection_gate", {})
            or {}
        ),
        "counterevidence": list(
            evaluation_evidence.get("counterevidence")
            or evidence.get("counterevidence", [])
            or []
        ),
        "opportunity_evaluation": opportunity_evaluation,
        "source_validation": source_validation,
        "build_ready": build_ready,
    }
