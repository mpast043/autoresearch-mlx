"""Canonical opportunity/research spec construction.

This module is intentionally small: it defines the contract that downstream
renderers and build-prep paths should consume instead of each stage inventing
its own partial payload.
"""

from __future__ import annotations

from typing import Any


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
    scorecard = evidence.get("opportunity_scorecard", {}) or {}
    selection_status = evidence.get("selection_status", "")
    source_validation = {
        "decision": evidence.get("decision", ""),
        "decision_reason": evidence.get("decision_reason", ""),
        "passed": bool(getattr(validation, "passed", False)),
        "overall_score": float(getattr(validation, "overall_score", 0.0) or 0.0),
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
        "market_gap_state": evidence.get("market_gap_state", "unknown"),
        "market_gap": evidence.get("market_gap", {}),
        "validation_plan": validation_plan,
        "opportunity_scorecard": scorecard,
        "evidence_assessment": evidence.get("evidence_assessment", {}),
        "selection_status": selection_status,
        "selection_reason": evidence.get("selection_reason", ""),
        "selection_gate": evidence.get("selection_gate", {}),
        "counterevidence": evidence.get("counterevidence", []),
        "source_validation": source_validation,
        "build_ready": build_ready,
    }
