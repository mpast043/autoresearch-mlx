"""Findings-first high-leverage scoring and reporting."""

from __future__ import annotations

import math
import re
from typing import Any

from src.database import Finding, ProblemAtom, RawSignal

HIGH_LEVERAGE_VERSION = "high_leverage_v1"
HIGH_LEVERAGE_WEIGHTS = {
    "specificity": 0.25,
    "workaround_intensity": 0.20,
    "operational_consequence": 0.20,
    "asymmetry": 0.15,
    "novelty": 0.10,
    "first_customer_clarity": 0.10,
}

HIGH_LEVERAGE_PRIOR_PATTERNS = [
    "reconcile",
    "reconciliation",
    "mismatch",
    "out of sync",
    "state drift",
    "restore",
    "recovery",
    "migration",
    "import",
    "export",
    "csv",
    "duplicate",
    "deleted order",
    "inventory",
    "fulfillment",
    "handoff",
    "version",
    "rollback",
    "unreachable",
]

REJECT_REASON_MARKERS = [
    "broad_buying_prompt_without_wedge_slice",
    "review_product_specific_issue",
    "github_product_specific_issue",
    "finance_tool_shopping_without_specific_failure",
    "broad_finance_visibility_without_specific_failure",
    "youtube_commentary_without_concrete_workflow",
    "stackoverflow_implementation_local_only",
    "listing_or_marketing_copy",
    "promo_or_generic_praise",
    "generic_or_low_signal_context",
    "help_or_generic_summary_content",
    "vendor_specific_complaint",
    "product_specific_issue",
]
EDITORIAL_NOISE_PATTERNS = [
    "what's new",
    "whats new",
    "latest release",
    "overview",
    "release notes",
    "release overview",
    "beginner guide",
    "ultimate guide",
]
FAILURE_SIGNAL_PATTERNS = [
    "mismatch",
    "reconcile",
    "restore",
    "recovery",
    "duplicate",
    "broken",
    "breaks",
    "out of sync",
    "unreachable",
    "deleted order",
    "inventory error",
    "import failure",
]

GENERIC_SEGMENTS = {
    "",
    "small business operations",
    "operators",
    "operations teams",
    "small business owners",
    "business owners",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalized(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _source_family(finding: Finding, signal: RawSignal | None, evidence: dict[str, Any]) -> str:
    source_name = _normalized(getattr(signal, "source_name", "") or finding.source)
    source_type = _normalized(getattr(signal, "source_type", ""))
    if "reddit" in source_name or source_type == "forum":
        return "reddit"
    if "github" in source_name or source_type == "github_issue":
        return "github"
    if "wordpress" in source_name:
        return "wordpress_reviews"
    if "shopify" in source_name:
        return "shopify_reviews"
    if "youtube" in source_name or source_type == "youtube":
        return "youtube"
    if "stackoverflow" in source_name:
        return "stackoverflow"
    if "web" in source_name or source_type == "web":
        return "web"
    return "unknown"


def _shape_tokens(
    finding: Finding,
    signal: RawSignal | None,
    atom: ProblemAtom | None,
    family: str,
) -> set[str]:
    fields = [
        getattr(atom, "segment", "") if atom else "",
        getattr(atom, "user_role", "") if atom else "",
        getattr(atom, "job_to_be_done", "") if atom else "",
        getattr(atom, "failure_mode", "") if atom else "",
        getattr(atom, "current_workaround", "") if atom else "",
        getattr(atom, "trigger_event", "") if atom else "",
        getattr(signal, "title", "") if signal else "",
        finding.product_built,
        finding.outcome_summary,
        family,
    ]
    text = _normalized(" ".join(part for part in fields if part))
    return {token for token in re.split(r"[^a-z0-9]+", text) if len(token) >= 4}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return overlap / union


def _extract_recent_shapes(cluster_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(cluster_context, dict):
        return []
    shapes = cluster_context.get("recent_shapes", [])
    if not isinstance(shapes, list):
        return []
    return [shape for shape in shapes if isinstance(shape, dict)]


def _compute_novelty(
    finding: Finding,
    signal: RawSignal | None,
    atom: ProblemAtom | None,
    family: str,
    cluster_context: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    recent_shapes = _extract_recent_shapes(cluster_context)
    if not recent_shapes:
        return 0.78, ["novel_shape_in_segment"]

    current_tokens = _shape_tokens(finding, signal, atom, family)
    current_cluster_key = _normalized(getattr(atom, "cluster_key", "") if atom else "")
    max_similarity = 0.0
    duplicate_cluster = False
    for shape in recent_shapes:
        tokens = set(shape.get("shape_tokens", []) or [])
        max_similarity = max(max_similarity, _jaccard_similarity(current_tokens, tokens))
        if current_cluster_key and current_cluster_key == _normalized(shape.get("cluster_key", "")):
            duplicate_cluster = True

    if duplicate_cluster:
        return 0.18, ["cluster_shape_already_seen"]
    novelty = _clamp(0.92 - (max_similarity * 0.9))
    if novelty >= 0.68:
        return novelty, ["uncommon_workflow_failure_combo"]
    if novelty <= 0.35:
        return novelty, ["common_workflow_shape"]
    return novelty, []


def _specificity_component(atom: ProblemAtom | None) -> tuple[float, list[str]]:
    if atom is None:
        return 0.0, ["missing_problem_atom"]
    specificity = float(getattr(atom, "specificity_score", 0.0) or 0.0)
    if specificity <= 0.0:
        filled = sum(
            bool(value)
            for value in [
                atom.segment,
                atom.user_role,
                atom.job_to_be_done,
                atom.trigger_event,
                atom.failure_mode,
                atom.current_workaround,
            ]
        )
        specificity = _clamp(filled / 6.0)
    reasons: list[str] = []
    if specificity >= 0.7:
        reasons.append("narrow_workflow_shape")
    elif specificity <= 0.35:
        reasons.append("generic_workflow_shape")
    return _clamp(specificity), reasons


def _workaround_component(atom: ProblemAtom | None, evidence: dict[str, Any]) -> tuple[float, list[str]]:
    if atom is None:
        return 0.0, []
    base = 0.0
    reasons: list[str] = []
    workaround_text = _normalized(getattr(atom, "current_workaround", ""))
    if workaround_text:
        base += 0.45
        reasons.append("visible_workaround")
    if _normalized(getattr(atom, "frequency_clues", "")):
        base += 0.18
    if _normalized(getattr(atom, "why_now_clues", "")):
        base += 0.08
    meaningful = ((evidence.get("candidate_meaningful") or {}) if isinstance(evidence, dict) else {}) or {}
    if meaningful.get("support_present"):
        base += 0.12
    if any(pattern in workaround_text for pattern in ["spreadsheet", "manual", "rollback", "copy", "paste"]):
        base += 0.12
    return _clamp(base), reasons


def _consequence_component(atom: ProblemAtom | None) -> tuple[float, list[str]]:
    if atom is None:
        return 0.0, []
    consequence = float(getattr(atom, "consequence_score", 0.0) or 0.0)
    if consequence <= 0.0:
        consequence_text = _normalized(getattr(atom, "cost_consequence_clues", ""))
        urgency_text = _normalized(getattr(atom, "urgency_clues", ""))
        consequence = 0.0
        if consequence_text:
            consequence += 0.45
        if urgency_text:
            consequence += 0.12
        if any(term in consequence_text for term in ["hours", "downtime", "revenue", "refund", "late", "penalty", "error"]):
            consequence += 0.22
        if "unreachable" in consequence_text:
            consequence += 0.12
    reasons = ["clear_operational_consequence"] if consequence >= 0.45 else []
    return _clamp(consequence), reasons


def _asymmetry_component(
    finding: Finding,
    signal: RawSignal | None,
    atom: ProblemAtom | None,
) -> tuple[float, list[str]]:
    haystack = _normalized(
        " ".join(
            [
                finding.product_built,
                finding.outcome_summary,
                getattr(signal, "title", "") if signal else "",
                getattr(signal, "body_excerpt", "") if signal else "",
                getattr(atom, "job_to_be_done", "") if atom else "",
                getattr(atom, "failure_mode", "") if atom else "",
                getattr(atom, "current_workaround", "") if atom else "",
            ]
        )
    )
    prior_hits = [pattern for pattern in HIGH_LEVERAGE_PRIOR_PATTERNS if pattern in haystack]
    score = 0.18 + min(0.55, 0.09 * len(prior_hits))
    if any(pattern in haystack for pattern in ["reconcile", "mismatch", "out of sync", "restore", "migration"]):
        score += 0.1
    if any(pattern in haystack for pattern in ["duplicate", "deleted order", "inventory", "fulfillment"]):
        score += 0.08
    reasons = ["high_tension_operational_failure"] if score >= 0.45 else []
    if len(prior_hits) >= 2:
        reasons.append("high_leverage_prior_match")
    return _clamp(score), reasons


def _first_customer_component(atom: ProblemAtom | None, finding: Finding) -> tuple[float, list[str]]:
    if atom is None:
        return 0.0, []
    role = _normalized(atom.user_role)
    segment = _normalized(atom.segment)
    entrepreneur = _normalized(finding.entrepreneur)
    score = 0.0
    if role:
        score += 0.45
    if segment and segment not in GENERIC_SEGMENTS:
        score += 0.4
    elif segment:
        score += 0.2
    if entrepreneur:
        score += 0.1
    reasons = ["clear_first_customer"] if score >= 0.65 else []
    return _clamp(score), reasons


def _extract_validation_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        return {}
    validation = evidence.get("validation")
    if isinstance(validation, dict):
        return validation
    return evidence


def _evidence_tier(
    finding: Finding,
    signal: RawSignal | None,
    atom: ProblemAtom | None,
    evidence: dict[str, Any],
    specificity: float,
    workaround_intensity: float,
    operational_consequence: float,
) -> tuple[str, float, list[str]]:
    validation = _extract_validation_evidence(evidence)
    corroboration = validation.get("corroboration") if isinstance(validation.get("corroboration"), dict) else {}
    family_confirmation_count = int(
        validation.get("family_confirmation_count")
        or corroboration.get("family_confirmation_count")
        or 0
    )
    source_family_diversity = int(
        validation.get("source_family_diversity")
        or corroboration.get("source_family_diversity")
        or 0
    )
    matched_results_by_source = validation.get("matched_results_by_source", {}) or {}
    independent_sources = [source for source, count in matched_results_by_source.items() if int(count or 0) > 0]
    if family_confirmation_count >= 2 or source_family_diversity >= 2 or len(independent_sources) >= 2:
        return "multi_family_confirmed", 0.92, ["independent_confirmation"]

    family = _source_family(finding, signal, evidence)
    if (
        family != "unknown"
        and specificity >= 0.70
        and (workaround_intensity >= 0.45 or operational_consequence >= 0.45)
    ):
        return "one_family_strong", 0.68, ["single_family_strong_structure"]

    return "thin", 0.24, ["thin_evidence"]


def _reject_reasons(finding: Finding, evidence: dict[str, Any]) -> list[str]:
    if finding.source_class != "pain_signal":
        return ["non_pain_signal"]
    reasons: list[str] = []
    source_classification = evidence.get("source_classification", {}) if isinstance(evidence, dict) else {}
    classification_reasons = list(source_classification.get("reasons", []) or [])
    screening = evidence.get("screening", {}) if isinstance(evidence, dict) else {}
    negative_signals = list(screening.get("negative_signals", []) or [])
    combined = [str(item) for item in [*classification_reasons, *negative_signals]]
    for item in combined:
        lowered = _normalized(item)
        if any(marker in lowered for marker in REJECT_REASON_MARKERS):
            reasons.append(item)
    text = _normalized(
        " ".join(
            [
                finding.product_built,
                finding.outcome_summary,
            ]
        )
    )
    if any(pattern in text for pattern in EDITORIAL_NOISE_PATTERNS) and not any(
        pattern in text for pattern in FAILURE_SIGNAL_PATTERNS
    ):
        reasons.append("editorial_or_overview_content")
    return reasons


def score_high_leverage_finding(
    finding: Finding,
    signal: RawSignal | None,
    atom: ProblemAtom | None,
    evidence: dict[str, Any],
    cluster_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reject_reasons = _reject_reasons(finding, evidence)
    specificity, specificity_reasons = _specificity_component(atom)
    workaround_intensity, workaround_reasons = _workaround_component(atom, evidence)
    operational_consequence, consequence_reasons = _consequence_component(atom)
    asymmetry, asymmetry_reasons = _asymmetry_component(finding, signal, atom)
    family = _source_family(finding, signal, evidence)
    novelty, novelty_reasons = _compute_novelty(finding, signal, atom, family, cluster_context)
    first_customer_clarity, customer_reasons = _first_customer_component(atom, finding)
    evidence_tier, evidence_confidence, evidence_reasons = _evidence_tier(
        finding,
        signal,
        atom,
        evidence,
        specificity,
        workaround_intensity,
        operational_consequence,
    )

    components = {
        "specificity": round(_clamp(specificity), 4),
        "workaround_intensity": round(_clamp(workaround_intensity), 4),
        "operational_consequence": round(_clamp(operational_consequence), 4),
        "asymmetry": round(_clamp(asymmetry), 4),
        "novelty": round(_clamp(novelty), 4),
        "first_customer_clarity": round(_clamp(first_customer_clarity), 4),
        "evidence_confidence": round(_clamp(evidence_confidence), 4),
    }

    weighted_score = sum(components[key] * weight for key, weight in HIGH_LEVERAGE_WEIGHTS.items())
    score = round(_clamp(weighted_score), 4)
    reasons = [
        *reject_reasons,
        *specificity_reasons,
        *workaround_reasons,
        *consequence_reasons,
        *asymmetry_reasons,
        *novelty_reasons,
        *customer_reasons,
        *evidence_reasons,
    ]
    deduped_reasons = list(dict.fromkeys(reason for reason in reasons if reason))

    if reject_reasons:
        band = "reject"
        status = "discarded"
    elif (
        score >= 0.68
        and specificity >= 0.70
        and (workaround_intensity >= 0.45 or operational_consequence >= 0.45)
        and evidence_tier == "multi_family_confirmed"
    ):
        band = "standout" if score >= 0.78 else "strong"
        status = "confirmed"
    elif (
        score >= 0.62
        and specificity >= 0.70
        and (workaround_intensity >= 0.45 or operational_consequence >= 0.45)
        and evidence_tier in {"one_family_strong", "multi_family_confirmed"}
    ):
        band = "standout" if score >= 0.78 else "strong"
        status = "candidate"
    else:
        band = "ordinary"
        status = "ordinary"

    return {
        "score": score,
        "band": band,
        "status": status,
        "reasons": deduped_reasons[:8],
        "components": components,
        "evidence_tier": evidence_tier,
        "version": HIGH_LEVERAGE_VERSION,
    }


def persist_high_leverage_assessment(
    db: Any,
    *,
    finding_id: int,
    assessment: dict[str, Any],
    signal_id: int | None = None,
) -> None:
    finding = db.get_finding(finding_id)
    if finding is None:
        return
    evidence = dict(finding.evidence or {})
    evidence["high_leverage"] = assessment
    db.update_finding_evidence(finding_id, evidence)

    if signal_id is None:
        signals = db.get_raw_signals_by_finding(finding_id)
        signal_id = signals[0].id if signals else None
    if signal_id:
        signal = db.get_raw_signal(signal_id)
        if signal is not None:
            metadata = dict(signal.metadata or {})
            metadata["high_leverage"] = assessment
            db.update_raw_signal_metadata(signal_id, metadata)


def _build_recent_shape_records(
    db: Any,
    *,
    segment: str,
    exclude_finding_id: int | None = None,
    limit: int = 120,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finding in db.get_findings(limit=max(limit * 2, 100)):
        if exclude_finding_id and getattr(finding, "id", None) == exclude_finding_id:
            continue
        if finding.source_class != "pain_signal" or finding.status == "screened_out":
            continue
        atoms = db.get_problem_atoms_by_finding(int(finding.id or 0))
        if not atoms:
            continue
        atom = atoms[0]
        if segment and _normalized(atom.segment) != _normalized(segment):
            continue
        signals = db.get_raw_signals_by_finding(int(finding.id or 0))
        signal = signals[0] if signals else None
        family = _source_family(finding, signal, finding.evidence or {})
        rows.append(
            {
                "finding_id": int(finding.id or 0),
                "cluster_key": getattr(atom, "cluster_key", ""),
                "shape_tokens": sorted(_shape_tokens(finding, signal, atom, family)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_high_leverage_cluster_context(db: Any, finding: Finding, atom: ProblemAtom | None) -> dict[str, Any]:
    if atom is None:
        return {"recent_shapes": []}
    cluster_context: dict[str, Any] = {
        "recent_shapes": _build_recent_shape_records(
            db,
            segment=getattr(atom, "segment", ""),
            exclude_finding_id=int(getattr(finding, "id", 0) or 0) or None,
        )
    }
    cluster = None
    if getattr(atom, "cluster_key", ""):
        cluster = db.get_cluster_by_key(atom.cluster_key)
    if cluster is not None:
        cluster_context.update(
            {
                "cluster_id": int(cluster.id or 0),
                "cluster_label": cluster.label,
                "cluster_key": cluster.cluster_key,
            }
        )
        opportunity = db.get_opportunity_by_cluster_id(int(cluster.id or 0))
        if opportunity is not None:
            cluster_context.update(
                {
                    "opportunity_id": int(opportunity.id or 0),
                    "recommendation": opportunity.recommendation,
                    "selection_status": opportunity.selection_status,
                }
            )
    return cluster_context


def build_high_leverage_report(db: Any, *, run_id: str | None = None, limit: int = 10) -> dict[str, Any]:
    findings = db.get_findings(limit=max(limit * 8, 100))
    rows: list[dict[str, Any]] = []
    for finding in findings:
        if run_id and (finding.evidence or {}).get("run_id") != run_id:
            continue
        signals = db.get_raw_signals_by_finding(int(finding.id or 0))
        signal = signals[0] if signals else None
        atoms = db.get_problem_atoms_by_finding(int(finding.id or 0))
        atom = atoms[0] if atoms else None
        cluster_context = build_high_leverage_cluster_context(db, finding, atom)
        assessment = dict((finding.evidence or {}).get("high_leverage") or {})
        if not assessment:
            assessment = score_high_leverage_finding(finding, signal, atom, finding.evidence or {}, cluster_context)
        row = {
            "finding_id": int(finding.id or 0),
            "source": finding.source,
            "title": finding.product_built or finding.outcome_summary or finding.source_url,
            "source_url": finding.source_url,
            "status": finding.status,
            "high_leverage_score": assessment.get("score", 0.0),
            "high_leverage_band": assessment.get("band", "ordinary"),
            "high_leverage_status": assessment.get("status", "ordinary"),
            "evidence_tier": assessment.get("evidence_tier", "thin"),
            "reasons": assessment.get("reasons", []),
            "components": assessment.get("components", {}),
            "cluster_id": cluster_context.get("cluster_id", 0),
            "cluster_label": cluster_context.get("cluster_label", ""),
            "opportunity_id": cluster_context.get("opportunity_id", 0),
            "recommendation": cluster_context.get("recommendation", ""),
            "selection_status": cluster_context.get("selection_status", ""),
            "version": assessment.get("version", HIGH_LEVERAGE_VERSION),
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            0 if item["high_leverage_status"] == "confirmed" else 1 if item["high_leverage_status"] == "candidate" else 2,
            -float(item.get("high_leverage_score", 0.0) or 0.0),
            -int(item.get("finding_id", 0) or 0),
        )
    )
    surfaced_rows = [
        row for row in rows
        if row["high_leverage_status"] in {"candidate", "confirmed", "ordinary"}
        and row["high_leverage_band"] != "reject"
    ]
    top_rows = surfaced_rows[:limit]
    band_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in top_rows:
        band_counts[row["high_leverage_band"]] = band_counts.get(row["high_leverage_band"], 0) + 1
        status_counts[row["high_leverage_status"]] = status_counts.get(row["high_leverage_status"], 0) + 1
    return {
        "run_id": run_id or "",
        "count": len(top_rows),
        "band_mix": band_counts,
        "status_mix": status_counts,
        "findings": top_rows,
    }
