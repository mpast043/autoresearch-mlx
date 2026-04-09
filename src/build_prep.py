"""Selection-state and build-prep helpers for post-validation workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.database import Finding, ProblemAtom

logger = logging.getLogger(__name__)

BUILD_BRIEF_SCHEMA_VERSION = "build_brief_v1"
BUILD_PREP_RULE_VERSION = "build_prep_v1"
PROTOTYPE_CANDIDATE_RULE_VERSION = "prototype_candidate_v1"


# Vague placeholder patterns to down-rank or reject
VAGUE_PATTERNS = {
    "workflow_reliability",
    "workflow_diagnostic",
    "operator_workflow_patch",
    "workflow",
    "manual workflow",
    "keep operations in sync",
    "keep sync",
    "keep data in sync",
    "spreadsheet hell",
    "copy-pasting",
    "repetitive tasks",
    "manual cleanup",
    "productivity",
    "operations automation",
    "data sync",
    "collaboration",
}

GENERIC_BUILD_READY_PATTERNS = {
    "daily operation",
    "manual execution of tasks",
    "workflow reliability",
    "workflow diagnostic",
    "workflow_diagnostic_prototype",
    "small businesses are spending excessive time",
    "responding to reviews, following up",
    "manual tasks like",
}


@dataclass
class PlatformFit:
    """Structured result from platform/format classification."""
    host_platform: str | None = None
    product_format: str | None = None
    product_name: str | None = None
    one_sentence_product: str | None = None
    why_this_format: str | None = None
    llm_used: bool = False
    fallback_used: bool = False
    classification_confidence: float | None = None
    raw_classification: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_vague(self) -> bool:
        """Check if this classification is too vague to be useful."""
        if not self.product_name:
            return True
        name_lower = self.product_name.lower()
        return any(pattern.lower() in name_lower for pattern in VAGUE_PATTERNS)

    @property
    def is_platform_native(self) -> bool:
        """Check if this is a plugin/add-on/extension format."""
        if not self.product_format:
            return False
        formats = self.product_format.lower()
        return any(fmt in formats for fmt in [
            "add-on", "add-in", "extension", "app", "plugin", "micro-saas"
        ])


def _looks_generic_build_ready_text(value: str | None) -> bool:
    """Detect placeholder or overly broad text in build-ready artifacts."""
    text = (value or "").strip().lower()
    if not text or len(text) < 18:
        return True
    return any(pattern in text for pattern in GENERIC_BUILD_READY_PATTERNS)


def evaluate_build_ready_sharpness(brief_payload: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic sharpness checks before any build-ready promotion.

    This protects the build-ready transition from broad or placeholder briefs even
    when later LLM outputs sound polished. The checks are intentionally conservative:
    a candidate must identify a specific product shape, a non-generic host context,
    a concrete failure mode, and enough corroboration to justify a build-ready handoff.
    """
    platform_fit = PlatformFit(**(brief_payload.get("platform_fit") or {}))
    pain = brief_payload.get("pain_workaround", {}) or {}
    corroboration = brief_payload.get("source_family_corroboration", {}) or {}

    reasons: list[str] = []

    host_platform = str(platform_fit.host_platform or "").strip()
    if not host_platform or host_platform.lower() == "unknown":
        reasons.append("unknown_host_platform")

    if platform_fit.is_vague:
        reasons.append("vague_product_name")

    job_to_be_done = str(brief_payload.get("job_to_be_done", "") or "")
    if _looks_generic_build_ready_text(job_to_be_done):
        reasons.append("generic_job_to_be_done")

    failure_mode = str(pain.get("failure_mode", "") or "")
    if _looks_generic_build_ready_text(failure_mode):
        reasons.append("generic_failure_mode")

    trigger_event = str(pain.get("trigger_event", "") or "")
    if _looks_generic_build_ready_text(trigger_event):
        reasons.append("generic_trigger_event")

    source_family_diversity = int(
        corroboration.get("source_family_diversity")
        or corroboration.get("core_source_family_diversity")
        or 0
    )
    if source_family_diversity < 2:
        reasons.append("insufficient_source_family_diversity")

    return {
        "passes": not reasons,
        "reasons": reasons,
        "host_platform": host_platform or "Unknown",
        "product_name": platform_fit.product_name or "",
        "source_family_diversity": source_family_diversity,
    }


# =============================================================================
# Provider Configuration for Platform Classification
# =============================================================================

def get_platform_classification_config() -> dict[str, str]:
    """Get configuration for platform classification LLM provider."""
    import os
    return {
        "provider": os.environ.get("PLATFORM_FIT_LLM_PROVIDER", "auto"),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"),
    }


# Optimized prompt for local/small models
PLATFORM_CLASSIFICATION_PROMPT = """Given this opportunity, output ONLY valid JSON with these exact fields:

{{"host_platform": "Google Docs|Gmail|Amazon Seller Central|Shopify Admin|Etsy Dashboard|Slack|Microsoft Word|Browser|Chrome extension|WordPress|internal dashboard", "product_format": "Google Docs add-on|Gmail add-on|Chrome extension|Shopify app|Slack app|Word add-in|Excel add-in|WordPress plugin|internal workflow tool|lightweight microSaaS", "product_name": "specific narrow name like Contract Guard", "one_sentence_product": "what it does", "why_this_format": "why better than standalone SaaS"}}

User role: {user_role}
Job: {job_to_be_done}
Failure: {failure_mode}
Trigger: {trigger_event}
Workaround: {current_workaround}
Pain: {pain_statement}

Output JSON only. No markdown. No prose."""


def _parse_platform_fit_response(raw: str, provider: str) -> PlatformFit | None:
    """Parse LLM response into PlatformFit. Handles markdown, partial JSON, etc."""
    import json

    if not raw:
        return None

    # Clean up response
    cleaned = raw.strip()

    # Remove markdown code blocks
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]

    # Try to extract JSON from potentially malformed response
    try:
        result = json.loads(cleaned.strip())

        # Validate we have at least a product_name
        if not result.get("product_name"):
            logger.warning(f"{provider}: No product_name in response")
            return None

        return PlatformFit(
            host_platform=result.get("host_platform"),
            product_format=result.get("product_format"),
            product_name=result.get("product_name"),
            one_sentence_product=result.get("one_sentence_product"),
            why_this_format=result.get("why_this_format"),
            llm_used=True,
            fallback_used=False,
            classification_confidence=0.7,
            raw_classification=result,
        )

    except json.JSONDecodeError as e:
        # Try to extract JSON object from response even if partial
        import re
        json_match = re.search(r'\{[^{}]*\}', cleaned)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if result.get("product_name"):
                    return PlatformFit(
                        host_platform=result.get("host_platform"),
                        product_format=result.get("product_format"),
                        product_name=result.get("product_name"),
                        one_sentence_product=result.get("one_sentence_product"),
                        why_this_format=result.get("why_this_format"),
                        llm_used=True,
                        fallback_used=False,
                        classification_confidence=0.5,  # Lower confidence for partial parse
                        raw_classification=result,
                    )
            except json.JSONDecodeError:
                pass

        logger.warning(f"{provider}: Failed to parse JSON response: {e}")
        return None


def _classify_via_ollama(
    *,
    job_to_be_done: str,
    failure_mode: str,
    trigger_event: str,
    current_workaround: str,
    pain_statement: str,
    user_role: str,
    cluster_summary: str,
) -> PlatformFit | None:
    """Classify platform fit using Ollama (local-first)."""
    import json

    config = get_platform_classification_config()
    base_url = config["ollama_base_url"]
    model = config["ollama_model"]

    # Format the prompt
    prompt = PLATFORM_CLASSIFICATION_PROMPT.format(
        user_role=user_role or "unknown",
        job_to_be_done=job_to_be_done or "unknown",
        failure_mode=failure_mode or "unknown",
        trigger_event=trigger_event or "none",
        current_workaround=current_workaround or "none",
        pain_statement=pain_statement or "none",
    )

    try:
        # Use synchronous request for simplicity
        import urllib.request
        import urllib.error

        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent output
                "num_predict": 300,   # Limit output length
            }
        }

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(request_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            raw = result.get("response", "").strip()

            if not raw:
                logger.warning("Ollama: Empty response")
                return None

            logger.info(f"Ollama classification succeeded with model {model}")
            return _parse_platform_fit_response(raw, "Ollama")

    except urllib.error.URLError as e:
        logger.debug(f"Ollama unavailable: {e}")
        return None
    except Exception as e:
        logger.warning(f"Ollama classification failed: {e}")
        return None


def _classify_via_anthropic(
    *,
    job_to_be_done: str,
    failure_mode: str,
    trigger_event: str,
    current_workaround: str,
    pain_statement: str,
    user_role: str,
    cluster_summary: str,
) -> PlatformFit | None:
    """Classify platform fit using Anthropic (cloud fallback)."""

    config = get_platform_classification_config()
    api_key = config["anthropic_api_key"]

    if not api_key:
        logger.debug("Anthropic: No API key available")
        return None

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        # Use the same optimized prompt
        prompt = PLATFORM_CLASSIFICATION_PROMPT.format(
            user_role=user_role or "unknown",
            job_to_be_done=job_to_be_done or "unknown",
            failure_mode=failure_mode or "unknown",
            trigger_event=trigger_event or "none",
            current_workaround=current_workaround or "none",
            pain_statement=pain_statement or "none",
        )

        response = client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=400,
            system="Output ONLY valid JSON. No markdown. No prose.",
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        logger.info("Anthropic classification succeeded")
        return _parse_platform_fit_response(raw, "Anthropic")

    except Exception as e:
        logger.warning(f"Anthropic classification failed: {e}")
        return None


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
    if decision == "promote":
        return (
            "prototype_candidate",
            "validated_selection_gate",
            {
                "eligible": True,
                "gate_version": BUILD_PREP_RULE_VERSION,
                "reasons": ["validation_recommended_promote"],
                "blocked_by": [],
            },
        )

    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    # Use source_family_diversity (total unique confirming sources) rather than
    # core_source_family_diversity (only "core" origin sources). A finding confirmed
    # by Reddit + Web should count as multi-family support regardless of origin.
    core_family_diversity = int(corroboration.get("source_family_diversity", 0) or 0)
    generalizability_class = str(corroboration.get("generalizability_class", "") or "")
    recurrence_state = str(corroboration.get("recurrence_state", "") or "")
    evidence_quality = float(scorecard.get("evidence_quality", 0.0) or 0.0)
    value_support = float(scorecard.get("value_support", 0.0) or 0.0)
    composite_score = float(scorecard.get("composite_score", 0.0) or 0.0)
    frequency_score = float(scorecard.get("frequency_score", 0.0) or 0.0)
    workaround_density = float(scorecard.get("workaround_density", 0.0) or 0.0)
    cost_of_inaction = float(scorecard.get("cost_of_inaction", 0.0) or 0.0)
    buildability = float(scorecard.get("buildability", 0.0) or 0.0)
    cross_source_match_score = float(corroboration.get("cross_source_match_score", 0.0) or 0.0)
    generalizability_score = float(corroboration.get("generalizability_score", 0.0) or 0.0)
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
    sharp_checkpoint_candidate = (
        core_family_diversity >= 2
        and recurrence_state in {"thin", "timeout", "supported", "strong"}
        and corroboration_score >= 0.22
        and cross_source_match_score >= 0.16
        and generalizability_score >= 0.58
        and frequency_score >= 0.25
        and value_support >= 0.46
        and evidence_quality >= 0.42
        and composite_score >= 0.31
        and workaround_density >= 0.34
        and cost_of_inaction >= 0.4
        and buildability >= 0.52
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
                sharp_checkpoint_candidate
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
        if sharp_checkpoint_candidate and not (
            core_family_diversity >= 2
            and (exploratory_recurrence_ok or timeout_checkpoint_candidate)
            and corroboration_score >= 0.25
            and value_support >= 0.55
            and evidence_quality >= 0.45
            and composite_score >= 0.34
        ):
            exploratory_reasons.append("prototype_candidate_sharp_checkpoint")
        elif core_family_diversity >= 2:
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
    # Use source_family_diversity (total unique confirming sources) rather than
    # core_source_family_diversity (only "core" origin sources). A finding confirmed
    # by Reddit + Web should count as multi-family support regardless of origin.
    core_family_diversity = int(corroboration.get("source_family_diversity", 0) or 0)
    generalizability_class = str(corroboration.get("generalizability_class", "") or "")
    recurrence_state = str(corroboration.get("recurrence_state", "") or "")
    evidence_quality = float(scorecard.get("evidence_quality", 0.0) or 0.0)
    value_support = float(scorecard.get("value_support", 0.0) or 0.0)
    composite_score = float(scorecard.get("composite_score", 0.0) or 0.0)
    frequency_score = float(scorecard.get("frequency_score", 0.0) or 0.0)
    workaround_density = float(scorecard.get("workaround_density", 0.0) or 0.0)
    cost_of_inaction = float(scorecard.get("cost_of_inaction", 0.0) or 0.0)
    buildability = float(scorecard.get("buildability", 0.0) or 0.0)
    cross_source_match_score = float(corroboration.get("cross_source_match_score", 0.0) or 0.0)
    generalizability_score = float(corroboration.get("generalizability_score", 0.0) or 0.0)
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
    sharp_checkpoint_candidate = (
        core_family_diversity >= 2
        and recurrence_state in {"thin", "timeout", "supported", "strong"}
        and corroboration_score >= 0.22
        and cross_source_match_score >= 0.16
        and generalizability_score >= 0.58
        and frequency_score >= 0.25
        and value_support >= 0.46
        and evidence_quality >= 0.42
        and composite_score >= 0.31
        and workaround_density >= 0.34
        and cost_of_inaction >= 0.4
        and buildability >= 0.52
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
            "id": "exploratory_sharp_checkpoint_branch",
            "pass": sharp_checkpoint_candidate,
            "detail": {
                "cross_source_match_score": round(cross_source_match_score, 4),
                "generalizability_score": round(generalizability_score, 4),
                "frequency_score": round(frequency_score, 4),
                "workaround_density": round(workaround_density, 4),
                "cost_of_inaction": round(cost_of_inaction, 4),
                "buildability": round(buildability, 4),
                "value_support_floor": 0.46,
                "evidence_quality_floor": 0.42,
                "composite_floor": 0.31,
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
    trigger_event: str = "",
    current_workaround: str = "",
    pain_statement: str = "",
    cluster_summary: str = "",
) -> PlatformFit:
    """
    Determines a specific product concept using LLM classification.
    Returns a structured PlatformFit object with host_platform, product_format, and product_name.
    Falls back to keyword matching if LLM is unavailable.
    """
    # Check trigger conditions - require more context for LLM
    has_trigger = any([trigger_event, pain_statement, current_workaround, cluster_summary])
    has_primary = job_to_be_done or failure_mode

    if has_primary and has_trigger:
        result = _determine_product_via_llm(
            job_to_be_done=job_to_be_done,
            failure_mode=failure_mode,
            trigger_event=trigger_event,
            current_workaround=current_workaround,
            pain_statement=pain_statement,
            user_role=user_role,
            cluster_summary=cluster_summary,
        )
        if result and result.product_name:
            logger.info(
                f"LLM classification succeeded: platform={result.host_platform}, "
                f"format={result.product_format}, name={result.product_name}"
            )
            return result

    # Fallback to keyword matching (legacy)
    fallback = _determine_product_via_keyword(
        wedge_name=wedge_name,
        job_to_be_done=job_to_be_done,
        failure_mode=failure_mode,
        user_role=user_role,
    )
    logger.info(
        f"Fallback classification: platform={fallback.host_platform}, "
        f"format={fallback.product_format}, name={fallback.product_name}"
    )
    return fallback


def _determine_product_via_keyword(
    *,
    wedge_name: str,
    job_to_be_done: str,
    failure_mode: str,
    user_role: str,
) -> PlatformFit:
    """Fallback keyword-based classification."""
    text = " ".join([wedge_name, job_to_be_done, failure_mode, user_role]).lower()

    if "backup" in text or "restore" in text or "recovery" in text:
        return PlatformFit(
            host_platform="Internal workflow",
            product_format="internal workflow tool",
            product_name="backup_reliability_console",
            one_sentence_product="Recover lost data from failed backups",
            why_this_format="Direct access to backup infrastructure",
            llm_used=False,
            fallback_used=True,
        )
    if "sync" in text or "handoff" in text or "import" in text:
        return PlatformFit(
            host_platform="Internal workflow",
            product_format="internal workflow tool",
            product_name="sync_handoff_assistant",
            one_sentence_product="Keep operations data in sync without manual cleanup",
            why_this_format="Direct integration with internal systems",
            llm_used=False,
            fallback_used=True,
        )
    if "compliance" in text or "evidence" in text or "monitoring" in text:
        return PlatformFit(
            host_platform="Internal workflow",
            product_format="internal workflow tool",
            product_name="compliance_evidence_workspace",
            one_sentence_product="Keep multi-framework compliance evidence and monitoring reliable",
            why_this_format="Direct access to compliance systems",
            llm_used=False,
            fallback_used=True,
        )
    if "shipping" in text or "postage" in text or "listing" in text:
        return PlatformFit(
            host_platform="E-commerce platforms",
            product_format="platform app",
            product_name="listing_workflow_patch",
            one_sentence_product="Streamline shipping and listing workflows",
            why_this_format="Built into seller workflow",
            llm_used=False,
            fallback_used=True,
        )

    # Default fallback
    return PlatformFit(
        host_platform="Unknown",
        product_format="lightweight microSaaS",
        product_name="workflow_diagnostic_prototype",
        one_sentence_product="Diagnostic tool for workflow reliability",
        why_this_format="Requires further validation for platform fit",
        llm_used=False,
        fallback_used=True,
    )


def _determine_product_via_llm(
    *,
    job_to_be_done: str,
    failure_mode: str,
    trigger_event: str,
    current_workaround: str,
    pain_statement: str,
    user_role: str,
    cluster_summary: str = "",
) -> PlatformFit | None:
    """
    Uses LLM to determine the best product concept for this opportunity.
    Tries providers in order: configured provider -> fallback -> keyword fallback.
    Returns a structured PlatformFit object.
    """
    config = get_platform_classification_config()
    provider = config["provider"].lower()

    logger.info(f"Platform classification: provider={provider}")

    # Try configured provider or auto-detect
    if provider == "ollama":
        result = _classify_via_ollama(
            job_to_be_done=job_to_be_done,
            failure_mode=failure_mode,
            trigger_event=trigger_event,
            current_workaround=current_workaround,
            pain_statement=pain_statement,
            user_role=user_role,
            cluster_summary=cluster_summary,
        )
        if result:
            logger.info("Ollama classification succeeded")
            return result
        logger.info("Ollama failed, falling back")

    elif provider == "anthropic":
        result = _classify_via_anthropic(
            job_to_be_done=job_to_be_done,
            failure_mode=failure_mode,
            trigger_event=trigger_event,
            current_workaround=current_workaround,
            pain_statement=pain_statement,
            user_role=user_role,
            cluster_summary=cluster_summary,
        )
        if result:
            logger.info("Anthropic classification succeeded")
            return result
        logger.info("Anthropic failed, falling back")

    else:  # "auto" mode
        # Try Ollama first (local-first)
        result = _classify_via_ollama(
            job_to_be_done=job_to_be_done,
            failure_mode=failure_mode,
            trigger_event=trigger_event,
            current_workaround=current_workaround,
            pain_statement=pain_statement,
            user_role=user_role,
            cluster_summary=cluster_summary,
        )
        if result:
            logger.info("Auto mode: Ollama succeeded")
            return result

        # Try Anthropic as fallback
        result = _classify_via_anthropic(
            job_to_be_done=job_to_be_done,
            failure_mode=failure_mode,
            trigger_event=trigger_event,
            current_workaround=current_workaround,
            pain_statement=pain_statement,
            user_role=user_role,
            cluster_summary=cluster_summary,
        )
        if result:
            logger.info("Auto mode: Anthropic fallback succeeded")
            return result

        logger.info("Auto mode: All LLM providers failed, will use keyword fallback")

    return None


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
    finding: Finding,
    cluster: dict[str, Any],
    anchor_atom: ProblemAtom,
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
    platform_fit = determine_narrow_output_type(
        wedge_name=wedge_name,
        job_to_be_done=cluster.get("job_to_be_done", ""),
        failure_mode=getattr(anchor_atom, "failure_mode", ""),
        user_role=cluster.get("user_role", ""),
        trigger_event=getattr(anchor_atom, "trigger_event", ""),
        current_workaround=getattr(anchor_atom, "current_workaround", ""),
        pain_statement=getattr(anchor_atom, "pain_statement", ""),
        cluster_summary=cluster.get("summary", ""),
    )
    # For backward compatibility, use product_name as the narrow output type
    recommended_output_type = platform_fit.product_name
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
            "source_family_diversity": corroboration.get("source_family_diversity", 0),
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
        "platform_fit": platform_fit.to_dict(),
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
