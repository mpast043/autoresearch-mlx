"""Selection-state and build-prep helpers for post-validation workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

from src.opportunity_evaluation import canonical_evidence_assessment

if TYPE_CHECKING:
    from src.database import Finding, ProblemAtom

logger = logging.getLogger(__name__)

BUILD_BRIEF_SCHEMA_VERSION = "build_brief_v1"
BUILD_PREP_RULE_VERSION = "build_prep_v1"
PROTOTYPE_CANDIDATE_RULE_VERSION = "prototype_candidate_v1"
_RUNTIME_CONFIG: dict[str, Any] = {}


# Vague placeholder patterns to down-rank or reject
VAGUE_PATTERNS = {
    "workflow_reliability",
    "workflow_diagnostic",
    "operator_workflow_patch",
    "sync_handoff_assistant",
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
    "managing and importing various business data",
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


def _prefer_specific_problem_summary(summary: str, pain_statement: str, failure_mode: str) -> str:
    summary_text = (summary or "").strip()
    lowered = summary_text.lower()
    if any(
        marker in lowered
        for marker in (
            "how do you prevent",
            "what's the one task",
            "manual tasks",
            "spreadsheet or csv errors",
            "problem:",
        )
    ):
        return (pain_statement or failure_mode or summary_text).strip()
    return summary_text


def _prefer_specific_job_to_be_done(job_to_be_done: str, failure_mode: str, current_tools: str, trigger_event: str) -> str:
    job = (job_to_be_done or "").strip()
    haystack = " ".join([job_to_be_done or "", failure_mode or "", current_tools or "", trigger_event or ""]).lower()
    if "various business data" in haystack:
        if "quickbooks" in haystack or "qbo" in haystack:
            return "Import Stripe and bank-feed CSVs into QuickBooks without reconciliation drift"
        if "shopify" in haystack:
            return "Import supplier and inventory CSVs into Shopify without corrupting downstream data"
        if "csv" in haystack and "import" in haystack:
            return "Validate CSV imports before they corrupt downstream business data"
        return failure_mode or trigger_event or job
    return job


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
    evidence_quality = float(corroboration.get("evidence_quality", 0.0) or 0.0)
    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    independent_confirmations = int(
        corroboration.get("family_confirmation_count", 0)
        or brief_payload.get("evidence_provenance", {}).get("family_confirmation_count", 0)
        or 0
    )

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
    if evidence_quality < 0.4:
        reasons.append("insufficient_evidence_quality")
    if corroboration_score < 0.3:
        reasons.append("insufficient_corroboration_score")
    if independent_confirmations < 2:
        reasons.append("insufficient_independent_confirmations")

    return {
        "passes": not reasons,
        "reasons": reasons,
        "host_platform": host_platform or "Unknown",
        "product_name": platform_fit.product_name or "",
        "source_family_diversity": source_family_diversity,
        "independent_confirmations": independent_confirmations,
        "corroboration_score": round(corroboration_score, 4),
        "evidence_quality": round(evidence_quality, 4),
    }


# =============================================================================
# Provider Configuration for Platform Classification
# =============================================================================

def get_platform_classification_config() -> dict[str, str]:
    """Get configuration for platform classification LLM provider."""
    import os
    build_prep_config = _RUNTIME_CONFIG.get("build_prep", {}) if _RUNTIME_CONFIG else {}
    classifier_config = build_prep_config.get("platform_classification", {}) if isinstance(build_prep_config, dict) else {}
    llm_config = _RUNTIME_CONFIG.get("llm", {}) if _RUNTIME_CONFIG else {}

    def _pick(*values: Any, default: str = "") -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return default

    return {
        "provider": _pick(
            classifier_config.get("provider"),
            llm_config.get("provider"),
            os.environ.get("PLATFORM_FIT_LLM_PROVIDER"),
            default="auto",
        ),
        "ollama_base_url": _pick(
            classifier_config.get("base_url"),
            llm_config.get("base_url"),
            os.environ.get("OLLAMA_BASE_URL"),
            default="http://127.0.0.1:11434",
        ),
        "ollama_model": _pick(
            classifier_config.get("model"),
            llm_config.get("model"),
            os.environ.get("OLLAMA_MODEL"),
            default="gemma4:latest",
        ),
        "ollama_api_key": _pick(
            classifier_config.get("api_key"),
            llm_config.get("api_key"),
            os.environ.get("OLLAMA_API_KEY"),
            default="",
        ),
        "timeout_seconds": float(
            _pick(
                classifier_config.get("timeout_seconds"),
                os.environ.get("PLATFORM_FIT_LLM_TIMEOUT"),
                default="60",
            )
        ),
        "anthropic_api_key": _pick(
            classifier_config.get("anthropic_api_key"),
            llm_config.get("anthropic_api_key"),
            os.environ.get("ANTHROPIC_API_KEY"),
            os.environ.get("CLAUDE_API_KEY"),
            default="",
        ),
    }


def configure_build_prep(config: dict[str, Any]) -> None:
    """Set runtime config for build-prep provider selection."""
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config or {}


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
    api_key = config.get("ollama_api_key", "")

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

        normalized_base_url = base_url.rstrip("/")
        use_openai_compat = normalized_base_url.endswith("/v1")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if use_openai_compat:
            request_data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Output ONLY valid JSON. No markdown. No prose."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            }
            endpoint = f"{normalized_base_url}/chat/completions"
        else:
            request_data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Output ONLY valid JSON. No markdown. No prose."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                },
            }
            endpoint = f"{normalized_base_url}/api/chat"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(request_data).encode("utf-8"),
            headers=headers,
        )

        with urllib.request.urlopen(req, timeout=float(config.get("timeout_seconds", 60) or 60)) as response:
            result = json.loads(response.read().decode("utf-8"))
            if use_openai_compat:
                raw = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
            else:
                raw = result.get("message", {}).get("content", "").strip()

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


def _selection_policy_inputs(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    opportunity_evaluation: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(opportunity_evaluation, dict) or not opportunity_evaluation:
        return decision, dict(scorecard or {}), dict(corroboration or {}), dict(market_enrichment or {})

    evaluation_inputs = opportunity_evaluation.get("inputs", {}) or {}
    validation_inputs = evaluation_inputs.get("validation", {}) or {}
    evaluation_corroboration = evaluation_inputs.get("corroboration", {}) or {}
    evaluation_market = evaluation_inputs.get("market_enrichment", {}) or {}
    evaluation_measures = opportunity_evaluation.get("measures", {}) or {}
    evaluation_scores = evaluation_measures.get("scores", {}) or {}
    evaluation_dimensions = evaluation_measures.get("dimensions", {}) or {}
    evaluation_transition = evaluation_measures.get("transition", {}) or {}
    evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
    evaluation_policy = opportunity_evaluation.get("policy", {}) or {}

    resolved_decision = str(evaluation_policy.get("decision") or decision or "")
    resolved_scorecard = dict(scorecard or {})
    resolved_corroboration = {**dict(corroboration or {}), **evaluation_corroboration}
    resolved_market = {**dict(market_enrichment or {}), **evaluation_market}

    if "decision_score" in evaluation_scores:
        resolved_scorecard["decision_score"] = float(evaluation_scores.get("decision_score", 0.0) or 0.0)
    if "problem_truth_score" in evaluation_scores:
        resolved_scorecard["problem_truth_score"] = float(evaluation_scores.get("problem_truth_score", 0.0) or 0.0)
    if "revenue_readiness_score" in evaluation_scores:
        resolved_scorecard["revenue_readiness_score"] = float(
            evaluation_scores.get("revenue_readiness_score", 0.0) or 0.0
        )

    for field in (
        "frequency_score",
        "cost_of_inaction",
        "workaround_density",
        "buildability",
        "value_support",
        "evidence_quality",
    ):
        if field in evaluation_dimensions:
            resolved_scorecard[field] = float(evaluation_dimensions.get(field, 0.0) or 0.0)

    if "composite_score" in evaluation_transition:
        resolved_scorecard["composite_score"] = float(evaluation_transition.get("composite_score", 0.0) or 0.0)
    if "cluster_signal_count" in validation_inputs:
        resolved_scorecard["cluster_signal_count"] = int(validation_inputs.get("cluster_signal_count", 0) or 0)
    if "cluster_atom_count" in validation_inputs:
        resolved_scorecard["cluster_atom_count"] = int(validation_inputs.get("cluster_atom_count", 0) or 0)

    resolved_corroboration["recurrence_state"] = str(
        evaluation_evidence.get("recurrence_state")
        or evaluation_corroboration.get("recurrence_state")
        or resolved_corroboration.get("recurrence_state", "")
    )

    return resolved_decision, resolved_scorecard, resolved_corroboration, resolved_market


def _selection_gate_context(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    opportunity_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision, scorecard, corroboration, market_enrichment = _selection_policy_inputs(
        decision=decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        opportunity_evaluation=opportunity_evaluation,
    )

    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    core_family_diversity = int(corroboration.get("source_family_diversity", 0) or 0)
    cluster_family_diversity = int(corroboration.get("cluster_source_family_diversity", 0) or 0)
    cluster_origin_diversity = int(corroboration.get("cluster_origin_family_diversity", 0) or 0)
    effective_family_diversity = max(core_family_diversity, cluster_family_diversity, cluster_origin_diversity)
    generalizability_class = str(corroboration.get("generalizability_class", "") or "")
    recurrence_state = str(corroboration.get("recurrence_state", "") or "")
    cluster_signal_count = int(scorecard.get("cluster_signal_count", scorecard.get("cluster_atom_count", 1)) or 1)
    cluster_atom_count = int(scorecard.get("cluster_atom_count", cluster_signal_count) or cluster_signal_count)
    decision_score = float(scorecard.get("decision_score", 0.0) or 0.0)
    problem_truth_score = float(scorecard.get("problem_truth_score", 0.0) or 0.0)
    revenue_readiness_score = float(scorecard.get("revenue_readiness_score", 0.0) or 0.0)
    frequency_score = float(scorecard.get("frequency_score", 0.0) or 0.0)
    workaround_density = float(scorecard.get("workaround_density", 0.0) or 0.0)
    cost_of_inaction = float(scorecard.get("cost_of_inaction", 0.0) or 0.0)
    buildability = float(scorecard.get("buildability", 0.0) or 0.0)
    cross_source_match_score = float(corroboration.get("cross_source_match_score", 0.0) or 0.0)
    generalizability_score = float(corroboration.get("generalizability_score", 0.0) or 0.0)
    wedge_active = bool(market_enrichment.get("wedge_active"))

    minimum_cluster_ok = cluster_signal_count >= 2 or cluster_atom_count >= 2 or effective_family_diversity >= 2
    reusable_workflow = generalizability_class == "reusable_workflow_pain"
    supported_recurrence = recurrence_state in {"supported", "strong"}
    checkpoint_recurrence = recurrence_state in {"thin", "supported", "strong", "timeout"}

    reasons: list[str] = []
    blocked_by: list[str] = []

    if decision == "promote":
        reasons.append("validation_recommended_promote")
    elif decision == "kill":
        blocked_by.append("decision_kill")
    else:
        blocked_by.append("decision_not_promote")

    if minimum_cluster_ok:
        reasons.append("cluster_size_threshold_met")
    else:
        blocked_by.append("single_signal_single_family_cluster")

    if reusable_workflow:
        reasons.append("generalizable_workflow_pain")
    else:
        blocked_by.append("not_generalizable_enough")

    if effective_family_diversity >= 2:
        reasons.append("multi_family_support")
    else:
        blocked_by.append("single_family_support")

    if supported_recurrence:
        reasons.append("recurrence_state_supported")
    elif recurrence_state == "timeout":
        blocked_by.append("recurrence_timeout")
    elif recurrence_state:
        blocked_by.append("recurrence_not_supported")
    else:
        blocked_by.append("recurrence_unknown")

    if corroboration_score >= 0.6:
        reasons.append("corroboration_threshold_met")
    elif corroboration_score >= 0.25:
        reasons.append("corroboration_checkpoint_met")
    else:
        blocked_by.append("corroboration_below_checkpoint")

    if wedge_active:
        reasons.append("wedge_active")

    validated_candidate = (
        decision == "promote"
        and minimum_cluster_ok
        and reusable_workflow
        and effective_family_diversity >= 2
        and supported_recurrence
        and corroboration_score >= 0.6
    )
    multifamily_checkpoint_candidate = (
        decision == "promote"
        and minimum_cluster_ok
        and reusable_workflow
        and effective_family_diversity >= 2
        and checkpoint_recurrence
        and corroboration_score >= 0.25
        and (cross_source_match_score >= 0.16 or generalizability_score >= 0.58)
    )
    sharp_checkpoint_candidate = (
        decision == "promote"
        and minimum_cluster_ok
        and reusable_workflow
        and effective_family_diversity >= 2
        and checkpoint_recurrence
        and corroboration_score >= 0.22
        and cross_source_match_score >= 0.16
        and generalizability_score >= 0.58
        and frequency_score >= 0.25
        and workaround_density >= 0.34
        and cost_of_inaction >= 0.4
        and buildability >= 0.52
    )
    single_family_checkpoint_candidate = (
        decision == "promote"
        and minimum_cluster_ok
        and reusable_workflow
        and effective_family_diversity == 1
        and supported_recurrence
        and corroboration_score >= 0.3
        and frequency_score >= 0.25
        and buildability >= 0.52
    )

    return {
        "decision": decision,
        "scorecard": scorecard,
        "corroboration": corroboration,
        "market_enrichment": market_enrichment,
        "decision_score": decision_score,
        "problem_truth_score": problem_truth_score,
        "revenue_readiness_score": revenue_readiness_score,
        "frequency_score": frequency_score,
        "workaround_density": workaround_density,
        "cost_of_inaction": cost_of_inaction,
        "buildability": buildability,
        "corroboration_score": corroboration_score,
        "effective_family_diversity": effective_family_diversity,
        "generalizability_class": generalizability_class,
        "generalizability_score": generalizability_score,
        "recurrence_state": recurrence_state,
        "cross_source_match_score": cross_source_match_score,
        "wedge_active": wedge_active,
        "cluster_signal_count": cluster_signal_count,
        "cluster_atom_count": cluster_atom_count,
        "minimum_cluster_ok": minimum_cluster_ok,
        "validated_candidate": validated_candidate,
        "multifamily_checkpoint_candidate": multifamily_checkpoint_candidate,
        "sharp_checkpoint_candidate": sharp_checkpoint_candidate,
        "single_family_checkpoint_candidate": single_family_checkpoint_candidate,
        "reasons": reasons,
        "blocked_by": blocked_by,
    }


def determine_selection_state(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    opportunity_evaluation: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Map validation output into the first build-prep lifecycle state."""
    context = _selection_gate_context(
        decision=decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        opportunity_evaluation=opportunity_evaluation,
    )
    decision = str(context["decision"] or "")

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

    if decision != "promote":
        return (
            "research_more",
            "selection_gate_not_met",
            {
                "eligible": False,
                "gate_version": BUILD_PREP_RULE_VERSION,
                "reasons": list(dict.fromkeys(context["reasons"])),
                "blocked_by": list(dict.fromkeys(context["blocked_by"])),
            },
        )

    if context["validated_candidate"]:
        return (
            "prototype_candidate",
            "validated_selection_gate",
            {
                "eligible": True,
                "gate_version": BUILD_PREP_RULE_VERSION,
                "reasons": list(
                    dict.fromkeys(
                        [*context["reasons"], "validated_multifamily_support"]
                    )
                ),
                "blocked_by": [],
            },
        )

    exploratory_reason = ""
    if context["sharp_checkpoint_candidate"]:
        exploratory_reason = "prototype_candidate_sharp_checkpoint"
    elif context["multifamily_checkpoint_candidate"]:
        exploratory_reason = "prototype_candidate_multifamily_checkpoint"
    elif context["single_family_checkpoint_candidate"]:
        exploratory_reason = "prototype_candidate_single_family_exception"

    if exploratory_reason:
        return (
            "prototype_candidate",
            "prototype_candidate_gate",
            {
                "eligible": True,
                "gate_version": PROTOTYPE_CANDIDATE_RULE_VERSION,
                "reasons": list(dict.fromkeys([*context["reasons"], exploratory_reason])),
                "blocked_by": [],
            },
        )

    return (
        "research_more",
        "selection_gate_not_met",
        {
            "eligible": False,
            "gate_version": BUILD_PREP_RULE_VERSION,
            "reasons": list(dict.fromkeys(context["reasons"])),
            "blocked_by": list(dict.fromkeys(context["blocked_by"])),
        },
    )


def explain_selection_gate_detail(
    *,
    decision: str,
    scorecard: dict[str, Any],
    corroboration: dict[str, Any],
    market_enrichment: dict[str, Any],
    opportunity_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured mirror of ``determine_selection_state`` thresholds for operator debugging."""
    status, reason, gate = determine_selection_state(
        decision=decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        opportunity_evaluation=opportunity_evaluation,
    )
    context = _selection_gate_context(
        decision=decision,
        scorecard=scorecard,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        opportunity_evaluation=opportunity_evaluation,
    )

    strict_checks: list[dict[str, Any]] = [
        {
            "id": "decision_promote",
            "pass": context["decision"] == "promote",
            "actual": context["decision"],
            "need": "promote",
        },
        {
            "id": "minimum_cluster_size",
            "pass": context["minimum_cluster_ok"],
            "actual": {
                "cluster_signal_count": context["cluster_signal_count"],
                "cluster_atom_count": context["cluster_atom_count"],
            },
            "need": ">= 2 signals/atoms or >= 2 source families",
        },
        {
            "id": "generalizability_class",
            "pass": context["generalizability_class"] == "reusable_workflow_pain",
            "actual": context["generalizability_class"],
            "need": "reusable_workflow_pain",
        },
        {
            "id": "source_family_diversity",
            "pass": context["effective_family_diversity"] >= 2,
            "actual": context["effective_family_diversity"],
            "need": ">= 2",
        },
        {
            "id": "recurrence_state",
            "pass": context["recurrence_state"] in {"supported", "strong"},
            "actual": context["recurrence_state"],
            "need": "supported/strong",
        },
        {
            "id": "corroboration_score_strict",
            "pass": context["corroboration_score"] >= 0.6,
            "actual": round(context["corroboration_score"], 4),
            "need": ">= 0.6",
        },
    ]

    score_language_checks: list[dict[str, Any]] = [
        {
            "id": "decision_score",
            "pass": context["decision"] == "promote",
            "actual": round(context["decision_score"], 4),
            "need": "promote policy already cleared this threshold",
        },
        {
            "id": "problem_truth_score",
            "pass": context["problem_truth_score"] >= 0.11,
            "actual": round(context["problem_truth_score"], 4),
            "need": ">= 0.11",
        },
        {
            "id": "revenue_readiness_score",
            "pass": context["revenue_readiness_score"] >= 0.22,
            "actual": round(context["revenue_readiness_score"], 4),
            "need": ">= 0.22",
        },
    ]

    exploratory_checks: list[dict[str, Any]] = [
        {
            "id": "multifamily_checkpoint_branch",
            "pass": context["multifamily_checkpoint_candidate"],
            "detail": {
                "decision": context["decision"],
                "recurrence_state": context["recurrence_state"],
                "effective_family_diversity": context["effective_family_diversity"],
                "corroboration_score": round(context["corroboration_score"], 4),
                "cross_source_match_score": round(context["cross_source_match_score"], 4),
                "generalizability_score": round(context["generalizability_score"], 4),
            },
        },
        {
            "id": "sharp_checkpoint_branch",
            "pass": context["sharp_checkpoint_candidate"],
            "detail": {
                "decision": context["decision"],
                "frequency_score": round(context["frequency_score"], 4),
                "workaround_density": round(context["workaround_density"], 4),
                "cost_of_inaction": round(context["cost_of_inaction"], 4),
                "buildability": round(context["buildability"], 4),
                "cross_source_match_score": round(context["cross_source_match_score"], 4),
                "generalizability_score": round(context["generalizability_score"], 4),
            },
        },
        {
            "id": "single_family_exception_branch",
            "pass": context["single_family_checkpoint_candidate"],
            "detail": {
                "decision": context["decision"],
                "recurrence_state": context["recurrence_state"],
                "effective_family_diversity": context["effective_family_diversity"],
                "frequency_score": round(context["frequency_score"], 4),
                "buildability": round(context["buildability"], 4),
                "corroboration_floor": 0.3,
                "requires_supported_recurrence": True,
            },
        },
    ]

    return {
        "resolved_selection_status": status,
        "resolved_selection_reason": reason,
        "selection_gate": gate,
        "score_language": {
            "primary": {
                "decision_score": round(context["decision_score"], 4),
                "problem_truth_score": round(context["problem_truth_score"], 4),
                "revenue_readiness_score": round(context["revenue_readiness_score"], 4),
            },
            "diagnostic": {
                "composite_score": round(float(context["scorecard"].get("composite_score", 0.0) or 0.0), 4),
                "value_support": round(float(context["scorecard"].get("value_support", 0.0) or 0.0), 4),
                "evidence_quality": round(float(context["scorecard"].get("evidence_quality", 0.0) or 0.0), 4),
            },
        },
        "recurrence_state": context["recurrence_state"],
        "score_language_checks": score_language_checks,
        "strict_path_checks": strict_checks,
        "exploratory_path_checks": exploratory_checks,
        "wedge_active": context["wedge_active"],
        "hints": [
            "prototype_candidate is only available when the canonical decision is promote.",
            "Selection uses canonical decision plus routing checks; composite/value/evidence remain diagnostic only.",
            "prototype_candidate_gate is a promoted checkpoint path, not a park override.",
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
    fallback = _determine_product_via_keyword(
        wedge_name=wedge_name,
        job_to_be_done=job_to_be_done,
        failure_mode=failure_mode,
        user_role=user_role,
    )

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
            result = _normalize_platform_fit_with_context(
                result=result,
                fallback=fallback,
                wedge_name=wedge_name,
                job_to_be_done=job_to_be_done,
                failure_mode=failure_mode,
                user_role=user_role,
                trigger_event=trigger_event,
                current_workaround=current_workaround,
                pain_statement=pain_statement,
                cluster_summary=cluster_summary,
            )
            logger.info(
                f"LLM classification succeeded: platform={result.host_platform}, "
                f"format={result.product_format}, name={result.product_name}"
            )
            return result

    logger.info(
        f"Fallback classification: platform={fallback.host_platform}, "
        f"format={fallback.product_format}, name={fallback.product_name}"
    )
    return fallback


def _normalize_platform_fit_with_context(
    *,
    result: PlatformFit,
    fallback: PlatformFit,
    wedge_name: str,
    job_to_be_done: str,
    failure_mode: str,
    user_role: str,
    trigger_event: str,
    current_workaround: str,
    pain_statement: str,
    cluster_summary: str,
) -> PlatformFit:
    """Repair obviously incompatible LLM classifications using deterministic context."""
    context_text = " ".join(
        str(value or "")
        for value in [
            wedge_name,
            job_to_be_done,
            failure_mode,
            user_role,
            trigger_event,
            current_workaround,
            pain_statement,
            cluster_summary,
        ]
    ).lower()
    result_text = " ".join(
        str(value or "")
        for value in [
            result.host_platform or "",
            result.product_format or "",
            result.product_name or "",
            result.one_sentence_product or "",
            result.why_this_format or "",
        ]
    ).lower()
    result_surface_text = " ".join(
        str(value or "") for value in [result.host_platform or "", result.product_format or ""]
    ).lower()

    accounting_markers = {
        "quickbooks",
        "qbo",
        "reconciliation",
        "bookkeeper",
        "accounting",
        "ledger",
        "invoice",
        "bank feed",
        "stripe",
    }
    shopify_markers = {"shopify", "merchant", "catalog", "inventory", "sku"}
    csv_markers = {"csv", "import", "spreadsheet", "vendor"}
    generic_doc_surfaces = {"google docs", "gmail", "slack", "word add-in", "docs add-on"}

    fallback_is_specific = (fallback.host_platform or "").lower() not in {"", "unknown"}
    if not fallback_is_specific:
        return result

    def _has_any(markers: set[str], text: str) -> bool:
        return any(marker in text for marker in markers)

    mismatch = False
    if _has_any(accounting_markers, context_text):
        mismatch = _has_any(generic_doc_surfaces, result_surface_text) or not _has_any(accounting_markers, result_text)
    elif _has_any(shopify_markers, context_text):
        mismatch = _has_any(generic_doc_surfaces, result_surface_text) or "shopify" not in result_text
    elif _has_any(csv_markers, context_text) and "csv import workflow" in (fallback.host_platform or "").lower():
        mismatch = _has_any(generic_doc_surfaces, result_text) or not _has_any(csv_markers, result_text)

    if not mismatch:
        return result

    merged_name = result.product_name
    if not merged_name or PlatformFit(product_name=merged_name).is_vague:
        merged_name = fallback.product_name

    merged_sentence = result.one_sentence_product or fallback.one_sentence_product
    if not merged_sentence or _looks_generic_build_ready_text(merged_sentence):
        merged_sentence = fallback.one_sentence_product

    logger.info(
        "Normalized incompatible LLM platform fit from %s/%s to %s/%s",
        result.host_platform,
        result.product_format,
        fallback.host_platform,
        fallback.product_format,
    )
    return PlatformFit(
        host_platform=fallback.host_platform,
        product_format=fallback.product_format,
        product_name=merged_name or fallback.product_name,
        one_sentence_product=merged_sentence,
        why_this_format=fallback.why_this_format or result.why_this_format,
        llm_used=result.llm_used,
        fallback_used=True,
        classification_confidence=result.classification_confidence,
        raw_classification=result.raw_classification,
    )


def _determine_product_via_keyword(
    *,
    wedge_name: str,
    job_to_be_done: str,
    failure_mode: str,
    user_role: str,
) -> PlatformFit:
    """Fallback keyword-based classification."""
    text = " ".join([wedge_name, job_to_be_done, failure_mode, user_role]).lower()

    csv_import_markers = [
        "csv",
        "import",
        "vendor",
        "inventory",
        "pricing",
        "catalog",
        "sku",
        "spreadsheet",
    ]
    accounting_import_markers = [
        "quickbooks",
        "qbo",
        "bookkeeper",
        "accounting",
        "invoice",
        "payment",
        "reconciliation",
    ]

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
    if "csv" in text and any(marker in text for marker in csv_import_markers):
        if any(marker in text for marker in accounting_import_markers):
            return PlatformFit(
                host_platform="QuickBooks",
                product_format="QuickBooks App",
                product_name="ledger_import_guard",
                one_sentence_product="Validate ledger import files before they create reconciliation drift",
                why_this_format="Close to the accounting import workflow and monthly review loop",
                llm_used=False,
                fallback_used=True,
            )
        if "shopify" in text or "merchant" in text or "store" in text:
            return PlatformFit(
                host_platform="Shopify Admin",
                product_format="Shopify App",
                product_name="catalog_import_guard",
                one_sentence_product="Catch bad catalog rows before they corrupt product and inventory imports",
                why_this_format="Runs at the exact merchant import step where bad feeds cause damage",
                llm_used=False,
                fallback_used=True,
            )
        return PlatformFit(
            host_platform="CSV import workflow",
            product_format="web-based CSV validator",
            product_name="csv_import_guard",
            one_sentence_product="Validate CSV rows before import so one bad value does not break downstream data",
            why_this_format="Fits a narrow pre-import checkpoint better than a generic internal workflow tool",
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
    if any(marker in text for marker in ["excel", "spreadsheet", "worksheet"]) and any(
        marker in text
        for marker in [
            "revision",
            "version",
            "submittal",
            "design",
            "calculation",
            "billing",
            "milestone",
            "contract",
            "tax rate",
            "field service",
            "parts usage",
        ]
    ):
        return PlatformFit(
            host_platform="Spreadsheet workflow",
            product_format="workflow add-on",
            product_name="spreadsheet_workflow_guard",
            one_sentence_product="Catch spreadsheet-driven workflow drift before revisions, billing, or calculations go wrong",
            why_this_format="Targets a narrow spreadsheet-heavy checkpoint instead of a generic internal workflow tool",
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
    opportunity_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_mode = "strict_validated" if selection_reason == "validated_selection_gate" else "prototype_candidate_exception"
    gate_reasons = list(selection_gate.get("reasons", []) or [])
    basis = "full_market_proof"
    if "prototype_candidate_single_family_exception" in gate_reasons:
        basis = "supported_single_family_workflow_pain"
    elif "prototype_candidate_multifamily_checkpoint" in gate_reasons:
        basis = "multifamily_near_miss"

    evidence_assessment = evidence_payload.get("evidence_assessment") or canonical_evidence_assessment(opportunity_evaluation)
    evidence_strength = {
        "recurrence_state": str(corroboration.get("recurrence_state", "") or ""),
        "corroboration_score": float(corroboration.get("corroboration_score", 0.0) or 0.0),
        "family_count": int(corroboration.get("core_source_family_diversity", 0) or 0),
        "value_support": float((evidence_assessment or {}).get("value_support", 0.0) or 0.0),
        "problem_plausibility": float((evidence_assessment or {}).get("problem_plausibility", 0.0) or 0.0),
        "composite_score": float((evidence_assessment or {}).get("composite_score", 0.0) or 0.0),
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
    opportunity_evaluation = evidence_payload.get("opportunity_evaluation")
    if not isinstance(opportunity_evaluation, dict):
        opportunity_evaluation = {}
    evaluation_measures = opportunity_evaluation.get("measures", {}) or {}
    evaluation_dimensions = evaluation_measures.get("dimensions", {}) or {}
    evaluation_transition = evaluation_measures.get("transition", {}) or {}
    evaluation_evidence = opportunity_evaluation.get("evidence", {}) or {}
    evaluation_selection = opportunity_evaluation.get("selection", {}) or {}
    resolved_selection_status = str(
        evaluation_selection.get("selection_status") or selection_status or ""
    )
    resolved_selection_reason = str(
        evaluation_selection.get("selection_reason") or selection_reason or ""
    )
    resolved_selection_gate = dict(
        evaluation_selection.get("selection_checks") or selection_gate or {}
    )
    counterevidence = list(
        evaluation_evidence.get("counterevidence")
        or evidence_payload.get("counterevidence", [])
        or []
    )
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
        selection_reason=resolved_selection_reason,
        selection_gate=resolved_selection_gate,
        corroboration=corroboration,
        market_enrichment=market_enrichment,
        evidence_payload=evidence_payload,
        opportunity_evaluation=opportunity_evaluation,
    )

    return {
        "schema_version": BUILD_BRIEF_SCHEMA_VERSION,
        "rule_version": BUILD_PREP_RULE_VERSION,
        "run_id": run_id,
        "opportunity_id": opportunity_id,
        "validation_id": validation_id,
        "cluster_id": cluster_id,
        "selection_status": resolved_selection_status,
        "selection_reason": resolved_selection_reason,
        "selection_gate": resolved_selection_gate,
        "prototype_gate": prototype_gate,
        "opportunity_evaluation": opportunity_evaluation,
        "linked_finding_ids": linked_finding_ids,
        "problem_summary": _prefer_specific_problem_summary(
            cluster.get("summary", {}).get("human_summary")
            or evidence_payload.get("summary", {}).get("problem_statement", ""),
            getattr(anchor_atom, "pain_statement", ""),
            getattr(anchor_atom, "failure_mode", ""),
        ),
        "job_to_be_done": _prefer_specific_job_to_be_done(
            cluster.get("job_to_be_done", ""),
            getattr(anchor_atom, "failure_mode", ""),
            getattr(anchor_atom, "current_tools", ""),
            getattr(anchor_atom, "trigger_event", ""),
        ),
        "user_role": cluster.get("user_role", ""),
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
            "recurrence_state": evaluation_evidence.get("recurrence_state", corroboration.get("recurrence_state", "")),
            "recurrence_gap_reason": evidence_payload.get("recurrence_gap_reason", ""),
            "recurrence_failure_class": evidence_payload.get("recurrence_failure_class", ""),
            "family_confirmation_count": evaluation_evidence.get(
                "family_confirmation_count",
                evidence_payload.get("family_confirmation_count", 0),
            ),
            "source_families": corroboration.get("source_families", []),
            "source_family_match_counts": corroboration.get("source_family_match_counts", {}),
            "core_source_families": corroboration.get("core_source_families", []),
            "core_source_family_diversity": corroboration.get("core_source_family_diversity", 0),
            "source_family_diversity": evaluation_evidence.get(
                "source_family_diversity",
                corroboration.get("source_family_diversity", 0),
            ),
            "cross_source_match_score": corroboration.get("cross_source_match_score", 0.0),
            "corroboration_score": corroboration.get("corroboration_score", 0.0),
            "generalizability_class": corroboration.get("generalizability_class", ""),
            "generalizability_score": corroboration.get("generalizability_score", 0.0),
            "evidence_quality": evaluation_dimensions.get(
                "evidence_quality",
                evidence_payload.get("evidence_assessment", {}).get("evidence_quality", 0.0),
            ),
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
            "value_support": evaluation_dimensions.get(
                "value_support",
                evidence_payload.get("evidence_assessment", {}).get("value_support", 0.0),
            ),
            "demand_score": market_enrichment.get("demand_score", 0.0),
            "buyer_intent_score": market_enrichment.get("buyer_intent_score", 0.0),
            "willingness_to_pay_signal": market_enrichment.get("willingness_to_pay_signal", 0.0),
            "multi_source_value_lift": market_enrichment.get("multi_source_value_lift", 0.0),
            "relevance_score": evaluation_transition.get(
                "problem_plausibility",
                evidence_payload.get("evidence_assessment", {}).get("problem_plausibility", 0.0),
            ),
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
