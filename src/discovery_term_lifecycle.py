"""Term lifecycle state machine for search-space optimization.

This module implements explicit state transitions for keywords and subreddits,
enabling forward+reverse adjustment of the discovery search space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Valid states for search terms
TERM_STATES = [
    "new",              # newly added, not yet searched
    "active",           # actively in use
    "high_performing",  # repeatedly produces validated opportunities
    "weak",             # produces low-yield/noisy output
    "exhausted",        # no longer yielding value after repeated failures
    "completed",        # has produced a build-worthy wedge, value likely exhausted
    "paused",           # temporarily paused due to poor performance
    "banned",           # manually or automatically excluded
    "used",             # has been searched but not high-performing
]

# Valid term types
TERM_TYPES = ["keyword", "subreddit"]

# Abstraction collapse patterns - terms that cause vague buckets
VAGUE_BUCKET_PATTERNS = [
    "manual work",
    "manual process",
    "manual workflow",
    "keep.*sync",
    "keep in sync",
    "data sync",
    "workflow reliability",
    "workflow automation",
    "spreadsheet hell",
    "duct tape",
    "held together",
    "repetitive task",
    "repetitive work",
    "busy work",
    "stuck",
    "frustrat",
    "time sink",
    "tedious",
    "pain point",
    "pain",
    "problem",
    "challenge",
    "struggle",
    "hard to",
    "difficult to",
    "annoying",
]

# Platform-native patterns for plugin/add-on/microSaaS detection
PLATFORM_PATTERNS = {
    "google docs": ["google docs", "gdocs", "google sheets", "google drive", "g sheets"],
    "gmail": ["gmail", "google mail", "gmail add-on"],
    "slack": ["slack", "slack app", "slack integration"],
    "shopify": ["shopify", "shopify app", "shopify store"],
    "wordpress": ["wordpress", "wp plugin", "wordpress plugin"],
    "chrome": ["chrome extension", "chrome add-on", "browser extension"],
    "notion": ["notion", "notion api", "notion integration"],
    "zapier": ["zapier", "zapier integration", "zap"],
    "airtable": ["airtable", "airtable base", "airtable automation"],
    "excel": ["excel", "microsoft excel", "excel addon", "excel plugin"],
    "teams": ["teams", "microsoft teams", "teams app"],
    "discord": ["discord", "discord bot", "discord app"],
}

# Plugin/add-on/microSaaS fit keywords
PLUGIN_FIT_KEYWORDS = [
    "addon", "add-on", "plugin", "extension", "app", "integration", "widget",
    "gadget", "tool", "utility", "microSaaS", "sidebar", "panel", "toolbar",
    "button", "automation", "workflow", "template", "script", "macro",
    "shortcut", "gadget", "embed", "inline", "popup", "modal",
]

# Specific consequence keywords - sharp, buildable problems
CONSEQUENCE_KEYWORDS = [
    "error", "mistake", "wrong", "incorrect", "lost", "missing", "failed",
    "delay", "late", "deadline", "penalty", "fee", "cost", "overcharge",
    "duplicate", "repeat", "reconcile", "match", "inconsistent", "mismatch",
    "out of sync", "version conflict", "file corruption", "data loss",
    "breach", "security", "compliance", "audit", "risk", "liability",
]

# Specificity indicators - specific user + workflow + failure mode
SPECIFICITY_KEYWORDS = [
    "invoice", "payment", "receipt", "contract", "quote", "estimate",
    "shipping", "tracking", "delivery", "order", "inventory", "stock",
    "customer", "client", "lead", "prospect", "sale", "revenue",
    "employee", "payroll", "expense", "budget", "forecast", "report",
    "tax", "vat", "gst", "quarterly", "annual", "fiscal",
    "approval", "sign-off", "authorization", "permit", "license",
    "renewal", "expiration", "reminder", "follow-up", "deadline",
]


@dataclass
class TransitionRule:
    """A single state transition rule."""

    from_states: list[str]
    to_state: str
    condition: str  # human-readable description of when this triggers
    predicate: Any = field(default=None)  # callable that takes term_metrics and returns bool


@dataclass
class TermMetrics:
    """Metrics for a search term, aggregated from discovery_feedback queries."""

    times_searched: int = 0
    findings_emitted: int = 0
    validations: int = 0
    passes: int = 0
    prototype_candidates: int = 0
    build_briefs: int = 0
    screened_out: int = 0
    low_yield_count: int = 0
    noisy_count: int = 0
    thin_validation_count: int = 0
    avg_validation_score: float = 0.0
    avg_screening_score: float = 0.0
    quality_score: float = 0.0


@dataclass
class LifecycleConfig:
    """Configuration for term lifecycle transitions."""

    # Promotion thresholds (promote to high_performing)
    min_passes_for_promotion: int = 2
    min_prototype_candidates_for_promotion: int = 1
    min_quality_score_for_promotion: float = 0.6
    min_validation_score_for_promotion: float = 0.5

    # Demotion thresholds (active -> weak)
    max_runs_before_demotion: int = 3
    max_low_yield_before_demotion: int = 2
    max_noisy_before_demotion: int = 2

    # Exhaustion thresholds (weak -> exhausted or active -> exhausted)
    max_weak_runs_before_exhausted: int = 3
    max_failed_validations_before_exhausted: int = 3

    # Pausing thresholds (weak -> paused)
    max_low_yield_before_pause: int = 2

    # Completion threshold (active -> completed)
    min_build_briefs_for_completion: int = 1

    # Misc
    min_runs_for_assessment: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_passes_for_promotion": self.min_passes_for_promotion,
            "min_prototype_candidates_for_promotion": self.min_prototype_candidates_for_promotion,
            "min_quality_score_for_promotion": self.min_quality_score_for_promotion,
            "min_validation_score_for_promotion": self.min_validation_score_for_promotion,
            "max_runs_before_demotion": self.max_runs_before_demotion,
            "max_low_yield_before_demotion": self.max_low_yield_before_demotion,
            "max_noisy_before_demotion": self.max_noisy_before_demotion,
            "max_weak_runs_before_exhausted": self.max_weak_runs_before_exhausted,
            "max_failed_validations_before_exhausted": self.max_failed_validations_before_exhausted,
            "max_low_yield_before_pause": self.max_low_yield_before_pause,
            "min_build_briefs_for_completion": self.min_build_briefs_for_completion,
            "min_runs_for_assessment": self.min_runs_for_assessment,
        }


DEFAULT_CONFIG = LifecycleConfig()


def should_promote_to_high_performing(metrics: TermMetrics, config: LifecycleConfig = DEFAULT_CONFIG) -> bool:
    """Determine if a term should be promoted to high_performing state."""
    # Has produced prototype candidates or passes
    if metrics.prototype_candidates >= config.min_prototype_candidates_for_promotion:
        return True
    if metrics.passes >= config.min_passes_for_promotion:
        return True
    # Or has high quality consistently
    if metrics.quality_score >= config.min_quality_score_for_promotion and metrics.times_searched >= config.min_runs_for_assessment:
        return True
    if metrics.avg_validation_score >= config.min_validation_score_for_promotion and metrics.validations >= config.min_runs_for_assessment:
        return True
    return False


def should_demote_to_weak(metrics: TermMetrics, config: LifecycleConfig = DEFAULT_CONFIG) -> bool:
    """Determine if a term should be demoted to weak state."""
    if metrics.times_searched < config.min_runs_for_assessment:
        return False

    # Repeated low yield
    if metrics.low_yield_count >= config.max_low_yield_before_demotion:
        return True
    # Repeated noisy output
    if metrics.noisy_count >= config.max_noisy_before_demotion:
        return True
    # Repeated thin validation (validations but no passes)
    if metrics.thin_validation_count >= config.max_noisy_before_demotion:
        return True

    return False


def should_mark_exhausted(metrics: TermMetrics, current_state: str, config: LifecycleConfig = DEFAULT_CONFIG) -> bool:
    """Determine if a term should be marked as exhausted."""
    if metrics.times_searched < config.min_runs_for_assessment:
        return False

    # From weak state: too many failed attempts
    if current_state == "weak":
        if metrics.low_yield_count + metrics.noisy_count >= config.max_weak_runs_before_exhausted:
            return True

    # From active state: consistent failure after enough runs
    if current_state == "active":
        if metrics.times_searched >= config.max_runs_before_demotion * 2:
            if metrics.findings_emitted == 0 and metrics.validations >= config.max_failed_validations_before_exhausted:
                return True

    return False


def should_mark_completed(metrics: TermMetrics, config: LifecycleConfig = DEFAULT_CONFIG) -> bool:
    """Determine if a term should be marked as completed (produced build-worthy wedge)."""
    return metrics.build_briefs >= config.min_build_briefs_for_completion


def should_mark_paused(metrics: TermMetrics, config: LifecycleConfig = DEFAULT_CONFIG) -> bool:
    """Determine if a term should be temporarily paused."""
    if metrics.times_searched < config.min_runs_for_assessment:
        return False
    if metrics.low_yield_count >= config.max_low_yield_before_pause:
        return True
    return False


def calculate_quality_score(metrics: TermMetrics) -> float:
    """Calculate a quality score for a term based on its metrics."""
    score = 0.0

    # Positive signals
    if metrics.prototype_candidates > 0:
        score += 0.4
    if metrics.build_briefs > 0:
        score += 0.3
    if metrics.passes > 0:
        score += 0.2
    if metrics.avg_validation_score > 0:
        score += min(metrics.avg_validation_score * 0.2, 0.2)

    # Negative signals
    if metrics.times_searched > 0:
        # Penalize if no findings despite runs
        if metrics.findings_emitted == 0:
            score -= 0.2
        # Penalize high screening-out rate
        if metrics.screened_out > 0:
            screening_rate = metrics.screened_out / (metrics.screened_out + metrics.findings_emitted + 1)
            score -= screening_rate * 0.3
        # Penalize low yield
        score -= min(metrics.low_yield_count * 0.1, 0.3)
        # Penalize noise
        score -= min(metrics.noisy_count * 0.1, 0.2)
        # Penalize thin validation trap
        score -= min(metrics.thin_validation_count * 0.1, 0.2)

    return max(0.0, min(1.0, score))


def calculate_specificity_score(term_value: str, opportunity_data: Optional[dict] = None) -> float:
    """Calculate specificity score for a term.

    Returns 0-1 score based on how specific/niche the term is.
    Higher scores = more specific, buildable problems.
    Lower scores = vague, generic abstractions.
    """
    term = term_value.lower()

    # Check for vague bucket patterns (penalize)
    vague_matches = sum(1 for pattern in VAGUE_BUCKET_PATTERNS if pattern in term)
    if vague_matches > 0:
        base_score = 0.2
    else:
        base_score = 0.5

    # Check for specificity keywords (bonus)
    specificity_bonus = sum(0.1 for keyword in SPECIFICITY_KEYWORDS if keyword in term)

    # Check for consequence keywords (bonus)
    consequence_bonus = sum(0.08 for keyword in CONSEQUENCE_KEYWORDS if keyword in term)

    # Penalize very short terms (likely too generic)
    if len(term) < 15:
        base_score *= 0.7

    # Check for specific entity patterns (bonus)
    if any(c.isdigit() for c in term):  # Contains numbers = specific
        base_score += 0.1
    if "$" in term or "fee" in term or "cost" in term:  # Money = specific
        base_score += 0.1

    score = base_score + specificity_bonus + consequence_bonus
    return max(0.0, min(1.0, score))


def calculate_consequence_score(term_value: str, opportunity_data: Optional[dict] = None) -> float:
    """Calculate consequence score for a term.

    Returns 0-1 score based on how consequence-heavy the term is.
    Higher scores = real business consequences (costs, errors, delays).
    Lower scores = vague annoyances.
    """
    term = term_value.lower()

    # Check for consequence keywords
    consequence_matches = sum(0.15 for keyword in CONSEQUENCE_KEYWORDS if keyword in term)
    base_score = 0.3 + consequence_matches

    # Check for vague/soft terms (penalize)
    soft_terms = ["annoying", "frustrat", "tedious", "time sink", "pain point", "struggle", "hard to"]
    soft_matches = sum(-0.1 for soft in soft_terms if soft in term)

    # Check for hard business terms (bonus)
    hard_terms = ["error", "mistake", "lost", "missing", "failed", "penalty", "fee", "overcharge", "risk", "liability", "breach"]
    hard_matches = sum(0.12 for hard in hard_terms if hard in term)

    score = base_score + soft_matches + hard_matches
    return max(0.0, min(1.0, score))


def calculate_platform_native_score(term_value: str) -> float:
    """Calculate platform-native score for a term.

    Returns 0-1 score based on how likely the term is to yield
    plugin/add-on/microSaaS opportunities.
    """
    term = term_value.lower()

    # Check for platform patterns
    platform_matches = 0
    for platform, patterns in PLATFORM_PATTERNS.items():
        if any(p in term for p in patterns):
            platform_matches += 0.25

    # Check for plugin/add-on keywords
    plugin_matches = sum(0.15 for keyword in PLUGIN_FIT_KEYWORDS if keyword in term)

    score = platform_matches + plugin_matches
    return min(1.0, score)


def calculate_plugin_fit_score(term_value: str, opportunity_data: Optional[dict] = None) -> float:
    """Calculate plugin/add-on/microSaaS fit score.

    Returns 0-1 score based on how buildable as a plugin/add-on the term is.
    """
    term = term_value.lower()

    # Plugin keywords
    plugin_keywords = sum(0.2 for kw in PLUGIN_FIT_KEYWORDS if kw in term)

    # Platform keywords
    platform_keywords = sum(0.15 for patterns in PLATFORM_PATTERNS.values() for p in patterns if p in term)

    # Specific narrow problems (good for plugins)
    narrow_problems = ["error", "sync", "duplicate", "missing", "reminder", "invoice", "payment", "approval"]
    narrow_matches = sum(0.1 for np in narrow_problems if np in term)

    score = plugin_keywords + platform_keywords + narrow_matches
    return min(1.0, score)


def calculate_wedge_quality_score(
    term_value: str,
    specificity: float,
    consequence: float,
    platform_native: float,
    plugin_fit: float,
) -> float:
    """Calculate overall wedge quality score.

    Combines specificity, consequence, platform-native, and plugin fit
    to produce a single score for ranking terms.
    """
    # Weighted combination:
    # - Specificity: 25% - must be sharp, not vague
    # - Consequence: 20% - must have real business impact
    # - Platform-native: 25% - must fit plugin/add-on shape
    # - Plugin-fit: 30% - must be buildable as microSaaS

    score = (
        specificity * 0.25
        + consequence * 0.20
        + platform_native * 0.25
        + plugin_fit * 0.30
    )

    return max(0.0, min(1.0, score))


def is_vague_bucket(term_value: str) -> bool:
    """Check if a term is a vague bucket pattern."""
    term = term_value.lower()
    return any(pattern in term for pattern in VAGUE_BUCKET_PATTERNS)


def get_platform_from_term(term_value: str) -> Optional[str]:
    """Extract platform from term if present."""
    term = term_value.lower()
    for platform, patterns in PLATFORM_PATTERNS.items():
        if any(p in term for p in patterns):
            return platform
    return None


def compute_next_state(
    current_state: str,
    metrics: TermMetrics,
    config: LifecycleConfig = DEFAULT_CONFIG,
) -> tuple[str, str]:
    """Compute the next state for a term based on its metrics.

    Returns (new_state, reason) tuple.
    """
    if current_state == "banned":
        return ("banned", "manually banned - no auto-transition")

    if current_state == "completed":
        return ("completed", "already completed - no auto-transition")

    # From new state
    if current_state == "new":
        if metrics.times_searched > 0:
            return ("active", "first use")
        return ("new", "not yet searched")

    # From active or used state - check for promotion
    if current_state in ("active", "used"):
        if should_promote_to_high_performing(metrics, config):
            return ("high_performing", "repeated success - promoting to high_performing")
        if should_mark_completed(metrics, config):
            return ("completed", "produced build-worthy wedge")
        if should_demote_to_weak(metrics, config):
            return ("weak", "repeated low-yield/noisy output - demoting to weak")
        if should_mark_exhausted(metrics, current_state, config):
            return ("exhausted", "consistent failure after many runs - marking exhausted")
        return (current_state, "no state change needed")

    # From high_performing - check for completion or demotion
    if current_state == "high_performing":
        if should_mark_completed(metrics, config):
            return ("completed", "produced build-worthy wedge")
        # Stay high_performing unless things go really bad
        if should_mark_exhausted(metrics, current_state, config):
            return ("exhausted", "even high-performer exhausted after consistent failure")
        return ("high_performing", "still high-performing")

    # From weak state - check for pause or exhaustion
    if current_state == "weak":
        if should_mark_exhausted(metrics, current_state, config):
            return ("exhausted", "too many weak runs - marking exhausted")
        if should_mark_paused(metrics, config):
            return ("paused", "pausing due to poor performance")
        return ("weak", "still weak but not yet exhausted")

    # From paused state - auto-reactivate after cooldown if metrics suggest viability
    if current_state == "paused":
        cooldown_waves = config.get("lifecycle", {}).get("paused_cooldown_waves", 3)
        waves_since_pause = metrics.get("waves_since_state_change", 0)
        if waves_since_pause >= cooldown_waves:
            return ("active", "auto-reactivated after cooldown period")
        return ("paused", f"paused - {cooldown_waves - waves_since_pause} waves until auto-reactivation")

    # From exhausted state - stays exhausted unless manually reactivated
    if current_state == "exhausted":
        return ("exhausted", "exhausted - requires manual reactivation")

    # Default
    return (current_state, "no transition rule matched")


def format_term_summary(term: dict[str, Any], metrics: Optional[TermMetrics] = None) -> str:
    """Format a human-readable summary of a term."""
    state = term.get("state", "unknown")
    term_type = term.get("term_type", "unknown")
    term_value = term.get("term_value", "")

    parts = [f"{term_type}: '{term_value}'"]
    parts.append(f"state={state}")
    parts.append(f"searched={term.get('times_searched', 0)}x")

    if metrics:
        parts.append(f"findings={metrics.findings_emitted}")
        parts.append(f"validations={metrics.validations}")
        parts.append(f"candidates={metrics.prototype_candidates}")
        parts.append(f"quality={metrics.quality_score:.2f}")
    else:
        parts.append(f"findings={term.get('findings_emitted', 0)}")
        parts.append(f"quality={term.get('quality_score', 0):.2f}")

    return " | ".join(parts)


class TermLifecycleManager:
    """Manager for term lifecycle operations."""

    def __init__(self, db, config: Optional[LifecycleConfig] = None):
        self.db = db
        self.config = config or DEFAULT_CONFIG

    def ensure_term_exists(self, term_type: str, term_value: str) -> None:
        """Ensure a term exists in the database."""
        self.db.upsert_search_term(term_type, term_value, state="new")

    def record_search_run(
        self,
        term_type: str,
        term_value: str,
        *,
        findings_emitted: int = 0,
        validations: int = 0,
        passes: int = 0,
        prototype_candidates: int = 0,
        build_briefs: int = 0,
        screened_out: int = 0,
        low_yield: bool = False,
        noisy: bool = False,
        thin_validation: bool = False,
        avg_validation_score: Optional[float] = None,
        avg_screening_score: Optional[float] = None,
        # Optional quality feedback from downstream
        is_vague: bool = False,
        is_abstraction_collapse: bool = False,
        is_buildable_wedge: bool = False,
        is_platform_native: bool = False,
    ) -> dict[str, Any]:
        """Record a search run and update term state."""
        # Ensure term exists
        self.ensure_term_exists(term_type, term_value)

        # Get current state
        current = self.db.get_search_term(term_type, term_value)
        current_state = current.get("state", "new") if current else "new"

        # Build metrics
        metrics = TermMetrics(
            times_searched=current.get("times_searched", 0) + 1 if current else 1,
            findings_emitted=current.get("findings_emitted", 0) + findings_emitted if current else findings_emitted,
            validations=current.get("validations", 0) + validations if current else validations,
            passes=current.get("passes", 0) + passes if current else passes,
            prototype_candidates=current.get("prototype_candidates", 0) + prototype_candidates if current else prototype_candidates,
            build_briefs=current.get("build_briefs", 0) + build_briefs if current else build_briefs,
            screened_out=current.get("screened_out", 0) + screened_out if current else screened_out,
            low_yield_count=current.get("low_yield_count", 0) + (1 if low_yield else 0) if current else (1 if low_yield else 0),
            noisy_count=current.get("noisy_count", 0) + (1 if noisy else 0) if current else (1 if noisy else 0),
            thin_validation_count=current.get("thin_validation_count", 0) + (1 if thin_validation else 0) if current else (1 if thin_validation else 0),
            avg_validation_score=current.get("avg_validation_score", 0.0) if current else 0.0,
            avg_screening_score=current.get("avg_screening_score", 0.0) if current else 0.0,
        )

        # Calculate quality score
        metrics.quality_score = calculate_quality_score(metrics)

        # Calculate niche quality scores
        specificity = calculate_specificity_score(term_value)
        consequence = calculate_consequence_score(term_value)
        platform_native = calculate_platform_native_score(term_value)
        plugin_fit = calculate_plugin_fit_score(term_value)
        wedge_quality = calculate_wedge_quality_score(term_value, specificity, consequence, platform_native, plugin_fit)

        # Determine next state
        new_state, reason = compute_next_state(current_state, metrics, self.config)

        # Update in database
        self.db.update_search_term_state(term_type, term_value, new_state, notes=reason)
        self.db.update_search_term_metrics(
            term_type,
            term_value,
            times_searched=metrics.times_searched,
            findings_emitted=metrics.findings_emitted,
            validations=metrics.validations,
            passes=metrics.passes,
            prototype_candidates=metrics.prototype_candidates,
            build_briefs=metrics.build_briefs,
            screened_out=metrics.screened_out,
            low_yield_count=metrics.low_yield_count,
            noisy_count=metrics.noisy_count,
            thin_validation_count=metrics.thin_validation_count,
            avg_validation_score=metrics.avg_validation_score,
            avg_screening_score=metrics.avg_screening_score,
            quality_score=metrics.quality_score,
            # Niche quality metrics
            specificity_score=specificity,
            consequence_score=consequence,
            platform_native_score=platform_native,
            plugin_fit_score=plugin_fit,
            wedge_quality_score=wedge_quality,
            vague_bucket_count=current.get("vague_bucket_count", 0) + (1 if is_vague else 0) if current else (1 if is_vague else 0),
            abstraction_collapse_count=current.get("abstraction_collapse_count", 0) + (1 if is_abstraction_collapse else 0) if current else (1 if is_abstraction_collapse else 0),
            buildable_opportunity_count=current.get("buildable_opportunity_count", 0) + (1 if is_buildable_wedge else 0) if current else (1 if is_buildable_wedge else 0),
            platform_native_count=current.get("platform_native_count", 0) + (1 if is_platform_native else 0) if current else (1 if is_platform_native else 0),
        )

        logger.debug(f"Term {term_type}:{term_value} state transition: {current_state} -> {new_state} ({reason})")

        return {
            "term_type": term_type,
            "term_value": term_value,
            "old_state": current_state,
            "new_state": new_state,
            "reason": reason,
            "metrics": {
                "times_searched": metrics.times_searched,
                "findings_emitted": metrics.findings_emitted,
                "validations": metrics.validations,
                "prototype_candidates": metrics.prototype_candidates,
                "quality_score": metrics.quality_score,
                "wedge_quality_score": wedge_quality,
                "specificity_score": specificity,
                "platform_native_score": platform_native,
            },
        }

    def get_available_terms(self, term_type: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get terms available for discovery (not exhausted/banned/paused)."""
        return self.db.get_active_terms(term_type, limit)

    def get_terms_for_expansion(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms that are candidates for expansion (high_performing, active).

        Now prioritizes by wedge_quality_score for better plugin/add-on fit.
        """
        return self.db.get_high_performing_terms(term_type, limit)

    def get_terms_for_expansion_by_wedge_quality(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms sorted by wedge quality (best plugin/add-on fit)."""
        return self.db.get_terms_by_wedge_quality(term_type, limit)

    def get_terms_by_specificity(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms sorted by specificity (sharpest niches)."""
        return self.db.get_terms_by_specificity(term_type, limit)

    def get_terms_by_platform_native(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms sorted by platform-native yield."""
        return self.db.get_terms_by_platform_native(term_type, limit)

    def get_abstraction_collapse_terms(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms most responsible for abstraction collapse."""
        return self.db.get_abstraction_collapse_terms(term_type, limit)

    def get_buildable_terms(self, term_type: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get terms most responsible for buildable plugin/add-on wedges."""
        return self.db.get_buildable_terms(term_type, limit)

    def filter_terms_for_discovery(
        self,
        terms: list[dict[str, Any]],
        exclude_states: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Filter terms to exclude those in specified states."""
        exclude_states = exclude_states or ["exhausted", "paused", "banned"]
        return [t for t in terms if t.get("state") not in exclude_states]

    def ban_term(self, term_type: str, term_value: str, reason: str = "manual ban") -> None:
        """Manually ban a term."""
        self.db.update_search_term_state(term_type, term_value, "banned", notes=reason)
        logger.info(f"Banned term {term_type}:{term_value} - {reason}")

    def reactivate_term(self, term_type: str, term_value: str, reason: str = "manual reactivation") -> None:
        """Manually reactivate a paused/exhausted term."""
        self.db.update_search_term_state(term_type, term_value, "active", notes=reason)
        logger.info(f"Reactivated term {term_type}:{term_value} - {reason}")

    def complete_term(self, term_type: str, term_value: str, reason: str = "manual completion") -> None:
        """Manually mark a term as completed."""
        self.db.update_search_term_state(term_type, term_value, "completed", notes=reason)
        logger.info(f"Completed term {term_type}:{term_value} - {reason}")

    def reset_term(self, term_type: str, term_value: str) -> None:
        """Reset a term to new state and clear metrics."""
        import time as time_module

        now = time_module.time()
        conn = self.db._get_connection()
        conn.execute(
            """UPDATE discovery_search_terms SET
            state = 'new',
            times_searched = 0,
            findings_emitted = 0,
            validations = 0,
            passes = 0,
            prototype_candidates = 0,
            build_briefs = 0,
            screened_out = 0,
            low_yield_count = 0,
            noisy_count = 0,
            thin_validation_count = 0,
            avg_validation_score = 0,
            avg_screening_score = 0,
            quality_score = 0,
            updated_ts = ?
            WHERE term_type = ? AND term_value = ?""",
            (now, term_type, term_value),
        )
        conn.commit()
        logger.info(f"Reset term {term_type}:{term_value}")