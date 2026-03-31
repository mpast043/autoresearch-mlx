"""Scoring module - validation scoring and decision logic."""

from __future__ import annotations

from typing import Any, Optional


def score_market_fit(evidence: dict[str, Any]) -> float:
    """Score market fit from evidence (0-1)."""
    score = 0.0

    # Revenue/cost mentions
    if evidence.get("mentions_revenue"):
        score += 0.3
    if evidence.get("mentions_cost"):
        score += 0.2

    # User count indicators
    user_count = evidence.get("user_count", 0)
    if user_count > 1000:
        score += 0.3
    elif user_count > 100:
        score += 0.2
    elif user_count > 10:
        score += 0.1

    # Frequency indicators
    if evidence.get("frequent_usage"):
        score += 0.2

    return min(1.0, score)


def score_technical_fit(evidence: dict[str, Any]) -> float:
    """Score technical fit from evidence (0-1)."""
    score = 0.0

    # Specificity of problem
    if evidence.get("specific_problem"):
        score += 0.4

    # Technical complexity indicator
    if evidence.get("technical_complexity"):
        score += 0.3

    # Current solution exists
    if evidence.get("has_current_solution"):
        score += 0.2

    # Willingness to pay signal
    if evidence.get("willingness_to_pay"):
        score += 0.1

    return min(1.0, score)


def score_distribution_fit(evidence: dict[str, Any]) -> float:
    """Score distribution fit from evidence (0-1)."""
    score = 0.0

    # Channel indicators
    if evidence.get("has_distribution_channel"):
        score += 0.4

    # Community size
    community_size = evidence.get("community_size", 0)
    if community_size > 10000:
        score += 0.3
    elif community_size > 1000:
        score += 0.2

    # Accessibility
    if evidence.get("accessible_segment"):
        score += 0.3

    return min(1.0, score)


def compute_composite_score(market: float, technical: float, distribution: float, weights: Optional[dict[str, float]] = None) -> float:
    """Compute weighted composite score."""
    if weights is None:
        weights = {"market": 0.40, "technical": 0.35, "distribution": 0.25}

    return (
        market * weights.get("market", 0.4)
        + technical * weights.get("technical", 0.35)
        + distribution * weights.get("distribution", 0.25)
    )


def make_decision(composite_score: float, promotion_threshold: float = 0.65, park_threshold: float = 0.35) -> str:
    """Make promote/park/kill decision from composite score."""
    if composite_score >= promotion_threshold:
        return "promote"
    elif composite_score <= park_threshold:
        return "kill"
    else:
        return "park"


def score_recurrence(documents: list[dict[str, Any]]) -> float:
    """Score recurrence evidence from documents."""
    if not documents:
        return 0.0

    recurrence_count = 0
    for doc in documents:
        text = f"{doc.get('title', '')} {doc.get('snippet', '')}".lower()
        if any(kw in text for kw in ["keep having", "every time", "over and over", "repeatedly"]):
            recurrence_count += 1

    return min(1.0, recurrence_count / 3)


def score_corroboration(matches: list[dict[str, Any]], required: int = 2) -> float:
    """Score corroboration from multiple sources."""
    if not matches:
        return 0.0

    # Weight by source diversity
    sources = set(m.get("source") for m in matches if m.get("source"))
    diversity_bonus = min(0.3, len(sources) * 0.1)

    # Base score from match count
    count_score = min(0.7, len(matches) / required * 0.7)

    return min(1.0, count_score + diversity_bonus)