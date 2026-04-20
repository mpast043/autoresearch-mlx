"""Classification module - signal classification and keyword matching."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.source_patterns import PAIN_KEYWORDS, contains_phrase

logger = logging.getLogger(__name__)


# Classification constants moved from research_tools.py
AI_TOOL_KEYWORDS = [
    "openai",
    "chatgpt",
    "gpt",
    "claude",
    "anthropic",
    "midjourney",
    "stable diffusion",
    "llm",
    "ai",
    "cursor",
    "v0",
    "bolt",
]

VALUE_KEYWORDS = [
    "daily",
    "every day",
    "every week",
    "hours",
    "cost",
    "revenue",
    "customers",
    "pipeline",
    "workflow",
    "automate",
    "save time",
    "save money",
    "productivity",
    "efficiency",
]

RECURRENCE_KEYWORDS = [
    "keep having",
    "happens every",
    "recurring",
    " repeatedly ",
    "ongoing",
    "constant",
    "always",
    "every time",
    "over and over",
    "continuously",
    "frequently",
    "day after day",
    "time and time again",
    "consistently",
]


def contains_ai_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in AI_TOOL_KEYWORDS)


def contains_pain_keyword(text: str) -> bool:
    return any(contains_phrase(text, kw) for kw in PAIN_KEYWORDS)


def contains_value_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in VALUE_KEYWORDS)


def contains_recurrence_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in RECURRENCE_KEYWORDS)


def classify_signal(title: str, body: str = "") -> str:
    """Classify a signal as pain, success, demand, competition, or low_signal."""
    combined = f"{title} {body}"
    has_pain = contains_pain_keyword(combined)
    has_value = contains_value_keyword(combined)
    has_ai = contains_ai_keyword(combined)

    if has_pain and not has_value:
        return "pain_signal"
    elif has_value and not has_pain:
        return "success_signal"
    elif has_ai:
        return "demand_signal"
    elif "competitor" in combined.lower() or "alternative" in combined.lower():
        return "competition_signal"
    else:
        return "low_signal_summary"


async def classify_signal_llm(
    title: str,
    body: str = "",
    *,
    llm_client: Optional[Any] = None,
) -> str:
    """LLM-augmented signal classification for ambiguous cases.

    Falls back to heuristic classify_signal() when LLM is unavailable
    or for unambiguous cases (clear pain or clear success).
    """
    heuristic_result = classify_signal(title, body)

    # Only call LLM for ambiguous cases
    if heuristic_result not in ("low_signal_summary", "competition_signal", "demand_signal"):
        return heuristic_result

    if not llm_client:
        return heuristic_result

    combined = f"{title} {body}"[:800]
    system_prompt = (
        "You are a signal classifier for a problem discovery pipeline. "
        "Classify this text as exactly one of: pain_signal, success_signal, "
        "demand_signal, competition_signal, low_signal_summary. "
        "pain_signal = someone describing a problem, frustration, or pain point. "
        "success_signal = someone describing a positive outcome or solution. "
        "demand_signal = someone asking for or seeking a tool. "
        "competition_signal = discussion of alternatives or competitors. "
        "low_signal_summary = vague or irrelevant content. "
        "Return only the classification label, nothing else."
    )
    user_prompt = f"Classify:\n{combined}"

    try:
        raw = await llm_client.reasoning_agenerate(system_prompt, user_prompt)
        if raw:
            label = raw.strip().lower()
            valid = {"pain_signal", "success_signal", "demand_signal", "competition_signal", "low_signal_summary"}
            if label in valid:
                return label
    except Exception as e:
        logger.warning("LLM signal classification failed: %s", e)

    return heuristic_result
