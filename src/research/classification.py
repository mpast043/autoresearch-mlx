"""Classification module - signal classification and keyword matching."""

from __future__ import annotations

from typing import Iterable


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

PAIN_KEYWORDS = [
    "manual",
    "frustrating",
    "annoying",
    "takes too long",
    "waste time",
    "repetitive",
    "error-prone",
    "wish there was",
    "need a way",
    "need a better way",
    "hate that",
    "struggle",
    "can't find",
    "too expensive",
    "overkill",
    "problem",
    "issue",
    "worried",
    "time consuming",
    "negative review",
    "destroyed my",
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
    text_lower = text.lower()
    return any(kw in text_lower for kw in PAIN_KEYWORDS)


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