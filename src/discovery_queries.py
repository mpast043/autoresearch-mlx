"""Shared query defaults for discovery lanes."""

from __future__ import annotations

from typing import Any

DEFAULT_REDDIT_PROBLEM_SUBREDDITS = ["accounting", "smallbusiness", "ecommerce", "shopify", "EtsySellers", "sysadmin"]
DEFAULT_REDDIT_PROBLEM_KEYWORDS = [
    "manual reconciliation",
    "spreadsheet reconciliation process",
    "bank reconciliation spreadsheet workflow",
    "month end close spreadsheet",
    "invoice reminder spreadsheet workflow",
    "manual handoff workflow",
    "sales channel reconciliation spreadsheet",
    "channel profitability reporting spreadsheet",
    "order level reconciliation spreadsheet",
    "bank deposit reconciliation spreadsheet",
    "returns workflow spreadsheet",
    "supplier data spreadsheet workflow",
    "pdf collaboration version control",
    '"order received" "label printed" whatsapp spreadsheet',
    "which spreadsheet is latest",
    "held together by spreadsheets",
    "copy paste workflow",
]
DEFAULT_REDDIT_SUCCESS_KEYWORDS: list[str] = []
CURATED_OPERATOR_SUBREDDITS = [
    "accounting",
    "smallbusiness",
    "ecommerce",
    "shopify",
    "EtsySellers",
]
CURATED_OPERATOR_KEYWORDS = [
    "manual reconciliation",
    "spreadsheet reconciliation process",
    "month end close spreadsheet",
    "bank reconciliation spreadsheet workflow",
    "sales channel reconciliation spreadsheet",
    "channel profitability reporting spreadsheet",
    "invoice reminder spreadsheet workflow",
    "order level reconciliation spreadsheet",
    "pdf collaboration version control",
    "manual handoff workflow",
    "bank deposit reconciliation spreadsheet",
    "returns workflow spreadsheet",
    "supplier data spreadsheet workflow",
    '"order received" "label printed" whatsapp spreadsheet',
    "which spreadsheet is latest",
]
META_REDDIT_SUBREDDITS = [
    "projectmanagement",
    "automation",
    "productivity",
    "indiehackers",
    "Entrepreneur",
    "SaaS",
    "startups",
]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _merge_unique(items: list[str]) -> list[str]:
    merged: list[str] = []
    for item in items:
        if item and item not in merged:
            merged.append(item)
    return merged


def _prioritize_subreddits(subreddits: list[str]) -> list[str]:
    preferred = [subreddit for subreddit in CURATED_OPERATOR_SUBREDDITS if subreddit in subreddits]
    remaining = [subreddit for subreddit in subreddits if subreddit not in preferred]
    meta = [subreddit for subreddit in remaining if subreddit in META_REDDIT_SUBREDDITS]
    neutral = [subreddit for subreddit in remaining if subreddit not in meta]
    return [*preferred, *neutral, *meta]


def reddit_problem_subreddits(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    subreddits = _string_list(reddit_config.get("problem_subreddits")) or _string_list(
        reddit_config.get("subreddits")
    )
    merged = _merge_unique([*(subreddits or list(DEFAULT_REDDIT_PROBLEM_SUBREDDITS)), *CURATED_OPERATOR_SUBREDDITS])
    return _prioritize_subreddits(merged)


def reddit_discovery_subreddits(config: dict[str, Any] | None = None) -> list[str]:
    """Subreddits for problem discovery: optional ``use_r_all`` searches all of Reddit (``r/all``)."""
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    if bool(reddit_config.get("use_r_all")):
        return ["all"]
    return reddit_problem_subreddits(config)


def reddit_problem_keywords(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    keywords = _string_list(reddit_config.get("problem_keywords")) or _string_list(
        reddit_config.get("keywords")
    )
    return _merge_unique([*(keywords or list(DEFAULT_REDDIT_PROBLEM_KEYWORDS)), *CURATED_OPERATOR_KEYWORDS])


def reddit_success_keywords(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    keywords = _string_list(reddit_config.get("success_keywords"))
    return keywords or list(DEFAULT_REDDIT_SUCCESS_KEYWORDS)
