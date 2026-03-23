"""Shared query defaults for discovery lanes."""

from __future__ import annotations

from typing import Any

DEFAULT_REDDIT_PROBLEM_SUBREDDITS = ["smallbusiness", "EtsySellers", "sysadmin"]
DEFAULT_REDDIT_PROBLEM_KEYWORDS = [
    "annoying manual task",
    "manual process",
    "held together by spreadsheets",
    "merging reports manually",
    "spreadsheet workaround",
    "manual reconciliation",
    "workaround",
    "copy paste workflow",
]
DEFAULT_REDDIT_SUCCESS_KEYWORDS: list[str] = []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def reddit_problem_subreddits(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    subreddits = _string_list(reddit_config.get("problem_subreddits")) or _string_list(
        reddit_config.get("subreddits")
    )
    return subreddits or list(DEFAULT_REDDIT_PROBLEM_SUBREDDITS)


def reddit_problem_keywords(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    keywords = _string_list(reddit_config.get("problem_keywords")) or _string_list(
        reddit_config.get("keywords")
    )
    return keywords or list(DEFAULT_REDDIT_PROBLEM_KEYWORDS)


def reddit_success_keywords(config: dict[str, Any] | None = None) -> list[str]:
    config = config or {}
    reddit_config = config.get("discovery", {}).get("reddit", {})
    keywords = _string_list(reddit_config.get("success_keywords"))
    return keywords or list(DEFAULT_REDDIT_SUCCESS_KEYWORDS)
