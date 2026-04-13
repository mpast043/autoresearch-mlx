"""Shared query defaults for discovery lanes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_PROBLEM_SUBREDDITS = ["accounting", "smallbusiness", "ecommerce", "shopify", "EtsySellers", "sysadmin"]
DEFAULT_REDDIT_PROBLEM_KEYWORDS = [
    # Mismatch-based keywords - platform specific failures
    "QuickBooks invoice does not match payment",
    "Shopify orders duplicated after import",
    "Excel formulas break when copying sheet",
    "CSV import creates duplicate entries",
    "Spreadsheet version conflict team",
    "Transaction mismatch reconciliation",
    "Bank statement doesn't match invoices",
    "Invoice payment mismatch error",
    "Data duplicated after CSV import",
    "Entries not matching after import",
    "Wrong totals after spreadsheet import",
    "Spreadsheet copy paste error",
    "Manual matching error problem",
    "Reconciliation fails after import",
    "Payment not matching invoice",
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
    # Mismatch-based keywords
    "Invoice does not match payment",
    "Orders duplicated after import",
    "Formulas break when copying",
    "CSV import creates duplicates",
    "Version conflict spreadsheet",
    "Transaction mismatch",
    "Bank statement doesn't match",
    "Data duplicated after import",
    "Wrong totals after import",
    "Copy paste error spreadsheet",
    "Reconciliation fails after import",
    "Payment not matching invoice",
    # Operator workflow keywords
    "month end close spreadsheet",
    "sales channel reconciliation spreadsheet",
    "invoice reminder spreadsheet workflow",
    "pdf collaboration version control",
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


def _string_list(value: list) -> list[str]:
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

    # Try to read from governed config first
    governed_path = Path("data/next_wave_governed_final.json")
    if governed_path.exists():
        try:
            governed = json.loads(governed_path.read_text())
            subs = governed.get("subreddits", [])
            if subs:
                logger.info(f"Using governed subreddits: {subs}")
                return subs
        except Exception as exc:
            logger.debug("Failed to load governed subreddits: %s", exc)

    # Fall back to config.yaml
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

    # Try to read from governed config first
    governed_path = Path("data/next_wave_governed_final.json")
    if governed_path.exists():
        try:
            governed = json.loads(governed_path.read_text())
            kw = governed.get("keywords", [])
            if kw:
                logger.info(f"Using governed keywords: {kw}")
                return kw
        except Exception as exc:
            logger.debug("Failed to load governed keywords: %s", exc)

    # Fall back to config.yaml
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
