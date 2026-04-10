"""Canonical source-policy keyword and phrase patterns."""

from __future__ import annotations

import re
from typing import Iterable


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

TRANSFERABLE_WORKFLOW_TERMS = [
    "import",
    "export",
    "csv",
    "spreadsheet",
    "invoice",
    "payment",
    "reconciliation",
    "bank",
    "inventory",
    "fulfillment",
    "shipment",
    "label",
    "order",
    "vendor",
    "customer",
    "backup",
    "restore",
    "recovery",
    "sync",
    "migration",
    "audit",
    "compliance",
    "handoff",
    "document",
    "contract",
]

TRANSFERABLE_FAILURE_TERMS = [
    "duplicate",
    "duplicating",
    "duplicates",
    "mismatch",
    "out of sync",
    "missing",
    "broken",
    "fails",
    "failed",
    "error",
    "wrong",
    "corrupt",
    "not matching",
    "still showing",
    "deleted",
    "late",
    "delay",
    "delayed",
    "unreachable",
    "fallback",
]

YOUTUBE_LOW_SIGNAL_PATTERNS = [
    "best shopify apps",
    "shopify app recommendations",
    "shopify app vs",
    "top 10",
    "top tools",
    "must-have apps",
    "must have apps",
    "roundup",
    "review roundup",
    "app roundup",
    "alternatives",
]

DATA_TOUCHPOINT_TERMS = [
    "import",
    "export",
    "csv",
    "file",
    "data",
    "record",
    "transaction",
]


def contains_phrase(text: str, phrase: str) -> bool:
    lowered = (text or "").lower()
    needle = (phrase or "").lower().strip()
    if not needle:
        return False
    if re.fullmatch(r"[\w\s'-]+", needle):
        pattern = re.escape(needle).replace(r"\ ", r"\s+")
        return re.search(rf"(?<!\w){pattern}(?!\w)", lowered) is not None
    return needle in lowered


def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)
