"""Content hashing utilities for deduplication.

This module provides functions for normalizing content, generating hashes,
and detecting duplicate or similar content for the Discovery agent.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher


def normalize_content(text: str) -> str:
    """Normalize text so equivalent content hashes and compares consistently."""
    if not text:
        return ""

    normalized = text.lower()
    normalized = re.sub(r"https?://\S+", "", normalized)
    normalized = re.sub(r"@\w+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    return normalized


def generate_content_hash(text: str) -> str:
    """Generate a stable SHA256 hash from normalized text."""
    normalized = normalize_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_similar_content(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Return True when two texts are similar after normalization."""
    if not text1 or not text2:
        return text1 == text2

    normalized1 = normalize_content(text1)
    normalized2 = normalize_content(text2)
    similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
    return similarity >= threshold


def find_duplicate_finding(
    text: str,
    existing_texts: list[str],
    threshold: float = 0.8,
) -> bool:
    """Return True when text duplicates or closely matches existing content."""
    if not text or not existing_texts:
        return False

    text_hash = generate_content_hash(text)
    for existing in existing_texts:
        if generate_content_hash(existing) == text_hash:
            return True

    for existing in existing_texts:
        if is_similar_content(text, existing, threshold):
            return True

    return False
