"""Text processing utilities extracted from research_tools."""

from __future__ import annotations

import re
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse, urlunparse


def compact_text(text: str, limit: int = 500) -> str:
    return " ".join((text or "").split())[:limit]


def slugify(value: str, fallback: str = "product") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return cleaned[:48] or fallback


def domain_for(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def unwrap_search_result_url(url: str) -> str:
    raw = (url or "").strip()
    if raw.startswith("//"):
        raw = f"https:{raw}"
    parsed = urlparse(raw)
    if "duckduckgo.com" in parsed.netloc:
        params = parse_qs(parsed.query)
        uddg = params.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return raw


def normalize_search_url(url: str) -> str:
    raw = unwrap_search_result_url(url)
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = parse_qs(parsed.query)
    keep_query_keys = {"id", "v", "p", "q"}
    kept_items = []
    for key in sorted(query):
        if key in keep_query_keys:
            for value in query[key]:
                kept_items.append((key, value))
    normalized = parsed._replace(
        query="&".join(f"{key}={value}" for key, value in kept_items),
        fragment="",
    )
    return urlunparse(normalized)


def query_phrases(text: str, QUERY_STOPWORDS: set[str]) -> list[str]:
    return [match.group(1).strip().lower() for match in re.finditer(r'"([^"]+)"', text or "") if match.group(1).strip()]


def query_terms(text: str, QUERY_STOPWORDS: set[str]) -> list[str]:
    phrase_spans = [match.span() for match in re.finditer(r'"([^"]+)"', text or "")]
    masked = list(text or "")
    for start, end in phrase_spans:
        for idx in range(start, end):
            masked[idx] = " "
    term_text = "".join(masked).lower()
    seen: list[str] = []
    for token in re.findall(r"[a-z0-9&/-]+", term_text):
        if token in QUERY_STOPWORDS or len(token) <= 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def _query_phrase(text: str, QUERY_STOPWORDS: set[str], *, max_words: int = 6) -> str:
    cleaned = compact_text(re.sub(r"[^a-z0-9\s/-]+", " ", text.lower()), 80)
    tokens = [token for token in cleaned.split() if token not in QUERY_STOPWORDS]
    if len(tokens) < 2:
        return ""
    return '"' + " ".join(tokens[:max_words]) + '"'


def _query_term_span(
    text: str,
    QUERY_STOPWORDS: set[str],
    WEAK_VALIDATION_TERMS: set[str],
    RECURRENCE_NOISE_TERMS: set[str],
    *,
    max_terms: int = 4,
) -> str:
    cleaned = compact_text(re.sub(r"[^a-z0-9\s/-]+", " ", text.lower()), 80)
    terms: list[str] = []
    for token in cleaned.split():
        if token in QUERY_STOPWORDS or token in WEAK_VALIDATION_TERMS or token in RECURRENCE_NOISE_TERMS:
            continue
        if len(token) <= 2:
            continue
        if token not in terms:
            terms.append(token)
    return " ".join(terms[:max_terms])


def _clean_recurrence_text(text: str, *, limit: int = 160) -> str:
    cleaned = compact_text((text or "").lower(), limit)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:v?\d+\.\d+(?:\.\d+)?|#\d+|[a-f0-9]{7,40})\b", " ", cleaned)
    cleaned = re.sub(r"\b(hey everyone|hope [a-z\s]{0,30} well|question for|anyone else|i want to complain|man alive)\b", " ", cleaned)
    cleaned = re.sub(r"\b(contact|logs stored|doctor output|issue checklist|provide as much information as possible)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s&/-]+", " ", cleaned)
    return " ".join(cleaned.split())


def topical_overlap(query: str, title: str, snippet: str, domain: str, QUERY_STOPWORDS: set[str]) -> int:
    haystack = f"{title} {snippet} {domain}".lower()
    term_hits = sum(1 for term in query_terms(query, QUERY_STOPWORDS) if term in haystack)
    phrase_hits = sum(2 for phrase in query_phrases(query, QUERY_STOPWORDS) if phrase in haystack)
    return term_hits + phrase_hits


def url_path(url: str) -> str:
    return urlparse(url).path.lower()


def contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = (text or "").lower()
    normalized_keyword = (keyword or "").lower()
    pattern = rf"\b{re.escape(normalized_keyword)}\b" if " " not in normalized_keyword else re.escape(normalized_keyword)
    return re.search(pattern, normalized_text) is not None


def first_match(patterns: Iterable[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def infer_recurrence_key(text: str) -> str:
    from utils.hashing import normalize_content
    normalized = normalize_content(text)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    stopwords = {
        "the",
        "and",
        "that",
        "with",
        "for",
        "from",
        "this",
        "have",
        "into",
        "your",
        "they",
        "just",
        "need",
        "wish",
        "tool",
        "software",
        "workflow",
        "keep",
        "reliable",
    }
    terms: list[str] = []
    for token in normalized.split():
        if len(token) <= 2 or token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return " ".join(terms[:6])