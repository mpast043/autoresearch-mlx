"""Shared research, extraction, and tooling helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import requests
import aiohttp
from bs4 import BeautifulSoup
from src.source_patterns import (
    PAIN_KEYWORDS as CANONICAL_PAIN_KEYWORDS,
    YOUTUBE_LOW_SIGNAL_PATTERNS as CANONICAL_YOUTUBE_LOW_SIGNAL_PATTERNS,
    contains_any_phrase,
)

try:
    from mcp_fetcher import MCPFetcher
except Exception:  # pragma: no cover - supports package and direct module usage
    try:
        from src.mcp_fetcher import MCPFetcher
    except Exception:  # pragma: no cover - optional runtime dependency
        MCPFetcher = None

try:
    from reddit_bridge import RedditBridgeClient
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_bridge import RedditBridgeClient

try:
    from reddit_transport import RedditTransport
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.reddit_transport import RedditTransport

try:
    from search_models import SearchDocument
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.search_models import SearchDocument

try:
    from discovery_queries import (
        reddit_discovery_subreddits,
        reddit_problem_keywords,
        reddit_success_keywords,
    )
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.discovery_queries import (
        reddit_discovery_subreddits,
        reddit_problem_keywords,
        reddit_success_keywords,
    )

try:
    from github_sources import GitHubIssueAdapter
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.github_sources import GitHubIssueAdapter

try:
    from review_sources import ShopifyAppReviewAdapter, WordPressPluginReviewAdapter
except Exception:  # pragma: no cover - supports package and direct module usage
    from src.review_sources import ShopifyAppReviewAdapter, WordPressPluginReviewAdapter

try:
    from utils.hashing import normalize_content
except Exception:  # pragma: no cover - supports package and direct module usage
    try:
        from src.utils.hashing import normalize_content
    except Exception:  # pragma: no cover - final fallback for recovery
        def normalize_content(text: str) -> str:
            return " ".join((text or "").lower().split())

# Re-export utilities for backward compatibility
from src.utils.search_plan import CorroborationPlan, CorroborationAction, DiscoveryQueryPlan, SkillAudit
from src.utils.text import (
    compact_text,
    contains_keyword,
    domain_for,
    first_match,
    infer_recurrence_key,
    normalize_search_url,
    query_phrases,
    query_terms,
    topical_overlap,
    unwrap_search_result_url,
    url_path,
    slugify,
    _query_phrase,
    _query_term_span,
    _clean_recurrence_text,
)


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

_RECURRENCE_SPLIT_TERM_REPAIRS: tuple[tuple[str, str], ...] = (
    (r"\breconcil\s+iation\b", "reconciliation"),
)


def clean_extracted_web_text(text: str, *, limit: int | None = None) -> str:
    """Normalize fetched web text before hashing or atom extraction.

    Reader/fetcher outputs occasionally collapse adjacent inline elements into a
    single token (for example ``spreadsheetprogram``). We repair the common
    boundary cases here so downstream source policy and atom extraction read the
    page more like a human would.
    """
    cleaned = str(text or "")
    if not cleaned:
        return ""

    cleaned = (
        cleaned.replace("\xa0", " ")
        .replace("\u200b", " ")
        .replace("\u200c", " ")
        .replace("\u200d", " ")
        .replace("\ufeff", " ")
    )
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", cleaned)
    for token in WEB_TEXT_JOIN_BREAKERS:
        cleaned = re.sub(
            rf"(?i)\b({re.escape(token)})(?=[a-z]{{4,}})",
            r"\1 ",
            cleaned,
        )
    cleaned = " ".join(cleaned.split())
    if limit is not None:
        return compact_text(cleaned, limit)
    return cleaned

PAIN_KEYWORDS = list(CANONICAL_PAIN_KEYWORDS)

VALUE_KEYWORDS = [
    "daily",
    "every day",
    "every week",
    "hours",
    "cost",
    "revenue",
    "customers",
    "pipeline",
    "leads",
    "ops",
    "sales",
    "billing",
    "support",
]

SOFTWARE_DOMAINS = {
    "g2.com",
    "capterra.com",
    "producthunt.com",
    "github.com",
    "chromewebstore.google.com",
    "apps.shopify.com",
    "wordpress.org",
}

REDDIT_MODES = {"bridge_with_fallback", "bridge_only", "public_direct"}

NOISY_SEARCH_DOMAINS = {
    "wikipedia.org",
    "en.wikipedia.org",
    "m.wikipedia.org",
    "grokipedia.com",
    "merriam-webster.com",
    "dictionary.cambridge.org",
    "collinsdictionary.com",
    "thefreedictionary.com",
    "wordreference.com",
    "youtube.com",
    "www.youtube.com",
    "support.google.com",
    "redditmedia.com",
}

SEARCH_ENGINE_RESULT_DOMAINS = {
    "duckduckgo.com",
    "html.duckduckgo.com",
    "bing.com",
    "www.bing.com",
    "google.com",
    "www.google.com",
}

STACKEXCHANGE_SITE_ALIASES = {
    "stackoverflow.com": "stackoverflow",
    "superuser.com": "superuser",
    "webapps.stackexchange.com": "webapps",
}

QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "there",
    "about",
    "how",
    "what",
    "when",
    "where",
    "which",
    "while",
    "using",
    "used",
    "does",
    "dont",
    "doesnt",
    "into",
    "after",
    "before",
    "have",
    "has",
    "had",
    "more",
    "some",
    "just",
    "than",
    "then",
    "them",
    "they",
    "been",
    "were",
    "want",
    "join",
    "complain",
}

DISCOVERY_QUERY_GENERIC_TERMS = {
    "manual",
    "workflow",
    "process",
    "tool",
    "tools",
    "software",
    "issue",
    "issues",
    "problem",
    "problems",
    "task",
    "tasks",
    "steps",
    "errors",
    "error",
    "forum",
    "forums",
    "review",
    "reviews",
    "latest",
    "best",
    "trying",
    "trying_to",
}

DISCOVERY_QUERY_ANCHOR_TERMS = [
    "reconciliation",
    "spreadsheet",
    "audit",
    "compliance",
    "export",
    "handoff",
    "excel",
    "csv",
    "shopify",
    "airtable",
    "notion",
]

WEAK_VALIDATION_TERMS = {
    "issue",
    "problem",
    "question",
    "request",
    "feature",
    "discussion",
    "thing",
    "stuff",
    "app",
    "plugin",
    "tool",
    "software",
    "solution",
}

RECURRENCE_NOISE_TERMS = {
    "anyone",
    "anybody",
    "else",
    "today",
    "yesterday",
    "please",
    "thanks",
    "thank",
    "hello",
    "help",
    "question",
    "thread",
    "post",
    "comment",
    "reddit",
    "github",
    "shopify",
    "wordpress",
    "apps",
    "plugin",
    "app",
}

COMPETITOR_INTENT_HINTS = [
    "software",
    "tool",
    "platform",
    "automation",
    "app",
    "saas",
    "solution",
    "pricing",
    "integrations",
    "workflow",
]

NON_PROBLEM_DISCOVERY_PATTERNS = [
    "i did it",
    "now makes",
    "mrr",
    "arr",
    "launched",
    "launching",
    "open-source company",
    "looking for solutions",
    "looking for solution",
    "any recommendations",
    "best and not too expensive",
    "what do you use",
    "planning to start",
    "resume help",
    "resume roast",
    "review my resume",
    "landed a senior accounting",
    "career advice",
    "ecommerce industry news recap",
    "this week's top ecommerce news stories",
]

LOW_VALUE_REDDIT_PREFILTER_PATTERNS = [
    "what are you using for",
    "what's the best",
    "what is the best",
    "best time to hire a bookkeeper",
    "when should i hire a bookkeeper",
    "hire a bookkeeper for a growing small business",
    "best inventory management app",
    "best software for",
    "best app for",
    "unbiased review",
    "automated software still makes us do",
    "what's the most outdated process",
    "what is the most outdated process",
    "how are you guys managing",
    "how are you reducing repetitive admin work",
    "how do you manage inventory as you scale",
    "how do you handle importing old data",
    "price / competitor monitoring",
    "good fit for me",
    "review my resume",
    "resume review",
    "resume roast",
    "critique my resume",
    "wholesale account owes me",
    "revenue concentration",
    "client concentration",
    "biggest client",
    "major client",
    "hiring first sales",
    "structuring msp business",
    "se fue un cliente",
]

LOW_VALUE_REDDIT_TITLE_PREFIXES = [
    "how do you handle",
    "how are you",
    "what's the one task",
    "what is the one task",
    "is anyone else's",
    "anyone else",
    "for those of you",
    "has anyone actually",
    "which ",
    "curious how",
    "looking for ",
]

REDDIT_PREFILTER_FAILURE_TERMS = [
    "still showing",
    "duplicate",
    "duplicates",
    "duplicated",
    "mismatch",
    "not matching",
    "does not match",
    "wrong",
    "missing",
    "break",
    "broken",
    "out of sync",
    "delay",
    "delayed",
    "failed",
    "fails",
    "error",
]

REDDIT_PREFILTER_OBJECT_TERMS = [
    "order",
    "orders",
    "analytics",
    "invoice",
    "invoices",
    "payment",
    "payments",
    "csv",
    "import",
    "export",
    "inventory",
    "report",
    "reports",
    "spreadsheet",
    "reconciliation",
]

WORKAROUND_SIGNAL_TERMS = [
    "manual",
    "manually",
    "spreadsheet",
    "spreadsheets",
    "copy paste",
    "copy and paste",
    "csv",
    "glue code",
    "workaround",
    "fallback",
    "script",
]

FREQUENCY_SIGNAL_TERMS = [
    "every day",
    "daily",
    "every week",
    "weekly",
    "repeated",
    "repetitive",
    "constantly",
    "every time",
]

COST_SIGNAL_TERMS = [
    "takes too long",
    "hours",
    "cost",
    "expensive",
    "overhead",
    "lost",
    "late",
    "risk",
    "broken",
    "fails",
    "failure",
    "error-prone",
    "downtime",
]

NON_OPERATIONAL_BUSINESS_RISK_TERMS = [
    "biggest client",
    "major client",
    "client concentration",
    "revenue concentration",
    "owes me",
    "wholesale account",
    "hiring first sales",
    "first sales",
    "structuring msp business",
    "good fit for me",
    "resume review",
    "resume roast",
    "review my resume",
    "career advice",
    "se fue un cliente",
]

OPERATIONAL_PROCESS_TERMS = [
    "reconcile",
    "reconciliation",
    "bank deposit",
    "bank deposits",
    "payout",
    "payouts",
    "ledger",
    "month end",
    "month-end",
    "import",
    "imports",
    "export",
    "exports",
    "invoice",
    "invoices",
    "payment",
    "payments",
    "order",
    "orders",
    "inventory",
    "approval",
    "approvals",
    "handoff",
    "handoffs",
    "label",
    "labels",
    "returns",
    "supplier data",
]


def _is_non_operational_business_risk_atom(atom: Optional[Any]) -> bool:
    if atom is None:
        return False
    haystack = normalize_content(
        " ".join(
            [
                getattr(atom, "segment", "") or "",
                getattr(atom, "user_role", "") or "",
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "trigger_event", "") or "",
                getattr(atom, "current_workaround", "") or "",
                getattr(atom, "cost_consequence_clues", "") or "",
                getattr(atom, "current_tools", "") or "",
            ]
        )
    )
    business_risk_pain = any(term in haystack for term in NON_OPERATIONAL_BUSINESS_RISK_TERMS)
    operational_process_pain = any(term in haystack for term in OPERATIONAL_PROCESS_TERMS)
    return business_risk_pain and not operational_process_pain

HELP_URL_PATTERNS = [
    "/help",
    "/support",
    "/docs",
    "/documentation",
    "/hc/",
    "/articles/",
    "/signin",
    "/log-in",
    "/login",
    "/download",
]

BLOG_LIKE_PATTERNS = [
    "blog.",
    "/blog/",
    "/news/",
    "/resources/",
    "/guide/",
    "/learn/",
    "/what-is/",
]

WEB_PROBLEM_REJECT_DOMAIN_TOKENS = [
    "learn.",
    "support.",
    "community.",
    "forum.",
    "g2.",
    "capterra.",
    "getapp.",
    "saashub.",
    "softwareadvice.",
    "trustradius.",
    "slashdot.",
]

WEB_PROBLEM_REJECT_PATH_TOKENS = [
    "/answers/",
    "/forum/",
    "/forums/",
    "/thread/",
    "/threads/",
    "/board/",
    "/blog/",
    "/blogs/",
    "/guide/",
    "/guides/",
    "/tutorial",
    "/article",
    "/articles/",
    "/content/",
    "/discussion",
    "/discussions/",
    "/best-",
    "/alternatives",
    "/compare",
    "/comparison",
    "/reviews/",
    "/top-",
    "/download",
    "/downloads/",
]

WEB_PROBLEM_REJECT_TEXT_PATTERNS = [
    "tutorial:",
    "unlock the secrets",
    "best software",
    "best for",
    "alternatives to",
    "top tools",
    "software directory",
    "read to know",
    "discover how to",
    "discover how",
    "important notice",
    "cookie policy",
    "indispensable tool",
    "navigating version control",
    "risks and challenges",
    "fails accounting teams",
    "solved:",
    "what's new in microsoft excel",
    "whats new in microsoft excel",
    "microsoft excel - download",
    "download hvac duct measurement excel sheet",
    "free download",
    "download now",
    "template gallery",
    "excel templates",
    "spreadsheet templates",
    "product review",
    "editor's rating",
    "editors' rating",
    "features and pricing",
    "pros and cons",
    "powerful spreadsheet",
    "find customizable templates",
    "browse templates",
    "microsoft 365 excel",
    "analyze data",
    "create spreadsheets",
    "duct tape phase",
    "held together with duct tape",
    "profitable business still feels like chaos",
    "not a permanent state",
    "not a character flaw",
    "making good money but feel like their business is held together with duct tape",
]

WEB_TEXT_JOIN_BREAKERS = [
    "spreadsheet",
    "spreadsheets",
    "template",
    "templates",
    "gallery",
    "download",
    "downloads",
    "worksheet",
    "worksheets",
    "software",
    "workflow",
    "invoice",
    "billing",
    "contract",
    "inventory",
    "pricing",
    "service",
    "services",
    "reconciliation",
    "audit",
    "report",
    "reports",
    "excel",
    "ductwork",
]

WEB_PROBLEM_REFERENCE_TEXT_PATTERNS = [
    "3rd edition",
    "third edition",
    "provided excel spreadsheet",
    "design calculations",
    "reference guide",
    "download worksheet",
    "design worksheet",
    "calculator",
]

WEB_PROBLEM_PRACTITIONER_TEXT_PATTERNS = [
    "our team",
    "we spend",
    "we have to",
    "i spend",
    "i have to",
    "manually",
    "every month",
    "every week",
    "ops team",
    "accounting team",
    "merchant",
    "bookkeeper",
    "manager",
]

WEB_PROBLEM_CONTENT_FARM_DOMAINS = {
    "fastercapital.com",
}
LOW_QUALITY_CORROBORATION_DOMAIN_TOKENS = [
    "g2.",
    "capterra.",
    "getapp.",
    "saashub.",
    "softwareadvice.",
    "trustradius.",
    "slashdot.",
]
LOW_QUALITY_CORROBORATION_PATH_TOKENS = [
    "/best-",
    "/alternatives",
    "/compare",
    "/comparison",
    "/reviews/",
    "/top-",
    "/list/",
]
LOW_QUALITY_CORROBORATION_TEXT_PATTERNS = [
    "best software",
    "top tools",
    "alternatives to",
    "software directory",
    "compare the best",
    "top 10",
]
YOUTUBE_LOW_SIGNAL_VIDEO_PATTERNS = list(CANONICAL_YOUTUBE_LOW_SIGNAL_PATTERNS)

HIGH_VALUE_DISCOVERY_QUERY_TERMS = {
    "reddit-problem": {
        "manual reconciliation": 0.7,
        "spreadsheet reconciliation": 0.7,
        "sales tax payment reconciliation": 0.7,
        "manual handoff": 0.65,
        "month end close": 0.7,
        "bank deposit": 0.65,
        "sales channel": 0.7,
        "channel profitability": 0.7,
        "invoice reminder": 0.6,
        "late payment": 0.6,
        "pdf collaboration": 0.6,
        "returns workflow": 0.65,
        "supplier data": 0.6,
        "label printed": 0.65,
    },
    "web-problem": {
        "spreadsheet version confusion": 0.7,
        "manual reconciliation forum": 0.7,
        "manual handoff workflow forum": 0.65,
        "month end close spreadsheet": 0.65,
        "sales channel reconciliation": 0.65,
        "sales tax payment reconciliation": 0.65,
        "channel profitability spreadsheet": 0.65,
        "invoice reminder spreadsheet": 0.6,
        "pdf collaboration version": 0.6,
        "returns workflow spreadsheet": 0.6,
    },
}

LOW_SIGNAL_DISCOVERY_QUERY_TERMS = {
    "reddit-problem": {
        "annoying manual task": 0.45,
        "manual process": 0.3,
        "workaround": 0.25,
        "wish there was a tool": 0.4,
        "which spreadsheet is latest": 0.8,
        "which spreadsheet is latest version": 0.8,
    },
    "web-problem": {
        '"manual process" every day': 0.35,
        '"too expensive" current tool': 0.25,
        "which spreadsheet is latest": 0.8,
        "which spreadsheet is latest version": 0.8,
    },
}

VALIDATION_QUERY_PHRASES = [
    (r"manual data entry", "manual data entry"),
    (r"manual evidence collection", "manual evidence collection"),
    (r"shared doc(?:ument)?", "shared doc"),
    (r"shipping profile", "shipping profile"),
    (r"cost tracking", "cost tracking"),
    (r"\bp&l\b", "p&l"),
    (r"gdpr", "gdpr"),
    (r"hipaa", "hipaa"),
    (r"soc 2", "soc 2"),
    (r"device imaging", "device imaging"),
    (r"image a laptop", "device imaging"),
    (r"backup restore", "backup restore"),
    (r"restore fails", "restore fails"),
    (r"sync fails", "sync fails"),
    (r"manual workaround", "manual workaround"),
]

logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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


def query_phrases(text: str) -> list[str]:
    return [match.group(1).strip().lower() for match in re.finditer(r'"([^"]+)"', text or "") if match.group(1).strip()]


def query_terms(text: str) -> list[str]:
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


def _query_phrase(text: str, *, max_words: int = 6) -> str:
    cleaned = compact_text(re.sub(r"[^a-z0-9\s/-]+", " ", text.lower()), 80)
    tokens = [token for token in cleaned.split() if token not in QUERY_STOPWORDS]
    if len(tokens) < 2:
        return ""
    return '"' + " ".join(tokens[:max_words]) + '"'


def _query_term_span(text: str, *, max_terms: int = 4) -> str:
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
    for pattern, replacement in _RECURRENCE_SPLIT_TERM_REPAIRS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return " ".join(cleaned.split())


def topical_overlap(query: str, title: str, snippet: str, domain: str) -> int:
    haystack = f"{title} {snippet} {domain}".lower()
    term_hits = sum(1 for term in query_terms(query) if term in haystack)
    phrase_hits = sum(2 for phrase in query_phrases(query) if phrase in haystack)
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


class ResearchToolkit:
    """Searches sources, fetches content, and scores validation evidence."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.fetcher = MCPFetcher() if MCPFetcher else None
        self.user_agent = (
            self.config.get("api_keys", {}).get("reddit", {}).get("user_agent")
            or "Mozilla/5.0 (compatible; AutoResearcher/1.0)"
        )
        self.skill_audit_config = self.config.get("security", {}).get("skill_audit", {})
        self.skill_audit_enabled = self.skill_audit_config.get("enabled", True)
        self.skill_min_score = float(self.skill_audit_config.get("minimum_score", 0.55))
        self.project_root = Path(__file__).resolve().parent.parent
        self.skills_root = self.project_root / ".agents" / "skills"
        self.skill_audits = self._audit_installed_skills() if self.skill_audit_enabled else {}
        self.reddit_readonly_script = self._approved_skill_path(
            "openclaw-skills-reddit-readonly",
            "scripts/reddit-readonly.mjs",
        )
        self.reddit_scraper_script = self._approved_skill_path(
            "openclaw-skills-reddit-scraper",
            "scripts/reddit_scraper.py",
        )
        self.youtube_transcript_script = self._approved_skill_path(
            "openclaw-skills-youtube-transcript-generator",
            "scripts/get_transcript.sh",
        )
        self.playwright_skill = self.skill_audits.get("openclaw-skills-playwright-mcp")
        self.github_search_skill = self.skill_audits.get("parcadei-continuous-claude-v3-github-search")
        self.security_testing_skill = self.skill_audits.get("proffesor-for-testing-agentic-qe-security-testing")
        self.node_bin = shutil.which("node")
        self.yt_dlp_bin = shutil.which("yt-dlp")
        self.yt_dlp_exec = self.yt_dlp_bin or str(
            Path.home() / "Library" / "Python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "bin" / "yt-dlp"
        )
        if not Path(self.yt_dlp_exec).exists():
            self.yt_dlp_exec = None
        self.yt_dlp_command = (
            [self.yt_dlp_exec]
            if self.yt_dlp_exec
            else ([sys.executable, "-m", "yt_dlp"] if importlib.util.find_spec("yt_dlp") else None)
        )
        bridge_config = self.config.get("reddit_bridge", {})
        self.reddit_bridge = RedditBridgeClient(bridge_config)
        configured_reddit_mode = str(bridge_config.get("mode", "") or "").strip().lower()
        if configured_reddit_mode in REDDIT_MODES:
            self.reddit_mode = configured_reddit_mode
        elif self.reddit_bridge.enabled:
            self.reddit_mode = "bridge_with_fallback"
        else:
            self.reddit_mode = "public_direct"
        self._search_cache: dict[tuple[str, int, Optional[str], str], list[SearchDocument]] = {}
        self._fetch_cache: dict[str, dict[str, Any]] = {}
        self._recurrence_attempt_cache: dict[str, tuple[list[SearchDocument], dict[str, Any]]] = {}
        self._discovery_feedback: dict[str, dict[str, dict[str, Any]]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl: float = 3600.0  # 1 hour
        self.reddit_transport = RedditTransport(
            config=self.config,
            reddit_bridge=self.reddit_bridge,
            reddit_mode=self.reddit_mode,
            node_bin=self.node_bin,
            readonly_script=self.reddit_readonly_script,
            user_agent=self.user_agent,
            run_json_command=self._run_json_command,
            compact_text=compact_text,
            normalize_query=self._normalize_recurrence_query,
            request_get=lambda *args, **kwargs: requests.get(*args, **kwargs),
            logger=logger,
        )
        self._reddit_search_cache = self.reddit_transport.search_cache
        self._reddit_thread_cache = self.reddit_transport.thread_cache
        self._reddit_metrics = self.reddit_transport.metrics
        self._reddit_validation_seeded_pairs = self.reddit_transport.validation_seeded_pairs
        validation_search = self.config.get("validation", {}).get("search", {})
        self.validation_query_terms = max(8, int(validation_search.get("query_terms", 14)))
        self.validation_recurrence_limit = max(4, int(validation_search.get("recurrence_results", 6)))
        self.validation_competitor_limit = max(6, int(validation_search.get("competitor_results", 8)))
        self.validation_evidence_sample = max(8, int(validation_search.get("evidence_sample", 20)))
        self.validation_recurrence_budget_seconds = float(validation_search.get("recurrence_budget_seconds", 8.0))
        self.validation_competitor_budget_seconds = float(
            validation_search.get("competitor_budget_seconds", 6.0)
        )
        self.search_domain_cap = max(1, int(validation_search.get("max_results_per_domain", 2)))
        self.ddgs_backend = validation_search.get("ddgs_backend", "duckduckgo")
        self.ddgs_extra_results = max(0, int(validation_search.get("ddgs_extra_results", 4)))
        self.result_title_length = max(20, int(validation_search.get("result_title_length", 180)))
        self.result_snippet_length = max(20, int(validation_search.get("result_snippet_length", 320)))
        self.search_user_agent = str(
            validation_search.get("search_user_agent", "Mozilla/5.0 (compatible; AutoResearcher/1.0)")
        )
        self.provider_timeout_recurrence = max(1, float(validation_search.get("provider_timeout_recurrence", 3)))
        self.provider_timeout_competitor = max(1, float(validation_search.get("provider_timeout_competitor", 4)))
        self.provider_timeout_general = max(1, float(validation_search.get("provider_timeout_general", 8)))
        self.request_timeout_recurrence = max(1, float(validation_search.get("request_timeout_recurrence", 4)))
        self.request_timeout_competitor = max(1, float(validation_search.get("request_timeout_competitor", 5)))
        self.request_timeout_general = max(1, float(validation_search.get("request_timeout_general", 12)))
        # YouTube API key
        self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self._session: aiohttp.ClientSession | None = None

        api_keys = self.config.get("api_keys", {}) if isinstance(self.config.get("api_keys", {}), dict) else {}
        github_cfg = api_keys.get("github", {}) if isinstance(api_keys.get("github", {}), dict) else {}
        self.github_adapter = GitHubIssueAdapter(
            search_web=self.search_web,
            fetch_content=self.fetch_content,
            token=str(github_cfg.get("token", "") or os.getenv("GITHUB_TOKEN", "")),
            user_agent=self.user_agent,
        )
        self.wordpress_review_adapter = WordPressPluginReviewAdapter(self.user_agent)
        shopify_review_config = self.config.get("discovery", {}).get("shopify_reviews", {}) or {}
        self.shopify_review_adapter = ShopifyAppReviewAdapter(
            self.user_agent,
            rate_limit_cooldown_seconds=max(
                60,
                int(shopify_review_config.get("rate_limit_cooldown_seconds", 900)),
            ),
        )

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self.reddit_transport:
            await self.reddit_transport.close()
        if self.github_adapter:
            await self.github_adapter.close()
        if self.wordpress_review_adapter:
            await self.wordpress_review_adapter.close()
        if self.shopify_review_adapter:
            await self.shopify_review_adapter.close()

    # ------------------------------------------------------------------
    # TTL-aware cache helpers
    # ------------------------------------------------------------------

    def _cache_key_str(self, key: Any) -> str:
        """Return a string representation usable as a timestamp-dict key."""
        if isinstance(key, str):
            return key
        return str(key)

    def _evict_expired_caches(self) -> None:
        """Remove entries from all caches whose TTL has expired."""
        now = time.time()
        expired: list[str] = []
        for k, ts in self._cache_timestamps.items():
            if now - ts > self._cache_ttl:
                expired.append(k)
        for k in expired:
            self._cache_timestamps.pop(k, None)
            # Remove from the appropriate cache dict.
            # We try each cache; the key type determines which one owns it.
            # Tuple keys belong to _search_cache; string keys may be in
            # _fetch_cache, _recurrence_attempt_cache, or _discovery_feedback.
            self._search_cache.pop(k, None)  # type: ignore[arg-type]
            self._fetch_cache.pop(k, None)
            self._recurrence_attempt_cache.pop(k, None)
            # _discovery_feedback is nested — skip; it is repopulated each run.

    def _cache_get(self, cache: dict, key: Any) -> Any:
        """Get a value from *cache*, returning ``None`` if expired."""
        self._evict_expired_caches()
        key_str = self._cache_key_str(key)
        if key_str not in self._cache_timestamps:
            # Not tracked — treat as expired / absent.
            cache.pop(key, None)
            return None
        if time.time() - self._cache_timestamps[key_str] > self._cache_ttl:
            # Expired — evict.
            self._cache_timestamps.pop(key_str, None)
            cache.pop(key, None)
            return None
        return cache.get(key)

    def _cache_set(self, cache: dict, key: Any, value: Any) -> None:
        """Store *value* in *cache* under *key* and record the timestamp."""
        key_str = self._cache_key_str(key)
        self._cache_timestamps[key_str] = time.time()
        cache[key] = value

    def get_reddit_runtime_metrics(self) -> dict[str, Any]:
        return self.reddit_transport.get_runtime_metrics()

    async def warm_reddit_validation_queries(
        self,
        *,
        subreddits: list[str],
        queries: list[str],
    ) -> dict[str, int]:
        return await self.reddit_transport.warm_validation_queries(
            subreddits=subreddits,
            queries=queries,
        )

    def _approved_skill_path(self, skill_name: str, relative_path: str) -> Path:
        skill_dir = self.skills_root / skill_name
        candidate = skill_dir / relative_path
        if not candidate.exists():
            return candidate
        if not self.skill_audit_enabled:
            return candidate
        audit = self.skill_audits.get(skill_name)
        return candidate if audit and audit.approved else (self.project_root / ".skill-blocked")

    def _audit_installed_skills(self) -> dict[str, SkillAudit]:
        if not self.skills_root.exists():
            return {}

        audits: dict[str, SkillAudit] = {}
        for skill_dir in sorted(self.skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            audits[skill_dir.name] = self._audit_skill_dir(skill_dir)
        return audits

    def _audit_skill_dir(self, skill_dir: Path) -> SkillAudit:
        skill_file = skill_dir / "SKILL.md"
        reasons: list[str] = []
        score = 1.0
        hard_fail = False
        missing_script_reference = False
        runnable = False

        skill_text = ""
        if skill_file.exists():
            skill_text = skill_file.read_text(encoding="utf-8", errors="ignore")
        else:
            score -= 0.5
            reasons.append("Missing SKILL.md")

        referenced_paths = set(re.findall(r"(?:scripts|references|assets)/[A-Za-z0-9_./-]+", skill_text))
        for relative_path in sorted(referenced_paths):
            if not (skill_dir / relative_path).exists():
                if relative_path.startswith("scripts/"):
                    missing_script_reference = True
                    score -= 0.45
                    reasons.append(f"References missing local script: {relative_path}")
                else:
                    score -= 0.18
                    reasons.append(f"References missing local resource: {relative_path}")

        file_text_chunks: list[str] = []
        script_count = 0
        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".py", ".sh", ".js", ".mjs", ".json", ".yaml", ".yml", ".md"}:
                if "scripts" in path.parts:
                    script_count += 1
                try:
                    file_text_chunks.append(path.read_text(encoding="utf-8", errors="ignore")[:12000])
                except Exception:
                    continue
        combined_text = "\n".join(file_text_chunks)
        runnable = script_count > 0 or not referenced_paths

        destructive_patterns = [
            r"rm\s+-rf\s+/",
            r"diskutil\s+eraseDisk",
            r"\bmkfs\.",
            r"dd\s+if=.*of=/dev/",
            r":\(\)\s*\{\s*:\|:\s*&\s*\};:",
        ]
        for pattern in destructive_patterns:
            if re.search(pattern, combined_text):
                hard_fail = True
                score = 0.0
                reasons.append("Contains destructive shell pattern")
                break

        risk_rules = [
            (r"(curl|wget)[^\n|]{0,200}\|\s*(sh|bash)", 0.35, "Pipes remote content directly into a shell"),
            (r"\bsudo\b", 0.25, "Requests elevated privileges"),
            (r"eval\s+[\"']?\$\(", 0.18, "Uses eval on command substitution"),
            (r"shell=True", 0.12, "Uses shell=True subprocess execution"),
            (
                r"\b(upvote|bulk-github-star|send direct message|send dm|auto[- ]?reply|post comments?|like posts?|follow accounts?)\b",
                0.20,
                "Automates engagement actions rather than readonly discovery",
            ),
        ]
        if not hard_fail:
            for pattern, penalty, reason in risk_rules:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    score -= penalty
                    reasons.append(reason)

        if re.search(r"\b(read[- ]?only|readonly)\b", combined_text, re.IGNORECASE):
            score += 0.08
            reasons.append("Declares readonly or read-only behavior")

        capabilities = self._infer_skill_capabilities(skill_dir.name, skill_text)
        score = max(0.0, min(score, 1.0))
        approved = (not hard_fail) and (not missing_script_reference) and score >= self.skill_min_score
        if not approved and not reasons:
            reasons.append("Fell below trust threshold")

        return SkillAudit(
            name=skill_dir.name,
            path=str(skill_dir),
            approved=approved,
            score=score,
            runnable=runnable,
            capabilities=capabilities,
            reasons=reasons,
        )

    def _infer_skill_capabilities(self, skill_name: str, text: str) -> list[str]:
        haystack = f"{skill_name} {text}".lower()
        capabilities: list[str] = []
        keyword_map = {
            "reddit": ["reddit"],
            "youtube": ["youtube", "transcript"],
            "github": ["github", "issues", "prs", "repository"],
            "browser": ["playwright", "browser", "chrome devtools"],
            "security": ["owasp", "security", "vulnerabilities", "audit"],
        }
        for capability, terms in keyword_map.items():
            if any(term in haystack for term in terms):
                capabilities.append(capability)
        return capabilities

    def audited_skill_summary(self) -> dict[str, Any]:
        approved = [audit.name for audit in self.skill_audits.values() if audit.approved]
        blocked = [audit.name for audit in self.skill_audits.values() if not audit.approved]
        return {
            "approved": approved,
            "blocked": blocked,
            "skills": {
                audit.name: {
                    "approved": audit.approved,
                    "score": audit.score,
                    "runnable": audit.runnable,
                    "capabilities": audit.capabilities,
                    "reasons": audit.reasons,
                }
                for audit in self.skill_audits.values()
            },
        }

    def set_discovery_feedback(self, rows: list[dict[str, Any]]) -> None:
        feedback: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            source_name = row.get("source_name")
            query_text = row.get("query_text")
            if not source_name or not query_text:
                continue
            feedback.setdefault(source_name, {})[query_text] = row
        self._discovery_feedback = feedback

    def _rank_discovery_queries(self, source_name: str, queries: list[str]) -> list[tuple[str, float, int, int]]:
        feedback = self._discovery_feedback.get(source_name, {})
        now = datetime.now(UTC)
        ranked: list[tuple[int, float, int, int, str]] = []
        for index, query in enumerate(queries):
            normalized_query = str(query or "").lower()
            row = feedback.get(query, {})
            runs = int(row.get("runs", 0) or 0)
            findings = int(row.get("findings_emitted", 0) or 0)
            screened_out = int(row.get("screened_out", 0) or 0)
            low_signal_count = int(row.get("low_signal_count", 0) or 0)
            pain_signal_count = int(row.get("pain_signal_count", 0) or 0)
            validations = int(row.get("validations", 0) or 0)
            passes = int(row.get("passes", 0) or 0)
            kills = int(row.get("kills", 0) or 0)
            parks = int(row.get("parks", 0) or 0)
            promotes = int(row.get("promotes", 0) or 0)
            thin_recurrence_count = int(row.get("thin_recurrence_count", 0) or 0)
            single_source_only_count = int(row.get("single_source_only_count", 0) or 0)
            prototype_candidates = int(row.get("prototype_candidates", 0) or 0)
            build_briefs = int(row.get("build_briefs", 0) or 0)
            avg_score = float(row.get("avg_validation_score", 0.0) or 0.0)
            avg_screening_score = float(row.get("avg_screening_score", 0.0) or 0.0)
            docs_seen = int(row.get("docs_seen", 0) or 0)
            cooldown_until = str(row.get("cooldown_until", "") or "").strip()
            cooldown_active = False
            if cooldown_until:
                try:
                    cooldown_dt = datetime.fromisoformat(cooldown_until)
                    if cooldown_dt.tzinfo is None:
                        cooldown_dt = cooldown_dt.replace(tzinfo=UTC)
                    cooldown_active = cooldown_dt > now
                except (ValueError, TypeError):
                    cooldown_active = False
            positive_yield = passes > 0 or findings > 0 or validations > 0 or prototype_candidates > 0 or build_briefs > 0
            low_yield = runs >= 2 and findings == 0 and validations == 0 and passes == 0 and docs_seen <= 1
            noisy_source = runs >= 2 and screened_out >= max(2, findings + 1) and pain_signal_count == 0
            thin_validation_trap = (
                validations >= 2
                and prototype_candidates == 0
                and build_briefs == 0
                and promotes == 0
                and (thin_recurrence_count >= 2 or single_source_only_count >= 2 or parks >= 2 or kills >= 2)
            )
            if positive_yield and not cooldown_active:
                bucket = 0
            elif runs == 0:
                bucket = 1
            elif cooldown_active:
                bucket = 4
            elif low_yield or noisy_source or thin_validation_trap:
                bucket = 3
            else:
                bucket = 2
            score = (
                build_briefs * 5.0
                + prototype_candidates * 3.5
                + passes * 4.0
                + promotes * 2.5
                + avg_score * 2.0
                + avg_screening_score * 0.8
                + findings * 1.2
                + pain_signal_count * 0.3
                + validations * 0.8
                + min(docs_seen, 20) * 0.05
                - runs * 0.08
            )
            for term, boost in (HIGH_VALUE_DISCOVERY_QUERY_TERMS.get(source_name, {}) or {}).items():
                if term in normalized_query:
                    score += boost
            for term, penalty in (LOW_SIGNAL_DISCOVERY_QUERY_TERMS.get(source_name, {}) or {}).items():
                if term in normalized_query:
                    score -= penalty
            if validations >= 2 and prototype_candidates == 0 and build_briefs == 0 and passes == 0:
                score -= 0.35
            score -= min(screened_out, 8) * 0.22
            score -= min(low_signal_count, 6) * 0.18
            score -= min(kills, 4) * 0.2
            score -= min(parks, 4) * 0.12
            score -= min(thin_recurrence_count, 4) * 0.18
            score -= min(single_source_only_count, 4) * 0.2
            if cooldown_active:
                score -= 5.0
            ranked.append((bucket, score, runs, index, query))
        ranked.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
        return [(query, score, runs, index) for bucket, score, runs, index, query in ranked]

    @staticmethod
    def discovery_query_family_key(query: str) -> str:
        tokens = ResearchToolkit._discovery_query_family_tokens(query)
        if not tokens:
            normalized = " ".join(str(query or "").lower().split())
            return normalized[:80]
        for anchor in DISCOVERY_QUERY_ANCHOR_TERMS:
            if anchor in tokens:
                return anchor
        return " ".join(sorted(tokens))

    @staticmethod
    def _discovery_query_family_tokens(query: str) -> set[str]:
        text = str(query or "").lower()
        raw_tokens = re.findall(r"[a-z0-9]+", text)
        tokens: list[str] = []
        for token in raw_tokens:
            if len(token) <= 2:
                continue
            if token in QUERY_STOPWORDS or token in DISCOVERY_QUERY_GENERIC_TERMS:
                continue
            normalized = token
            if normalized.endswith("ies") and len(normalized) > 4:
                normalized = normalized[:-3] + "y"
            elif normalized.endswith("s") and len(normalized) > 4:
                normalized = normalized[:-1]
            tokens.append(normalized)
        return set(tokens)

    @classmethod
    def _queries_are_near_duplicates(cls, left: str, right: str) -> bool:
        left_tokens = cls._discovery_query_family_tokens(left)
        right_tokens = cls._discovery_query_family_tokens(right)
        if not left_tokens or not right_tokens:
            left_norm = " ".join(str(left or "").lower().split())
            right_norm = " ".join(str(right or "").lower().split())
            return bool(left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm))
        overlap = len(left_tokens & right_tokens)
        if overlap == 0:
            return False
        union = len(left_tokens | right_tokens)
        if union == 0:
            return False
        similarity = overlap / union
        return similarity >= 0.5 or left_tokens <= right_tokens or right_tokens <= left_tokens

    @staticmethod
    def _is_discovery_query_searchable(query: str) -> bool:
        normalized = " ".join(str(query or "").strip().lower().split())
        if not normalized:
            return False
        if len(normalized) > 90:
            return False
        tokens = re.findall(r"[a-z0-9]+", normalized)
        if len(tokens) > 10:
            return False
        conversational_markers = (
            "i am",
            "i'm",
            "we are",
            "we're",
            "trying to",
            "how do you",
            "does anyone",
            "what are you",
            "when i am",
            "please help",
        )
        if any(marker in normalized for marker in conversational_markers):
            return False
        return True

    @staticmethod
    def _rotate_candidates(items: list[dict[str, Any]], take: int, cycle_index: int) -> tuple[list[dict[str, Any]], int]:
        if take <= 0 or not items:
            return [], 0
        if len(items) <= take:
            return list(items), 0
        offset = (cycle_index * take) % len(items)
        rotated = [items[(offset + step) % len(items)] for step in range(take)]
        return rotated, offset

    def build_discovery_query_plan(
        self,
        source_name: str,
        queries: list[str],
        *,
        limit: int,
        cycle_index: int = 0,
    ) -> DiscoveryQueryPlan:
        sanitized_queries: list[str] = []
        for query in queries:
            normalized = " ".join(str(query or "").strip().split())
            if not normalized or normalized in sanitized_queries:
                continue
            if not self._is_discovery_query_searchable(normalized):
                continue
            sanitized_queries.append(normalized)

        if limit <= 0 or not sanitized_queries:
            return DiscoveryQueryPlan(source_name=source_name, queries=[], slice_size=max(0, limit), cycle_index=cycle_index)

        ranked = self._rank_discovery_queries(source_name, sanitized_queries)
        feedback = self._discovery_feedback.get(source_name, {})
        now = datetime.now(UTC)
        candidates: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        max_per_concept = max(1, int(self.config.get("discovery", {}).get("max_queries_per_concept", 1)))
        concept_counts: dict[str, int] = {}
        for query, score, runs, index in ranked:
            row = feedback.get(query, {}) or {}
            cooldown_until = str(row.get("cooldown_until", "") or "").strip()
            if cooldown_until:
                try:
                    cooldown_dt = datetime.fromisoformat(cooldown_until)
                    if cooldown_dt.tzinfo is None:
                        cooldown_dt = cooldown_dt.replace(tzinfo=UTC)
                    if cooldown_dt > now:
                        continue
                except (ValueError, TypeError):
                    pass
            findings = int(row.get("findings_emitted", 0) or 0)
            validations = int(row.get("validations", 0) or 0)
            passes = int(row.get("passes", 0) or 0)
            screened_out = int(row.get("screened_out", 0) or 0)
            pain_signal_count = int(row.get("pain_signal_count", 0) or 0)
            prototype_candidates = int(row.get("prototype_candidates", 0) or 0)
            build_briefs = int(row.get("build_briefs", 0) or 0)
            promotes = int(row.get("promotes", 0) or 0)
            parks = int(row.get("parks", 0) or 0)
            kills = int(row.get("kills", 0) or 0)
            thin_recurrence_count = int(row.get("thin_recurrence_count", 0) or 0)
            single_source_only_count = int(row.get("single_source_only_count", 0) or 0)
            docs_seen = int(row.get("docs_seen", 0) or 0)
            family_key = self.discovery_query_family_key(query)
            low_yield = runs >= 2 and findings == 0 and validations == 0 and passes == 0 and docs_seen <= 1
            noisy_source = runs >= 2 and screened_out >= max(2, findings + 1) and pain_signal_count == 0
            thin_validation_trap = (
                validations >= 2
                and prototype_candidates == 0
                and build_briefs == 0
                and promotes == 0
                and (thin_recurrence_count >= 2 or single_source_only_count >= 2 or parks >= 2 or kills >= 2)
            )
            candidate = {
                "query": query,
                "score": score,
                "runs": runs,
                "index": index,
                "family_key": family_key,
                "novel": runs == 0 or (runs <= 1 and findings == 0 and validations == 0),
                "deprioritized": low_yield or noisy_source or thin_validation_trap,
            }
            current_count = concept_counts.get(family_key, 0)
            duplicate_family = any(
                self._queries_are_near_duplicates(query, existing["query"])
                for existing in candidates
                if existing["family_key"] == family_key
            )
            if candidate["deprioritized"] or current_count >= max_per_concept or duplicate_family:
                overflow.append(candidate)
                continue
            concept_counts[family_key] = current_count + 1
            candidates.append(candidate)
        if not candidates and overflow:
            candidates = list(overflow)
            overflow = []
        if not candidates:
            return DiscoveryQueryPlan(source_name=source_name, queries=[], slice_size=0, cycle_index=cycle_index)
        exploration_slots = max(0, int(self.config.get("discovery", {}).get("exploration_slots_per_cycle", 0)))
        novel_candidates = [candidate for candidate in candidates if candidate["novel"]]
        exploration_slots = min(exploration_slots, max(0, limit - 1), len(novel_candidates))

        anchor = candidates[0] if candidates else None
        if anchor and anchor["novel"]:
            stable_anchor = next((candidate for candidate in candidates if not candidate["novel"]), None)
            if stable_anchor is not None:
                anchor = stable_anchor

        remaining = [candidate for candidate in candidates if not anchor or candidate["query"] != anchor["query"]]
        exploration_pool = [candidate for candidate in remaining if candidate["novel"]]
        exploration_selected, _exploration_offset = self._rotate_candidates(exploration_pool, exploration_slots, cycle_index)
        selected_queries = {candidate["query"] for candidate in exploration_selected}
        if anchor is not None:
            selected_queries.add(anchor["query"])

        fill_pool = [candidate for candidate in remaining if candidate["query"] not in selected_queries]
        fill_selected, offset = self._rotate_candidates(fill_pool, max(0, limit - len(selected_queries)), cycle_index)

        selected: list[dict[str, Any]] = []
        if anchor is not None:
            selected.append(anchor)
        selected.extend(exploration_selected)
        selected.extend(fill_selected)

        if len(selected) < limit and overflow:
            overflow_fill = [candidate for candidate in overflow if candidate["query"] not in {item["query"] for item in selected}]
            extra_selected, _ = self._rotate_candidates(overflow_fill, limit - len(selected), cycle_index)
            selected.extend(extra_selected)

        ordered = [candidate["query"] for candidate in selected[:limit]]
        slice_size = min(limit, len(ordered))
        if len(candidates) <= limit and not overflow:
            return DiscoveryQueryPlan(
                source_name=source_name,
                queries=ordered,
                slice_size=slice_size,
                cycle_index=cycle_index,
                rotated_queries_used=list(ordered),
            )
        return DiscoveryQueryPlan(
            source_name=source_name,
            queries=ordered,
            slice_size=slice_size,
            cycle_index=cycle_index,
            query_offset=offset,
            rotation_applied=cycle_index > 0 and len(fill_pool) > max(0, limit - len(selected_queries)),
            rotated_queries_used=[candidate["query"] for candidate in fill_selected],
        )

    def choose_query_plan(self, source_name: str, queries: list[str], *, limit: int) -> list[str]:
        return self.build_discovery_query_plan(source_name, queries, limit=limit).queries

    def source_learning_insights(self, source_names: list[str]) -> list[str]:
        insights: list[str] = []
        for source_name in source_names:
            feedback = list(self._discovery_feedback.get(source_name, {}).values())
            if not feedback:
                continue
            best = max(
                feedback,
                key=lambda row: (
                    int(row.get("passes", 0) or 0),
                    float(row.get("avg_validation_score", 0.0) or 0.0),
                    int(row.get("findings_emitted", 0) or 0),
                ),
            )
            worst = min(
                feedback,
                key=lambda row: (
                    int(row.get("passes", 0) or 0),
                    float(row.get("avg_validation_score", 0.0) or 0.0),
                    int(row.get("findings_emitted", 0) or 0),
                ),
            )
            if best.get("query_text"):
                insights.append(f"{source_name}: favor '{best['query_text']}'")
            if worst.get("query_text") and worst.get("query_text") != best.get("query_text"):
                insights.append(f"{source_name}: explore beyond '{worst['query_text']}'")
        return insights[:12]

    def _has_any_term(self, haystack: str, terms: Iterable[str]) -> bool:
        return any(contains_keyword(haystack, term) for term in terms)

    def _is_help_or_reference_page(self, *, title: str, snippet: str, domain: str, url: str) -> bool:
        haystack = f"{title} {snippet} {domain} {url}".lower()
        return any(pattern in haystack for pattern in HELP_URL_PATTERNS)

    def _is_junk_result(self, title: str, snippet: str, domain: str, url: str) -> bool:
        haystack = f"{title} {snippet} {domain} {url}".lower()
        if domain in NOISY_SEARCH_DOMAINS or domain in SEARCH_ENGINE_RESULT_DOMAINS:
            return True
        if any(pattern in haystack for pattern in BLOG_LIKE_PATTERNS) and self._pain_score(haystack) == 0:
            return True
        return False

    def _is_low_quality_web_problem_page(self, *, title: str, snippet: str, body: str, url: str) -> bool:
        domain = domain_for(url)
        path = url_path(url)
        cleaned_title = clean_extracted_web_text(title, limit=300)
        cleaned_snippet = clean_extracted_web_text(snippet, limit=600)
        cleaned_body = clean_extracted_web_text(body, limit=1800)
        haystack = compact_text(f"{cleaned_title} {cleaned_snippet} {cleaned_body} {domain} {path}".lower(), 2200)
        title_lower = compact_text(cleaned_title.lower(), 300)

        if domain in WEB_PROBLEM_CONTENT_FARM_DOMAINS:
            return True
        if any(token in domain for token in WEB_PROBLEM_REJECT_DOMAIN_TOKENS):
            return True
        if "engineersedge." in domain:
            return True
        if any(token in path for token in WEB_PROBLEM_REJECT_PATH_TOKENS):
            return True
        if any(pattern in haystack for pattern in WEB_PROBLEM_REJECT_TEXT_PATTERNS):
            return True
        has_practitioner_context = any(pattern in haystack for pattern in WEB_PROBLEM_PRACTITIONER_TEXT_PATTERNS)
        looks_like_reference = any(pattern in haystack for pattern in WEB_PROBLEM_REFERENCE_TEXT_PATTERNS)
        discussion_surface = any(token in path for token in ["/forum/", "/forums/", "/thread/", "/threads/", "/discussion", "/discussions/"])
        looks_like_download = (
            title_lower.startswith("download ")
            or title_lower.endswith(" - download")
            or "free download" in haystack
            or "download now" in haystack
            or "/download" in path
        )
        looks_like_release_editorial = (
            title_lower.startswith("what's new in ")
            or title_lower.startswith("whats new in ")
            or "release notes" in haystack
            or "new features" in haystack
        )
        looks_like_vendor_product_page = (
            any(token in domain for token in ["microsoft.com", "office.com"])
            and ("excel" in haystack or "microsoft 365" in haystack)
            and not has_practitioner_context
        )
        looks_like_template_gallery = (
            ("template" in title_lower or "templates" in title_lower or "gallery" in title_lower)
            and ("excel" in haystack or "spreadsheet" in haystack)
            and not has_practitioner_context
        )
        looks_like_editorial_review = (
            ("review" in title_lower or "rating" in haystack or "pros and cons" in haystack)
            and ("excel" in haystack or "microsoft 365" in haystack or "spreadsheet" in haystack)
            and not has_practitioner_context
        )
        looks_like_marketing_copy = (
            any(
                phrase in haystack
                for phrase in [
                    "powerful spreadsheet",
                    "find customizable templates",
                    "browse templates",
                    "create spreadsheets",
                    "analyze data",
                    "microsoft 365 excel",
                ]
            )
            and not has_practitioner_context
        )
        if looks_like_reference and not has_practitioner_context and not discussion_surface:
            return True
        if looks_like_download and not has_practitioner_context and not discussion_surface:
            return True
        if looks_like_release_editorial and not has_practitioner_context:
            return True
        if looks_like_vendor_product_page:
            return True
        if looks_like_template_gallery:
            return True
        if looks_like_editorial_review:
            return True
        if looks_like_marketing_copy:
            return True
        if title_lower.startswith("why ") and ("tutorial" in haystack or "compatibility version" in haystack):
            return True
        return False

    def _is_low_quality_corroboration_page(self, *, title: str, snippet: str, domain: str, url: str) -> bool:
        haystack = compact_text(f"{title} {snippet} {domain} {url}".lower(), 1200)
        path = url_path(url)
        if any(token in domain for token in LOW_QUALITY_CORROBORATION_DOMAIN_TOKENS):
            return True
        if any(token in path for token in LOW_QUALITY_CORROBORATION_PATH_TOKENS):
            return True
        if any(pattern in haystack for pattern in LOW_QUALITY_CORROBORATION_TEXT_PATTERNS):
            return True
        return False

    def _is_problem_candidate(self, title: str, body: str, *, source_url: str = "") -> bool:
        """Lightweight heuristic gate used at discovery time.

        This should be permissive enough to emit candidates for downstream policy screening,
        while still filtering obvious non-problems.
        """
        haystack = compact_text(f"{title} {body} {source_url}".lower(), 1600)
        if any(pattern in haystack for pattern in NON_PROBLEM_DISCOVERY_PATTERNS):
            return False

        discovery_cfg = (self.config.get("discovery") or {}).get("candidate_filter") or {}
        min_score = int(discovery_cfg.get("min_score", 2))
        behavioral_min = int(discovery_cfg.get("behavioral_min_signals", 1))
        behavioral_penalty = int(discovery_cfg.get("behavioral_penalty", 1))

        score = 0
        pain_hits = sum(1 for term in PAIN_KEYWORDS if contains_keyword(haystack, term))
        workaround_hits = sum(1 for term in WORKAROUND_SIGNAL_TERMS if contains_keyword(haystack, term))
        frequency_hits = sum(1 for term in FREQUENCY_SIGNAL_TERMS if contains_keyword(haystack, term))
        cost_hits = sum(1 for term in COST_SIGNAL_TERMS if contains_keyword(haystack, term))
        score = pain_hits + workaround_hits + frequency_hits + cost_hits

        behavioral = sum(
            1
            for terms in [WORKAROUND_SIGNAL_TERMS, FREQUENCY_SIGNAL_TERMS, COST_SIGNAL_TERMS]
            if self._has_any_term(haystack, terms)
        )
        if behavioral < behavioral_min:
            score -= behavioral_penalty
        if "?" in title and not self._has_any_term(haystack, WORKAROUND_SIGNAL_TERMS):
            score -= 1

        # Debug logging for candidate filtering
        if score < min_score:
            logger = logging.getLogger("research_tools")
            logger.debug(
                f"candidate_filtered title={title[:50]!r} score={score} min={min_score} "
                f"pain={pain_hits} workaround={workaround_hits} freq={frequency_hits} cost={cost_hits} behavioral={behavioral}"
            )
        return score >= min_score

    def _is_concrete_youtube_problem_comment(self, text: str) -> bool:
        haystack = normalize_content(text)
        failure_terms = [
            "broken",
            "fails",
            "failed",
            "error",
            "missing",
            "duplicate",
            "duplicating",
            "out of sync",
            "wrong",
            "still showing",
            "manual workaround",
        ]
        artifact_terms = [
            "csv",
            "import",
            "export",
            "invoice",
            "payment",
            "inventory",
            "shipment",
            "label",
            "order",
            "backup",
            "restore",
            "sync",
            "reconciliation",
            "spreadsheet",
        ]
        consequence_terms = [
            "hours",
            "manual",
            "lost",
            "refund",
            "risk",
            "revenue",
            "downtime",
            "customers",
        ]
        return any(term in haystack for term in failure_terms) and (
            any(term in haystack for term in artifact_terms)
            or any(term in haystack for term in consequence_terms)
        )

    def _should_keep_youtube_comment_candidate(self, *, title: str, snippet: str, comments: list[dict[str, Any]]) -> bool:
        haystack = normalize_content(f"{title} {snippet}")
        if contains_any_phrase(haystack, YOUTUBE_LOW_SIGNAL_VIDEO_PATTERNS):
            return False
        concrete_comments = [
            comment
            for comment in comments
            if self._is_concrete_youtube_problem_comment(str(comment.get("text", "") or ""))
        ]
        return len(concrete_comments) >= 2

    def _should_hydrate_reddit_problem_doc(self, doc: SearchDocument) -> bool:
        """Cheap pre-hydration screen for Reddit problem discovery.

        Uses only title/snippet/url so we can skip thread fetches for obvious
        non-problem results like recap/news/summary posts.
        """
        title = getattr(doc, "title", "") or ""
        snippet = getattr(doc, "snippet", "") or ""
        url = getattr(doc, "url", "") or ""
        cheap_text = compact_text(f"{title} {snippet}", 800)
        cheap_haystack = compact_text(f"{title} {snippet} {url}".lower(), 900)
        title_lower = compact_text(title.lower(), 240)
        if any(pattern in cheap_haystack for pattern in LOW_VALUE_REDDIT_PREFILTER_PATTERNS):
            return False
        if any(pattern in cheap_haystack for pattern in NON_OPERATIONAL_BUSINESS_RISK_TERMS):
            has_operational_process = any(term in cheap_haystack for term in OPERATIONAL_PROCESS_TERMS)
            if not has_operational_process:
                return False
        has_specific_failure = any(term in cheap_haystack for term in REDDIT_PREFILTER_FAILURE_TERMS)
        has_specific_object = any(term in cheap_haystack for term in REDDIT_PREFILTER_OBJECT_TERMS)
        has_workaround_signal = any(term in cheap_haystack for term in WORKAROUND_SIGNAL_TERMS)
        has_cost_signal = any(term in cheap_haystack for term in COST_SIGNAL_TERMS)
        if has_specific_failure and has_specific_object:
            return True
        if any(title_lower.startswith(prefix) for prefix in LOW_VALUE_REDDIT_TITLE_PREFIXES):
            if has_workaround_signal and has_cost_signal and has_specific_object:
                return True
            return False
        return self._is_problem_candidate(title, cheap_text, source_url=url)

    @staticmethod
    def _reddit_query_matches_subreddit(subreddit: str, query: str) -> bool:
        subreddit_norm = str(subreddit or "").strip().lower()
        query_norm = " ".join(str(query or "").strip().lower().split())
        if not subreddit_norm or not query_norm:
            return True

        finance_subreddits = {"accounting"}
        ecommerce_subreddits = {"ecommerce", "shopify", "etsysellers"}
        smallbiz_subreddits = {"smallbusiness"}
        practitioner_subreddits = finance_subreddits | ecommerce_subreddits | smallbiz_subreddits

        finance_terms = (
            "bank reconciliation",
            "month end close",
            "sales tax",
            "invoice reminder",
            "late payment",
        )
        ecommerce_terms = (
            "sales channel",
            "channel profitability",
            "order level reconciliation",
            "returns workflow",
            "supplier data",
            "label printed",
        )
        pdf_terms = (
            "pdf collaboration",
            "latest version",
        )

        if any(term in query_norm for term in finance_terms):
            return subreddit_norm in finance_subreddits | smallbiz_subreddits
        if any(term in query_norm for term in ecommerce_terms):
            return subreddit_norm in ecommerce_subreddits | smallbiz_subreddits
        if any(term in query_norm for term in pdf_terms):
            return subreddit_norm in smallbiz_subreddits
        if subreddit_norm in practitioner_subreddits:
            return True
        return True

    def _extract_validation_phrases(self, text: str) -> list[str]:
        lowered = compact_text(text.lower(), 500)
        phrases: list[str] = []
        for pattern, label in VALIDATION_QUERY_PHRASES:
            if re.search(pattern, lowered) and label not in phrases:
                phrases.append(label)
        return phrases[:3]

    @staticmethod
    def _source_diverse_sample(docs: list, limit: int) -> list:
        """Sample docs round-robin from each source family for diversity."""
        if len(docs) <= limit:
            return docs
        by_source: dict[str, list[Any]] = {}
        for doc in docs:
            src = getattr(doc, "source", "") or (doc.get("source", "") if isinstance(doc, dict) else "")
            by_source.setdefault(src, []).append(doc)
        result: list[Any] = []
        sources = list(by_source.keys())
        idx = {s: 0 for s in sources}
        while len(result) < limit:
            added = False
            for s in sources:
                if idx[s] < len(by_source[s]) and len(result) < limit:
                    result.append(by_source[s][idx[s]])
                    idx[s] += 1
                    added = True
            if not added:
                break
        return result

    def _is_relevant_search_result(
        self,
        *,
        query: str,
        title: str,
        snippet: str,
        domain: str,
        url: str,
        intent: str,
    ) -> bool:
        if self._is_junk_result(title, snippet, domain, url):
            return False
        if self._is_help_or_reference_page(title=title, snippet=snippet, domain=domain, url=url):
            return False

        overlap = topical_overlap(query, title, snippet, domain)
        if intent == "general":
            return overlap >= 1 or bool(snippet.strip())

        if intent == "validation_recurrence":
            haystack = f"{title} {snippet} {domain} {url}".lower()
            # High topical overlap is itself strong evidence of recurrence —
            # a result closely matching the problem query corroborates
            # the problem even without explicit pain language or on a
            # blog-like page.  Accept these before other filters run.
            if overlap >= 3:
                if not self._is_low_quality_corroboration_page(title=title, snippet=snippet, domain=domain, url=url):
                    return True
            if any(pattern in haystack for pattern in BLOG_LIKE_PATTERNS):
                return False
            if self._is_low_quality_corroboration_page(title=title, snippet=snippet, domain=domain, url=url):
                return False
            pain_ok = (
                self._pain_score(haystack) >= 1
                or self._has_any_term(haystack, WORKAROUND_SIGNAL_TERMS + COST_SIGNAL_TERMS)
            )
            return overlap >= 2 and pain_ok

        if intent == "validation_competitor":
            haystack = f"{title} {snippet} {domain} {url}".lower()
            path = url_path(url)
            has_competitor_hint = domain in SOFTWARE_DOMAINS or any(term in haystack for term in COMPETITOR_INTENT_HINTS)
            blog_like = any(pattern in domain or pattern in path for pattern in BLOG_LIKE_PATTERNS)
            forum_like = any(token in domain for token in ["stackoverflow.com", "reddit.com", "news.ycombinator.com"])
            return overlap >= 2 and has_competitor_hint and not blog_like and not forum_like

        return overlap >= 1

    async def _run_json_command(self, command: list[str], timeout: int = 20) -> Optional[dict[str, Any]]:
        def _run() -> Optional[dict[str, Any]]:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = completed.stdout.strip()
            if not stdout:
                return None
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return None

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return None

    async def _run_text_command(
        self,
        command: list[str],
        timeout: int = 40,
        env: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        def _run() -> Optional[str]:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if completed.returncode != 0 and not completed.stdout:
                return None
            return completed.stdout.strip() or None

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return None

    def _stackexchange_site_key(self, site: Optional[str]) -> Optional[str]:
        if not site:
            return None
        normalized = site.lower().strip()
        return STACKEXCHANGE_SITE_ALIASES.get(normalized)

    async def _search_stackexchange_site(
        self,
        *,
        query: str,
        site: str,
        max_results: int,
        request_timeout: int,
    ) -> list[dict[str, str]]:
        site_key = self._stackexchange_site_key(site)
        if not site_key:
            return []

        def _run() -> list[dict[str, str]]:
            response = requests.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "order": "desc",
                    "sort": "relevance",
                    "q": query,
                    "site": site_key,
                    "pagesize": max_results + 2,
                    "filter": "withbody",
                },
                timeout=request_timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AutoResearcher/1.0)"},
            )
            response.raise_for_status()
            payload = response.json()
            rows: list[dict[str, str]] = []
            for item in payload.get("items", []) or []:
                title = BeautifulSoup(str(item.get("title", "")), "html.parser").get_text(" ", strip=True)
                body = BeautifulSoup(str(item.get("body", "")), "html.parser").get_text(" ", strip=True)
                rows.append(
                    {
                        "title": title,
                        "href": str(item.get("link", "") or ""),
                        "body": compact_text(body, 320),
                    }
                )
                if len(rows) >= max_results + 2:
                    break
            return rows

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return []

    async def search_web(
        self,
        query: str,
        max_results: int = 8,
        site: Optional[str] = None,
        intent: str = "general",
    ) -> list[SearchDocument]:
        cache_key = (query, max_results, site, intent)
        cached = self._cache_get(self._search_cache, cache_key)
        if cached is not None:
            return list(cached)

        full_query = f"site:{site} {query}" if site else query
        results: list[SearchDocument] = []
        seen_urls: set[str] = set()
        domain_counts: Counter[str] = Counter()
        validation_intent = intent.startswith("validation_")
        if intent == "validation_recurrence":
            provider_timeout = self.provider_timeout_recurrence
            request_timeout = self.request_timeout_recurrence
        elif intent == "validation_competitor":
            provider_timeout = self.provider_timeout_competitor
            request_timeout = self.request_timeout_competitor
        else:
            provider_timeout = self.provider_timeout_general
            request_timeout = self.request_timeout_general

        def _append_result(title: str, url: str, snippet: str) -> bool:
            normalized_url = normalize_search_url(url)
            if not normalized_url:
                return False
            domain = domain_for(normalized_url)
            if not domain or domain in NOISY_SEARCH_DOMAINS or domain in SEARCH_ENGINE_RESULT_DOMAINS:
                logger.debug(
                    "[search_dbg] NOISY/ENGINE domain=%s query=%r", domain, query[:60]
                )
                return False
            relevant = self._is_relevant_search_result(
                query=query,
                title=title,
                snippet=snippet,
                domain=domain,
                url=normalized_url,
                intent=intent,
            )
            if not relevant:
                ov = topical_overlap(query, title, snippet, domain)
                logger.debug(
                    "[search_dbg] FILTERED intent=%s overlap=%d domain=%s title=%r query=%r",
                    intent, ov, domain, title[:60], query[:60],
                )
                return False
            if normalized_url in seen_urls or domain_counts[domain] >= self.search_domain_cap:
                return False
            results.append(
                SearchDocument(
                    title=compact_text(title, self.result_title_length),
                    url=normalized_url,
                    snippet=compact_text(snippet, self.result_snippet_length),
                    source=site or "web",
                )
            )
            seen_urls.add(normalized_url)
            domain_counts[domain] += 1
            return True

        if intent == "validation_recurrence" and site:
            stackexchange_rows = await self._search_stackexchange_site(
                query=query,
                site=site,
                max_results=max_results,
                request_timeout=request_timeout,
            )
            for row in stackexchange_rows:
                _append_result(row.get("title", ""), row.get("href", ""), row.get("body", ""))
                if len(results) >= max_results:
                    break
            if results:
                self._cache_set(self._search_cache, cache_key, list(results))
                return list(results)

        async def _run_ddg_provider(import_path: str, timeout: int) -> list[dict[str, Any]]:
            def _collect() -> list[dict[str, Any]]:
                if import_path == "ddgs":
                    from ddgs import DDGS  # type: ignore
                else:
                    from duckduckgo_search import DDGS  # type: ignore

                rows: list[dict[str, Any]] = []
                with DDGS() as ddgs:
                    extra = self.ddgs_extra_results
                    for row in ddgs.text(full_query, backend=self.ddgs_backend, max_results=max_results + extra):
                        rows.append(row)
                        if len(rows) >= max_results + extra:
                            break
                return rows

            return await asyncio.wait_for(asyncio.to_thread(_collect), timeout=timeout)

        try:
            for row in await _run_ddg_provider("ddgs", provider_timeout):
                _append_result(row.get("title", ""), row.get("href", ""), row.get("body", ""))
                if len(results) >= max_results:
                    break
            if results:
                self._cache_set(self._search_cache, cache_key, list(results))
                return list(results)
        except Exception as exc:
            logger.debug("ddgs search provider failed: %s", exc)

        try:
            for row in await _run_ddg_provider("duckduckgo_search", provider_timeout):
                _append_result(row.get("title", ""), row.get("href", ""), row.get("body", ""))
                if len(results) >= max_results:
                    break
        except Exception as exc:
            logger.debug("duckduckgo_search provider failed: %s", exc)

        if results:
            self._cache_set(self._search_cache, cache_key, list(results))
            return list(results)

        try:
            def _ddg_html_search() -> str:
                response = requests.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": full_query},
                    timeout=request_timeout,
                    headers={"User-Agent": self.search_user_agent},
                )
                response.raise_for_status()
                return response.text

            html = await asyncio.to_thread(_ddg_html_search)
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.select("a.result__a"):
                href = unwrap_search_result_url(link.get("href", ""))
                title = link.get_text(" ", strip=True)
                snippet_node = link.find_parent(class_="result")
                snippet = ""
                if snippet_node:
                    snippet_el = snippet_node.select_one(".result__snippet")
                    if snippet_el:
                        snippet = snippet_el.get_text(" ", strip=True)
                if _append_result(title, href, snippet) and len(results) >= max_results:
                    break
        except Exception as exc:
            logger.debug("DDG HTML search failed: %s", exc)

        if validation_intent and not (intent == "validation_recurrence" and site):
            self._cache_set(self._search_cache, cache_key, list(results))
            return list(results)

        if not results:
            try:
                def _bing_search() -> str:
                    response = requests.get(
                        "https://www.bing.com/search",
                        params={"q": full_query},
                        timeout=request_timeout,
                        headers={"User-Agent": self.search_user_agent},
                    )
                    response.raise_for_status()
                    return response.text

                html = await asyncio.to_thread(_bing_search)
                soup = BeautifulSoup(html, "html.parser")
                for item in soup.select("li.b_algo"):
                    link = item.select_one("h2 a")
                    if link is None:
                        continue
                    href = unwrap_search_result_url(link.get("href", ""))
                    snippet_el = item.select_one(".b_caption p")
                    if _append_result(
                        link.get_text(" ", strip=True),
                        href,
                        snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    ) and len(results) >= max_results:
                        break
            except Exception:
                return []

        self._cache_set(self._search_cache, cache_key, list(results))
        return list(results)

    async def fetch_content(self, url: str) -> dict[str, Any]:
        cached = self._cache_get(self._fetch_cache, url)
        if cached is not None:
            return dict(cached)

        # Check for Jina Reader config
        use_jina = self.config.get("discovery", {}).get("web", {}).get("use_jina_reader", False)

        if use_jina:
            try:
                from utils.jina_reader import read_url_async
                result = await read_url_async(url)
                if result.success:
                    normalized = {
                        "url": url,
                        "title": clean_extracted_web_text(result.title or url, limit=180),
                        "description": clean_extracted_web_text(result.markdown[:900] if result.markdown else "", limit=900),
                        "text": clean_extracted_web_text(result.markdown or "", limit=2500),
                    }
                    self._cache_set(self._fetch_cache, url, normalized)
                    return dict(normalized)
            except Exception:
                pass  # Fall through to other methods

        if self.fetcher:
            try:
                payload = await asyncio.to_thread(self.fetcher.fetch, url)
                if payload:
                    normalized = {
                        "url": url,
                        "title": clean_extracted_web_text(str(payload.get("title", "") or url), limit=180),
                        "description": clean_extracted_web_text(
                            str(payload.get("description", "") or payload.get("text", "")),
                            limit=900,
                        ),
                        "text": clean_extracted_web_text(
                            str(payload.get("text", "") or payload.get("description", "")),
                            limit=2500,
                        ),
                    }
                    self._cache_set(self._fetch_cache, url, normalized)
                    return dict(normalized)
            except Exception as exc:
                logger.debug("async web fetch failed for %s: %s", url, exc)

        def _request() -> dict[str, Any]:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = clean_extracted_web_text(soup.title.string.strip() if soup.title and soup.title.string else url, limit=180)
            text = clean_extracted_web_text(soup.get_text(" ", strip=True), limit=2500)
            return {"url": url, "title": title, "description": text[:900], "text": text}

        try:
            payload = await asyncio.to_thread(_request)
        except Exception:
            payload = {"url": url, "title": url, "description": "", "text": ""}
        self._cache_set(self._fetch_cache, url, payload)
        return dict(payload)

    async def reddit_search(
        self,
        subreddit: str,
        query: str,
        limit: int = 2,
        sort: str = "relevance",
    ) -> list[SearchDocument]:
        return await self.reddit_transport.search(
            subreddit=subreddit,
            query=query,
            limit=limit,
            sort=sort,
        )

    async def reddit_thread_context(self, url: str) -> dict[str, Any]:
        return await self.reddit_transport.thread_context(url)

    async def youtube_transcript(self, video_id: str) -> str:
        if self.youtube_transcript_script.exists() and self.yt_dlp_exec:
            command_env = os.environ.copy()
            command_env["PATH"] = f"{Path(self.yt_dlp_exec).parent}:{command_env.get('PATH', '')}"
            with tempfile.NamedTemporaryFile(prefix="yt-transcript-", suffix=".txt", delete=False) as handle:
                output_path = Path(handle.name)
            try:
                await self._run_text_command(
                    ["bash", str(self.youtube_transcript_script), f"https://www.youtube.com/watch?v={video_id}", str(output_path)],
                    timeout=45,
                    env=command_env,
                )
                if output_path.exists():
                    cleaned = output_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if cleaned:
                        return compact_text(cleaned, 2500)
            finally:
                output_path.unlink(missing_ok=True)

        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return compact_text(" ".join(chunk["text"] for chunk in transcript), 2500)
        except Exception:
            return ""

    async def youtube_comments(self, video_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch YouTube video comments via Data API v3."""
        if not self.youtube_api_key:
            return []

        comments: list[dict[str, Any]] = []
        try:
            session = self._get_session()
            # Get comment threads
            url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(limit, 100),
                "key": self.youtube_api_key,
            }
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
                    comments.append({
                        "author": top_comment.get("authorDisplayName", ""),
                        "text": top_comment.get("textDisplay", ""),
                        "like_count": top_comment.get("likeCount", 0),
                        "published_at": top_comment.get("publishedAt", ""),
                    })
        except Exception as exc:
            logger.debug("YouTube comment fetch failed: %s", exc)
        return comments

    async def youtube_search(self, query: str, limit: int = 5) -> list[SearchDocument]:
        cache_key = (f"youtube:{query}", limit, "youtube", "general")
        cached = self._cache_get(self._search_cache, cache_key)
        if cached is not None:
            return list(cached)

        if self.yt_dlp_command:
            payload = await self._run_text_command(
                [*self.yt_dlp_command, "--flat-playlist", "--dump-single-json", f"ytsearch{limit}:{query}"],
                timeout=25,
            )
            if payload:
                try:
                    data = json.loads(payload)
                    docs: list[SearchDocument] = []
                    for entry in data.get("entries", [])[:limit]:
                        video_id = entry.get("id")
                        if not video_id:
                            continue
                        channel = entry.get("channel") or entry.get("uploader") or ""
                        duration = entry.get("duration_string") or ""
                        views = entry.get("view_count")
                        snippet_parts = [part for part in [channel, duration, f"{views} views" if views else ""] if part]
                        docs.append(
                            SearchDocument(
                                title=entry.get("title", ""),
                                url=f"https://www.youtube.com/watch?v={video_id}",
                                snippet=" | ".join(snippet_parts),
                                source="youtube",
                            )
                        )
                    if docs:
                        self._cache_set(self._search_cache, cache_key, list(docs))
                        return docs
                except json.JSONDecodeError:
                    pass

        docs = await self.search_web(query, max_results=limit, site="youtube.com")
        # Fallback to YouTube Data API if no results
        if not docs and self.youtube_api_key:
            try:
                session = self._get_session()
                url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "part": "snippet",
                    "q": query,
                    "maxResults": min(limit, 10),
                    "type": "video",
                    "key": self.youtube_api_key,
                }
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("items", [])[:limit]:
                            snippet = item.get("snippet", {})
                            docs.append(
                                SearchDocument(
                                    title=snippet.get("title", ""),
                                    url=f'https://www.youtube.com/watch?v={item["id"]["videoId"]}',
                                    snippet=snippet.get("description", "")[:200],
                                    source="youtube",
                                )
                            )
            except Exception as exc:
                logger.debug("YouTube search API failed: %s", exc)
        self._cache_set(self._search_cache, cache_key, list(docs))
        return docs

    async def discover_success_signals(self) -> list[dict[str, Any]]:
        tasks = [
            self._discover_youtube_successes(),
            self._discover_reddit_successes(),
            self._discover_success_stories_on_web(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        findings: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            findings.extend(result)
        return findings

    async def discover_problem_signals(self) -> list[dict[str, Any]]:
        tasks = [
            self._discover_reddit_problem_threads(),
            self._discover_github_problem_threads(),
            self._discover_web_problem_threads(),
            self._discover_marketplace_problem_threads(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        findings: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            findings.extend(result)
        return findings

    async def _discover_youtube_successes(
        self,
        keywords: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        keywords = keywords or ["AI business", "AI startup revenue", "make money with AI"]
        docs: list[tuple[str, SearchDocument]] = []
        for keyword in keywords[:4]:
            started = asyncio.get_running_loop().time()
            status = "ok"
            error = ""
            query_docs: list[SearchDocument] = []
            try:
                query_docs = await self.youtube_search(keyword, limit=3)
            except Exception as exc:
                status = "error"
                error = str(exc)
            if observer:
                observer(
                    {
                        "source_name": "youtube-success",
                        "query_text": keyword,
                        "docs_seen": len(query_docs),
                        "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                        "status": status,
                        "error": error,
                    }
                )
            docs.extend((keyword, doc) for doc in query_docs)

        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for keyword, doc in docs:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            video_id_match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", doc.url)
            transcript = await self.youtube_transcript(video_id_match.group(1)) if video_id_match else ""
            full_text = compact_text(f"{doc.title} {doc.snippet} {transcript}", 2200)
            if not any(keyword in full_text.lower() for keyword in AI_TOOL_KEYWORDS):
                continue
            money = first_match(
                [r"\$[\d,.]+(?:[kKmM])?\s*(?:mrr|arr|revenue|monthly)", r"\d+\s+(?:customers|users|clients)"],
                full_text,
            )
            if not money:
                continue
            findings.append(
                {
                    "source": "youtube",
                    "source_url": doc.url,
                    "entrepreneur": "Video creator",
                    "tool_used": self._extract_tools(full_text),
                    "product_built": doc.title,
                    "monetization_method": money,
                    "outcome_summary": compact_text(full_text, 420),
                    "finding_kind": "success_signal",
                    "recurrence_key": infer_recurrence_key(doc.title),
                    "evidence": {
                        "source_plan": "youtube-success",
                        "discovery_query": keyword,
                        "channel_or_site": "youtube",
                        "snippet": doc.snippet,
                        "transcript_excerpt": transcript[:800],
                    },
                }
            )
        return findings

    async def _discover_youtube_comments(
        self,
        keywords: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        """Discover problem signals from YouTube video comments."""
        keywords = keywords or [
            "shopify app review", "shopify problems", "ecommerce tools",
            "shopify alternative", "shopify frustration",
        ]
        docs: list[tuple[str, SearchDocument]] = []
        for keyword in keywords[:4]:
            started = asyncio.get_running_loop().time()
            status = "ok"
            error = ""
            query_docs: list[SearchDocument] = []
            try:
                query_docs = await self.youtube_search(keyword, limit=3)
            except Exception as exc:
                status = "error"
                error = str(exc)
            if observer:
                observer(
                    {
                        "source_name": "youtube-comments",
                        "query_text": keyword,
                        "docs_seen": len(query_docs),
                        "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                        "status": status,
                        "error": error,
                    }
                )
            docs.extend((keyword, doc) for doc in query_docs)

        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        problem_keywords = [
            "problem", "issue", "broken", "bug", "frustrat", "annoying", "slow",
            "expensive", "missing", "need", "wish", "want", "can't", "cannot",
            "doesn work", "doesn't work", "no option", "hard to", "difficult",
        ]
        for keyword, doc in docs:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            video_id_match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", doc.url)
            if not video_id_match:
                continue
            comments = await self.youtube_comments(video_id_match.group(1), limit=20)
            # Filter for problem-focused comments
            problem_comments = [
                c for c in comments
                if any(pk in c.get("text", "").lower() for pk in problem_keywords)
            ]
            if not problem_comments:
                continue
            if not self._should_keep_youtube_comment_candidate(
                title=doc.title,
                snippet=doc.snippet,
                comments=problem_comments,
            ):
                continue
            comment_texts = " | ".join(c["text"][:200] for c in problem_comments[:5])
            findings.append({
                "source": "youtube",
                "source_url": doc.url,
                "entrepreneur": "Video commenter",
                "product_built": doc.title,
                "monetization_method": "",
                "outcome_summary": compact_text(comment_texts, 420),
                "finding_kind": "pain_signal",
                "recurrence_key": infer_recurrence_key(doc.title),
                "evidence": {
                    "source_plan": "youtube-comments",
                    "discovery_query": keyword,
                    "channel_or_site": "youtube",
                    "snippet": doc.snippet,
                    "comments": problem_comments[:5],
                },
            })
        return findings

    async def _discover_reddit_successes(
        self,
        subreddits: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        subreddits = subreddits or ["entrepreneur", "SideProject", "IndieHackers", "SaaS"]
        keywords = keywords or reddit_success_keywords(self.config) or []
        if not keywords:
            return []
        docs: list[tuple[str, SearchDocument]] = []
        seen_urls: set[str] = set()
        discovery_reddit_cfg = (self.config.get("discovery", {}) or {}).get("reddit", {}) or {}
        configured_sorts = discovery_reddit_cfg.get("search_sorts") or ["relevance", "new", "top", "comments"]
        allowed_sorts = {"relevance", "new", "top", "comments"}
        search_sorts = [str(item).strip().lower() for item in configured_sorts if str(item).strip().lower() in allowed_sorts]
        if not search_sorts:
            search_sorts = ["relevance", "new", "top", "comments"]
        per_sort_limit = max(1, int(discovery_reddit_cfg.get("per_sort_limit", 2)))
        max_docs_per_pair = max(1, int(discovery_reddit_cfg.get("max_docs_per_pair", 6)))

        for subreddit in subreddits[:4]:
            for keyword in keywords[:3]:
                started = asyncio.get_running_loop().time()
                docs_seen = 0
                pair_docs: list[SearchDocument] = []
                pair_seen: set[str] = set()
                for sort_mode in search_sorts:
                    for doc in await self.reddit_search(subreddit, keyword, limit=per_sort_limit, sort=sort_mode):
                        if not doc.url or doc.url in pair_seen:
                            continue
                        pair_seen.add(doc.url)
                        pair_docs.append(doc)
                        if len(pair_docs) >= max_docs_per_pair:
                            break
                    if len(pair_docs) >= max_docs_per_pair:
                        break
                for doc in pair_docs:
                    if doc.url and doc.url not in seen_urls:
                        seen_urls.add(doc.url)
                        docs.append((keyword, doc))
                        docs_seen += 1
                if observer:
                    observer(
                        {
                            "source_name": "reddit-success",
                            "query_text": keyword,
                            "docs_seen": docs_seen,
                            "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                            "status": "ok",
                            "error": "",
                        }
                    )

        findings: list[dict[str, Any]] = []
        for keyword, doc in docs:
            content = await self.reddit_thread_context(doc.url)
            full_text = compact_text(f"{doc.title} {doc.snippet} {content.get('text', '')}", 2400)
            money = first_match(
                [r"\$[\d,.]+(?:[kKmM])?\s*(?:mrr|arr|revenue|monthly)", r"\d+\s+(?:customers|users|clients)"],
                full_text,
            )
            if not money:
                continue
            findings.append(
                {
                    "source": "reddit-success",
                    "source_url": doc.url,
                    "entrepreneur": "Reddit founder",
                    "tool_used": self._extract_tools(full_text),
                    "product_built": doc.title,
                    "monetization_method": money,
                    "outcome_summary": compact_text(full_text, 420),
                    "finding_kind": "success_signal",
                    "recurrence_key": infer_recurrence_key(doc.title),
                    "evidence": {
                        "source_plan": "reddit-success",
                        "discovery_query": keyword,
                        "snippet": doc.snippet,
                        "page_excerpt": content.get("description", ""),
                        "comments": content.get("comments", []),
                        "comment_metadata": content.get("comment_metadata", []),
                    },
                }
            )
        return findings

    async def _discover_success_stories_on_web(
        self,
        queries: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        queries = queries or [
            "AI startup success story revenue",
            "GPT business revenue story",
            "AI side project customers",
        ]
        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for query in queries[:4]:
            started = asyncio.get_running_loop().time()
            docs = await self.search_web(query, max_results=3)
            if observer:
                observer(
                    {
                        "source_name": "web-success",
                        "query_text": query,
                        "docs_seen": len(docs),
                        "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                        "status": "ok",
                        "error": "",
                    }
                )
            for doc in docs:
                if doc.url in seen_urls:
                    continue
                seen_urls.add(doc.url)
                content = await self.fetch_content(doc.url)
                full_text = compact_text(f"{doc.title} {doc.snippet} {content.get('text', '')}", 2200)
                money = first_match(
                    [r"\$[\d,.]+(?:[kKmM])?\s*(?:mrr|arr|revenue|monthly)", r"\d+\s+(?:customers|users|clients)"],
                    full_text,
                )
                if not money:
                    continue
                findings.append(
                    {
                        "source": "web-success",
                        "source_url": doc.url,
                        "entrepreneur": "Founder or operator",
                        "tool_used": self._extract_tools(full_text),
                        "product_built": doc.title,
                        "monetization_method": money,
                        "outcome_summary": compact_text(full_text, 420),
                        "finding_kind": "success_signal",
                        "recurrence_key": infer_recurrence_key(doc.title),
                        "evidence": {
                            "source_plan": "web-success",
                            "discovery_query": query,
                            "snippet": doc.snippet,
                            "page_excerpt": content.get("description", ""),
                        },
                    }
                )
        return findings

    async def _discover_reddit_problem_threads(
        self,
        subreddits: Optional[list[str]] = None,
        queries: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        subreddits = subreddits or reddit_discovery_subreddits(self.config)
        queries = queries or reddit_problem_keywords(self.config)
        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        discovery_config = self.config.get("discovery", {}).get("reddit", {})
        # max_* 0 or negative = use entire configured list (no artificial cap).
        _ms = discovery_config.get("max_subreddits_per_wave", 6)
        _mk = discovery_config.get("max_keywords_per_wave", 4)
        try:
            max_subs_raw = int(_ms)
        except (TypeError, ValueError):
            max_subs_raw = 6
        try:
            max_kw_raw = int(_mk)
        except (TypeError, ValueError):
            max_kw_raw = 4
        max_subs_wave = len(subreddits) if max_subs_raw <= 0 else min(len(subreddits), max(1, max_subs_raw))
        max_kw_wave = len(queries) if max_kw_raw <= 0 else min(len(queries), max(1, max_kw_raw))
        pair_concurrency = max(1, int(discovery_config.get("pair_concurrency", 6)))
        context_concurrency = max(1, int(discovery_config.get("context_concurrency", 6)))
        configured_sorts = discovery_config.get("search_sorts") or ["relevance", "new", "top", "comments"]
        allowed_sorts = {"relevance", "new", "top", "comments"}
        search_sorts = [str(item).strip().lower() for item in configured_sorts if str(item).strip().lower() in allowed_sorts]
        if not search_sorts:
            search_sorts = ["relevance", "new", "top", "comments"]
        per_sort_limit = max(1, int(discovery_config.get("per_sort_limit", 2)))
        max_docs_per_pair = max(1, int(discovery_config.get("max_docs_per_pair", 6)))
        pair_sem = asyncio.Semaphore(pair_concurrency)
        context_sem = asyncio.Semaphore(context_concurrency)

        async def _fetch_pair(subreddit: str, query: str) -> tuple[str, str, list[tuple[str, SearchDocument]]]:
            started = asyncio.get_running_loop().time()
            status = "ok"
            error = ""
            docs_seen = 0
            try:
                async with pair_sem:
                    pair_docs: list[tuple[str, SearchDocument]] = []
                    pair_seen: set[str] = set()
                    for sort_mode in search_sorts:
                        direct_docs = await self.reddit_search(
                            subreddit,
                            query,
                            limit=per_sort_limit,
                            sort=sort_mode,
                        )
                        for doc in direct_docs:
                            if not doc.url or doc.url in pair_seen:
                                continue
                            if not self._should_hydrate_reddit_problem_doc(doc):
                                continue
                            pair_seen.add(doc.url)
                            pair_docs.append((sort_mode, doc))
                            if len(pair_docs) >= max_docs_per_pair:
                                break
                        if len(pair_docs) >= max_docs_per_pair:
                            break
            except Exception as exc:
                pair_docs = []
                status = "error"
                error = str(exc)

            docs: list[tuple[str, SearchDocument]] = []
            for sort_mode, doc in pair_docs:
                if doc.url and doc.url not in seen_urls:
                    docs.append((sort_mode, doc))
                    seen_urls.add(doc.url)
                    docs_seen += 1
            if observer:
                observer(
                    {
                        "source_name": "reddit-problem",
                        "query_text": query,
                        "docs_seen": docs_seen,
                        "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                        "status": status,
                        "error": error,
                    }
                )
            return subreddit, query, docs

        compatible_pairs = [
            (subreddit, query)
            for subreddit in subreddits[:max_subs_wave]
            for query in queries[:max_kw_wave]
            if self._reddit_query_matches_subreddit(subreddit, query)
        ]

        pair_results = await asyncio.gather(
            *[
                _fetch_pair(subreddit, query)
                for subreddit, query in compatible_pairs
            ]
        )

        async def _build_finding(subreddit: str, query: str, sort_mode: str, doc: SearchDocument) -> Optional[dict[str, Any]]:
            async with context_sem:
                content = await self.reddit_thread_context(doc.url)
            full_text = compact_text(f"{doc.title} {doc.snippet} {content.get('text', '')}", 2400)
            if not self._is_problem_candidate(doc.title, full_text, source_url=doc.url):
                return None
            return {
                "source": f"reddit-problem/{subreddit}",
                "source_url": doc.url,
                "entrepreneur": "Operator or practitioner",
                "tool_used": self._extract_tools(full_text),
                "product_built": doc.title,
                "monetization_method": "",
                "outcome_summary": compact_text(full_text, 420),
                "finding_kind": "problem_signal",
                "recurrence_key": infer_recurrence_key(doc.title + " " + full_text[:200]),
                "evidence": {
                    "source_plan": "reddit-problem",
                    "discovery_query": query,
                    "discovery_sort": sort_mode,
                    "pain_score": self._pain_score(full_text),
                    "snippet": doc.snippet,
                    "page_excerpt": content.get("description", ""),
                    "comments": content.get("comments", []),
                    "comment_metadata": content.get("comment_metadata", []),
                },
            }

        built = await asyncio.gather(
            *[
                _build_finding(subreddit, query, sort_mode, doc)
                for subreddit, query, docs in pair_results
                for sort_mode, doc in docs
            ]
        )
        for item in built:
            if item:
                findings.append(item)
        return findings

    async def _discover_github_problem_threads(
        self,
        queries: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        queries = queries or [
            '"feature request" automation',
            '"wish there was" workflow',
            '"manual process" tool',
            '"too expensive" software',
            '"time consuming" issue',
        ]
        findings = await self.github_adapter.discover_items(
            queries=queries[:4],
            max_results_per_query=4,
            observer=observer,
        )
        return [finding for finding in findings if self._is_problem_candidate(finding.get("product_built", ""), finding.get("outcome_summary", ""), source_url=finding.get("source_url", ""))]

    async def _discover_web_problem_threads(
        self,
        queries: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        queries = queries or [
            '"wish there was" software for',
            '"too expensive" current tool',
            '"manual process" every day',
            '"need a better way" automate',
            '"frustrating" workflow',
        ]
        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for query in queries[:4]:
            started = asyncio.get_running_loop().time()
            docs = await self.search_web(query, max_results=3)
            if observer:
                observer(
                    {
                        "source_name": "web-problem",
                        "query_text": query,
                        "docs_seen": len(docs),
                        "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
                        "status": "ok",
                        "error": "",
                    }
                )
            for doc in docs:
                if doc.url in seen_urls:
                    continue
                seen_urls.add(doc.url)
                content = await self.fetch_content(doc.url)
                full_text = compact_text(f"{doc.title} {doc.snippet} {content.get('text', '')}", 2200)
                if self._is_low_quality_web_problem_page(
                    title=doc.title,
                    snippet=doc.snippet,
                    body=content.get("text", ""),
                    url=doc.url,
                ):
                    continue
                if not self._is_problem_candidate(doc.title, full_text, source_url=doc.url):
                    continue
                findings.append(
                    {
                        "source": "web-problem",
                        "source_url": doc.url,
                        "entrepreneur": "Operator or practitioner",
                        "tool_used": self._extract_tools(full_text),
                        "product_built": doc.title,
                        "monetization_method": "",
                        "outcome_summary": compact_text(full_text, 420),
                        "finding_kind": "problem_signal",
                        "recurrence_key": infer_recurrence_key(doc.title + " " + full_text[:200]),
                        "evidence": {
                            "source_plan": "web-problem",
                            "discovery_query": query,
                            "pain_score": self._pain_score(full_text),
                            "snippet": doc.snippet,
                            "page_excerpt": content.get("description", ""),
                        },
                    }
                )
        return findings

    async def _discover_wordpress_review_threads(
        self,
        plugin_slugs: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        wordpress_config = self.config.get("discovery", {}).get("wordpress_reviews", {})
        plugin_slugs = plugin_slugs or wordpress_config.get("plugin_slugs") or ["woocommerce", "updraftplus", "contact-form-7"]
        reviews_per_plugin = int(wordpress_config.get("reviews_per_plugin", 2))
        reviews = await self.wordpress_review_adapter.fetch_reviews(
            plugin_slugs=list(plugin_slugs)[:3],
            reviews_per_plugin=reviews_per_plugin,
            star_filters=wordpress_config.get("star_filters") or [1],
        )
        if observer:
            observer(
                {
                    "source_name": "wordpress-reviews",
                    "query_text": "wordpress_reviews",
                    "docs_seen": len(reviews),
                    "status": "ok",
                    "error": "",
                }
            )
        return [review.as_finding() for review in reviews]

    async def _discover_shopify_review_threads(
        self,
        app_handles: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        shopify_config = self.config.get("discovery", {}).get("shopify_reviews", {})
        # Allow empty list to trigger sitemap discovery
        if app_handles is None:
            app_handles = shopify_config.get("app_handles")
        if app_handles is None:
            app_handles = ["parcel-intelligence", "backup-and-sync"]
        max_apps = int(shopify_config.get("max_apps", 2))
        reviews = await self.shopify_review_adapter.fetch_reviews(
            app_handles=list(app_handles)[:max_apps],
            max_apps=max_apps,
            reviews_per_app=int(shopify_config.get("reviews_per_app", 2)),
            rating_filters=shopify_config.get("rating_filters") or [1],
            sort_by=str(shopify_config.get("sort_by", "newest")),
        )
        if observer:
            observer(
                {
                    "source_name": "shopify-reviews",
                    "query_text": "shopify_reviews",
                    "docs_seen": len(reviews),
                    "status": "ok",
                    "error": "",
                }
            )
        return [review.as_finding() for review in reviews]

    async def _discover_marketplace_problem_threads(
        self,
        queries: Optional[list[str]] = None,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        wordpress_findings, shopify_findings = await asyncio.gather(
            self._discover_wordpress_review_threads(observer=observer),
            self._discover_shopify_review_threads(observer=observer),
        )
        findings = wordpress_findings + shopify_findings
        if queries:
            # Keep a lightweight web lane for marketplace-adjacent pain when explicitly queried.
            findings.extend(await self._discover_web_problem_threads(queries=queries[:2], observer=observer))
        return findings

    async def validate_problem(
        self,
        title: str,
        summary: str,
        finding_kind: str,
        audience_hint: str = "",
        atom: Optional[Any] = None,
    ) -> dict[str, Any]:
        evidence_query = self._build_validation_query(title, summary)
        recurrence_queries = self.build_recurrence_queries(title=title, summary=summary, atom=atom)
        recurrence_timeout = False
        competitor_timeout = False
        try:
            recurrence_docs, recurrence_meta = await asyncio.wait_for(
                self.gather_recurrence_evidence(
                    recurrence_queries,
                    finding_kind=finding_kind,
                    atom=atom,
                ),
                timeout=self.validation_recurrence_budget_seconds,
            )
        except asyncio.TimeoutError:
            recurrence_timeout = True
            recurrence_docs = []
            recurrence_meta = {
                "recurrence_score": 0.0,
                "recurrence_state": "timeout",
                "recurrence_gap_reason": "recurrence_budget_timeout",
                "recurrence_failure_class": "budget_exhausted",
                "query_coverage": 0.0,
                "doc_count": 0,
                "domain_count": 0,
                "results_by_query": {query: 0 for query in recurrence_queries},
                "results_by_source": {
                    "web": 0,
                    "reddit": 0,
                    "github": 0,
                    "stackoverflow": 0,
                    "etsy": 0,
                    "forum_fallback": 0,
                },
                "matched_results_by_source": {},
                "partial_results_by_source": {},
                "matched_docs_by_source": {},
                "partial_docs_by_source": {},
                "family_confirmation_count": 0,
                "source_yield": {},
                "reshaped_query_history": [],
                "warmed_validation_queries": {
                    "seed_runs": 0,
                    "seeded_pairs": 0,
                    "seeded_searches": 0,
                    "uncovered_before": 0,
                    "uncovered_after": 0,
                },
                "queries_considered": list(recurrence_queries),
                "queries_executed": [],
                "recurrence_budget_profile": self._recurrence_budget_profile(atom),
                "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
                "recurrence_probe_summary": {},
                "recurrence_source_branch": {
                    "triggered": False,
                    "missing_sources": [],
                    "queries": [],
                    "added_docs": 0,
                    "reshaped_query_history": [],
                    "last_action": "STOP_FOR_BUDGET",
                    "last_transition_reason": "recurrence_budget_timeout",
                    "chosen_family": "",
                    "expected_gain_class": "low",
                    "source_attempts_snapshot": {},
                    "skipped_families": {},
                    "controller_actions": [],
                    "budget_snapshot": self._recurrence_budget_profile(atom),
                    "fallback_strategy_used": "",
                    "decomposed_atom_queries": [],
                    "routing_override_reason": "",
                    "cohort_query_pack_used": False,
                    "cohort_query_pack_name": "",
                    "web_query_strategy_path": [],
                    "specialized_surface_targeting_used": False,
                    "promotion_gap_class": "mixed_gap",
                    "near_miss_enrichment_action": "",
                    "sufficiency_priority_reason": "",
                    "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
                },
                "last_action": "STOP_FOR_BUDGET",
                "last_transition_reason": "recurrence_budget_timeout",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": self._recurrence_budget_profile(atom),
                "fallback_strategy_used": "",
                "decomposed_atom_queries": [],
                "routing_override_reason": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "promotion_gap_class": "mixed_gap",
                "near_miss_enrichment_action": "",
                "sufficiency_priority_reason": "",
            }
        competitor_query = self._build_competitor_query(title=title, summary=summary, atom=atom)
        try:
            competitor_docs = await asyncio.wait_for(
                self.search_web(
                    competitor_query,
                    max_results=self.validation_competitor_limit,
                    intent="validation_competitor",
                ),
                timeout=self.validation_competitor_budget_seconds,
            )
        except asyncio.TimeoutError:
            competitor_timeout = True
            competitor_docs = []

        recurrence_domains = {domain_for(doc.url) for doc in recurrence_docs if doc.url}
        competitor_domains = {domain_for(doc.url) for doc in competitor_docs if doc.url}
        competitor_domains = {domain for domain in competitor_domains if domain}
        recurrence_score = recurrence_meta["recurrence_score"]

        pain_text = f"{title} {summary}".lower()
        value_hits = sum(1 for keyword in VALUE_KEYWORDS if contains_keyword(pain_text, keyword))
        pain_hits = sum(1 for keyword in PAIN_KEYWORDS if contains_keyword(pain_text, keyword))
        money_hits = len(re.findall(r"\$[\d,.]+(?:[kKmM])?", pain_text))
        value_score = min(1.0, 0.25 + pain_hits * 0.08 + value_hits * 0.07 + money_hits * 0.1)
        promotion_gap_class = self._classify_promotion_gap(
            atom=atom,
            recurrence_state=str(recurrence_meta.get("recurrence_state", "") or ""),
            recurrence_score=float(recurrence_score or 0.0),
            family_confirmation_count=int(recurrence_meta.get("family_confirmation_count", 0) or 0),
            strong_match_count=sum(int(v or 0) for v in (recurrence_meta.get("matched_results_by_source", {}) or {}).values()),
            partial_match_count=sum(int(v or 0) for v in (recurrence_meta.get("partial_results_by_source", {}) or {}).values()),
            query_coverage=float(recurrence_meta.get("query_coverage", 0.0) or 0.0),
            value_signal=value_score,
        )
        near_miss_enrichment_action = self._choose_near_miss_enrichment_action(
            promotion_gap_class=promotion_gap_class,
            family_confirmation_count=int(recurrence_meta.get("family_confirmation_count", 0) or 0),
            matched_results_by_source=recurrence_meta.get("matched_results_by_source", {}) or {},
            partial_results_by_source=recurrence_meta.get("partial_results_by_source", {}) or {},
            source_attempts_by_family={
                source: int(details.get("attempts", 0) or 0)
                for source, details in (recurrence_meta.get("source_yield", {}) or {}).items()
            },
            budget_profile=recurrence_meta.get("recurrence_budget_profile", {}) or {},
            available_families=list((recurrence_meta.get("source_yield", {}) or {}).keys()) or ["web", "github"],
        )
        value_enrichment_docs: list[SearchDocument] = []
        value_enrichment_queries: list[str] = []
        value_enrichment_used = False
        if near_miss_enrichment_action.get("action") == "GATHER_MARKET_ENRICHMENT":
            value_enrichment_queries = self._build_value_enrichment_queries(
                title=title,
                summary=summary,
                atom=atom,
                audience_hint=audience_hint,
            )
            per_query_budget = max(1.0, self.validation_competitor_budget_seconds / max(len(value_enrichment_queries), 2))
            for query in value_enrichment_queries:
                try:
                    docs = await asyncio.wait_for(
                        self.search_web(
                            query,
                            max_results=max(2, min(4, self.validation_competitor_limit // 2 or 2)),
                            intent="validation_value_enrichment",
                        ),
                        timeout=per_query_budget,
                    )
                except asyncio.TimeoutError:
                    continue
                value_enrichment_docs.extend(docs)
            if value_enrichment_docs:
                value_enrichment_used = True
                value_score = min(1.0, value_score + self._value_enrichment_signal_bonus(value_enrichment_docs))
                promotion_gap_class = self._classify_promotion_gap(
                    atom=atom,
                    recurrence_state=str(recurrence_meta.get("recurrence_state", "") or ""),
                    recurrence_score=float(recurrence_score or 0.0),
                    family_confirmation_count=int(recurrence_meta.get("family_confirmation_count", 0) or 0),
                    strong_match_count=sum(int(v or 0) for v in (recurrence_meta.get("matched_results_by_source", {}) or {}).values()),
                    partial_match_count=sum(int(v or 0) for v in (recurrence_meta.get("partial_results_by_source", {}) or {}).values()),
                    query_coverage=float(recurrence_meta.get("query_coverage", 0.0) or 0.0),
                    value_signal=value_score,
                )

        software_hits = sum(
            1 for domain in competitor_domains if domain in SOFTWARE_DOMAINS or "app" in domain or "tool" in domain
        )
        solution_gap_score = max(0.0, 1.0 - min(software_hits / 6.0, 1.0))
        saturation_score = max(0.0, 1.0 - min(len(competitor_domains) / 10.0, 1.0))

        ai_fit = 0.55 if any(contains_keyword(pain_text, keyword) for keyword in AI_TOOL_KEYWORDS) else 0.4
        operational_fit = 0.1 if audience_hint or any(contains_keyword(pain_text, keyword) for keyword in VALUE_KEYWORDS) else 0.0
        feasibility_score = min(1.0, ai_fit + operational_fit + solution_gap_score * 0.2)

        evidence = {
            "query": evidence_query,
            "recurrence_queries": recurrence_queries,
            "competitor_query": competitor_query,
            "recurrence_docs": [doc.__dict__ for doc in self._source_diverse_sample(recurrence_docs, self.validation_evidence_sample)],
            "_all_recurrence_docs": [doc.__dict__ for doc in recurrence_docs],
            "competitor_docs": [doc.__dict__ for doc in competitor_docs[: self.validation_evidence_sample]],
            "recurrence_domains": sorted(recurrence_domains),
            "competitor_domains": sorted(competitor_domains),
            "recurrence_state": recurrence_meta["recurrence_state"],
            "recurrence_query_coverage": recurrence_meta["query_coverage"],
            "recurrence_doc_count": recurrence_meta["doc_count"],
            "recurrence_domain_count": recurrence_meta["domain_count"],
            "recurrence_results_by_query": recurrence_meta["results_by_query"],
            "recurrence_results_by_source": recurrence_meta["results_by_source"],
            "matched_results_by_source": recurrence_meta.get("matched_results_by_source", {}),
            "partial_results_by_source": recurrence_meta.get("partial_results_by_source", {}),
            "matched_docs_by_source": recurrence_meta.get("matched_docs_by_source", {}),
            "partial_docs_by_source": recurrence_meta.get("partial_docs_by_source", {}),
            "family_confirmation_count": recurrence_meta.get("family_confirmation_count", 0),
            "source_yield": recurrence_meta.get("source_yield", {}),
            "reshaped_query_history": recurrence_meta.get("reshaped_query_history", []),
            "recurrence_gap_reason": recurrence_meta.get("recurrence_gap_reason", ""),
            "recurrence_failure_class": recurrence_meta.get("recurrence_failure_class", ""),
            "queries_considered": recurrence_meta.get("queries_considered", []),
            "queries_executed": recurrence_meta.get("queries_executed", []),
            "recurrence_budget_profile": recurrence_meta.get("recurrence_budget_profile", {}),
            "candidate_meaningful": recurrence_meta.get("candidate_meaningful", {}),
            "recurrence_probe_summary": recurrence_meta.get("recurrence_probe_summary", {}),
            "recurrence_source_branch": recurrence_meta.get("recurrence_source_branch", {}),
            "last_action": recurrence_meta.get("last_action", ""),
            "last_transition_reason": recurrence_meta.get("last_transition_reason", ""),
            "chosen_family": recurrence_meta.get("chosen_family", ""),
            "expected_gain_class": recurrence_meta.get("expected_gain_class", ""),
            "source_attempts_snapshot": recurrence_meta.get("source_attempts_snapshot", {}),
            "skipped_families": recurrence_meta.get("skipped_families", {}),
            "controller_actions": recurrence_meta.get("controller_actions", []),
            "budget_snapshot": recurrence_meta.get("budget_snapshot", {}),
            "fallback_strategy_used": recurrence_meta.get("fallback_strategy_used", ""),
            "decomposed_atom_queries": recurrence_meta.get("decomposed_atom_queries", []),
            "routing_override_reason": recurrence_meta.get("routing_override_reason", ""),
            "cohort_query_pack_used": recurrence_meta.get("cohort_query_pack_used", False),
            "cohort_query_pack_name": recurrence_meta.get("cohort_query_pack_name", ""),
            "web_query_strategy_path": recurrence_meta.get("web_query_strategy_path", []),
            "specialized_surface_targeting_used": recurrence_meta.get("specialized_surface_targeting_used", False),
            "promotion_gap_class": promotion_gap_class or recurrence_meta.get("promotion_gap_class", ""),
            "near_miss_enrichment_action": near_miss_enrichment_action.get("action", "") or recurrence_meta.get("near_miss_enrichment_action", ""),
            "sufficiency_priority_reason": near_miss_enrichment_action.get("sufficiency_priority_reason", "") or recurrence_meta.get("sufficiency_priority_reason", ""),
            "pain_hits": pain_hits,
            "value_hits": value_hits,
            "software_hits": software_hits,
            "value_enrichment_used": value_enrichment_used,
            "value_enrichment_queries": value_enrichment_queries,
            "value_enrichment_docs": [doc.__dict__ for doc in value_enrichment_docs[: self.validation_evidence_sample]],
            "recurrence_timeout": recurrence_timeout,
            "competitor_timeout": competitor_timeout,
            "warmed_validation_queries": recurrence_meta.get("warmed_validation_queries", {}),
        }

        return {
            "problem_score": recurrence_score,
            "solution_gap_score": solution_gap_score,
            "saturation_score": saturation_score,
            "feasibility_score": feasibility_score,
            "value_score": value_score,
            "evidence": evidence,
        }

    def build_recurrence_queries(
        self,
        *,
        title: str,
        summary: str,
        atom: Optional[Any] = None,
        limit: int = 4,
    ) -> list[str]:
        queries: list[str] = []

        def add(query: str) -> None:
            normalized = self._normalize_recurrence_query(query)
            if normalized and normalized not in queries:
                queries.append(normalized)

        if atom is not None:
            cohort_haystack = normalize_content(
                " ".join(
                    [
                        title or "",
                        summary or "",
                        getattr(atom, "segment", "") or "",
                        getattr(atom, "user_role", "") or "",
                        getattr(atom, "job_to_be_done", "") or "",
                        getattr(atom, "failure_mode", "") or "",
                        getattr(atom, "trigger_event", "") or "",
                        getattr(atom, "current_workaround", "") or "",
                        getattr(atom, "current_tools", "") or "",
                    ]
                )
            )
            accounting_focus = any(marker in cohort_haystack for marker in [
                "reconciliation", "bank", "deposit", "invoice", "payment", "payout", "quickbooks", "qbo", "month end", "close",
            ])
            seller_reporting_focus = (
                any(marker in cohort_haystack for marker in ["shopify", "amazon", "etsy", "seller", "merchant", "store"])
                and any(marker in cohort_haystack for marker in ["profitability", "revenue", "payout", "bank deposit", "spreadsheet", "reporting", "reconciliation"])
            )
            role_terms = self._recurrence_role_terms(getattr(atom, "user_role", "") or "")
            segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
            job_terms = _query_term_span(getattr(atom, "job_to_be_done", "") or "", max_terms=5)
            tool_terms = _query_term_span(getattr(atom, "current_tools", "") or "", max_terms=3)
            job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6)
            failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)
            trigger_phrase = self._recurrence_focus_phrase(getattr(atom, "trigger_event", "") or "", max_words=5)
            workaround_phrase = self._recurrence_focus_phrase(getattr(atom, "current_workaround", "") or "", max_words=4)
            cost_phrase = self._recurrence_focus_phrase(getattr(atom, "cost_consequence_clues", "") or "", max_words=3)

            if seller_reporting_focus:
                for query in [
                    '"shopify amazon etsy" "payout reconciliation"',
                    '"sales channel profitability" spreadsheet',
                    '"bank deposits" payouts spreadsheet',
                    '"channel profitability reporting" spreadsheet',
                ]:
                    add(query)
            elif accounting_focus:
                for query in [
                    '"manual reconciliation" "small business"',
                    '"payment reconciliation" "small business"',
                    '"bank deposits not matching invoices" spreadsheet',
                    '"quickbooks payout reconciliation" forum',
                ]:
                    add(query)

            for parts in [
                [failure_phrase, workaround_phrase or tool_terms, segment_terms],
                [trigger_phrase or failure_phrase, job_phrase or job_terms, role_terms],
                [job_terms, tool_terms or workaround_phrase, segment_terms],
                [failure_phrase or trigger_phrase, segment_terms, cost_phrase],
            ]:
                add(" ".join(part for part in parts if part))

            # Add broader, concept-level queries that generalize beyond the
            # specific finding.  These use the core problem nouns/verbs rather
            # than exact phrases, so they find corroborating pain signals even
            # when the vocabulary differs.
            core_concepts = self._extract_core_problem_concepts(atom)
            if core_concepts:
                for combo in [
                    " ".join(core_concepts[:3]),
                    " ".join(core_concepts[:2]) + " pain point",
                    " ".join(core_concepts[:2]) + " frustration",
                ]:
                    add(combo)

        base_query = self._build_validation_query(title, summary)
        if not queries:
            add(base_query)
        elif len(queries) < 2:
            shortened_base = " ".join(base_query.split()[: min(self.validation_query_terms, 5)])
            add(shortened_base)
        profile = self._recurrence_budget_profile(atom)
        effective_limit = min(limit, profile["query_limit"])
        return queries[:effective_limit]

    def _build_competitor_query(self, *, title: str, summary: str, atom: Optional[Any] = None) -> str:
        if atom is None:
            return f"{self._build_validation_query(title, summary)} software tool alternative"

        parts: list[str] = []

        def add(part: str) -> None:
            normalized = self._normalize_recurrence_query(part)
            if normalized and normalized not in parts:
                parts.append(normalized)

        failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)
        trigger_phrase = self._recurrence_focus_phrase(getattr(atom, "trigger_event", "") or "", max_words=5)
        job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6)
        segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
        tool_terms = _query_term_span(getattr(atom, "current_tools", "") or "", max_terms=3)

        add(" ".join(part for part in [failure_phrase or trigger_phrase, segment_terms] if part))
        add(" ".join(part for part in [job_phrase, tool_terms or segment_terms] if part))

        base = next((part for part in parts if part), self._build_validation_query(title, summary))
        base_terms = query_phrases(base) + query_terms(base)
        compact_base = " ".join(base_terms[: max(4, self.validation_query_terms - 3)]).strip()
        return f"{compact_base} software tool alternative".strip()

    def _build_value_enrichment_queries(
        self,
        *,
        title: str,
        summary: str,
        atom: Optional[Any] = None,
        audience_hint: str = "",
    ) -> list[str]:
        queries: list[str] = []
        query_limit = 2

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        if atom is not None:
            fragility_plan = self._build_corroboration_plan(
                atom=atom,
                queries=[self._build_validation_query(title, summary)],
                finding_kind="problem_signal",
            )
            job_phrase = self._recurrence_query_seed(getattr(atom, "job_to_be_done", "") or "", max_terms=4)
            failure_phrase = self._recurrence_query_seed(getattr(atom, "failure_mode", "") or "", max_terms=4)
            workaround_phrase = self._recurrence_query_seed(getattr(atom, "current_workaround", "") or "", max_terms=3)
            role_terms = self._recurrence_role_terms(getattr(atom, "user_role", "") or "")
            segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
            cost_terms = self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=2)
            cost_seed = " ".join(cost_terms) or "time cost"
            if self._is_workflow_fragility_cohort(atom=atom, plan=fragility_plan):
                query_limit = 5
                add("outgrew excel operations", "hours lost")
                add("spreadsheet version confusion", "error cost")
                add("manual handoff workflow", "hours lost")
                add("status chasing manual handoff", "headcount drag")
                add("replace spreadsheets workflow software", "too expensive")
                add("shared workbook conflict", "time loss")
            add(job_phrase or failure_phrase, workaround_phrase or "manual workflow", cost_seed)
            add(failure_phrase or job_phrase, role_terms or segment_terms, "hours risk")
            add(job_phrase or self._build_validation_query(title, summary), "software replacement")
        if not queries:
            base = self._build_validation_query(title, summary)
            add(base, "manual hours")
            add(base, audience_hint, "cost risk")
        return queries[:query_limit]

    def _value_enrichment_signal_bonus(self, docs: list[SearchDocument]) -> float:
        if not docs:
            return 0.0
        text = normalize_content(" ".join(" ".join([doc.title or "", doc.snippet or ""]) for doc in docs))
        if not text:
            return 0.0
        value_hits = sum(1 for keyword in VALUE_KEYWORDS if contains_keyword(text, keyword))
        cost_hits = sum(1 for keyword in COST_SIGNAL_TERMS if contains_keyword(text, keyword))
        workaround_hits = sum(1 for keyword in WORKAROUND_SIGNAL_TERMS if contains_keyword(text, keyword))
        workflow_fragility_value_terms = [
            "outgrew excel",
            "hours lost",
            "hours a week",
            "manual reconciliation",
            "shared workbook conflict",
            "version confusion",
            "wrong spreadsheet",
            "missed step",
            "status chasing",
            "headcount",
            "hiring",
            "time spent",
            "costly error",
            "tool to replace spreadsheets",
            "replaced spreadsheets",
            "pay for software",
            "willing to pay",
        ]
        fragility_hits = sum(1 for term in workflow_fragility_value_terms if term in text)
        return min(0.18, value_hits * 0.02 + cost_hits * 0.025 + workaround_hits * 0.015 + fragility_hits * 0.018)

    def _recurrence_focus_phrase(self, text: str, *, max_words: int = 5) -> str:
        cleaned = _clean_recurrence_text(text)
        if not cleaned:
            return ""

        validation_phrases = self._extract_validation_phrases(cleaned)
        if validation_phrases:
            return f'"{validation_phrases[0]}"'

        terms = [
            token
            for token in re.findall(r"[a-z0-9&/-]+", cleaned)
            if token not in QUERY_STOPWORDS
            and token not in WEAK_VALIDATION_TERMS
            and token not in RECURRENCE_NOISE_TERMS
            and token not in self._PERSONAL_NARRATIVE_TERMS
            and len(token) > 2
        ]
        if len(terms) < 2:
            return ""
        return '"' + " ".join(terms[:max_words]) + '"'

    # Terms that signal a specific first-person narrative rather than a
    # generalizable problem concept.  When these dominate, the extracted
    # focus phrase is unlikely to match external corroborating docs.
    _PERSONAL_NARRATIVE_TERMS = frozenset({
        "ive", "id", "im", "my", "myself", "kept", "telling",
        "tried", "finally", "decided", "wanted",
        "years", "day", "one", "wish", "wished",
    })

    def _extract_core_problem_concepts(self, atom: Optional[Any]) -> list[str]:
        """Extract generalized problem-concept nouns/verbs from the atom.

        These are domain-agnostic terms that describe WHAT hurts (e.g.,
        "google contacts backup", "manual csv export") rather than the
        specific narrative (e.g., "years kept telling myself").
        """
        if atom is None:
            return []
        # Combine job_to_be_done + failure_mode + trigger_event, but
        # strip personal-narrative noise and pick the most meaningful
        # noun-like tokens.
        raw = " ".join([
            getattr(atom, "job_to_be_done", "") or "",
            getattr(atom, "trigger_event", "") or "",
            getattr(atom, "failure_mode", "") or "",
            getattr(atom, "current_workaround", "") or "",
            getattr(atom, "current_tools", "") or "",
        ])
        cleaned = _clean_recurrence_text(raw)
        if not cleaned:
            return []
        tokens = [
            t for t in re.findall(r"[a-z0-9&/-]+", cleaned)
            if t not in QUERY_STOPWORDS
            and t not in RECURRENCE_NOISE_TERMS
            and t not in WEAK_VALIDATION_TERMS
            and t not in self._PERSONAL_NARRATIVE_TERMS
            and len(t) > 2
        ]
        # Prioritize domain/product nouns (capitalized in original) and
        # action verbs.  Simple heuristic: longer tokens and those that
        # appear in the job_to_be_done are more likely core concepts.
        jtbd_tokens = set(
            re.findall(r"[a-z0-9]+", (getattr(atom, "job_to_be_done", "") or "").lower())
        )
        scored = sorted(
            tokens,
            key=lambda t: (
                2 if t in jtbd_tokens else 0,
                min(len(t), 8),
            ),
            reverse=True,
        )
        return scored[:5]

    def _recurrence_role_terms(self, text: str) -> str:
        cleaned = _clean_recurrence_text(text)
        if not cleaned:
            return ""
        tokens = [
            token
            for token in re.findall(r"[a-z0-9&/-]+", cleaned)
            if token not in QUERY_STOPWORDS and token not in RECURRENCE_NOISE_TERMS and len(token) > 2
        ]
        if not tokens:
            return ""
        return " ".join(tokens[:3])

    def _recurrence_segment_terms(self, text: str) -> str:
        cleaned = _clean_recurrence_text(text)
        if not cleaned:
            return ""
        if "small business" in cleaned:
            return '"small business"'
        if "etsy" in cleaned and "seller" in cleaned:
            return '"etsy seller"'
        return _query_term_span(cleaned, max_terms=3)

    def _normalize_recurrence_query(self, query: str) -> str:
        phrases = [f'"{" ".join(phrase.split())}"' for phrase in query_phrases(query)[:2]]
        unquoted = re.sub(r'"[^"]+"', " ", query.lower())
        tokens: list[str] = []
        for token in re.findall(r"[a-z0-9&/-]+", _clean_recurrence_text(unquoted)):
            if token in QUERY_STOPWORDS or token in WEAK_VALIDATION_TERMS or token in RECURRENCE_NOISE_TERMS:
                continue
            if len(token) <= 2:
                continue
            if token not in tokens:
                tokens.append(token)

        parts = phrases + tokens[: max(0, self.validation_query_terms - len(phrases) * 2)]
        normalized = " ".join(parts[: self.validation_query_terms]).strip()
        if len(query_terms(normalized)) + len(query_phrases(normalized)) < 2:
            return ""
        return normalized

    def _recurrence_query_seed(self, text: str, *, max_terms: int = 4) -> str:
        cleaned = _clean_recurrence_text(text)
        if not cleaned:
            return ""
        phrase_terms = query_phrases(text)
        if phrase_terms:
            seeded = _query_term_span(phrase_terms[0], max_terms=max_terms)
            if seeded:
                return seeded
        return _query_term_span(cleaned, max_terms=max_terms)

    def _corroboration_terms(self, text: str, *, max_terms: int = 4) -> list[str]:
        cleaned = _clean_recurrence_text(text)
        if not cleaned:
            return []
        terms: list[str] = []
        for token in re.findall(r"[a-z0-9&/-]+", cleaned):
            if token in QUERY_STOPWORDS or token in WEAK_VALIDATION_TERMS or token in RECURRENCE_NOISE_TERMS:
                continue
            if len(token) <= 2 or token in terms:
                continue
            terms.append(token)
        return terms[:max_terms]

    def _build_corroboration_plan(
        self,
        *,
        atom: Optional[Any],
        queries: list[str],
        finding_kind: str,
    ) -> CorroborationPlan:
        role_terms = self._corroboration_terms(getattr(atom, "user_role", "") or "", max_terms=3) if atom is not None else []
        segment_terms = self._corroboration_terms(getattr(atom, "segment", "") or "", max_terms=3) if atom is not None else []
        job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6) if atom is not None else ""
        failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5) if atom is not None else ""
        workaround_phrase = self._recurrence_focus_phrase(getattr(atom, "current_workaround", "") or "", max_words=4) if atom is not None else ""
        cost_terms = self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=3) if atom is not None else []
        ecosystem_hints = self._corroboration_terms(getattr(atom, "current_tools", "") or "", max_terms=4) if atom is not None else []

        signature_terms: list[str] = []
        for pool in (
            self._corroboration_terms(getattr(atom, "job_to_be_done", "") or "", max_terms=4) if atom is not None else [],
            self._corroboration_terms(getattr(atom, "failure_mode", "") or "", max_terms=4) if atom is not None else [],
            self._corroboration_terms(getattr(atom, "current_workaround", "") or "", max_terms=3) if atom is not None else [],
            cost_terms,
            ecosystem_hints[:2],
        ):
            for term in pool:
                if term not in signature_terms:
                    signature_terms.append(term)
        for query in queries:
            for term in self._corroboration_terms(query, max_terms=4):
                if term not in signature_terms:
                    signature_terms.append(term)
        family_queries = {
            "github": self._github_recurrence_queries_from_atom(
                atom=atom,
                signature_terms=signature_terms,
                role_terms=role_terms,
                segment_terms=segment_terms,
                job_phrase=job_phrase,
                failure_phrase=failure_phrase,
                workaround_phrase=workaround_phrase,
                cost_terms=cost_terms,
                ecosystem_hints=ecosystem_hints,
            ),
            "web": self._web_recurrence_queries_from_atom(
                atom=atom,
                signature_terms=signature_terms,
                role_terms=role_terms,
                segment_terms=segment_terms,
                job_phrase=job_phrase,
                failure_phrase=failure_phrase,
                workaround_phrase=workaround_phrase,
                cost_terms=cost_terms,
                ecosystem_hints=ecosystem_hints,
            ),
        }
        source_priority = ("reddit", "web", "github", "stackoverflow", "etsy")
        max_attempts_per_family = 2
        cohort_haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(signature_terms),
                    " ".join(ecosystem_hints),
                ]
            )
        ) if atom is not None else ""
        accounting_focus = any(marker in cohort_haystack for marker in [
            "reconciliation", "bank", "deposit", "month end", "close", "invoice", "payment", "payout", "quickbooks", "qbo", "stripe",
        ])
        seller_reporting_focus = (
            any(marker in cohort_haystack for marker in ["shopify", "etsy", "amazon", "seller", "merchant", "sales channel", "channel"])
            and any(marker in cohort_haystack for marker in ["profitability", "revenue", "reconciliation", "payout", "bank deposit", "spreadsheet", "reporting"])
        )
        if atom is not None and not self._atom_supports_github_recurrence(
            atom=atom,
            role_terms=role_terms,
            segment_terms=segment_terms,
            ecosystem_hints=ecosystem_hints,
            signature_terms=signature_terms,
        ):
            source_priority = ("reddit", "web", "stackoverflow", "etsy", "github")
        if atom is not None:
            budget_profile = self._recurrence_budget_profile(atom)
            meaningful = self._meaningful_candidate_snapshot(atom)
            specificity_score = float(budget_profile.get("specificity_score", 0.0) or 0.0)
            if meaningful["meaningful_candidate"] and specificity_score >= 0.72:
                if accounting_focus or seller_reporting_focus:
                    source_priority = ("reddit", "web", "github", "stackoverflow", "etsy")
                    # Preserve retry slot: high-specificity atoms need at least 2
                    # attempts so _choose_corroboration_action can reach the
                    # high_specificity_cross_source_retry / practitioner_retry path.
                    max_attempts_per_family = 2
                elif self._atom_supports_stackoverflow_recurrence(atom):
                    source_priority = ("web", "stackoverflow", "github", "reddit", "etsy")
                else:
                    source_priority = ("web", "github", "reddit", "stackoverflow", "etsy")
            elif accounting_focus or seller_reporting_focus:
                source_priority = ("reddit", "web", "github", "stackoverflow", "etsy")
                max_attempts_per_family = 1
        if finding_kind == "pain_point":
            source_priority = ("web", "github", "reddit", "stackoverflow", "etsy")
        return CorroborationPlan(
            signature_terms=signature_terms[:10],
            role_terms=role_terms,
            segment_terms=segment_terms,
            job_phrase=job_phrase,
            failure_phrase=failure_phrase,
            workaround_phrase=workaround_phrase,
            cost_terms=cost_terms,
            ecosystem_hints=ecosystem_hints,
            family_queries=family_queries,
            max_attempts_per_family=max_attempts_per_family,
            source_priority=source_priority,
        )

    def _atom_supports_github_recurrence(
        self,
        *,
        atom: Optional[Any],
        role_terms: Optional[list[str]] = None,
        segment_terms: Optional[list[str]] = None,
        ecosystem_hints: Optional[list[str]] = None,
        signature_terms: Optional[list[str]] = None,
    ) -> bool:
        if atom is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(role_terms or []),
                    " ".join(segment_terms or []),
                    " ".join(ecosystem_hints or []),
                    " ".join(signature_terms or []),
                ]
            )
        )
        haystack_terms = set(haystack.split())

        def contains_marker(marker: str) -> bool:
            normalized = normalize_content(marker)
            if not normalized:
                return False
            if " " in normalized:
                return normalized in haystack
            return normalized in haystack_terms

        workflow_fragility_markers = [
            "duct tape",
            "spreadsheet",
            "spreadsheets",
            "excel",
            "google sheets",
            "csv",
            "copy paste",
            "handoff",
            "latest version",
            "latest file",
            "out of sync",
            "missed step",
            "manual follow-up",
            "version confusion",
        ]
        explicit_engineering_markers = [
            "developer",
            "engineer",
            "devops",
            "sysadmin",
            "api",
            "sdk",
            "webhook",
            "oauth",
            "plugin",
            "extension",
            "integration",
            "deployment",
            "script",
            "cli",
            "terraform",
            "kubernetes",
            "docker",
            "repo",
            "github",
        ]
        technical_markers = [
            "developer",
            "engineer",
            "sysadmin",
            "devops",
            "admin console",
            "api",
            "sdk",
            "webhook",
            "oauth",
            "plugin",
            "extension",
            "integration",
            "deployment",
            "config",
            "configuration",
            "script",
            "cli",
            "terraform",
            "kubernetes",
            "docker",
            "m365",
            "automation",
            "repo",
            "github",
        ]
        tool_surface_markers = [
            "zapier",
            "airtable",
            "notion",
            "hubspot",
            "salesforce",
            "shopify",
            "woocommerce",
            "wordpress",
            "slack",
            "jira",
            "stripe",
        ]
        # Spreadsheet-glue workflow pain often references automation tools, but that
        # should not push corroboration into GitHub unless the atom is also clearly
        # technical in its own right.
        if any(contains_marker(marker) for marker in workflow_fragility_markers) and not any(
            contains_marker(marker) for marker in explicit_engineering_markers
        ):
            return False
        if any(contains_marker(marker) for marker in technical_markers):
            return True
        tool_integration_markers = [
            "api",
            "sdk",
            "webhook",
            "oauth",
            "plugin",
            "extension",
            "integration",
            "deployment",
            "config",
            "configuration",
            "script",
            "cli",
            "automation",
            "import",
            "export",
            "sync",
        ]
        if any(contains_marker(marker) for marker in tool_surface_markers) and any(
            contains_marker(marker) for marker in tool_integration_markers
        ):
            return True
        return False

    def _meaningful_candidate_snapshot(self, atom: Optional[Any]) -> dict[str, Any]:
        if atom is None:
            return {
                "identity_context": False,
                "jtbd_present": False,
                "failure_present": False,
                "support_present": False,
                "meaningful_candidate": False,
            }
        identity_context = bool((getattr(atom, "segment", "") or "").strip() or (getattr(atom, "user_role", "") or "").strip())
        jtbd_present = bool((getattr(atom, "job_to_be_done", "") or "").strip())
        failure_present = bool((getattr(atom, "failure_mode", "") or "").strip())
        support_present = bool((getattr(atom, "current_workaround", "") or "").strip() or (getattr(atom, "cost_consequence_clues", "") or "").strip())
        return {
            "identity_context": identity_context,
            "jtbd_present": jtbd_present,
            "failure_present": failure_present,
            "support_present": support_present,
            "meaningful_candidate": bool(identity_context and jtbd_present and failure_present and support_present),
        }

    def _recurrence_corroboration_proxy(
        self,
        *,
        recurrence_score: float,
        family_confirmation_count: int,
        strong_match_count: int,
        partial_match_count: int,
        query_coverage: float,
    ) -> float:
        return min(
            1.0,
            max(
                0.0,
                recurrence_score * 0.42
                + min(family_confirmation_count / 2.0, 1.0) * 0.26
                + min(strong_match_count / 3.0, 1.0) * 0.18
                + min(partial_match_count / 4.0, 1.0) * 0.06
                + query_coverage * 0.08,
            ),
        )

    def _classify_promotion_gap(
        self,
        *,
        atom: Optional[Any],
        recurrence_state: str,
        recurrence_score: float,
        family_confirmation_count: int,
        strong_match_count: int,
        partial_match_count: int,
        query_coverage: float,
        value_signal: float,
        economic_signal: float = 0.0,
    ) -> str:
        meaningful = self._meaningful_candidate_snapshot(atom)
        corroboration_proxy = self._recurrence_corroboration_proxy(
            recurrence_score=recurrence_score,
            family_confirmation_count=family_confirmation_count,
            strong_match_count=strong_match_count,
            partial_match_count=partial_match_count,
            query_coverage=query_coverage,
        )
        if not meaningful["meaningful_candidate"] and recurrence_score < 0.2 and value_signal < 0.3:
            return "economically_weak"
        if economic_signal < 0.18 and recurrence_score < 0.2 and value_signal < 0.28:
            return "economically_weak"
        supported_single_family_near_miss = (
            recurrence_state in {"supported", "strong"}
            and family_confirmation_count == 1
            and query_coverage >= 0.45
            and (
                strong_match_count >= 2
                or (strong_match_count >= 1 and partial_match_count >= 1)
            )
        )
        corroboration_short = (
            recurrence_state == "timeout"
            or family_confirmation_count < 2
            or corroboration_proxy < 0.55
        )
        value_short = value_signal < 0.45
        strict_value_short = value_signal < 0.55
        single_family_value_weak = value_signal < 0.38
        if supported_single_family_near_miss and single_family_value_weak:
            return "value_gap"
        if supported_single_family_near_miss and not value_short:
            return "corroboration_gap"
        if supported_single_family_near_miss and value_short:
            return "corroboration_gap"
        if recurrence_state in {"supported", "strong"} and not corroboration_short and strict_value_short:
            return "value_gap"
        if recurrence_state == "thin" and family_confirmation_count >= 2:
            return "evidence_sufficiency_gap"
        if recurrence_state in {"supported", "strong"} and family_confirmation_count >= 2 and corroboration_proxy < 0.55:
            return "evidence_sufficiency_gap"
        if recurrence_state in {"thin", "supported", "strong"} and corroboration_short and not value_short:
            return "corroboration_gap"
        if corroboration_short and value_short:
            return "mixed_gap"
        return "confirmed"

    def _choose_near_miss_enrichment_action(
        self,
        *,
        promotion_gap_class: str,
        family_confirmation_count: int,
        matched_results_by_source: dict[str, int],
        partial_results_by_source: dict[str, int],
        source_attempts_by_family: dict[str, int],
        budget_profile: dict[str, Any],
        available_families: list[str],
    ) -> dict[str, Any]:
        remaining_beta = sum(
            max(0, 2 - int(source_attempts_by_family.get(family, 0) or 0))
            for family in available_families
        )
        if promotion_gap_class == "value_gap" and family_confirmation_count >= 1:
            return {
                "action": "GATHER_MARKET_ENRICHMENT",
                "reason": "supported_recurrence_value_gap",
                "sufficiency_priority_reason": "value_signal_below_gate_after_supported_recurrence",
            }
        if promotion_gap_class in {"corroboration_gap", "evidence_sufficiency_gap"} and family_confirmation_count >= 1:
            if remaining_beta <= 0:
                return {
                    "action": "STOP_FOR_BUDGET",
                    "reason": "near_miss_budget_exhausted",
                    "sufficiency_priority_reason": "no_remaining_attempts_for_confirmation_completion",
                }
            if any(int(count or 0) > 0 for count in partial_results_by_source.values()):
                return {
                    "action": "GATHER_CORROBORATION",
                    "reason": "complete_partial_confirmation_path",
                    "sufficiency_priority_reason": "partial_match_family_can_raise_sufficiency",
                }
            if len(matched_results_by_source) <= family_confirmation_count:
                return {
                    "action": "GATHER_CORROBORATION",
                    "reason": "seek_independent_family_confirmation",
                    "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
                }
        return {
            "action": "",
            "reason": "",
            "sufficiency_priority_reason": "",
        }

    def _current_source_yield_summary(
        self,
        *,
        collection_meta: dict[str, Any],
        atom: Optional[Any],
        corroboration_plan: CorroborationPlan,
        source_attempts_by_family: Optional[dict[str, int]] = None,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, int], int]:
        source_yield: dict[str, Any] = {}
        matched_results_by_source: dict[str, int] = {}
        partial_results_by_source: dict[str, int] = {}
        family_labels = set((collection_meta.get("docs_by_source", {}) or {}).keys()) | set((collection_meta.get("queries_by_source", {}) or {}).keys())
        for label in family_labels:
            docs_for_label = list((collection_meta.get("docs_by_source", {}) or {}).get(label, []))
            family_yield = self._evaluate_source_yield(
                source_label=label,
                docs=docs_for_label,
                docs_retrieved=int((collection_meta.get("retrieved_by_source", {}) or {}).get(label, 0) or 0),
                docs_after_dedupe=int((collection_meta.get("deduped_by_source", {}) or {}).get(label, 0) or 0),
                queries_attempted=list((collection_meta.get("queries_by_source", {}) or {}).get(label, [])),
                atom=atom,
                signature_terms=corroboration_plan.signature_terms,
                attempts=max(1, int((source_attempts_by_family or {}).get(label, 0) or 0)),
            )
            source_yield[label] = family_yield
            matched_results_by_source[label] = family_yield["docs_strong_match"]
            partial_results_by_source[label] = family_yield["docs_partial_match"]
        family_confirmation_count = sum(1 for details in source_yield.values() if details.get("confirmed"))
        return source_yield, matched_results_by_source, partial_results_by_source, family_confirmation_count

    def _atom_supports_stackoverflow_recurrence(self, atom: Optional[Any]) -> bool:
        if atom is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "current_tools", "") or "",
                ]
            )
        )
        technical_markers = [
            "developer",
            "engineer",
            "sysadmin",
            "admin",
            "deployment",
            "script",
            "api",
            "oauth",
            "webhook",
            "plugin",
            "extension",
            "integration",
            "config",
            "configuration",
            "m365",
            "export",
            "debug",
            "error",
            "automation",
        ]
        operator_transfer_markers = [
            "import",
            "export",
            "csv",
            "spreadsheet",
            "reconciliation",
            "invoice",
            "payment",
            "order",
            "inventory",
            "shipment",
            "sync",
            "backup",
            "restore",
        ]
        implementation_only_markers = [
            "unit test",
            "mock",
            "fixture",
            "lint",
            "refactor",
            "typescript",
            "react component",
            "css",
        ]
        return (
            any(marker in haystack for marker in technical_markers)
            and any(marker in haystack for marker in operator_transfer_markers)
            and not any(marker in haystack for marker in implementation_only_markers)
        )

    def _family_fit_score(
        self,
        *,
        atom: Optional[Any],
        source_label: str,
        corroboration_plan: CorroborationPlan,
    ) -> tuple[float, str]:
        if atom is None:
            return (0.2, "no_atom_context")
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(corroboration_plan.signature_terms),
                    " ".join(corroboration_plan.ecosystem_hints),
                ]
            )
        )
        spreadsheet_like = any(term in haystack for term in ["spreadsheet", "excel", "google sheets", "csv", "duplicate", "import"])
        operator_like = any(term in haystack for term in ["operations", "operator", "admin", "small business", "vendor", "partner", "follow-up", "handoff"])
        compliance_like = any(term in haystack for term in ["compliance", "audit", "evidence", "m365", "export"])
        if source_label == "github":
            if self._atom_supports_github_recurrence(
                atom=atom,
                role_terms=corroboration_plan.role_terms,
                segment_terms=corroboration_plan.segment_terms,
                ecosystem_hints=corroboration_plan.ecosystem_hints,
                signature_terms=corroboration_plan.signature_terms,
            ):
                return (0.82, "technical_issue_surface")
            return (0.08, "low_public_issue_fit")
        if source_label == "stackoverflow":
            if self._atom_supports_stackoverflow_recurrence(atom):
                return (0.76, "setup_or_debug_surface")
            return (0.14, "not_setup_or_debug_shaped")
        if source_label == "web":
            if spreadsheet_like or operator_like:
                return (0.84, "operator_workflow_fit")
            if compliance_like:
                return (0.72, "compliance_practitioner_fit")
            return (0.48, "generic_practitioner_fit")
        if source_label == "reddit":
            return (0.7, "community_operator_fit")
        if source_label == "etsy":
            if "etsy" in haystack or "seller" in haystack:
                return (0.74, "seller_ecosystem_fit")
            return (0.1, "not_seller_shaped")
        if source_label == "forum_fallback":
            return (0.36, "broad_forum_backstop")
        return (0.22, "generic_fit")

    def _choose_corroboration_action(
        self,
        *,
        atom: Optional[Any],
        corroboration_plan: CorroborationPlan,
        source_yield: dict[str, Any],
        matched_results_by_source: dict[str, int],
        partial_results_by_source: dict[str, int],
        family_confirmation_count: int,
        source_attempts_by_family: dict[str, int],
        budget_profile: dict[str, Any],
        available_families: list[str],
        current_family: str = "",
        promotion_gap_class: str = "",
    ) -> CorroborationAction:
        meaningful = self._meaningful_candidate_snapshot(atom)
        max_attempts = max(1, corroboration_plan.max_attempts_per_family)
        specificity_score = float(budget_profile.get("specificity_score", 0.0) or 0.0)
        remaining_beta = sum(max(0, max_attempts - int(source_attempts_by_family.get(family, 0) or 0)) for family in available_families)
        budget_snapshot = {
            "remaining_beta": remaining_beta,
            "target_sources": int(budget_profile.get("target_sources", 1) or 1),
            "specificity_score": specificity_score,
            "family_attempts": {family: int(source_attempts_by_family.get(family, 0) or 0) for family in available_families},
        }
        if not meaningful["meaningful_candidate"]:
            return CorroborationAction(
                action="PARK",
                reason="atom_under_specified",
                expected_gain_class="low",
                budget_snapshot=budget_snapshot,
                promotion_gap_class=promotion_gap_class,
            )
        near_miss_action = self._choose_near_miss_enrichment_action(
            promotion_gap_class=promotion_gap_class,
            family_confirmation_count=family_confirmation_count,
            matched_results_by_source=matched_results_by_source,
            partial_results_by_source=partial_results_by_source,
            source_attempts_by_family=source_attempts_by_family,
            budget_profile=budget_profile,
            available_families=available_families,
        )
        practitioner_retry_candidate = (
            specificity_score >= 0.72
            and (
                self._is_accounting_reconciliation_cohort(atom=atom, plan=corroboration_plan)
                or self._is_multichannel_seller_reporting_cohort(atom=atom, plan=corroboration_plan)
            )
        )
        if current_family:
            family_yield = source_yield.get(current_family, {})
            attempts = int(source_attempts_by_family.get(current_family, 0) or 0)
            docs_retrieved = int(family_yield.get("docs_retrieved", 0) or 0)
            strong_matches = int(family_yield.get("docs_strong_match", 0) or 0)
            partial_matches = int(family_yield.get("docs_partial_match", 0) or 0)
            high_specificity_cross_source_retry = (
                (specificity_score >= 0.85 or practitioner_retry_candidate)
                and promotion_gap_class in {"corroboration_gap", "evidence_sufficiency_gap"}
                and family_confirmation_count >= 1
                and attempts < max_attempts
            )
            if attempts >= max_attempts:
                return CorroborationAction(
                    action="STOP_FOR_BUDGET",
                    target_family=current_family,
                    reason="family_retry_limit_reached",
                    expected_gain_class="low",
                    budget_snapshot=budget_snapshot,
                    promotion_gap_class=promotion_gap_class,
                    sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                )
            if remaining_beta <= 0:
                return CorroborationAction(
                    action="STOP_FOR_BUDGET",
                    target_family=current_family,
                    reason="family_attempt_budget_exhausted",
                    expected_gain_class="low",
                    budget_snapshot=budget_snapshot,
                    promotion_gap_class=promotion_gap_class,
                    sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                )
            if (
                docs_retrieved == 0
                and current_family == "web"
                and attempts >= 1
                and promotion_gap_class in {"corroboration_gap", "evidence_sufficiency_gap"}
                and family_confirmation_count >= 1
            ):
                if high_specificity_cross_source_retry:
                    return CorroborationAction(
                        action="RETRY_WITH_RESHAPED_QUERY",
                        target_family=current_family,
                        reason="high_specificity_cross_source_retry",
                        expected_gain_class="medium",
                        budget_snapshot=budget_snapshot,
                        fallback_strategy="decomposed_query_switch",
                        promotion_gap_class=promotion_gap_class,
                        sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                    )
                return CorroborationAction(
                    action="STOP_FOR_BUDGET",
                    target_family=current_family,
                    reason="zero_retrieval_confirmation_family_low_yield",
                    expected_gain_class="low",
                    budget_snapshot=budget_snapshot,
                    promotion_gap_class=promotion_gap_class,
                    sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                )
            if docs_retrieved == 0:
                return CorroborationAction(
                    action="RETRY_WITH_RESHAPED_QUERY",
                    target_family=current_family,
                    reason="zero_retrieval_strategy_switch",
                    expected_gain_class="medium",
                    budget_snapshot=budget_snapshot,
                    fallback_strategy="decomposed_query_switch",
                    promotion_gap_class=promotion_gap_class,
                    sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                )
            if strong_matches == 0 and (partial_matches > 0 or docs_retrieved > 0):
                if (
                    current_family == "web"
                    and partial_matches > 0
                    and attempts >= 1
                    and promotion_gap_class in {"corroboration_gap", "evidence_sufficiency_gap"}
                    and family_confirmation_count >= 1
                ):
                    if high_specificity_cross_source_retry:
                        return CorroborationAction(
                            action="RETRY_WITH_RESHAPED_QUERY",
                            target_family=current_family,
                            reason="high_specificity_partial_confirmation_retry",
                            expected_gain_class="medium",
                            budget_snapshot=budget_snapshot,
                            fallback_strategy="tighten_match_grammar",
                            promotion_gap_class=promotion_gap_class,
                            sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                        )
                    return CorroborationAction(
                        action="STOP_FOR_BUDGET",
                        target_family=current_family,
                        reason="partial_only_confirmation_family_low_yield",
                        expected_gain_class="low",
                        budget_snapshot=budget_snapshot,
                        promotion_gap_class=promotion_gap_class,
                        sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                    )
                return CorroborationAction(
                    action="RETRY_WITH_RESHAPED_QUERY",
                    target_family=current_family,
                    reason="retrieval_without_confirmation",
                    expected_gain_class="medium",
                    budget_snapshot=budget_snapshot,
                    fallback_strategy="tighten_match_grammar",
                    promotion_gap_class=promotion_gap_class,
                    sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
                )
            return CorroborationAction(
                action=near_miss_action["action"] or ("GATHER_MARKET_ENRICHMENT" if family_confirmation_count >= 1 else "PARK"),
                target_family=current_family,
                reason=near_miss_action["reason"] or "family_attempt_complete",
                expected_gain_class="low",
                budget_snapshot=budget_snapshot,
                promotion_gap_class=promotion_gap_class,
                sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
            )

        if remaining_beta <= 0:
            return CorroborationAction(
                action="STOP_FOR_BUDGET",
                reason="family_attempt_budget_exhausted",
                expected_gain_class="low",
                budget_snapshot=budget_snapshot,
                promotion_gap_class=promotion_gap_class,
                sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
            )

        skipped_families: dict[str, str] = {}
        candidates: list[tuple[float, str, str]] = []
        any_partials = any(int(partial_results_by_source.get(label, 0) or 0) > 0 for label in partial_results_by_source)
        for family in available_families:
            attempts = int(source_attempts_by_family.get(family, 0) or 0)
            if attempts >= max_attempts:
                skipped_families[family] = "retry_budget_exhausted"
                continue
            fit_score, fit_reason = self._family_fit_score(atom=atom, source_label=family, corroboration_plan=corroboration_plan)
            if fit_score < 0.2:
                skipped_families[family] = fit_reason
                continue
            score = fit_score - attempts * 0.18
            if family_confirmation_count == 0 and family != "reddit":
                score += 0.12
            if any_partials and family not in partial_results_by_source:
                score += 0.08
            if promotion_gap_class in {"corroboration_gap", "evidence_sufficiency_gap"} and family_confirmation_count >= 1:
                if family == "reddit":
                    score -= 0.28
                if family == "web":
                    score += 0.22
                elif family == "stackoverflow" and self._atom_supports_stackoverflow_recurrence(atom):
                    score += 0.1
                elif family == "github" and self._atom_supports_github_recurrence(atom=atom):
                    score += 0.06
                if int(matched_results_by_source.get(family, 0) or 0) == 0:
                    score += 0.16
                if int(partial_results_by_source.get(family, 0) or 0) > 0:
                    score += 0.12
                if fit_score < 0.4:
                    score -= 0.18
            if specificity_score >= 0.85 and family_confirmation_count >= 1 and family in {"web", "github", "stackoverflow"}:
                score += 0.14
            if family == "web" and not self._atom_supports_github_recurrence(atom=atom):
                score += 0.08
            candidates.append((score, family, fit_reason))

        if not candidates:
            return CorroborationAction(
                action="PARK" if family_confirmation_count > 0 else "REQUEST_HUMAN_REVIEW",
                reason="no_high_gain_family_available",
                expected_gain_class="low",
                skipped_families=skipped_families,
                budget_snapshot=budget_snapshot,
                promotion_gap_class=promotion_gap_class,
                sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, family, fit_reason = candidates[0]
        expected_gain = "high" if score >= 0.78 else "medium" if score >= 0.5 else "low"
        return CorroborationAction(
            action="GATHER_CORROBORATION",
            target_family=family,
            reason=near_miss_action["reason"] or f"highest_information_gain:{fit_reason}",
            expected_gain_class=expected_gain,
            skipped_families=skipped_families,
            budget_snapshot=budget_snapshot,
            promotion_gap_class=promotion_gap_class,
            sufficiency_priority_reason=near_miss_action.get("sufficiency_priority_reason", ""),
        )

    def _decomposed_recurrence_queries(
        self,
        *,
        atom: Optional[Any],
        source_label: str,
        role_terms: list[str],
        segment_terms: list[str],
        job_phrase: str,
        failure_phrase: str,
        workaround_phrase: str,
        cost_terms: list[str],
        ecosystem_hints: list[str],
    ) -> list[str]:
        if atom is None:
            return []
        queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        segment_seed = " ".join(segment_terms[:2])
        role_seed = " ".join(role_terms[:2])
        cost_seed = " ".join(cost_terms[:2])
        tool_seed = _query_term_span(" ".join(ecosystem_hints[:3]), max_terms=3)
        job_seed = self._recurrence_query_seed(job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(workaround_phrase, max_terms=4)

        if source_label == "web":
            add(job_seed, segment_seed or role_seed)
            add(failure_seed, workaround_seed or cost_seed)
            add(workaround_seed or "manual workaround", cost_seed or "time risk")
            add(tool_seed or failure_seed or job_seed, "workflow")
        elif source_label == "github":
            add(tool_seed or failure_seed or job_seed, "issue")
            add(failure_seed or tool_seed, "script")
            add(workaround_seed or job_seed, "automation script")
            add(job_seed or failure_seed, "integration")
        return queries[:4]

    def _decompose_low_information_atom(
        self,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> list[str]:
        if atom is None or plan is None:
            return []
        queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        segment_seed = " ".join((plan.segment_terms or [])[:2])
        job_seed = self._recurrence_query_seed(plan.job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(plan.failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(plan.workaround_phrase, max_terms=3)
        cost_seed = " ".join((plan.cost_terms or [])[:2])
        tool_seed = _query_term_span(" ".join((plan.ecosystem_hints or [])[:3]), max_terms=3)

        add(segment_seed, job_seed)
        add(failure_seed, workaround_seed or cost_seed or "manual workaround")
        add(tool_seed or failure_seed or job_seed, cost_seed or "time risk")
        return queries[:3]

    def _web_zero_retrieval_fallback_queries(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
        prior_queries: list[str],
    ) -> list[str]:
        if atom is None or plan is None:
            return []
        queries: list[str] = []
        prior_text = normalize_content(" ".join(prior_queries))

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries and normalized not in prior_text:
                queries.append(normalized)

        role_seed = " ".join((plan.role_terms or [])[:2])
        segment_seed = " ".join((plan.segment_terms or [])[:2])
        job_seed = self._recurrence_query_seed(plan.job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(plan.failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(plan.workaround_phrase, max_terms=3)
        cost_seed = " ".join((plan.cost_terms or [])[:2])

        add(workaround_seed or "manual workaround", cost_seed or "time risk", job_seed)
        add(failure_seed or "slow workflow", cost_seed or "friction", job_seed or "workflow")
        add(segment_seed or "operators", failure_seed or job_seed or "manual process")
        if not queries:
            for query in self._decompose_low_information_atom(atom, plan):
                add(query)
        return queries[:4]

    def _spreadsheet_operator_admin_web_queries(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> list[str]:
        if not self._is_spreadsheet_operator_admin_cohort(atom=atom, plan=plan):
            return []

        queries: list[str] = []
        prioritized_queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        def prioritize(query: str) -> None:
            if query in queries and query not in prioritized_queries:
                prioritized_queries.append(query)

        role_seed = " ".join((plan.role_terms or [])[:2])
        segment_seed = " ".join((plan.segment_terms or [])[:2])
        job_seed = self._recurrence_query_seed(plan.job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(plan.failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(plan.workaround_phrase, max_terms=3)
        cost_seed = " ".join((plan.cost_terms or [])[:2]) or "time loss"
        focus_haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                    getattr(atom, "current_tools", "") or "",
                ]
            )
        )

        add("spreadsheet", job_seed or "tracking workflow", failure_seed or "manual cleanup")
        add("manual re-entry", job_seed or "reporting", role_seed or segment_seed or "operations")
        add(role_seed or "admin", "tracking", "manual", segment_seed or "workflow")
        add("using spreadsheets for", job_seed or "status updates", cost_seed)
        add("replace spreadsheet for", job_seed or "reporting", "workflow")
        add("back office workflow", failure_seed or "bottleneck", cost_seed)
        add(workaround_seed or "copy paste workflow", cost_seed, job_seed or "operations")
        add("bank reconciliation spreadsheet workflow")
        add("month end close spreadsheet workflow")
        add("sales tax payment reconciliation workflow")
        add("sales channel profitability spreadsheet")
        add("shopify amazon etsy payout reconciliation")
        add("channel profitability reporting spreadsheet")
        add("invoice reminder spreadsheet workflow")
        add("late payment follow up spreadsheet")
        add("pdf collaboration version control")
        add("shared pdf latest version approval")

        is_channel_profitability_focus = any(
            term in focus_haystack
            for term in [
                "amazon",
                "shopify",
                "etsy",
                "sales channel",
                "channel",
                "payout",
                "profitability",
                "profitable",
            ]
        )
        is_accounting_reconciliation_focus = any(
            term in focus_haystack
            for term in [
                "reconciliation",
                "bank",
                "deposit",
                "month end",
                "close",
                "sales tax",
                "payment reconciliation",
                "books",
            ]
        )

        if is_channel_profitability_focus:
            prioritize(self._normalize_recurrence_query("sales channel profitability spreadsheet"))
            prioritize(self._normalize_recurrence_query("shopify amazon etsy payout reconciliation"))
            prioritize(self._normalize_recurrence_query("channel profitability reporting spreadsheet"))

        if is_accounting_reconciliation_focus:
            prioritize(self._normalize_recurrence_query("bank reconciliation spreadsheet workflow"))
            prioritize(self._normalize_recurrence_query("month end close spreadsheet workflow"))
            prioritize(self._normalize_recurrence_query("sales tax payment reconciliation workflow"))

        if any(
            term in focus_haystack
            for term in [
                "invoice",
                "invoices",
                "late payment",
                "unpaid",
                "reminder",
                "accounts receivable",
                "client reminder",
            ]
        ):
            prioritize(self._normalize_recurrence_query("invoice reminder spreadsheet workflow"))
            prioritize(self._normalize_recurrence_query("late payment follow up spreadsheet"))

        if any(
            term in focus_haystack
            for term in [
                "pdf",
                "document",
                "approval",
                "latest version",
                "wrong version",
                "collaboration",
            ]
        ):
            prioritize(self._normalize_recurrence_query("pdf collaboration version control"))
            prioritize(self._normalize_recurrence_query("shared pdf latest version approval"))

        ordered_queries = prioritized_queries + [query for query in queries if query not in prioritized_queries]
        return ordered_queries[:8]

    def _is_spreadsheet_operator_admin_cohort(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> bool:
        if atom is None or plan is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.signature_terms),
                ]
            )
        )
        cohort_markers = [
            "spreadsheet",
            "excel",
            "google sheets",
            "sheets",
            "manual entry",
            "re-entry",
            "duplicate",
            "tracking",
            "reporting",
            "reconciliation",
            "admin",
            "administrator",
            "office manager",
            "coordinator",
            "operations",
            "back office",
            "approvals",
            "batch",
            "status updates",
            "copy paste",
            "audit trail",
            "schedule",
            "workflow bottleneck",
            "vendor",
            "supplier",
        ]
        return any(marker in haystack for marker in cohort_markers)

    def _is_accounting_reconciliation_cohort(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> bool:
        if atom is None or plan is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.signature_terms),
                ]
            )
        )
        accounting_markers = [
            "reconciliation",
            "bank",
            "deposit",
            "month end",
            "close",
            "sales tax",
            "accounts receivable",
            "invoice",
            "payment",
            "payout",
            "quickbooks",
            "qbo",
            "books",
            "stripe",
        ]
        return any(marker in haystack for marker in accounting_markers)

    def _is_state_drift_operator_cohort(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> bool:
        if atom is None or plan is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.signature_terms),
                ]
            )
        )
        drift_markers = [
            "deleted order",
            "analytics",
            "out of sync",
            "drift",
            "inventory mismatch",
            "wrong count",
            "still showing",
            "status mismatch",
            "duplicate order",
            "order status",
            "fulfillment",
            "shipment",
            "inventory",
        ]
        return any(marker in haystack for marker in drift_markers)

    def _is_multichannel_seller_reporting_cohort(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> bool:
        if atom is None or plan is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.signature_terms),
                ]
            )
        )
        seller_markers = [
            "shopify",
            "etsy",
            "amazon",
            "seller",
            "merchant",
            "store",
            "orders",
            "sales channel",
            "channel",
        ]
        reporting_markers = [
            "profitability",
            "revenue",
            "reconciliation",
            "payout",
            "bank deposit",
            "spreadsheet",
            "manual work",
            "tracking",
            "reporting",
        ]
        return any(marker in haystack for marker in seller_markers) and any(
            marker in haystack for marker in reporting_markers
        )

    def _specialized_operator_surface_queries(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> list[str]:
        if not self._is_spreadsheet_operator_admin_cohort(atom=atom, plan=plan):
            return []

        queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        role_seed = " ".join((plan.role_terms or [])[:2]) or "operations admin"
        segment_seed = " ".join((plan.segment_terms or [])[:2]) or "small business"
        job_seed = self._recurrence_query_seed(plan.job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(plan.failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(plan.workaround_phrase, max_terms=3)
        cost_seed = " ".join((plan.cost_terms or [])[:2]) or "time loss"

        add("replace spreadsheet", job_seed or "workflow", "software")
        add("excel workflow", failure_seed or "manual tracking", "forum")
        add(role_seed, job_seed or "reporting workflow", "community")
        add("back office workflow", failure_seed or "manual approvals", "software")
        add(workaround_seed or "using spreadsheets for reporting", cost_seed, "tool")
        add(segment_seed, "manual reporting workflow", "software")
        if self._is_accounting_reconciliation_cohort(atom=atom, plan=plan):
            add("quickbooks community reconciliation workflow")
            add("stripe quickbooks payout reconciliation forum")
            add("month end close csv mismatch forum")
        if self._is_state_drift_operator_cohort(atom=atom, plan=plan):
            add("deleted order still showing analytics forum")
            add("inventory counts out of sync after import forum")
            add("order status mismatch reporting community")
        return queries[:4]

    def _is_workflow_fragility_cohort(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> bool:
        if atom is None or plan is None:
            return False
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.signature_terms),
                ]
            )
        )
        fragility_anchors = [
            "duct tape",
            "spreadsheet",
            "excel",
            "handoff",
            "manual handoff",
            "copy paste",
            "version chaos",
            "out of sync",
            "missed step",
            "missed steps",
            "manual cleanup",
            "vendor file",
        ]
        fragility_burden = [
            "manual",
            "workflow",
            "workaround",
            "follow-up",
            "approvals",
            "status update",
            "status updates",
            "duplicate",
            "bottleneck",
            "cleanup",
            "time loss",
            "missed follow-up",
            "broken",
            "brittle",
        ]
        return any(marker in haystack for marker in fragility_anchors) and any(
            marker in haystack for marker in fragility_burden
        )

    def _workflow_fragility_web_queries(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
    ) -> list[str]:
        if not self._is_workflow_fragility_cohort(atom=atom, plan=plan):
            return []

        queries: list[str] = []
        prioritized_queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        def prioritize(query: str) -> None:
            if query in queries and query not in prioritized_queries:
                prioritized_queries.append(query)

        role_seed = " ".join((plan.role_terms or [])[:2]) or "operations"
        segment_seed = " ".join((plan.segment_terms or [])[:2]) or "small business"
        job_seed = self._recurrence_query_seed(plan.job_phrase, max_terms=4)
        failure_seed = self._recurrence_query_seed(plan.failure_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(plan.workaround_phrase, max_terms=3)
        cost_seed = " ".join((plan.cost_terms or [])[:2]) or "time loss"
        focus_haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                ]
            )
        )

        add("excel shared workbook conflict")
        add("shared spreadsheet saving conflicts")
        add("google sheets collaborator changes not showing")
        add("which spreadsheet is latest version")
        add("latest spreadsheet version confusion")
        add("multiple people editing same spreadsheet latest version")
        add("spreadsheet version confusion teams")
        add("manual handoff workflow software")
        add("spreadsheet handoff errors forum")
        add("spreadsheet handoff errors", role_seed or segment_seed)
        add("manual handoff workflow", failure_seed or job_seed)
        add("duct tape spreadsheets", job_seed or "workflow")
        add("workflow handoff tool too expensive")
        add("missed steps", job_seed or "approval workflow")
        add("copy paste workflow handoff")
        add("manual data entry spreadsheet handoff")

        if any(
            term in focus_haystack
            for term in [
                "latest version",
                "latest file",
                "which spreadsheet is the latest",
                "which spreadsheet is latest",
                "version confusion",
                "out of sync",
                "wrong file",
            ]
        ):
            prioritize(self._normalize_recurrence_query("excel shared workbook conflict"))
            prioritize(self._normalize_recurrence_query("shared spreadsheet saving conflicts"))
            prioritize(self._normalize_recurrence_query("google sheets collaborator changes not showing"))
            prioritize(self._normalize_recurrence_query("which spreadsheet is latest version"))
            prioritize(self._normalize_recurrence_query("latest spreadsheet version confusion"))
            prioritize(self._normalize_recurrence_query("multiple people editing same spreadsheet latest version"))

        if any(
            term in focus_haystack
            for term in [
                "handoff",
                "follow-up",
                "missed step",
                "missed steps",
                "approval",
            ]
        ):
            prioritize(self._normalize_recurrence_query("manual handoff workflow software"))
            prioritize(self._normalize_recurrence_query("spreadsheet handoff errors forum"))
            prioritize(self._normalize_recurrence_query(f"missed steps {job_seed or 'approval workflow'}"))

        if any(
            term in focus_haystack
            for term in [
                "manual data entry",
                "manual re-entry",
                "copy paste",
            ]
        ):
            prioritize(self._normalize_recurrence_query("copy paste workflow handoff"))
            prioritize(self._normalize_recurrence_query("duct tape spreadsheets workflow"))

        ordered_queries = prioritized_queries + [query for query in queries if query not in prioritized_queries]
        return ordered_queries[:8]

    def _record_web_strategy_step(self, path: list[str], step: str) -> None:
        if step and step not in path:
            path.append(step)

    def _specialized_web_routing_sites(
        self,
        *,
        atom: Optional[Any],
        plan: Optional[CorroborationPlan],
        attempt_index: int = 0,
    ) -> tuple[list[tuple[Optional[str], str]], str]:
        if atom is None or plan is None:
            return ([(None, "web")], "")
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_tools", "") or "",
                    " ".join(plan.ecosystem_hints),
                ]
            )
        )
        if self._is_multichannel_seller_reporting_cohort(atom=atom, plan=plan):
            sites = [
                ("community.shopify.com", "web"),
                ("community.etsy.com", "web"),
                (None, "web"),
            ]
            return (sites, "seller_reporting_surface_first")
        if any(term in haystack for term in ["shopify", "merchant", "storefront", "app store"]):
            sites = [("community.shopify.com", "web")]
            if attempt_index > 0:
                sites.append((None, "web"))
            return (sites, "shopify_community_first")
        if any(term in haystack for term in ["wordpress", "woocommerce", "plugin"]):
            sites = [("wordpress.org/support", "web")]
            if attempt_index > 0:
                sites.append((None, "web"))
            return (sites, "wordpress_support_first")
        if any(term in haystack for term in ["etsy", "seller"]):
            sites = [("community.etsy.com", "web")]
            if attempt_index > 0:
                sites.append((None, "web"))
            return (sites, "seller_community_first")
        if self._is_accounting_reconciliation_cohort(atom=atom, plan=plan):
            sites = [
                ("community.intuit.com", "web"),
                ("quickbooks.intuit.com/learn-support", "web"),
                ("community.oracle.com", "web"),
                ("community.sap.com", "web"),
                (None, "web"),
            ]
            return (sites, "accounting_practitioner_surface_first")
        if self._is_workflow_fragility_cohort(atom=atom, plan=plan):
            sites = [
                ("superuser.com", "web"),
                ("webapps.stackexchange.com", "web"),
                ("community.atlassian.com", "web"),
                ("community.monday.com", "web"),
                (None, "web"),
            ]
            return (sites, "workflow_fragility_surface_first")
        if attempt_index > 0 and self._is_spreadsheet_operator_admin_cohort(atom=atom, plan=plan):
            return ([(None, "web")], "operator_surface_queries_first")
        return ([(None, "web")], "")

    def _github_recurrence_queries_from_atom(
        self,
        *,
        atom: Optional[Any],
        signature_terms: list[str],
        role_terms: list[str],
        segment_terms: list[str],
        job_phrase: str,
        failure_phrase: str,
        workaround_phrase: str,
        cost_terms: list[str],
        ecosystem_hints: list[str],
        reshape_reason: str = "",
    ) -> list[str]:
        if atom is None:
            return []
        if not self._atom_supports_github_recurrence(
            atom=atom,
            role_terms=role_terms,
            segment_terms=segment_terms,
            ecosystem_hints=ecosystem_hints,
            signature_terms=signature_terms,
        ):
            return []
        queries: list[str] = []

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        hint_pool = {
            *(term.lower() for term in signature_terms),
            *(term.lower() for term in ecosystem_hints),
            *(term.lower() for term in role_terms),
            *(term.lower() for term in segment_terms),
        }
        is_spreadsheet = any(term in hint_pool for term in ("spreadsheet", "excel", "csv", "import", "google", "sheets"))
        is_compliance = any(term in hint_pool for term in ("audit", "m365", "compliance", "evidence", "export"))

        failure_seed = self._recurrence_query_seed(failure_phrase, max_terms=4)
        job_seed = self._recurrence_query_seed(job_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(workaround_phrase, max_terms=4)
        artifact_seed = _query_term_span(" ".join(ecosystem_hints[:3] + signature_terms[:3]), max_terms=4)
        role_seed = " ".join(role_terms[:2] or segment_terms[:2])
        cost_seed = " ".join(cost_terms[:2])

        if reshape_reason:
            for query in self._decomposed_recurrence_queries(
                atom=atom,
                source_label="github",
                role_terms=role_terms,
                segment_terms=segment_terms,
                job_phrase=job_phrase,
                failure_phrase=failure_phrase,
                workaround_phrase=workaround_phrase,
                cost_terms=cost_terms,
                ecosystem_hints=ecosystem_hints,
            ):
                add(query)
            if reshape_reason == "role_missing" and role_seed:
                add(role_seed, artifact_seed or failure_seed or job_seed, "issue")
            elif reshape_reason == "workaround_missing" and workaround_seed:
                add(workaround_seed, artifact_seed or failure_seed or "script issue")
            elif reshape_reason == "cost_missing" and cost_seed:
                add(artifact_seed or job_seed or failure_seed, cost_seed, "issue")
            elif reshape_reason == "failure_missing" and failure_seed:
                add(failure_seed, artifact_seed or role_seed, "issue")
            elif reshape_reason == "job_missing" and job_seed:
                add(job_seed, artifact_seed or role_seed, "workflow issue")

        if is_spreadsheet:
            add("csv import duplicate rows", "issue")
            add("excel import cleanup", "script")
            add("spreadsheet sync automation", "issue")
            add("vendor spreadsheet cleanup", "script")
        elif is_compliance:
            add("m365 audit export", "script issue")
            add("compliance evidence automation", "issue")
            add("audit export workflow", "script")
            add("m365 compliance export", "issue")
        else:
            add(failure_seed or artifact_seed or "workflow failure", "issue")
            add(job_seed or "manual workflow automation", "issue")
            add(workaround_seed or "manual cleanup", "script")
            add(artifact_seed or role_seed or "operations workflow", "issue")

        if artifact_seed and not is_spreadsheet and not is_compliance:
            add(artifact_seed, "issue")
        if workaround_seed:
            add(workaround_seed, "workaround script")
        if cost_seed:
            add(failure_seed or artifact_seed or job_seed, cost_seed, "issue")
        return queries[:4]

    def _web_recurrence_queries_from_atom(
        self,
        *,
        atom: Optional[Any],
        signature_terms: list[str],
        role_terms: list[str],
        segment_terms: list[str],
        job_phrase: str,
        failure_phrase: str,
        workaround_phrase: str,
        cost_terms: list[str],
        ecosystem_hints: list[str],
        reshape_reason: str = "",
    ) -> list[str]:
        if atom is None:
            return []
        queries: list[str] = []
        plan = CorroborationPlan(
            signature_terms=signature_terms[:10],
            role_terms=role_terms,
            segment_terms=segment_terms,
            job_phrase=job_phrase,
            failure_phrase=failure_phrase,
            workaround_phrase=workaround_phrase,
            cost_terms=cost_terms,
            ecosystem_hints=ecosystem_hints,
            family_queries={},
            source_priority=("web",),
        )

        def add(*parts: str) -> None:
            normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
            if normalized and normalized not in queries:
                queries.append(normalized)

        hint_pool = {
            *(term.lower() for term in signature_terms),
            *(term.lower() for term in ecosystem_hints),
            *(term.lower() for term in role_terms),
            *(term.lower() for term in segment_terms),
        }
        is_spreadsheet = any(term in hint_pool for term in ("spreadsheet", "excel", "csv", "import", "google", "sheets"))
        is_compliance = any(term in hint_pool for term in ("audit", "m365", "compliance", "evidence", "export"))

        operator_terms = " ".join(role_terms[:2] or segment_terms[:2]) or "operators"
        role_seed = operator_terms
        failure_seed = self._recurrence_query_seed(failure_phrase, max_terms=4)
        job_seed = self._recurrence_query_seed(job_phrase, max_terms=4)
        workaround_seed = self._recurrence_query_seed(workaround_phrase, max_terms=4)
        artifact_seed = _query_term_span(" ".join(ecosystem_hints[:3] + signature_terms[:3]), max_terms=4)
        cost_seed = " ".join(cost_terms[:2])
        fragility_queries = self._workflow_fragility_web_queries(atom=atom, plan=plan)
        cohort_queries = self._spreadsheet_operator_admin_web_queries(atom=atom, plan=plan)
        specialized_queries = self._specialized_operator_surface_queries(atom=atom, plan=plan)
        accounting_focus = self._is_accounting_reconciliation_cohort(atom=atom, plan=plan)
        state_drift_focus = self._is_state_drift_operator_cohort(atom=atom, plan=plan)
        fragility_focus_haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                ]
            )
        )
        fragility_focus_case = any(
            term in fragility_focus_haystack
            for term in [
                "handoff",
                "latest version",
                "latest file",
                "out of sync",
                "missed step",
                "version confusion",
            ]
        )

        # Keep workflow-fragility corroboration visible on the first web branch, not
        # only after a retry path. Retry mode still needs to prioritize the actual
        # reshape/fallback queries so a second attempt is meaningfully different.
        if not reshape_reason:
            if accounting_focus:
                add("quickbooks stripe payout reconciliation")
                add("bank reconciliation spreadsheet workflow")
                add("month end close csv mismatch")
            if state_drift_focus:
                add("deleted orders still showing analytics")
                add("inventory counts out of sync after import")
                add("order analytics mismatch after delete")
            if fragility_queries:
                fragility_limit = 1 if (accounting_focus or state_drift_focus) else 3
                for query in fragility_queries[:fragility_limit]:
                    add(query)
            if cohort_queries:
                cohort_limit = 1 if (accounting_focus or state_drift_focus) else 2
                for query in cohort_queries[:cohort_limit]:
                    add(query)
            if specialized_queries:
                specialized_limit = 3 if (accounting_focus or state_drift_focus) else 2
                for query in specialized_queries[:specialized_limit]:
                    add(query)

        if reshape_reason:
            if fragility_focus_case:
                for query in fragility_queries[:3]:
                    add(query)
                for query in specialized_queries:
                    add(query)
                for query in cohort_queries:
                    add(query)
            else:
                for query in fragility_queries[:1]:
                    add(query)
                for query in cohort_queries:
                    add(query)
                for query in fragility_queries[1:]:
                    add(query)
                for query in specialized_queries:
                    add(query)
            fallback_queries = self._web_zero_retrieval_fallback_queries(
                atom=atom,
                plan=plan,
                prior_queries=[],
            )
            for query in fallback_queries:
                add(query)
            for query in self._decomposed_recurrence_queries(
                atom=atom,
                source_label="web",
                role_terms=role_terms,
                segment_terms=segment_terms,
                job_phrase=job_phrase,
                failure_phrase=failure_phrase,
                workaround_phrase=workaround_phrase,
                cost_terms=cost_terms,
                ecosystem_hints=ecosystem_hints,
            ):
                add(query)
            if not self._meaningful_candidate_snapshot(atom)["meaningful_candidate"]:
                for query in self._decompose_low_information_atom(atom, plan):
                    add(query)
            if reshape_reason == "role_missing" and role_seed:
                add(failure_seed or job_seed or artifact_seed, role_seed, "workflow")
            elif reshape_reason == "workaround_missing" and workaround_seed:
                add(workaround_seed, "manual workaround", role_seed)
            elif reshape_reason == "cost_missing" and cost_seed:
                add(job_seed or failure_seed or artifact_seed, cost_seed, "risk")
            elif reshape_reason == "failure_missing" and failure_seed:
                add(failure_seed, artifact_seed or role_seed, "workflow")
            elif reshape_reason == "job_missing" and job_seed:
                add(job_seed, role_seed, artifact_seed or "workflow")

        if is_spreadsheet:
            if accounting_focus:
                add("quickbooks stripe payout reconciliation")
                add("bank deposit reconciliation spreadsheet")
                add("month end close csv mismatch")
            if state_drift_focus:
                add("deleted orders still showing analytics")
                add("order analytics mismatch after delete")
                add("inventory counts out of sync after import")
            add("spreadsheet import duplicates", "manual cleanup")
            add("csv import cleanup workflow", role_seed)
            add("vendor spreadsheet cleanup", "excel")
            add("manual spreadsheet reconciliation", role_seed)
        elif is_compliance:
            add("m365 audit export", "manual evidence collection")
            add("compliance evidence export", "workflow")
            add("audit evidence collection", "manual exports")
            add("m365 audit export", "checklist team")
        else:
            add(failure_seed or artifact_seed or "manual workflow", role_seed)
            add(workaround_seed or job_seed or "manual workaround", "workflow")
            add(job_seed or artifact_seed or "operations workflow", "operators")
            add(artifact_seed or failure_seed or "manual process", "complaints")

        if cost_seed:
            add(job_seed or failure_seed or artifact_seed, cost_seed, role_seed)
        return queries[:5]

    def _select_empty_recurrence_reshape_reason(
        self,
        *,
        atom: Optional[Any],
        source_label: str,
        attempted_queries: list[str],
    ) -> str:
        if atom is None:
            return ""
        attempted_text = normalize_content(" ".join(attempted_queries))
        if source_label == "github":
            checks = [
                ("workaround_missing", self._corroboration_terms(getattr(atom, "current_workaround", "") or "", max_terms=3)),
                ("failure_missing", self._corroboration_terms(getattr(atom, "failure_mode", "") or "", max_terms=3)),
                ("cost_missing", self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=2)),
                ("role_missing", self._corroboration_terms(getattr(atom, "user_role", "") or "", max_terms=2)),
            ]
        else:
            checks = [
                ("failure_missing", self._corroboration_terms(getattr(atom, "failure_mode", "") or "", max_terms=3)),
                ("workaround_missing", self._corroboration_terms(getattr(atom, "current_workaround", "") or "", max_terms=3)),
                ("role_missing", self._corroboration_terms(getattr(atom, "user_role", "") or "", max_terms=2)),
                ("cost_missing", self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=2)),
            ]
        for reason, terms in checks:
            if terms and not any(term in attempted_text for term in terms):
                return reason
        return "job_missing"

    def _infer_doc_source_family(self, doc: SearchDocument) -> str:
        source = (doc.source or "").lower()
        url = (doc.url or "").lower()
        if "github" in source or "github.com" in url:
            return "github"
        if "stackoverflow" in source or "stackoverflow.com" in url:
            return "stackoverflow"
        if "reddit" in source or "reddit.com" in url:
            return "reddit"
        if "etsy" in source or "etsy" in url:
            return "etsy"
        if "forum_fallback" in source:
            return "forum_fallback"
        return "web"

    def _classify_recurrence_match(
        self,
        doc: SearchDocument,
        atom: Optional[Any],
        signature_terms: list[str],
    ) -> str:
        text = normalize_content(" ".join([doc.title or "", doc.snippet or "", doc.url or ""]))
        if not text:
            return "none"
        workflow_terms = [
            "manual",
            "workflow",
            "workaround",
            "export",
            "rollback",
            "restore",
            "sync",
            "cleanup",
            "spreadsheet",
            "csv",
            "audit",
            "handoff",
            "reconcile",
            "brittle",
            "duct tape",
            "copy paste",
            "out of sync",
            "version chaos",
            "missed step",
        ]
        cosmetic_noise = ["settings page", "plugin page", "theme", "font", "css", "layout", "button color", "visual bug"]
        if any(term in text for term in cosmetic_noise) and not any(term in text for term in workflow_terms):
            return "none"

        atom_text = normalize_content(
            " ".join(
                [
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    " ".join(signature_terms or []),
                ]
            )
        )
        workflow_fragility_fit = (
            any(term in atom_text for term in ["spreadsheet", "excel", "copy paste", "handoff", "latest version", "out of sync"])
            and any(term in atom_text for term in ["manual", "workflow", "workaround", "cleanup", "time loss", "brittle"])
        )
        collaboration_conflict_terms = [
            "shared workbook",
            "edit conflict",
            "saving conflict",
            "collaborator changes not showing",
            "latest spreadsheet version",
            "latest file",
            "wrong file",
            "version confusion",
            "multiple people editing same spreadsheet",
        ]
        if workflow_fragility_fit and any(term in text for term in collaboration_conflict_terms):
            spreadsheet_terms = ["spreadsheet", "excel", "google sheets", "workbook", "sheet"]
            collaboration_terms = ["shared", "collaborator", "conflict", "version", "changes not showing", "out of sync"]
            burden_terms = ["manual", "workflow", "handoff", "copy paste", "cleanup", "workaround"]
            if (
                any(term in text for term in spreadsheet_terms)
                and any(term in text for term in collaboration_terms)
                and (any(term in text for term in burden_terms) or any(term in atom_text for term in burden_terms))
            ):
                return "strong"

        buckets = 0
        signature_hits = sum(1 for term in signature_terms if term and term in text)
        if atom is not None:
            failure_terms = self._corroboration_terms(getattr(atom, "failure_mode", "") or "", max_terms=4)
            job_terms = self._corroboration_terms(getattr(atom, "job_to_be_done", "") or "", max_terms=4)
            workaround_terms = self._corroboration_terms(getattr(atom, "current_workaround", "") or "", max_terms=3)
            cost_terms = self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=3)
            role_segment_terms = self._corroboration_terms(
                " ".join(
                    [
                        getattr(atom, "user_role", "") or "",
                        getattr(atom, "segment", "") or "",
                    ]
                ),
                max_terms=4,
            )
            for pool in (failure_terms, job_terms, workaround_terms, cost_terms, role_segment_terms):
                if any(term in text for term in pool):
                    buckets += 1
        workflow_hits = sum(1 for term in workflow_terms if term in text)
        logger.debug(
            "[match_dbg] atom_none=%s sig_hits=%d buckets=%d wf_hits=%d url=%s",
            atom is None, signature_hits, buckets, workflow_hits, (doc.url or "")[:80],
        )
        if signature_hits >= 2 and buckets >= 2 and workflow_hits >= 1:
            return "strong"
        if (signature_hits >= 1 and buckets >= 1) or workflow_hits >= 2:
            return "partial"
        return "none"

    def _source_repo_count(self, docs: list[SearchDocument]) -> int:
        repos: set[str] = set()
        for doc in docs:
            parsed = urlparse(doc.url or "")
            if "github.com" not in (parsed.netloc or "").lower():
                continue
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2:
                repos.add(f"{parts[0]}/{parts[1]}")
        return len(repos)

    def _evaluate_source_yield(
        self,
        *,
        source_label: str,
        docs: list[SearchDocument],
        docs_retrieved: int,
        docs_after_dedupe: int,
        queries_attempted: list[str],
        atom: Optional[Any],
        signature_terms: list[str],
        attempts: int,
        reshape_reason: str = "",
    ) -> dict[str, Any]:
        partial = 0
        strong = 0
        for doc in docs:
            match = self._classify_recurrence_match(doc, atom, signature_terms)
            if match == "strong":
                strong += 1
            elif match == "partial":
                partial += 1
        result = {
            "attempts": attempts,
            "queries_attempted": list(queries_attempted),
            "docs_retrieved": docs_retrieved,
            "docs_after_dedupe": docs_after_dedupe,
            "docs_after_problem_filter": strong + partial,
            "docs_partial_match": partial,
            "docs_strong_match": strong,
            "confirmed": strong > 0,
        }
        if source_label == "github":
            result["repo_count"] = self._source_repo_count(docs)
        else:
            result["independent_domains"] = len({domain_for(doc.url) for doc in docs if doc.url})
        if reshape_reason:
            result["reshape_reason"] = reshape_reason
        return result

    def _inspectable_recurrence_matches(
        self,
        *,
        source_label: str,
        docs: list[SearchDocument],
        atom: Optional[Any],
        signature_terms: list[str],
        queries_attempted: list[str],
        limit: int = 5,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        strong_matches: list[dict[str, Any]] = []
        partial_matches: list[dict[str, Any]] = []
        for doc in docs:
            match_class = self._classify_recurrence_match(doc, atom, signature_terms)
            if match_class not in {"strong", "partial"}:
                continue
            record = {
                "source_family": source_label,
                "source": doc.source,
                "query_text": getattr(doc, "retrieval_query", "") or (queries_attempted[0] if queries_attempted else ""),
                "normalized_url": normalize_search_url(doc.url),
                "title": doc.title,
                "snippet": doc.snippet,
                "match_class": match_class,
            }
            if match_class == "strong":
                if len(strong_matches) < limit:
                    strong_matches.append(record)
            elif len(partial_matches) < limit:
                partial_matches.append(record)
        return strong_matches, partial_matches

    def _select_recurrence_reshape_reason(
        self,
        *,
        docs: list[SearchDocument],
        atom: Optional[Any],
    ) -> str:
        if atom is None or not docs:
            return ""
        text = normalize_content(" ".join(" ".join([doc.title or "", doc.snippet or ""]) for doc in docs))
        checks = [
            ("failure_missing", self._corroboration_terms(getattr(atom, "failure_mode", "") or "", max_terms=3)),
            ("workaround_missing", self._corroboration_terms(getattr(atom, "current_workaround", "") or "", max_terms=3)),
            ("role_missing", self._corroboration_terms(getattr(atom, "user_role", "") or "", max_terms=2)),
            ("cost_missing", self._corroboration_terms(getattr(atom, "cost_consequence_clues", "") or "", max_terms=2)),
        ]
        for reason, terms in checks:
            if terms and not any(term in text for term in terms):
                return reason
        return "job_missing"

    async def gather_recurrence_evidence(
        self,
        queries: list[str],
        *,
        finding_kind: str,
        atom: Optional[Any] = None,
    ) -> tuple[list[SearchDocument], dict[str, Any]]:
        corroboration_plan = self._build_corroboration_plan(atom=atom, queries=queries, finding_kind=finding_kind)
        budget_profile = self._recurrence_budget_profile(atom)
        selected_queries = queries[: budget_profile["query_limit"]]
        all_subreddits = self._recurrence_subreddits(atom, limit=5) if atom is not None else []
        subreddit_plan = all_subreddits[: budget_profile["subreddit_limit"]]
        all_sites = self._recurrence_site_plan(
            atom,
            subreddit_plan=subreddit_plan,
            limit=max(4, budget_profile["site_limit"] + 1),
        )
        site_plan = all_sites[: budget_profile["site_limit"]]
        cache_key = self._recurrence_attempt_cache_key(
            queries=selected_queries,
            subreddit_plan=subreddit_plan,
            site_plan=site_plan,
            finding_kind=finding_kind,
            budget_profile=budget_profile,
        )
        cached_attempt = self._cache_get(self._recurrence_attempt_cache, cache_key)
        if cached_attempt is not None:
            docs, meta = cached_attempt
            return list(docs), dict(meta)
        warmed_validation_queries = {
            "seed_runs": 0,
            "seeded_pairs": 0,
            "seeded_searches": 0,
            "uncovered_before": 0,
            "uncovered_after": 0,
        }
        if subreddit_plan and selected_queries:
            warmed_validation_queries = await self.warm_reddit_validation_queries(
                subreddits=subreddit_plan,
                queries=selected_queries,
            )
        probe_queries = selected_queries[: budget_profile.get("probe_query_limit", 1)]
        probe_subreddits = subreddit_plan[: budget_profile.get("probe_subreddit_limit", 1)]
        probe_sites = site_plan[: budget_profile.get("probe_site_limit", 1)]
        probe_docs, probe_results_by_query, probe_results_by_source, _probe_collection_meta = await self._run_recurrence_collection(
            queries=probe_queries,
            subreddit_plan=probe_subreddits,
            site_plan=probe_sites,
            atom=atom,
            per_source_limit=budget_profile.get("probe_max_results", 2),
            stop_after_docs=max(2, budget_profile.get("probe_max_results", 2)),
            allow_fallback=False,
        )
        branch_subreddits = list(subreddit_plan)
        branch_sites = list(site_plan)
        branch_available = len(all_subreddits) > len(branch_subreddits) or len(all_sites) > len(branch_sites)
        probe_hit_count = len(probe_docs)
        branched_after_probe = False

        if probe_hit_count == 0 and budget_profile["specificity_score"] < 0.25:
            recurrence_docs = []
            results_by_query = {query: 0 for query in selected_queries}
            results_by_source = dict(probe_results_by_source)
            collection_meta = self._merge_recurrence_collection_meta({}, _probe_collection_meta)
            source_family_branch = {
                "triggered": False,
                "missing_sources": [],
                "queries": [],
                "added_docs": 0,
                "source_attempts": [],
                "last_action": "PARK",
                "last_transition_reason": "probe_miss_very_low_specificity",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": budget_profile,
                "fallback_strategy_used": "",
                "decomposed_atom_queries": [],
                "routing_override_reason": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "promotion_gap_class": "mixed_gap",
                "near_miss_enrichment_action": "",
                "sufficiency_priority_reason": "",
                "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
            }
        else:
            if probe_hit_count == 0 and budget_profile["specificity_score"] >= 0.25:
                if len(selected_queries) > len(probe_queries):
                    branched_after_probe = True
                if len(all_subreddits) > len(branch_subreddits):
                    branch_subreddits = all_subreddits[: min(len(all_subreddits), len(branch_subreddits) + 1)]
                    branched_after_probe = True
                if len(all_sites) > len(branch_sites):
                    branch_sites = all_sites[: min(len(all_sites), len(branch_sites) + 1)]
                    branched_after_probe = True
            recurrence_docs, results_by_query, results_by_source, collection_meta = await self._run_recurrence_collection(
                queries=selected_queries,
                subreddit_plan=branch_subreddits,
                site_plan=branch_sites,
                atom=atom,
                per_source_limit=max(2, self.validation_recurrence_limit // 2),
                stop_after_docs=budget_profile["early_stop_docs"],
                allow_fallback=probe_hit_count == 0 and not branched_after_probe,
            )
            recurrence_docs, results_by_query, results_by_source, collection_meta, source_family_branch = await self._expand_recurrence_source_families(
                selected_queries=selected_queries,
                atom=atom,
                all_subreddits=all_subreddits,
                all_sites=all_sites,
                current_docs=recurrence_docs,
                current_results_by_query=results_by_query,
                current_results_by_source=results_by_source,
                current_collection_meta=collection_meta,
                budget_profile=budget_profile,
                corroboration_plan=corroboration_plan,
            )

        recurrence_domains = {domain_for(doc.url) for doc in recurrence_docs if doc.url}
        query_coverage = sum(1 for count in results_by_query.values() if count > 0) / max(len(selected_queries), 1)
        doc_count = len(recurrence_docs)
        domain_count = len(recurrence_domains)
        source_yield: dict[str, Any] = {}
        matched_results_by_source: dict[str, int] = {}
        partial_results_by_source: dict[str, int] = {}
        matched_docs_by_source: dict[str, list[dict[str, Any]]] = {}
        partial_docs_by_source: dict[str, list[dict[str, Any]]] = {}
        reshaped_query_history = list(source_family_branch.get("reshaped_query_history", [])) if source_family_branch else []
        family_labels = set((collection_meta.get("docs_by_source", {}) or {}).keys()) | set((collection_meta.get("queries_by_source", {}) or {}).keys())
        for label in family_labels:
            docs_for_label = list((collection_meta.get("docs_by_source", {}) or {}).get(label, []))
            queries_attempted = list((collection_meta.get("queries_by_source", {}) or {}).get(label, []))
            family_yield = self._evaluate_source_yield(
                source_label=label,
                docs=docs_for_label,
                docs_retrieved=int((collection_meta.get("retrieved_by_source", {}) or {}).get(label, 0) or 0),
                docs_after_dedupe=int((collection_meta.get("deduped_by_source", {}) or {}).get(label, 0) or 0),
                queries_attempted=queries_attempted,
                atom=atom,
                signature_terms=corroboration_plan.signature_terms,
                attempts=max(1, len((collection_meta.get("queries_by_source", {}) or {}).get(label, [])) and 1 or 0),
            )
            prior_attempt = next(
                (attempt for attempt in source_family_branch.get("source_attempts", []) if attempt.get("source") == label),
                None,
            )
            if prior_attempt:
                family_yield["attempts"] = prior_attempt.get("attempts", family_yield["attempts"])
                if prior_attempt.get("reshape_reason"):
                    family_yield["reshape_reason"] = prior_attempt["reshape_reason"]
            source_yield[label] = family_yield
            matched_results_by_source[label] = family_yield["docs_strong_match"]
            partial_results_by_source[label] = family_yield["docs_partial_match"]
            strong_docs, partial_docs = self._inspectable_recurrence_matches(
                source_label=label,
                docs=docs_for_label,
                atom=atom,
                signature_terms=corroboration_plan.signature_terms,
                queries_attempted=queries_attempted,
            )
            matched_docs_by_source[label] = strong_docs
            partial_docs_by_source[label] = partial_docs
        family_confirmation_count = sum(1 for details in source_yield.values() if details.get("confirmed"))
        strong_match_count = sum(matched_results_by_source.values())
        partial_match_count = sum(partial_results_by_source.values())
        problem_filtered_count = sum(details.get("docs_after_problem_filter", 0) for details in source_yield.values())
        base = 0.08 if finding_kind == "pain_point" else 0.0
        recurrence_score = min(
            1.0,
            base
            + min(strong_match_count / 4.0, 1.0) * 0.42
            + min(partial_match_count / 5.0, 1.0) * 0.14
            + min(family_confirmation_count / 2.0, 1.0) * 0.22
            + min(problem_filtered_count / 6.0, 1.0) * 0.1
            + query_coverage * 0.12,
        )
        if recurrence_score >= 0.65 and family_confirmation_count >= 2:
            recurrence_state = "strong"
        elif recurrence_score >= 0.42 and strong_match_count >= 1:
            recurrence_state = "supported"
        elif recurrence_score >= 0.2 or partial_match_count >= 1:
            recurrence_state = "thin"
        else:
            recurrence_state = "weak"
        recurrence_failure_class = ""
        if strong_match_count == 0 and partial_match_count == 0:
            if branch_available and not branched_after_probe:
                recurrence_gap_reason = "search_breadth_likely_insufficient"
                recurrence_failure_class = "breadth_limited"
            else:
                recurrence_gap_reason = "no_independent_confirmations"
                recurrence_failure_class = "no_corroboration_found"
        elif family_confirmation_count <= 1:
            recurrence_gap_reason = "single_source_confirmation_only"
            recurrence_failure_class = "single_source_only"
        elif strong_match_count == 0 and partial_match_count > 0:
            recurrence_gap_reason = "partial_confirmation_only"
            recurrence_failure_class = "partial_confirmation_only"
        else:
            recurrence_gap_reason = ""
        if not recurrence_failure_class and recurrence_state in {"supported", "strong"}:
            recurrence_failure_class = "confirmed"

        result_meta = {
            "recurrence_score": round(recurrence_score, 4),
            "recurrence_state": recurrence_state,
            "recurrence_gap_reason": recurrence_gap_reason,
            "recurrence_failure_class": recurrence_failure_class,
            "query_coverage": round(query_coverage, 4),
            "doc_count": doc_count,
            "domain_count": domain_count,
            "results_by_query": results_by_query,
            "results_by_source": results_by_source,
            "matched_results_by_source": matched_results_by_source,
            "partial_results_by_source": partial_results_by_source,
            "matched_docs_by_source": matched_docs_by_source,
            "partial_docs_by_source": partial_docs_by_source,
            "family_confirmation_count": family_confirmation_count,
            "source_yield": source_yield,
            "reshaped_query_history": reshaped_query_history,
            "warmed_validation_queries": warmed_validation_queries,
            "queries_considered": list(queries),
            "queries_executed": list(selected_queries),
            "recurrence_budget_profile": budget_profile,
            "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
            "recurrence_probe_summary": {
                "probe_queries": probe_queries,
                "probe_subreddits": probe_subreddits,
                "probe_sites": probe_sites,
                "probe_hit_count": probe_hit_count,
                "branched_after_probe": branched_after_probe,
                "branch_available": branch_available,
            },
            "recurrence_source_branch": source_family_branch,
            "last_action": source_family_branch.get("last_action", ""),
            "last_transition_reason": source_family_branch.get("last_transition_reason", ""),
            "chosen_family": source_family_branch.get("chosen_family", ""),
            "expected_gain_class": source_family_branch.get("expected_gain_class", ""),
            "source_attempts_snapshot": source_family_branch.get("source_attempts_snapshot", {}),
            "skipped_families": source_family_branch.get("skipped_families", {}),
            "controller_actions": source_family_branch.get("controller_actions", []),
            "budget_snapshot": source_family_branch.get("budget_snapshot", {}),
            "fallback_strategy_used": source_family_branch.get("fallback_strategy_used", ""),
            "decomposed_atom_queries": source_family_branch.get("decomposed_atom_queries", []),
            "routing_override_reason": source_family_branch.get("routing_override_reason", ""),
            "cohort_query_pack_used": source_family_branch.get("cohort_query_pack_used", False),
            "cohort_query_pack_name": source_family_branch.get("cohort_query_pack_name", ""),
            "web_query_strategy_path": source_family_branch.get("web_query_strategy_path", []),
            "specialized_surface_targeting_used": source_family_branch.get("specialized_surface_targeting_used", False),
            "promotion_gap_class": source_family_branch.get("promotion_gap_class", ""),
            "near_miss_enrichment_action": source_family_branch.get("near_miss_enrichment_action", ""),
            "sufficiency_priority_reason": source_family_branch.get("sufficiency_priority_reason", ""),
        }
        self._cache_set(self._recurrence_attempt_cache, cache_key, (list(recurrence_docs), dict(result_meta)))
        return recurrence_docs, result_meta

    def _recurrence_site_plan(
        self,
        atom: Optional[Any],
        *,
        subreddit_plan: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[tuple[Optional[str], str]]:
        sites: list[tuple[Optional[str], str]] = []
        if not subreddit_plan:
            sites.append(("reddit.com", "reddit"))
        if atom is None:
            sites.insert(0, (None, "web"))
            return sites[:limit] if limit else sites

        plan = self._build_corroboration_plan(atom=atom, queries=[], finding_kind="problem_signal")
        specialized_web_sites, _reason = self._specialized_web_routing_sites(atom=atom, plan=plan, attempt_index=0)
        for site in specialized_web_sites:
            if site not in sites:
                sites.append(site)
        if (None, "web") not in sites:
            sites.append((None, "web"))

        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "current_tools", "") or "",
                ]
            )
        )
        github_enabled = self._atom_supports_github_recurrence(atom=atom)
        if github_enabled:
            sites.append(("github.com", "github"))
        if self._atom_supports_stackoverflow_recurrence(atom):
            sites.append(("stackoverflow.com", "stackoverflow"))
        if "etsy" in haystack:
            sites.append(("community.etsy.com", "etsy"))
        if not limit:
            return sites
        limited_sites = list(sites[:limit])
        if (None, "web") in sites and (None, "web") not in limited_sites:
            if limited_sites:
                limited_sites[-1] = (None, "web")
            else:
                limited_sites.append((None, "web"))
        return limited_sites

    def _recurrence_subreddits(self, atom: Optional[Any], *, limit: int = 5) -> list[str]:
        if atom is None:
            return []
        plan = self._build_corroboration_plan(atom=atom, queries=[], finding_kind="problem_signal")
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                ]
            )
        )
        if self._is_accounting_reconciliation_cohort(atom=atom, plan=plan):
            subreddits = ["accounting", "Bookkeeping", "quickbooksonline", "Netsuite", "smallbusiness"]
        elif self._is_multichannel_seller_reporting_cohort(atom=atom, plan=plan):
            subreddits = ["ecommerce", "shopify", "EtsySellers", "smallbusiness", "accounting"]
        elif "compliance" in haystack:
            subreddits = ["compliance", "sysadmin", "smallbusiness"]
        elif "developer" in haystack or "engineer" in haystack:
            subreddits = ["sysadmin", "devops", "webdev"]
        elif "etsy" in haystack or "seller" in haystack:
            subreddits = ["EtsySellers", "smallbusiness"]
        else:
            subreddits = ["smallbusiness", "sysadmin"]
        deduped: list[str] = []
        for subreddit in subreddits:
            if subreddit not in deduped:
                deduped.append(subreddit)
        return deduped[:limit]

    def _recurrence_budget_profile(self, atom: Optional[Any]) -> dict[str, Any]:
        if atom is None:
            return {
                "specificity_score": 0.35,
                "query_limit": 2,
                "subreddit_limit": 2,
                "site_limit": 2,
                "probe_query_limit": 1,
                "probe_subreddit_limit": 1,
                "probe_site_limit": 1,
                "probe_max_results": 2,
                "target_docs": 4,
                "target_sources": 2,
                "early_stop_docs": max(4, self.validation_recurrence_limit),
            }

        specificity = 0.1
        for field, weight in [
            (getattr(atom, "failure_mode", "") or "", 0.2),
            (getattr(atom, "trigger_event", "") or "", 0.16),
            (getattr(atom, "current_workaround", "") or "", 0.16),
            (getattr(atom, "cost_consequence_clues", "") or "", 0.12),
            (getattr(atom, "user_role", "") or "", 0.1),
            (getattr(atom, "segment", "") or "", 0.08),
        ]:
            if str(field).strip():
                specificity += weight
        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                ]
            )
        )
        if any(term in haystack for term in ["backup", "restore", "recovery", "compliance", "audit", "sync", "handoff"]):
            specificity += 0.1
        specificity = min(1.0, specificity)
        specific_context_markers = [
            "receipt",
            "issue",
            "invoice",
            "payment",
            "reconciliation",
            "bank",
            "csv",
            "import",
            "export",
            "order",
            "shipment",
            "label",
            "stock",
            "inventory",
            "contract",
            "quantity",
            "duplicate",
            "goods receipt",
            "goods issue",
            "vendor",
        ]
        has_specific_context = any(term in haystack for term in specific_context_markers)
        is_generic_manual = (
            any(term in haystack for term in ["manual work", "manual process", "manual task"])
            and not any(term in haystack for term in ["backup", "restore", "compliance", "sync", "shipping", "pricing", "audit"])
            and not has_specific_context
        )
        if specificity >= 0.85 and not is_generic_manual:
            return {
                "specificity_score": round(specificity, 4),
                "query_limit": 5,
                "subreddit_limit": 3,
                "site_limit": 4,
                "probe_query_limit": 1,
                "probe_subreddit_limit": 1,
                "probe_site_limit": 1,
                "probe_max_results": 2,
                "target_docs": 7,
                "target_sources": 2,
                "early_stop_docs": max(7, self.validation_recurrence_limit + 1),
            }
        if specificity >= 0.72 and not is_generic_manual:
            return {
                "specificity_score": round(specificity, 4),
                "query_limit": 4,
                "subreddit_limit": 3,
                "site_limit": 3,
                "probe_query_limit": 1,
                "probe_subreddit_limit": 1,
                "probe_site_limit": 1,
                "probe_max_results": 2,
                "target_docs": 6,
                "target_sources": 2,
                "early_stop_docs": max(6, self.validation_recurrence_limit),
            }
        if specificity >= 0.45 and not is_generic_manual:
            return {
                "specificity_score": round(specificity, 4),
                "query_limit": 3,
                "subreddit_limit": 3,
                "site_limit": 2,
                "probe_query_limit": 1,
                "probe_subreddit_limit": 1,
                "probe_site_limit": 1,
                "probe_max_results": 2,
                "target_docs": 5,
                "target_sources": 2,
                "early_stop_docs": max(5, self.validation_recurrence_limit),
            }
        return {
            "specificity_score": round(specificity, 4),
            "query_limit": 3,
            "subreddit_limit": 2,
            "site_limit": 2,
            "probe_query_limit": 1,
            "probe_subreddit_limit": 1,
            "probe_site_limit": 1,
            "probe_max_results": 2,
            "target_docs": 4,
            "target_sources": 2,
            "early_stop_docs": max(4, self.validation_recurrence_limit - 1),
        }

    def _recurrence_attempt_cache_key(
        self,
        *,
        queries: list[str],
        subreddit_plan: list[str],
        site_plan: list[tuple[Optional[str], str]],
        finding_kind: str,
        budget_profile: dict[str, Any],
    ) -> str:
        payload = {
            "queries": queries,
            "subreddit_plan": subreddit_plan,
            "site_plan": site_plan,
            "finding_kind": finding_kind,
            "reddit_mode": self.reddit_mode,
            "specificity_score": budget_profile.get("specificity_score", 0.0),
        }
        return json.dumps(payload, sort_keys=True)

    async def _run_recurrence_collection(
        self,
        *,
        queries: list[str],
        subreddit_plan: list[str],
        site_plan: list[tuple[Optional[str], str]],
        atom: Optional[Any],
        per_source_limit: int,
        stop_after_docs: int,
        allow_fallback: bool,
    ) -> tuple[list[SearchDocument], dict[str, int], dict[str, int], dict[str, Any]]:
        seen_urls: set[str] = set()
        recurrence_docs: list[SearchDocument] = []
        results_by_query: dict[str, int] = {}
        results_by_source: dict[str, int] = {
            "web": 0,
            "reddit": 0,
            "github": 0,
            "stackoverflow": 0,
            "etsy": 0,
            "forum_fallback": 0,
        }
        retrieved_by_source = {label: 0 for label in results_by_source}
        deduped_by_source = {label: 0 for label in results_by_source}
        docs_by_source: dict[str, list[SearchDocument]] = {label: [] for label in results_by_source}
        queries_by_source: dict[str, list[str]] = {label: [] for label in results_by_source}

        async def _collect_query(query: str) -> tuple[str, list[tuple[str, SearchDocument]], int]:
            kept = 0
            collected: list[tuple[str, SearchDocument]] = []
            if subreddit_plan:
                queries_by_source.setdefault("reddit", [])
                if query not in queries_by_source["reddit"]:
                    queries_by_source["reddit"].append(query)
                reddit_results = await asyncio.gather(
                    *[
                        self.reddit_search(
                            subreddit,
                            query,
                            limit=per_source_limit,
                            sort="relevance",
                        )
                        for subreddit in subreddit_plan
                    ]
                )
                for docs in reddit_results:
                    for doc in docs:
                        collected.append(("reddit", doc))
                        kept += 1
                        retrieved_by_source["reddit"] = retrieved_by_source.get("reddit", 0) + 1

            if site_plan:
                site_tasks = [
                    asyncio.create_task(
                        self.search_web(
                            query,
                            max_results=per_source_limit if site else max(per_source_limit, 2),
                            site=site,
                            intent="validation_recurrence",
                        )
                    )
                    for site, _ in site_plan
                ]
                for _site, source_label in site_plan:
                    queries_by_source.setdefault(source_label, [])
                    if query not in queries_by_source[source_label]:
                        queries_by_source[source_label].append(query)
                done, pending = await asyncio.wait(
                    site_tasks,
                    timeout=min(4.0, self.validation_recurrence_budget_seconds * 0.5),
                )
                for task in pending:
                    task.cancel()
                for task, (_site, source_label) in zip(site_tasks, site_plan):
                    if task not in done:
                        continue
                    try:
                        docs = task.result()
                    except Exception:
                        docs = []
                    for doc in docs:
                        collected.append((source_label, doc))
                        kept += 1
                        retrieved_by_source[source_label] = retrieved_by_source.get(source_label, 0) + 1

            if kept == 0 and allow_fallback:
                for fallback_query in self._recurrence_fallback_queries(query, atom=atom):
                    queries_by_source.setdefault("forum_fallback", [])
                    if fallback_query not in queries_by_source["forum_fallback"]:
                        queries_by_source["forum_fallback"].append(fallback_query)
                    docs = await self.search_web(
                        fallback_query,
                        max_results=max(2, per_source_limit),
                        intent="validation_recurrence",
                    )
                    for doc in docs:
                        collected.append(("forum_fallback", doc))
                        kept += 1
                        retrieved_by_source["forum_fallback"] = retrieved_by_source.get("forum_fallback", 0) + 1
                    if kept:
                        break

            return query, collected, kept

        for query, collected, kept in await asyncio.gather(*[_collect_query(query) for query in queries]):
            results_by_query[query] = kept
            for source_label, doc in collected:
                normalized_url = normalize_search_url(doc.url)
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                recurrence_docs.append(
                    SearchDocument(
                        title=doc.title,
                        url=normalized_url,
                        snippet=doc.snippet,
                        source=doc.source,
                        source_family=source_label,
                        retrieval_query=query,
                    )
                )
                results_by_source[source_label] = results_by_source.get(source_label, 0) + 1
                deduped_by_source[source_label] = deduped_by_source.get(source_label, 0) + 1
                docs_by_source.setdefault(source_label, []).append(recurrence_docs[-1])
                if len(recurrence_docs) >= stop_after_docs:
                    break
            if len(recurrence_docs) >= stop_after_docs:
                break

        return recurrence_docs, results_by_query, results_by_source, {
            "retrieved_by_source": retrieved_by_source,
            "deduped_by_source": deduped_by_source,
            "docs_by_source": docs_by_source,
            "queries_by_source": queries_by_source,
        }


    def _active_recurrence_source_labels(self, results_by_source: dict[str, int], *, min_per_source: int = 1) -> set[str]:
        return {label for label, count in results_by_source.items() if count >= min_per_source}

    def _merge_recurrence_results(
        self,
        *,
        existing_docs: list[SearchDocument],
        existing_results_by_query: dict[str, int],
        existing_results_by_source: dict[str, int],
        new_docs: list[SearchDocument],
        new_results_by_query: dict[str, int],
        new_results_by_source: dict[str, int],
        stop_after_docs: int,
    ) -> tuple[list[SearchDocument], dict[str, int], dict[str, int]]:
        merged_docs: list[SearchDocument] = []
        seen_urls: set[str] = set()
        for doc in existing_docs + new_docs:
            normalized_url = normalize_search_url(doc.url)
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged_docs.append(
                SearchDocument(
                    title=doc.title,
                    url=normalized_url,
                    snippet=doc.snippet,
                    source=doc.source,
                    source_family=getattr(doc, "source_family", ""),
                    retrieval_query=getattr(doc, "retrieval_query", ""),
                )
            )
            if len(merged_docs) >= stop_after_docs:
                break

        merged_results_by_query = dict(existing_results_by_query)
        for query, count in new_results_by_query.items():
            merged_results_by_query[query] = merged_results_by_query.get(query, 0) + count

        merged_results_by_source = dict(existing_results_by_source)
        for source, count in new_results_by_source.items():
            merged_results_by_source[source] = merged_results_by_source.get(source, 0) + count

        return merged_docs, merged_results_by_query, merged_results_by_source

    def _merge_recurrence_collection_meta(
        self,
        existing: Optional[dict[str, Any]],
        new: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        existing = existing or {}
        new = new or {}
        merged = {
            "retrieved_by_source": dict(existing.get("retrieved_by_source", {})),
            "deduped_by_source": dict(existing.get("deduped_by_source", {})),
            "docs_by_source": {label: list(docs) for label, docs in (existing.get("docs_by_source", {}) or {}).items()},
            "queries_by_source": {label: list(queries) for label, queries in (existing.get("queries_by_source", {}) or {}).items()},
        }
        for key in ("retrieved_by_source", "deduped_by_source"):
            for label, count in (new.get(key, {}) or {}).items():
                merged[key][label] = merged[key].get(label, 0) + int(count or 0)
        for label, docs in (new.get("docs_by_source", {}) or {}).items():
            merged["docs_by_source"].setdefault(label, [])
            seen = {normalize_search_url(doc.url) for doc in merged["docs_by_source"][label] if doc.url}
            for doc in docs:
                normalized = normalize_search_url(doc.url)
                if normalized and normalized not in seen:
                    merged["docs_by_source"][label].append(doc)
                    seen.add(normalized)
        for label, queries in (new.get("queries_by_source", {}) or {}).items():
            merged["queries_by_source"].setdefault(label, [])
            for query in queries:
                if query not in merged["queries_by_source"][label]:
                    merged["queries_by_source"][label].append(query)
        return merged

    def _recurrence_cross_source_queries(
        self,
        *,
        selected_queries: list[str],
        atom: Optional[Any],
        limit: int = 2,
    ) -> list[str]:
        candidates: list[str] = []
        for query in selected_queries:
            normalized = self._normalize_recurrence_query(query)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if atom is not None:
            job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6)
            failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)
            segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
            role_terms = self._recurrence_role_terms(getattr(atom, "user_role", "") or "")
            for parts in (
                [failure_phrase, segment_terms],
                [job_phrase, role_terms or segment_terms],
            ):
                normalized = self._normalize_recurrence_query(" ".join(part for part in parts if part))
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates[:limit]

    def _recurrence_source_specific_queries(
        self,
        *,
        selected_queries: list[str],
        atom: Optional[Any],
        source_label: str,
        limit: int = 2,
        plan: Optional[CorroborationPlan] = None,
        reshape_reason: str = "",
    ) -> list[str]:
        plan = plan or self._build_corroboration_plan(atom=atom, queries=selected_queries, finding_kind="problem_signal")
        if source_label == "github":
            return self._github_recurrence_queries_from_atom(
                atom=atom,
                signature_terms=plan.signature_terms,
                role_terms=plan.role_terms,
                segment_terms=plan.segment_terms,
                job_phrase=plan.job_phrase,
                failure_phrase=plan.failure_phrase,
                workaround_phrase=plan.workaround_phrase,
                cost_terms=plan.cost_terms,
                ecosystem_hints=plan.ecosystem_hints,
                reshape_reason=reshape_reason,
            )[:limit]
        if source_label == "web":
            return self._web_recurrence_queries_from_atom(
                atom=atom,
                signature_terms=plan.signature_terms,
                role_terms=plan.role_terms,
                segment_terms=plan.segment_terms,
                job_phrase=plan.job_phrase,
                failure_phrase=plan.failure_phrase,
                workaround_phrase=plan.workaround_phrase,
                cost_terms=plan.cost_terms,
                ecosystem_hints=plan.ecosystem_hints,
                reshape_reason=reshape_reason,
            )[:limit]

        base_candidates = self._recurrence_cross_source_queries(
            selected_queries=selected_queries,
            atom=atom,
            limit=max(limit * 2, 4),
        )
        candidates: list[str] = []

        def add(query: str) -> None:
            normalized = self._normalize_recurrence_query(query)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        source_shaped_candidates = self._recurrence_source_archetype_queries(atom=atom, source_label=source_label)
        for query in source_shaped_candidates:
            add(query)

        hint_map = {
            "github": ["script workaround", "restore fails", "rollback script"],
            "web": ["forum", "runbook checklist", "manual workaround"],
            "stackoverflow": ["script error", "cli workaround", "automation fails"],
            "etsy": ["seller forum", "community workaround"],
            "reddit": ["manual workaround", "operator workflow"],
        }
        source_hints = hint_map.get(source_label, [])
        for query in base_candidates:
            add(query)
            for hint in source_hints:
                add(f"{query} {hint}")

        if atom is not None:
            failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)
            job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6)
            workaround_phrase = self._recurrence_focus_phrase(getattr(atom, "current_workaround", "") or "", max_words=4)
            segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
            role_terms = self._recurrence_role_terms(getattr(atom, "user_role", "") or "")
            source_specific_parts = {
                "github": [
                    [failure_phrase, "script workaround"],
                    [job_phrase, "rollback script"],
                ],
                "web": [
                    [failure_phrase or workaround_phrase, "forum"],
                    [job_phrase, "runbook checklist", segment_terms],
                ],
                "stackoverflow": [
                    [failure_phrase, "script error"],
                    [workaround_phrase or job_phrase, "cli workaround"],
                ],
                "etsy": [
                    [failure_phrase or workaround_phrase, "seller forum"],
                    [job_phrase, "community workaround", segment_terms or role_terms],
                ],
                "reddit": [
                    [failure_phrase or workaround_phrase, "manual workaround"],
                    [job_phrase, "operator workflow", role_terms or segment_terms],
                ],
            }
            for parts in source_specific_parts.get(source_label, []):
                add(" ".join(part for part in parts if part))

        return candidates[:limit]

    def _recurrence_source_archetype_queries(
        self,
        *,
        atom: Optional[Any],
        source_label: str,
    ) -> list[str]:
        if atom is None:
            return []

        haystack = normalize_content(
            " ".join(
                [
                    getattr(atom, "segment", "") or "",
                    getattr(atom, "user_role", "") or "",
                    getattr(atom, "job_to_be_done", "") or "",
                    getattr(atom, "failure_mode", "") or "",
                    getattr(atom, "trigger_event", "") or "",
                    getattr(atom, "current_workaround", "") or "",
                    getattr(atom, "cost_consequence_clues", "") or "",
                    getattr(atom, "current_tools", "") or "",
                ]
            )
        )
        role_terms = self._recurrence_role_terms(getattr(atom, "user_role", "") or "")
        segment_terms = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
        job_phrase = self._recurrence_focus_phrase(getattr(atom, "job_to_be_done", "") or "", max_words=6)
        failure_phrase = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)

        archetype_queries: list[str] = []

        def push(*parts: str) -> None:
            query = " ".join(part for part in parts if part)
            if query and query not in archetype_queries:
                archetype_queries.append(query)

        spreadsheet_pain = any(term in haystack for term in ["spreadsheet", "excel", "google sheets", "csv", "duplicate", "manual data entry"])
        ops_admin_pain = any(term in haystack for term in ["operations", "operator", "admin", "supplier", "vendor", "partner", "follow-up", "handoff"])
        compliance_pain = any(term in haystack for term in ["compliance", "audit", "soc 2", "m365", "evidence monitoring", "export"])
        generic_manual_pain = any(term in haystack for term in ["manual work", "manual process", "manual task", "repetitive"])
        if _is_non_operational_business_risk_atom(atom):
            return archetype_queries

        if source_label == "web":
            if spreadsheet_pain:
                push('"spreadsheet cleanup workflow"', segment_terms or role_terms)
                push('"excel csv cleanup"', '"manual import"', segment_terms)
                push('"manual data entry"', '"spreadsheet cleanup"', segment_terms or '"small business"')
            if ops_admin_pain:
                push('"operations admin workflow"', '"manual follow-up"', segment_terms or role_terms)
                push('"vendor spreadsheet cleanup"', '"manual workflow"', segment_terms)
            if compliance_pain:
                push('"manual audit exports"', '"compliance evidence collection"', role_terms or "compliance")
                push('"m365 audit exports"', '"manual compliance evidence"', "forum")
            if any(term in haystack for term in ["reconciliation", "quickbooks", "qbo", "bank", "month end", "payout", "sales tax"]):
                push('"quickbooks stripe payout reconciliation"', "forum")
                push('"bank reconciliation spreadsheet workflow"', "community")
                push('"month end close csv mismatch"', "forum")
            if any(term in haystack for term in ["deleted order", "analytics", "still showing", "inventory mismatch", "status mismatch"]):
                push('"deleted orders still showing analytics"', "community")
                push('"inventory counts out of sync after import"', "forum")
                push('"order analytics mismatch after delete"', "merchant")
            if generic_manual_pain and not (spreadsheet_pain or ops_admin_pain or compliance_pain):
                push(job_phrase or '"manual process"', '"workflow forum"', segment_terms or role_terms)
                push(failure_phrase or '"manual workaround"', '"operator workflow"', segment_terms)

        if source_label == "github":
            if spreadsheet_pain:
                push('"csv import"', "dedupe script issue")
                push('"excel import"', '"manual cleanup"', "automation issue")
                push('"spreadsheet sync"', "script issue")
            if ops_admin_pain:
                push('"operations workflow"', "automation script issue")
                push('"vendor data sync"', "script workaround")
            if compliance_pain:
                push('"audit export"', "script issue")
                push('"compliance evidence"', "automation script")
                push('"m365 audit"', "export script")
            if generic_manual_pain and not (spreadsheet_pain or ops_admin_pain or compliance_pain):
                push(job_phrase or '"manual workflow"', "automation issue")
                push(failure_phrase or '"manual workaround"', "script issue")

        if source_label == "stackoverflow":
            if spreadsheet_pain:
                push('"csv import"', "dedupe script")
                push('"excel import"', "cleanup automation")
            if compliance_pain:
                push('"audit export"', "automation script")
            if generic_manual_pain and not (spreadsheet_pain or compliance_pain):
                push(job_phrase or '"manual workflow"', "script workaround")

        if source_label == "reddit":
            if ops_admin_pain or spreadsheet_pain:
                push('"manual workflow"', segment_terms or role_terms)
                push('"spreadsheet workaround"', segment_terms)
            if compliance_pain:
                push('"manual audit exports"', "compliance teams")
            if generic_manual_pain and not (spreadsheet_pain or ops_admin_pain or compliance_pain):
                push('"manual workaround"', role_terms or segment_terms)

        if source_label == "etsy":
            push('"seller workflow"', '"manual workaround"', segment_terms or role_terms)

        return archetype_queries

    async def _expand_recurrence_source_families(
        self,
        *,
        selected_queries: list[str],
        atom: Optional[Any],
        all_subreddits: list[str],
        all_sites: list[tuple[Optional[str], str]],
        current_docs: list[SearchDocument],
        current_results_by_query: dict[str, int],
        current_results_by_source: dict[str, int],
        current_collection_meta: dict[str, Any],
        budget_profile: dict[str, Any],
        corroboration_plan: Optional[CorroborationPlan] = None,
    ) -> tuple[list[SearchDocument], dict[str, int], dict[str, int], dict[str, Any], dict[str, Any]]:
        if _is_non_operational_business_risk_atom(atom):
            return current_docs, current_results_by_query, current_results_by_source, current_collection_meta, {
                "triggered": False,
                "missing_sources": [],
                "queries": [],
                "reshaped_query_history": [],
                "source_attempts": [],
                "last_action": "GATHER_MARKET_ENRICHMENT",
                "last_transition_reason": "non_operational_business_risk_atom",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": budget_profile,
                "fallback_strategy_used": "",
                "decomposed_atom_queries": [],
                "routing_override_reason": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "promotion_gap_class": "weak",
                "near_miss_enrichment_action": "GATHER_MARKET_ENRICHMENT",
                "sufficiency_priority_reason": "business_risk_without_transferable_workflow",
                "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
            }

        active_sources = self._active_recurrence_source_labels(current_results_by_source, min_per_source=2)
        target_sources = max(2, int(budget_profile.get("target_sources", 1) or 1))
        corroboration_plan = corroboration_plan or self._build_corroboration_plan(
            atom=atom,
            queries=selected_queries,
            finding_kind="problem_signal",
        )
        if len(active_sources) >= target_sources:
            return current_docs, current_results_by_query, current_results_by_source, current_collection_meta, {
                "triggered": False,
                "missing_sources": [],
                "queries": [],
                "reshaped_query_history": [],
                "source_attempts": [],
                "last_action": "GATHER_MARKET_ENRICHMENT",
                "last_transition_reason": "target_source_diversity_already_met",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": budget_profile,
                "fallback_strategy_used": "",
                "decomposed_atom_queries": [],
                "routing_override_reason": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "promotion_gap_class": "confirmed",
                "near_miss_enrichment_action": "GATHER_MARKET_ENRICHMENT",
                "sufficiency_priority_reason": "",
                "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
            }

        missing_sources: list[str] = []
        if "reddit" not in active_sources and all_subreddits:
            missing_sources.append("reddit")
        for _site, label in all_sites:
            if label in active_sources or label in missing_sources:
                continue
            missing_sources.append(label)
        priority_order = {label: idx for idx, label in enumerate(corroboration_plan.source_priority)}
        missing_sources.sort(key=lambda label: priority_order.get(label, len(priority_order)))

        if not missing_sources:
            return current_docs, current_results_by_query, current_results_by_source, current_collection_meta, {
                "triggered": False,
                "missing_sources": [],
                "queries": [],
                "reshaped_query_history": [],
                "source_attempts": [],
                "last_action": "GATHER_MARKET_ENRICHMENT",
                "last_transition_reason": "target_source_diversity_already_met",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": budget_profile,
                "fallback_strategy_used": "",
                "decomposed_atom_queries": [],
                "routing_override_reason": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "promotion_gap_class": "confirmed",
                "near_miss_enrichment_action": "GATHER_MARKET_ENRICHMENT",
                "sufficiency_priority_reason": "",
                "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
            }

        merged_docs = list(current_docs)
        merged_results_by_query = dict(current_results_by_query)
        merged_results_by_source = dict(current_results_by_source)
        merged_collection_meta = self._merge_recurrence_collection_meta({}, current_collection_meta)
        attempted_queries: list[str] = []
        source_attempts: list[dict[str, Any]] = []
        added_docs_total = 0
        reshaped_query_history: list[dict[str, Any]] = []
        skipped_families: dict[str, str] = {}
        controller_actions: list[dict[str, Any]] = []
        source_attempts_by_family: dict[str, int] = {}
        last_action = ""
        last_transition_reason = ""
        chosen_family = ""
        expected_gain_class = ""
        fallback_strategy_used = ""
        decomposed_atom_queries: list[str] = []
        routing_override_reason = ""
        cohort_query_pack_used = False
        cohort_query_pack_name = ""
        web_query_strategy_path: list[str] = []
        specialized_surface_targeting_used = False
        promotion_gap_class = ""
        sufficiency_priority_reason = ""
        near_miss_enrichment_action = ""
        while missing_sources:
            current_source_yield, current_matched_results, current_partial_results, current_family_confirmation_count = self._current_source_yield_summary(
                collection_meta=merged_collection_meta,
                atom=atom,
                corroboration_plan=corroboration_plan,
                source_attempts_by_family=source_attempts_by_family,
            )
            promotion_gap_class = self._classify_promotion_gap(
                atom=atom,
                recurrence_state="supported" if current_family_confirmation_count >= 2 else ("thin" if current_family_confirmation_count >= 1 else "weak"),
                recurrence_score=min(1.0, max(0.0, len(merged_docs) / 6.0)),
                family_confirmation_count=current_family_confirmation_count,
                strong_match_count=sum(current_matched_results.values()),
                partial_match_count=sum(current_partial_results.values()),
                query_coverage=sum(1 for count in merged_results_by_query.values() if count > 0) / max(len(selected_queries), 1),
                value_signal=0.5 if self._meaningful_candidate_snapshot(atom)["support_present"] else 0.35,
            )
            action = self._choose_corroboration_action(
                atom=atom,
                corroboration_plan=corroboration_plan,
                source_yield=current_source_yield,
                matched_results_by_source=current_matched_results,
                partial_results_by_source=current_partial_results,
                family_confirmation_count=current_family_confirmation_count,
                source_attempts_by_family=source_attempts_by_family,
                budget_profile=budget_profile,
                available_families=missing_sources,
                promotion_gap_class=promotion_gap_class,
            )
            controller_actions.append(action.as_dict())
            skipped_families.update(action.skipped_families)
            last_action = action.action
            last_transition_reason = action.reason
            chosen_family = action.target_family
            expected_gain_class = action.expected_gain_class
            fallback_strategy_used = action.fallback_strategy or fallback_strategy_used
            sufficiency_priority_reason = action.sufficiency_priority_reason or sufficiency_priority_reason
            near_miss_enrichment_action = action.action or near_miss_enrichment_action
            if action.action != "GATHER_CORROBORATION" or not action.target_family:
                break
            source_label = action.target_family
            missing_sources = [label for label in missing_sources if label != source_label]
            family_queries = self._recurrence_source_specific_queries(
                selected_queries=selected_queries,
                atom=atom,
                source_label=source_label,
                limit=max(3, min(4, len(selected_queries) + 2)),
                plan=corroboration_plan,
            )
            if not family_queries:
                source_attempts.append({"source": source_label, "queries": [], "added_docs": 0})
                continue

            family_subreddits = list(all_subreddits) if source_label == "reddit" else []
            family_sites = [(site, label) for site, label in all_sites if label == source_label]
            if not family_subreddits and not family_sites:
                source_attempts.append({"source": source_label, "queries": family_queries, "added_docs": 0})
                continue

            aggregate_attempt = {
                "source": source_label,
                "queries": [],
                "added_docs": 0,
                "attempts": 0,
                "controller_reason": action.reason,
                "expected_gain_class": action.expected_gain_class,
            }
            reshape_reason = ""
            family_docs: list[SearchDocument] = []
            family_queries_attempted: list[str] = []
            family_retrieved = 0
            family_deduped = 0
            for attempt_index in range(corroboration_plan.max_attempts_per_family):
                if source_label == "web":
                    family_sites, attempt_routing_override = self._specialized_web_routing_sites(
                        atom=atom,
                        plan=corroboration_plan,
                        attempt_index=attempt_index,
                    )
                    routing_override_reason = attempt_routing_override or routing_override_reason
                    if attempt_routing_override:
                        aggregate_attempt["routing_override_reason"] = attempt_routing_override
                    cohort_queries = self._spreadsheet_operator_admin_web_queries(atom=atom, plan=corroboration_plan)
                    specialized_queries = self._specialized_operator_surface_queries(atom=atom, plan=corroboration_plan)
                    if attempt_index == 0:
                        self._record_web_strategy_step(web_query_strategy_path, "atom_shaped")
                    else:
                        if cohort_queries:
                            self._record_web_strategy_step(web_query_strategy_path, "cohort_pack")
                        if specialized_queries:
                            self._record_web_strategy_step(web_query_strategy_path, "specialized_surface_targeting")
                            specialized_surface_targeting_used = True
                        self._record_web_strategy_step(web_query_strategy_path, "fallback_workaround_friction")
                        self._record_web_strategy_step(web_query_strategy_path, "decomposition")
                    if cohort_queries:
                        cohort_query_pack_used = True
                        cohort_query_pack_name = "spreadsheet_operator_admin"
                    aggregate_attempt["web_query_strategy_path"] = list(web_query_strategy_path)
                    aggregate_attempt["cohort_query_pack_used"] = cohort_query_pack_used
                    aggregate_attempt["cohort_query_pack_name"] = cohort_query_pack_name
                    aggregate_attempt["specialized_surface_targeting_used"] = specialized_surface_targeting_used
                if family_subreddits:
                    await self.warm_reddit_validation_queries(
                        subreddits=family_subreddits,
                        queries=family_queries,
                    )

                extra_docs, extra_results_by_query, extra_results_by_source, extra_collection_meta = await self._run_recurrence_collection(
                    queries=family_queries,
                    subreddit_plan=family_subreddits,
                    site_plan=family_sites,
                    atom=atom,
                    per_source_limit=max(2, self.validation_recurrence_limit // 2),
                    stop_after_docs=max(2, budget_profile["target_docs"]),
                    allow_fallback=False,
                )
                prior_doc_count = len(merged_docs)
                merged_docs, merged_results_by_query, merged_results_by_source = self._merge_recurrence_results(
                    existing_docs=merged_docs,
                    existing_results_by_query=merged_results_by_query,
                    existing_results_by_source=merged_results_by_source,
                    new_docs=extra_docs,
                    new_results_by_query=extra_results_by_query,
                    new_results_by_source=extra_results_by_source,
                    stop_after_docs=budget_profile["early_stop_docs"],
                )
                merged_collection_meta = self._merge_recurrence_collection_meta(merged_collection_meta, extra_collection_meta)
                added_docs = max(0, len(merged_docs) - prior_doc_count)
                added_docs_total += added_docs
                attempted_queries.extend(family_queries)
                aggregate_attempt["queries"].extend(query for query in family_queries if query not in aggregate_attempt["queries"])
                aggregate_attempt["added_docs"] += added_docs
                aggregate_attempt["attempts"] = attempt_index + 1
                family_docs = list((merged_collection_meta.get("docs_by_source", {}) or {}).get(source_label, []))
                family_queries_attempted = list((merged_collection_meta.get("queries_by_source", {}) or {}).get(source_label, []))
                family_retrieved = int((merged_collection_meta.get("retrieved_by_source", {}) or {}).get(source_label, 0) or 0)
                family_deduped = int((merged_collection_meta.get("deduped_by_source", {}) or {}).get(source_label, 0) or 0)
                family_yield = self._evaluate_source_yield(
                    source_label=source_label,
                    docs=family_docs,
                    docs_retrieved=family_retrieved,
                    docs_after_dedupe=family_deduped,
                    queries_attempted=family_queries_attempted,
                    atom=atom,
                    signature_terms=corroboration_plan.signature_terms,
                    attempts=attempt_index + 1,
                    reshape_reason=reshape_reason,
                )
                aggregate_attempt.update({
                    "docs_retrieved": family_retrieved,
                    "docs_after_dedupe": family_deduped,
                    "docs_after_problem_filter": family_yield["docs_after_problem_filter"],
                    "docs_partial_match": family_yield["docs_partial_match"],
                    "docs_strong_match": family_yield["docs_strong_match"],
                    "confirmed": family_yield["confirmed"],
                })
                if "independent_domains" in family_yield:
                    aggregate_attempt["independent_domains"] = family_yield["independent_domains"]
                if "repo_count" in family_yield:
                    aggregate_attempt["repo_count"] = family_yield["repo_count"]
                if reshape_reason:
                    aggregate_attempt["reshape_reason"] = reshape_reason
                source_attempts_by_family[source_label] = attempt_index + 1
                if family_yield["confirmed"] or attempt_index + 1 >= corroboration_plan.max_attempts_per_family:
                    break
                if family_yield["docs_strong_match"] == 0:
                    retry_action = self._choose_corroboration_action(
                        atom=atom,
                        corroboration_plan=corroboration_plan,
                        source_yield={source_label: family_yield},
                        matched_results_by_source={source_label: family_yield.get("docs_strong_match", 0)},
                        partial_results_by_source={source_label: family_yield.get("docs_partial_match", 0)},
                        family_confirmation_count=sum(1 for details in [family_yield] if details.get("confirmed")),
                        source_attempts_by_family=source_attempts_by_family,
                        budget_profile=budget_profile,
                        available_families=[source_label],
                        current_family=source_label,
                        promotion_gap_class=promotion_gap_class,
                    )
                    controller_actions.append(retry_action.as_dict())
                    last_action = retry_action.action
                    last_transition_reason = retry_action.reason
                    chosen_family = retry_action.target_family or source_label
                    expected_gain_class = retry_action.expected_gain_class or expected_gain_class
                    fallback_strategy_used = retry_action.fallback_strategy or fallback_strategy_used
                    promotion_gap_class = retry_action.promotion_gap_class or promotion_gap_class
                    sufficiency_priority_reason = retry_action.sufficiency_priority_reason or sufficiency_priority_reason
                    if retry_action.action != "RETRY_WITH_RESHAPED_QUERY":
                        break
                    if family_yield["docs_retrieved"] > 0:
                        reshape_reason = self._select_recurrence_reshape_reason(docs=family_docs, atom=atom)
                    else:
                        reshape_reason = self._select_empty_recurrence_reshape_reason(
                            atom=atom,
                            source_label=source_label,
                            attempted_queries=family_queries,
                        )
                    if not reshape_reason:
                        break
                    family_queries = self._recurrence_source_specific_queries(
                        selected_queries=selected_queries,
                        atom=atom,
                        source_label=source_label,
                        limit=max(3, min(4, len(selected_queries) + 2)),
                        plan=corroboration_plan,
                        reshape_reason=reshape_reason,
                    )
                    if not family_queries:
                        break
                    reshaped_query_history.append(
                        {
                            "source": source_label,
                            "attempt": attempt_index + 2,
                            "reason": reshape_reason,
                            "queries": family_queries,
                        }
                    )
                    if source_label == "web":
                        decomposed_atom_queries = self._decompose_low_information_atom(atom, corroboration_plan)
                    family_queries = self._recurrence_source_specific_queries(
                        selected_queries=selected_queries,
                        atom=atom,
                        source_label=source_label,
                        limit=max(3, min(4, len(selected_queries) + 2)),
                        plan=corroboration_plan,
                        reshape_reason=reshape_reason,
                    )
                    continue
                break

            source_attempts.append(aggregate_attempt)

            if len(self._active_recurrence_source_labels(merged_results_by_source)) >= target_sources:
                break

        return merged_docs, merged_results_by_query, merged_results_by_source, merged_collection_meta, {
            "triggered": True,
            "missing_sources": missing_sources,
            "queries": attempted_queries,
            "added_docs": added_docs_total,
            "source_attempts": source_attempts,
            "reshaped_query_history": reshaped_query_history,
            "last_action": last_action,
            "last_transition_reason": last_transition_reason,
            "chosen_family": chosen_family,
            "expected_gain_class": expected_gain_class,
            "skipped_families": skipped_families,
            "source_attempts_snapshot": {item.get("source", ""): item for item in source_attempts},
            "controller_actions": controller_actions,
            "budget_snapshot": action.budget_snapshot if 'action' in locals() else {},
            "fallback_strategy_used": fallback_strategy_used,
            "decomposed_atom_queries": decomposed_atom_queries,
            "routing_override_reason": routing_override_reason,
            "cohort_query_pack_used": cohort_query_pack_used,
            "cohort_query_pack_name": cohort_query_pack_name,
            "web_query_strategy_path": web_query_strategy_path,
            "specialized_surface_targeting_used": specialized_surface_targeting_used,
            "promotion_gap_class": promotion_gap_class,
            "near_miss_enrichment_action": near_miss_enrichment_action,
            "sufficiency_priority_reason": sufficiency_priority_reason,
            "candidate_meaningful": self._meaningful_candidate_snapshot(atom),
        }

    def _recurrence_fallback_queries(self, query: str, *, atom: Optional[Any] = None) -> list[str]:
        fallbacks = [query]
        if atom is not None:
            segment = self._recurrence_segment_terms(getattr(atom, "segment", "") or "")
            failure = self._recurrence_focus_phrase(getattr(atom, "failure_mode", "") or "", max_words=5)
            if segment and failure:
                fallbacks.append(f"{failure} {segment}")
        forum_variant = f'{query} forum'
        if forum_variant not in fallbacks:
            fallbacks.append(forum_variant)
        return [item for item in fallbacks if item]

    def _build_validation_query(self, title: str, summary: str) -> str:
        raw_text = compact_text(f"{summary} {title}", 500)
        raw_text = re.sub(r"\[[^\]]+\]", " ", raw_text)
        raw_text = re.sub(r"#\d+", " ", raw_text)
        raw_text = re.sub(r"\b(feature request|request|issue|bug|help wanted|discussion)\b", " ", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\b(i did it|now makes|looking for|best and not too expensive|anyone knows|any recommendations)\b", " ", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(
            r"\b(what manual process would you like to automate|what is the most annoying manual task|what’s the one manual process|question for|how do you track|what are people actually running|anyone want to join me|i want to complain)\b",
            " ",
            raw_text,
            flags=re.IGNORECASE,
        )
        raw_text = re.sub(r"https?://\S+", " ", raw_text)
        lowered = raw_text.lower()
        phrase_tokens = [f'"{phrase}"' for phrase in self._extract_validation_phrases(lowered)]

        prioritized_terms: list[str] = []
        for keyword in PAIN_KEYWORDS + VALUE_KEYWORDS + AI_TOOL_KEYWORDS:
            if contains_keyword(lowered, keyword):
                for token in keyword.split():
                    if token not in prioritized_terms and token not in QUERY_STOPWORDS and token not in WEAK_VALIDATION_TERMS:
                        prioritized_terms.append(token)

        tokens: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[a-z0-9][a-z0-9+.-]{1,24}", lowered):
            if token.isdigit() or token in QUERY_STOPWORDS:
                continue
            if len(token) <= 2:
                continue
            if token in WEAK_VALIDATION_TERMS:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)

        merged_terms: list[str] = []
        for token in phrase_tokens + prioritized_terms + tokens:
            if token not in merged_terms:
                merged_terms.append(token)

        return " ".join(merged_terms[: self.validation_query_terms]) or compact_text(title, 120)

    def _extract_tools(self, text: str) -> str:
        tool_patterns = [
            ("excel", "Excel"),
            ("google sheets", "Google Sheets"),
            ("sheets", "Google Sheets"),
            ("stripe", "Stripe"),
            ("quickbooks", "QuickBooks"),
            ("square", "Square"),
            ("etsy", "Etsy"),
            ("gmail", "Gmail"),
            ("slack", "Slack"),
            ("notion", "Notion"),
            ("airtable", "Airtable"),
            ("dropbox", "Dropbox"),
            ("zapier", "Zapier"),
            ("claude", "Claude"),
            ("chatgpt", "ChatGPT"),
        ]
        lowered = (text or "").lower()
        deduped: list[str] = []
        for needle, label in tool_patterns:
            if contains_keyword(lowered, needle) and label not in deduped:
                deduped.append(label)
        return ", ".join(deduped[:4])

    def _pain_score(self, text: str) -> int:
        normalized = (text or "").lower()
        return sum(1 for keyword in PAIN_KEYWORDS if keyword in normalized)

    def _comment_corroboration_score(self, comment_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        """Score comment corroboration for a Reddit finding.

        Returns:
            - comment_count: total comments fetched
            - agreement_count: comments expressing agreement ("me too", "same", etc.)
            - total_upvotes: sum of all comment scores
            - workaround_count: comments suggesting workarounds
            - corroboration_score: 0-1 normalized score
        """
        if not comment_metadata:
            return {
                "comment_count": 0,
                "agreement_count": 0,
                "total_upvotes": 0,
                "workaround_count": 0,
                "corroboration_score": 0.0,
            }

        agreement_patterns = ["me too", "same here", "i have this", "same problem", "experiencing this", "happening to me"]
        workaround_patterns = ["what i do is", "workaround", "i use", "i've been using", "solution", "fix", "resolved by"]

        agreement_count = 0
        total_upvotes = 0
        workaround_count = 0

        for comment in comment_metadata:
            body = (comment.get("body", "") or "").lower()
            score = comment.get("score", 0) or 0

            total_upvotes += score

            # Check for agreement
            if any(pattern in body for pattern in agreement_patterns):
                agreement_count += 1

            # Check for workarounds
            if any(pattern in body for pattern in workaround_patterns):
                workaround_count += 1

        comment_count = len(comment_metadata)

        # Calculate corroboration score (0-1)
        # Weight: agreement (0.4) + upvotes normalized (0.3) + workarounds (0.3)
        upvote_score = min(total_upvotes / 100, 1.0) * 0.3  # Cap at 100 upvotes
        agreement_score = min(agreement_count / 5, 1.0) * 0.4  # Cap at 5 agreements
        workaround_score = min(workaround_count / 3, 1.0) * 0.3  # Cap at 3 workarounds

        corroboration_score = agreement_score + upvote_score + workaround_score

        return {
            "comment_count": comment_count,
            "agreement_count": agreement_count,
            "total_upvotes": total_upvotes,
            "workaround_count": workaround_count,
            "corroboration_score": round(corroboration_score, 2),
        }

    def skill_and_tooling_manifest(self, title: str, audience: str, problem_statement: str) -> dict[str, Any]:
        return {
            "title": title,
            "audience": audience,
            "problem_statement": problem_statement,
            "value_hypothesis": f"Help {audience or 'operators'} solve {problem_statement}",
        }
