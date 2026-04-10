"""Problem space model for LLM-driven autonomous discovery expansion.

A ProblemSpace represents a semantic problem domain (e.g., "financial
reconciliation") that owns derived search queries.  After each validation
cycle, an LLM analyses validated opportunities and proposes new adjacent
problem spaces.  Keywords, subreddits, and web queries become derived
properties of a space rather than the primary organising unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json


# Lifecycle states
EXPLORING = "exploring"
VALIDATED = "validated"
EXHAUSTED = "exhausted"
ARCHIVED = "archived"

VALID_STATES = {EXPLORING, VALIDATED, EXHAUSTED, ARCHIVED}

# Source types
SOURCE_LLM = "llm"
SOURCE_MANUAL = "manual"
SOURCE_THEME_MIGRATION = "theme_migration"

# Term types
TERM_KEYWORD = "keyword"
TERM_SUBREDDIT = "subreddit"
TERM_WEB_QUERY = "web_query"
TERM_GITHUB_QUERY = "github_query"

VALID_TERM_TYPES = {TERM_KEYWORD, TERM_SUBREDDIT, TERM_WEB_QUERY, TERM_GITHUB_QUERY}


def _coerce_json_list(raw: str, *, subreddit: bool = False) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        data = raw
    if isinstance(data, str):
        items = [piece for piece in data.replace("\n", ",").split(",") if piece]
    elif isinstance(data, (list, tuple, set)):
        items = list(data)
    else:
        items = [data]
    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if subreddit:
            text = text.removeprefix("r/").removeprefix("/r/").strip()
            if len(text) < 2:
                continue
        if text not in normalized:
            normalized.append(text)
    return normalized


@dataclass
class ProblemSpace:
    """A semantic problem domain that drives its own search queries."""

    space_key: str  # unique slug, e.g. "financial_reconciliation"
    label: str  # human-readable, e.g. "Financial reconciliation"
    description: str = ""  # 1-2 sentence semantic description
    parent_space_key: str | None = None  # for hierarchical spaces
    status: str = EXPLORING  # exploring | validated | exhausted | archived

    # LLM-generated context
    semantic_summary: str = ""  # what this space is about
    adjacent_spaces_json: str = "[]"  # suggested adjacent spaces (not yet created)

    # Derived search configuration
    keywords_json: str = "[]"  # derived search keywords
    subreddits_json: str = "[]"  # derived subreddits
    web_queries_json: str = "[]"  # derived web search queries
    github_queries_json: str = "[]"  # derived GitHub search queries

    # Provenance
    source: str = SOURCE_LLM  # llm | manual | theme_migration
    llm_provider: str = ""  # ollama | anthropic | ""
    llm_model: str = ""  # model used for generation
    generation_prompt_hash: str = ""  # hash of the prompt (dedup)

    # Metrics (populated from linked search terms)
    total_findings: int = 0
    total_validations: int = 0
    total_prototype_candidates: int = 0
    total_build_briefs: int = 0
    yield_score: float = 0.0

    # Timestamps
    created_at: str | None = None
    updated_at: str | None = None

    id: int | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATES:
            raise ValueError(f"Invalid status: {self.status}")
        self.keywords_json = json.dumps(_coerce_json_list(self.keywords_json))
        self.subreddits_json = json.dumps(_coerce_json_list(self.subreddits_json, subreddit=True))
        self.web_queries_json = json.dumps(_coerce_json_list(self.web_queries_json))
        self.github_queries_json = json.dumps(_coerce_json_list(self.github_queries_json))
        self.adjacent_spaces_json = json.dumps(_coerce_json_list(self.adjacent_spaces_json))

    @property
    def keywords(self) -> list[str]:
        return _coerce_json_list(self.keywords_json)

    @keywords.setter
    def keywords(self, value: list[str]) -> None:
        self.keywords_json = json.dumps(_coerce_json_list(json.dumps(value) if not isinstance(value, str) else value))

    @property
    def subreddits(self) -> list[str]:
        return _coerce_json_list(self.subreddits_json, subreddit=True)

    @subreddits.setter
    def subreddits(self, value: list[str]) -> None:
        self.subreddits_json = json.dumps(
            _coerce_json_list(json.dumps(value) if not isinstance(value, str) else value, subreddit=True)
        )

    @property
    def web_queries(self) -> list[str]:
        return _coerce_json_list(self.web_queries_json)

    @web_queries.setter
    def web_queries(self, value: list[str]) -> None:
        self.web_queries_json = json.dumps(_coerce_json_list(json.dumps(value) if not isinstance(value, str) else value))

    @property
    def github_queries(self) -> list[str]:
        return _coerce_json_list(self.github_queries_json)

    @github_queries.setter
    def github_queries(self, value: list[str]) -> None:
        self.github_queries_json = json.dumps(
            _coerce_json_list(json.dumps(value) if not isinstance(value, str) else value)
        )

    @property
    def adjacent_spaces(self) -> list[str]:
        return _coerce_json_list(self.adjacent_spaces_json)

    @adjacent_spaces.setter
    def adjacent_spaces(self, value: list[str]) -> None:
        self.adjacent_spaces_json = json.dumps(
            _coerce_json_list(json.dumps(value) if not isinstance(value, str) else value)
        )


@dataclass
class ProblemSpaceTerm:
    """A derived search term linked to a problem space."""

    space_key: str
    term_type: str  # keyword | subreddit | web_query | github_query
    term_value: str
    source: str = "derived"  # derived | manual | fallback
    created_at: str | None = None

    id: int | None = None
