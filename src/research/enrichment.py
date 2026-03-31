"""Enrichment module - recurrence queries and corroboration planning."""

from __future__ import annotations

from typing import Any, Optional


class RecurrenceQueryBuilder:
    """Builds recurrence-focused search queries."""

    def __init__(self):
        self.queries: list[str] = []

    def add(self, query: str) -> None:
        """Add a query component."""
        if query and query not in self.queries:
            self.queries.append(query)

    def build(self) -> list[str]:
        """Build the final query list."""
        return self.queries.copy()


class CorroborationPlanner:
    """Plans corroboration actions for validation."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.actions: list[dict[str, Any]] = []

    def add_action(self, action_type: str, query: str, priority: int = 1) -> None:
        """Add a corroboration action."""
        self.actions.append({
            "type": action_type,
            "query": query,
            "priority": priority,
        })

    def build_plan(self) -> list[dict[str, Any]]:
        """Build the final corroboration plan."""
        return sorted(self.actions, key=lambda x: x.get("priority", 1))


def build_recurrence_query(atom: dict[str, Any]) -> str:
    """Build a recurrence query from an atom."""
    product = atom.get("product_built", "")
    pain_point = atom.get("pain_point", "")

    parts = []
    if product:
        parts.append(product)
    if pain_point:
        parts.append(pain_point)

    query = " ".join(parts)
    return query


def build_competitor_query(atom: dict[str, Any]) -> str:
    """Build a competitor query from an atom."""
    product = atom.get("product_built", "")
    return f"alternative to {product}" if product else ""


def build_value_enrichment_query(atom: dict[str, Any]) -> str:
    """Build a value enrichment query from an atom."""
    product = atom.get("product_built", "")
    pain_point = atom.get("pain_point", "")
    return f"{product} {pain_point} ROI"


# Query decomposition strategies
def decompose_recurrence_queries(text: str, max_queries: int = 4) -> list[str]:
    """Decompose text into recurrence-focused queries."""
    queries = []
    words = text.split()

    # Generate phrase combinations
    for i in range(len(words)):
        for length in [2, 3, 4]:
            if i + length <= len(words):
                phrase = " ".join(words[i:i + length])
                queries.append(phrase)
                if len(queries) >= max_queries:
                    return queries

    return queries[:max_queries]


def decompose_low_info_atom(text: str, max_queries: int = 3) -> list[str]:
    """Decompose low-information atoms into queries."""
    queries = []
    words = text.split()

    # Single term queries
    for word in words[:max_queries]:
        queries.append(word)

    return queries


# Query prioritization
def prioritize_queries(queries: list[str], focus_keywords: list[str]) -> list[str]:
    """Prioritize queries based on focus keywords."""
    scored = []
    for q in queries:
        score = sum(1 for kw in focus_keywords if kw.lower() in q.lower())
        scored.append((score, q))

    scored.sort(key=lambda x: -x[0])
    return [q for _, q in scored]