"""Shared search/result model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchDocument:
    title: str
    url: str
    snippet: str
    source: str
    source_family: str = ""
    retrieval_query: str = ""
