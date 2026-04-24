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
    query_origin: str = ""
    distinguishing_source_field: str = ""
    distinguishing_concept: str = ""
    distinguishing_concept_span: str = ""
    domain_key: str = ""
    workflow_cluster_key: str = ""
    retrieval_strategy_key: str = ""
    corroboration_strategy_key: str = ""
