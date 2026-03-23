"""Legacy discovery dataclasses kept for compatibility helpers and older paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


def parse_json_blob(raw: str | None, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


@dataclass
class RawSignal:
    source_type: str
    source_name: str
    source_url: str
    title: str
    source_class: str = "pain_signal"
    author: str = ""
    role_hint: str = ""
    published_at: Optional[str] = None
    content_hash: str = ""
    content_text: str = ""
    quotes_json: str = "[]"
    metadata_json: str = "{}"
    finding_id: Optional[int] = None
    id: Optional[int] = None
    collected_at: Optional[str] = None


@dataclass
class ProblemAtom:
    signal_id: int
    segment: str
    user_role: str
    job_to_be_done: str
    trigger_event: str
    pain_statement: str
    failure_mode: str
    current_workaround: str
    current_tools: str
    urgency_clues: str
    frequency_clues: str
    emotional_intensity: float
    cost_consequence_clues: str
    why_now_clues: str
    confidence: float
    finding_id: Optional[int] = None
    cluster_id: Optional[int] = None
    supporting_quote: str = ""
    extraction_method: str = "heuristic"
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class PatternCluster:
    cluster_key: str
    label: str
    summary: str
    segment: str
    user_role: str
    job_to_be_done: str
    trigger_pattern: str
    workaround_pattern: str
    failure_pattern: str
    source_count: int
    signal_count: int
    atom_count: int
    evidence_quality: float
    status: str = "candidate"
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class OpportunityRecord:
    cluster_id: int
    title: str
    market_gap_state: str
    status: str
    total_score: float
    evidence_quality: float
    score_json: str = "{}"
    support_json: str = "{}"
    counter_json: str = "{}"
    rationale: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class ValidationExperiment:
    opportunity_id: int
    test_type: str
    hypothesis: str
    smallest_test: str
    falsification_condition: str
    status: str = "proposed"
    priority: int = 3
    plan_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class EvidenceLedgerEntry:
    entity_type: str
    entity_id: int
    entry_kind: str
    stance: str
    summary: str
    evidence_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class CorroborationRecord:
    finding_id: int
    recurrence_state: str
    recurrence_score: float
    corroboration_score: float
    evidence_sufficiency: float
    query_coverage: float
    independent_confirmations: int
    source_diversity: int
    query_set_hash: str
    evidence_json: str = "{}"
    run_id: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class MarketEnrichment:
    finding_id: int
    demand_score: float
    buyer_intent_score: float
    competition_score: float
    trend_score: float
    review_signal_score: float
    value_signal_score: float
    evidence_json: str = "{}"
    run_id: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
