"""Database schema and operations for the evidence-first discovery pipeline."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.build_prep import is_allowed_selection_transition
from src.database_views import (
    build_candidate_workbench_item,
    build_recent_validation_row,
    build_validation_corroboration_digest,
    build_validation_review_row,
)


def _json_loads(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {})


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {row[1] for row in rows}


def _dedupe_evidence_ledger(conn: sqlite3.Connection) -> None:
    """Collapse duplicate ledger rows (same run/entity/kind); keep smallest id."""
    try:
        conn.execute(
            """
            DELETE FROM evidence_ledger
            WHERE id NOT IN (
                SELECT MIN(id) FROM evidence_ledger GROUP BY run_id, entity_type, entity_id, entry_kind
            )
            """
        )
    except sqlite3.OperationalError:
        pass


def _ensure_evidence_ledger_unique_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ledger_run_entity_kind
        ON evidence_ledger(run_id, entity_type, entity_id, entry_kind)
        """
    )


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    column_sql: str,
    *,
    backfill_sql: str | None = None,
) -> None:
    columns = _table_columns(conn, table)
    if column in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")
    if backfill_sql:
        conn.execute(backfill_sql)


@dataclass
class Finding:
    source: str
    source_url: str
    entrepreneur: str = ""
    tool_used: str = ""
    product_built: str = ""
    monetization_method: str = ""
    outcome_summary: str = ""
    content_hash: str = ""
    status: str = "new"
    finding_kind: str = "problem_signal"
    source_class: str = "pain_signal"
    recurrence_key: str = ""
    evidence: dict[str, Any] | None = None
    evidence_json: str = "{}"
    id: Optional[int] = None
    discovered_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence is None and self.evidence_json not in (None, ""):
            self.evidence = _json_loads(self.evidence_json, {})
        self.evidence_json = _json_dumps(self.evidence)


@dataclass
class RawSignal:
    finding_id: int = 0
    source_name: str = ""
    source_type: str = ""
    source_url: str = ""
    title: str = ""
    body_excerpt: str = ""
    content_hash: str = ""
    source_class: str = "pain_signal"
    quote_text: str = ""
    role_hint: str = ""
    published_at: str = ""
    timestamp_hint: str = ""
    metadata: dict[str, Any] | None = None
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata is None and self.metadata_json not in (None, ""):
            self.metadata = _json_loads(self.metadata_json, {})
        self.metadata_json = _json_dumps(self.metadata)


@dataclass
class ProblemAtom:
    finding_id: int = 0
    raw_signal_id: int = 0
    signal_id: int = 0
    cluster_key: str = ""
    segment: str = ""
    user_role: str = ""
    job_to_be_done: str = ""
    pain_statement: str = ""
    trigger_event: str = ""
    failure_mode: str = ""
    current_workaround: str = ""
    current_tools: str = ""
    source_quote: str = ""
    urgency_clues: str = ""
    frequency_clues: str = ""
    emotional_intensity: float = 0.0
    cost_consequence_clues: str = ""
    why_now_clues: str = ""
    confidence: float = 0.0
    confidence_score: float = 0.0
    atom_json: str = "{}"
    score_json: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.score_json is None:
            self.score_json = {}
        if self.metadata is None and self.metadata_json not in (None, ""):
            self.metadata = _json_loads(self.metadata_json, {})
        self.metadata_json = _json_dumps(self.metadata)


@dataclass
class OpportunityCluster:
    label: str
    cluster_key: str = ""
    status: str = "active"
    segment: str = ""
    summary: dict[str, Any] | None = None
    summary_json: str = "{}"
    user_role: str = ""
    job_to_be_done: str = ""
    trigger_summary: str = ""
    signal_count: int = 0
    atom_count: int = 0
    evidence_quality: float = 0.0
    metadata: dict[str, Any] | None = None
    metadata_json: str = "{}"
    id: Optional[int] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.summary is None and self.summary_json not in (None, ""):
            self.summary = _json_loads(self.summary_json, {})
        if self.metadata is None and self.metadata_json not in (None, ""):
            self.metadata = _json_loads(self.metadata_json, {})
        self.summary_json = _json_dumps(self.summary)
        self.metadata_json = _json_dumps(
            {
                **(self.metadata or {}),
                "cluster_key": self.cluster_key,
                "segment": self.segment,
                "trigger_summary": self.trigger_summary,
                "signal_count": self.signal_count,
                "atom_count": self.atom_count,
                "evidence_quality": self.evidence_quality,
            }
        )


@dataclass
class Opportunity:
    cluster_id: int
    title: str
    market_gap: str
    recommendation: str
    status: str
    pain_severity: float = 0.0
    frequency_score: float = 0.0
    cost_of_inaction: float = 0.0
    workaround_density: float = 0.0
    urgency_score: float = 0.0
    segment_concentration: float = 0.0
    reachability: float = 0.0
    timing_shift: float = 0.0
    buildability: float = 0.0
    expansion_potential: float = 0.0
    education_burden: float = 0.0
    dependency_risk: float = 0.0
    adoption_friction: float = 0.0
    evidence_quality: float = 0.0
    composite_score: float = 0.0
    confidence: float = 0.0
    scoring_version: str = "0"  # "0" = not yet validated, set by validation agent
    # v4: PTS/RRS split scoring
    problem_truth_score: float = 0.0
    revenue_readiness_score: float = 0.0
    decision_score: float = 0.0
    problem_plausibility: float = 0.0
    value_support: float = 0.0
    corroboration_strength: float = 0.0
    evidence_sufficiency: float = 0.0
    willingness_to_pay_proxy: float = 0.0
    # Version tracking
    formula_version: str = "original"
    threshold_version: str = "2025_q1"
    evaluated_at: Optional[str] = None
    last_rescored_at: Optional[str] = None
    # Legacy fields
    selection_status: str = "research_more"
    selection_reason: str = ""
    notes: dict[str, Any] | None = None
    notes_json: str = "{}"
    id: Optional[int] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.notes is None and self.notes_json not in (None, ""):
            self.notes = _json_loads(self.notes_json, {})
        self.notes_json = _json_dumps(self.notes)


@dataclass
class Validation:
    finding_id: int
    run_id: str = ""
    market_score: float = 0.0
    technical_score: float = 0.0
    distribution_score: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    evidence: dict[str, Any] | None = None
    evidence_json: str = "{}"
    id: Optional[int] = None
    validated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence is None and self.evidence_json not in (None, ""):
            self.evidence = _json_loads(self.evidence_json, {})
        self.evidence_json = _json_dumps(self.evidence)


@dataclass
class ValidationExperiment:
    opportunity_id: int
    cluster_id: int
    test_type: str
    hypothesis: str
    falsifier: str
    smallest_test: str
    success_signal: str
    failure_signal: str
    run_id: str = ""
    status: str = "proposed"
    result: dict[str, Any] | None = None
    result_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.result is None and self.result_json not in (None, ""):
            self.result = _json_loads(self.result_json, {})
        self.result_json = _json_dumps(self.result)


@dataclass
class EvidenceLedgerEntry:
    entity_type: str
    entity_id: int
    entry_kind: str
    entry_json: dict[str, Any] | None = None
    stance: str = "neutral"
    source_name: str = ""
    source_url: str = ""
    quote_text: str = ""
    summary: str = ""
    metadata_json: str = "{}"
    run_id: str = ""
    id: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.entry_json is None:
            self.entry_json = _json_loads(self.metadata_json, {})
        payload = dict(self.entry_json or {})
        if self.stance:
            payload.setdefault("stance", self.stance)
        if self.source_name:
            payload.setdefault("source_name", self.source_name)
        if self.source_url:
            payload.setdefault("source_url", self.source_url)
        if self.quote_text:
            payload.setdefault("quote_text", self.quote_text)
        if self.summary:
            payload.setdefault("summary", self.summary)
        self.entry_json = payload
        self.metadata_json = _json_dumps(payload)


@dataclass
class CorroborationRecord:
    finding_id: int
    recurrence_state: str
    run_id: str = ""
    recurrence_score: float = 0.0
    corroboration_score: float = 0.0
    evidence_sufficiency: float = 0.0
    query_coverage: float = 0.0
    independent_confirmations: int = 0
    source_diversity: int = 0
    query_set_hash: str = ""
    evidence: dict[str, Any] | None = None
    evidence_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence is None and self.evidence_json not in (None, ""):
            self.evidence = _json_loads(self.evidence_json, {})
        self.evidence_json = _json_dumps(self.evidence)


@dataclass
class MarketEnrichment:
    finding_id: int
    run_id: str = ""
    demand_score: float = 0.0
    buyer_intent_score: float = 0.0
    competition_score: float = 0.0
    trend_score: float = 0.0
    review_signal_score: float = 0.0
    value_signal_score: float = 0.0
    evidence: dict[str, Any] | None = None
    evidence_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.evidence is None and self.evidence_json not in (None, ""):
            self.evidence = _json_loads(self.evidence_json, {})
        self.evidence_json = _json_dumps(self.evidence)


@dataclass
class ReviewFeedback:
    finding_id: int
    review_label: str
    note: str = ""
    cluster_id: Optional[int] = None
    opportunity_id: Optional[int] = None
    validation_id: Optional[int] = None
    run_id: str = ""
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class BuildBrief:
    opportunity_id: int
    validation_id: int
    cluster_id: int
    run_id: str = ""
    status: str = "prototype_candidate"
    recommended_output_type: str = ""
    schema_version: str = "build_brief_v1"
    brief_hash: str = ""
    brief_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def brief(self) -> dict[str, Any]:
        return _json_loads(self.brief_json, {})


@dataclass
class BuildPrepOutput:
    build_brief_id: int
    opportunity_id: int
    validation_id: int
    agent_name: str
    prep_stage: str
    run_id: str = ""
    status: str = "ready"
    output_hash: str = ""
    output_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def output(self) -> dict[str, Any]:
        return _json_loads(self.output_json, {})


@dataclass
class DiscoveryTheme:
    theme_key: str
    label: str
    query_seeds: list[str] | None = None
    source_signals: list[str] | None = None
    times_seen: int = 0
    yield_score: float = 0.0
    run_id: str = ""
    id: Optional[int] = None
    updated_at: Optional[str] = None

    @property
    def query_seeds_json(self) -> str:
        return _json_dumps(self.query_seeds or [])

    @property
    def source_signals_json(self) -> str:
        return _json_dumps(self.source_signals or [])


@dataclass
class Idea:
    description: str
    title: str = ""
    slug: str = ""
    name: str = ""
    pattern_ids: str = ""
    estimated_market_size: str = ""
    technical_complexity: str = ""
    status: str = "proposed"
    audience: str = ""
    monetization_strategy: str = ""
    confidence_score: float = 0.0
    product_type: str = "solution"
    spec_json: str = "{}"
    build_brief_id: int = 0
    id: Optional[int] = None
    created_at: Optional[str] = None

    @property
    def spec(self) -> dict[str, Any]:
        return _json_loads(self.spec_json, {})


@dataclass
class Product:
    idea_id: int = 0
    build_brief_id: int = 0
    opportunity_id: int = 0
    validation_id: int = 0
    name: str = ""
    location: str = ""
    status: str = "proposed"
    metadata: dict[str, Any] | None = None
    metadata_json: str = "{}"
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata is None and self.metadata_json not in (None, ""):
            self.metadata = _json_loads(self.metadata_json, {})
        self.metadata_json = _json_dumps(self.metadata)


@dataclass
class ResourceRecord:
    query: str = ""
    resource_type: str = ""
    source_url: str = ""
    relevance_score: float = 0.0
    access_method: str = ""
    cost_info: str = ""
    doc_quality: str = ""
    metadata_json: str = "{}"
    id: Optional[int] = None
    last_verified: Optional[str] = None
    cached_at: Optional[str] = None


class Database:
    """SQLite database manager with thread-local connections."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self.active_run_id = ""

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def init_schema(self) -> None:
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                entrepreneur TEXT,
                tool_used TEXT,
                product_built TEXT,
                monetization_method TEXT,
                outcome_summary TEXT,
                content_hash TEXT UNIQUE,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'new',
                finding_kind TEXT DEFAULT 'problem_signal',
                source_class TEXT DEFAULT 'pain_signal',
                recurrence_key TEXT,
                evidence_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
            CREATE INDEX IF NOT EXISTS idx_findings_source ON findings(source);
            CREATE TABLE IF NOT EXISTS raw_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id INTEGER REFERENCES findings(id),
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_class TEXT DEFAULT 'pain_signal',
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                body_excerpt TEXT NOT NULL,
                quote_text TEXT DEFAULT '',
                role_hint TEXT DEFAULT '',
                published_at TEXT,
                timestamp_hint TEXT DEFAULT '',
                content_hash TEXT UNIQUE,
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS problem_atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id INTEGER REFERENCES findings(id),
                raw_signal_id INTEGER DEFAULT 0,
                user_role TEXT DEFAULT '',
                job_to_be_done TEXT DEFAULT '',
                pain_statement TEXT DEFAULT '',
                trigger_event TEXT DEFAULT '',
                failure_mode TEXT DEFAULT '',
                current_workaround TEXT DEFAULT '',
                current_tools TEXT DEFAULT '',
                source_quote TEXT DEFAULT '',
                confidence_score REAL DEFAULT 0,
                atom_json TEXT DEFAULT '{}',
                score_json TEXT DEFAULT '{}',
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS opportunity_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                summary_json TEXT DEFAULT '{}',
                user_role TEXT DEFAULT '',
                job_to_be_done TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                summary_json TEXT DEFAULT '{}',
                user_role TEXT DEFAULT '',
                job_to_be_done TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS cluster_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                problem_atom_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL REFERENCES opportunity_clusters(id),
                title TEXT NOT NULL,
                market_gap TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                status TEXT NOT NULL,
                pain_severity REAL DEFAULT 0,
                frequency_score REAL DEFAULT 0,
                cost_of_inaction REAL DEFAULT 0,
                workaround_density REAL DEFAULT 0,
                urgency_score REAL DEFAULT 0,
                segment_concentration REAL DEFAULT 0,
                reachability REAL DEFAULT 0,
                timing_shift REAL DEFAULT 0,
                buildability REAL DEFAULT 0,
                expansion_potential REAL DEFAULT 0,
                education_burden REAL DEFAULT 0,
                dependency_risk REAL DEFAULT 0,
                adoption_friction REAL DEFAULT 0,
                evidence_quality REAL DEFAULT 0,
                composite_score REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                scoring_version TEXT DEFAULT 'v1',
                -- v4 additions: PTS/RRS split scoring
                problem_truth_score REAL DEFAULT 0,
                revenue_readiness_score REAL DEFAULT 0,
                decision_score REAL DEFAULT 0,
                problem_plausibility REAL DEFAULT 0,
                value_support REAL DEFAULT 0,
                corroboration_strength REAL DEFAULT 0,
                evidence_sufficiency REAL DEFAULT 0,
                willingness_to_pay_proxy REAL DEFAULT 0,
                -- Version tracking
                formula_version TEXT DEFAULT 'original',
                threshold_version TEXT DEFAULT '2025_q1',
                evaluated_at TIMESTAMP,
                last_rescored_at TIMESTAMP,
                -- Legacy fields
                selection_status TEXT DEFAULT 'research_more',
                selection_reason TEXT DEFAULT '',
                notes_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_opportunities_selection_status ON opportunities(selection_status);
            CREATE INDEX IF NOT EXISTS idx_opportunities_composite_score ON opportunities(composite_score);
            CREATE INDEX IF NOT EXISTS idx_opportunities_decision_score ON opportunities(decision_score);
            CREATE INDEX IF NOT EXISTS idx_opportunities_scoring_version ON opportunities(scoring_version);
            CREATE INDEX IF NOT EXISTS idx_opportunities_formula_version ON opportunities(formula_version);
            -- Scoring run audit table
            CREATE TABLE IF NOT EXISTS scoring_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                formula_version TEXT DEFAULT 'pts_rrs_v1',
                threshold_version TEXT DEFAULT '2025_q2',
                scoring_version TEXT DEFAULT 'v4',
                opportunity_count INTEGER DEFAULT 0,
                promote_count INTEGER DEFAULT 0,
                park_count INTEGER DEFAULT 0,
                kill_count INTEGER DEFAULT 0,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_scoring_runs_computed_at ON scoring_runs(computed_at);
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                finding_id INTEGER REFERENCES findings(id),
                market_score REAL,
                technical_score REAL,
                distribution_score REAL,
                overall_score REAL,
                passed BOOLEAN,
                evidence TEXT,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_validations_run_finding ON validations(run_id, finding_id);
            CREATE TABLE IF NOT EXISTS validation_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                opportunity_id INTEGER NOT NULL REFERENCES opportunities(id),
                cluster_id INTEGER NOT NULL REFERENCES opportunity_clusters(id),
                test_type TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                falsifier TEXT NOT NULL,
                smallest_test TEXT NOT NULL,
                success_signal TEXT NOT NULL,
                failure_signal TEXT NOT NULL,
                status TEXT DEFAULT 'proposed',
                result_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS evidence_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                entry_kind TEXT NOT NULL,
                entry_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS corroborations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                finding_id INTEGER NOT NULL REFERENCES findings(id),
                recurrence_state TEXT NOT NULL,
                recurrence_score REAL DEFAULT 0,
                corroboration_score REAL DEFAULT 0,
                evidence_sufficiency REAL DEFAULT 0,
                query_coverage REAL DEFAULT 0,
                independent_confirmations INTEGER DEFAULT 0,
                source_diversity INTEGER DEFAULT 0,
                query_set_hash TEXT DEFAULT '',
                evidence_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_corroborations_run_finding ON corroborations(run_id, finding_id);
            CREATE TABLE IF NOT EXISTS market_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                finding_id INTEGER NOT NULL REFERENCES findings(id),
                demand_score REAL DEFAULT 0,
                buyer_intent_score REAL DEFAULT 0,
                competition_score REAL DEFAULT 0,
                trend_score REAL DEFAULT 0,
                review_signal_score REAL DEFAULT 0,
                value_signal_score REAL DEFAULT 0,
                evidence_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_market_enrichments_run_finding ON market_enrichments(run_id, finding_id);
            CREATE TABLE IF NOT EXISTS review_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                finding_id INTEGER NOT NULL REFERENCES findings(id),
                cluster_id INTEGER,
                opportunity_id INTEGER,
                validation_id INTEGER,
                review_label TEXT NOT NULL,
                note TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS build_briefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                opportunity_id INTEGER NOT NULL REFERENCES opportunities(id),
                validation_id INTEGER NOT NULL REFERENCES validations(id),
                cluster_id INTEGER NOT NULL REFERENCES opportunity_clusters(id),
                status TEXT DEFAULT 'prototype_candidate',
                recommended_output_type TEXT DEFAULT '',
                schema_version TEXT DEFAULT 'build_brief_v1',
                brief_hash TEXT DEFAULT '',
                brief_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_build_briefs_run_opportunity ON build_briefs(run_id, opportunity_id);
            CREATE TABLE IF NOT EXISTS build_prep_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                build_brief_id INTEGER NOT NULL REFERENCES build_briefs(id),
                opportunity_id INTEGER NOT NULL REFERENCES opportunities(id),
                validation_id INTEGER NOT NULL REFERENCES validations(id),
                agent_name TEXT NOT NULL,
                prep_stage TEXT NOT NULL,
                status TEXT DEFAULT 'ready',
                output_hash TEXT DEFAULT '',
                output_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_build_prep_outputs_run_agent
            ON build_prep_outputs(run_id, build_brief_id, agent_name);
            CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                slug TEXT UNIQUE,
                name TEXT,
                description TEXT NOT NULL,
                pattern_ids TEXT,
                estimated_market_size TEXT,
                technical_complexity TEXT,
                status TEXT DEFAULT 'proposed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                audience TEXT DEFAULT '',
                monetization_strategy TEXT DEFAULT '',
                confidence_score REAL DEFAULT 0,
                product_type TEXT DEFAULT 'solution',
                spec_json TEXT DEFAULT '{}',
                build_brief_id INTEGER REFERENCES build_briefs(id)
            );
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id INTEGER DEFAULT 0,
                build_brief_id INTEGER DEFAULT 0,
                opportunity_id INTEGER DEFAULT 0,
                validation_id INTEGER DEFAULT 0,
                name TEXT,
                location TEXT DEFAULT '',
                status TEXT DEFAULT 'proposed',
                test_results TEXT,
                tooling_manifest TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                built_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT DEFAULT '',
                resource_type TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                relevance_score REAL DEFAULT 0,
                access_method TEXT DEFAULT '',
                cost_info TEXT DEFAULT '',
                doc_quality TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                last_verified TEXT,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS discovery_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                query_text TEXT NOT NULL,
                runs INTEGER DEFAULT 0,
                docs_seen INTEGER DEFAULT 0,
                findings_emitted INTEGER DEFAULT 0,
                validations INTEGER DEFAULT 0,
                passes INTEGER DEFAULT 0,
                prototype_candidates INTEGER DEFAULT 0,
                build_briefs INTEGER DEFAULT 0,
                avg_validation_score REAL DEFAULT 0,
                last_latency_ms REAL DEFAULT 0,
                last_status TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_name, query_text)
            );
            CREATE TABLE IF NOT EXISTS discovery_themes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme_key TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL,
                query_seeds_json TEXT DEFAULT '[]',
                source_signals_json TEXT DEFAULT '[]',
                times_seen INTEGER DEFAULT 0,
                yield_score REAL DEFAULT 0,
                run_id TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                opportunity_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                test_type TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                falsifier TEXT NOT NULL,
                smallest_test TEXT NOT NULL,
                success_signal TEXT NOT NULL,
                failure_signal TEXT NOT NULL,
                status TEXT DEFAULT 'proposed',
                result_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        _ensure_column(
            conn,
            "problem_atoms",
            "raw_signal_id",
            "INTEGER DEFAULT 0",
            backfill_sql="UPDATE problem_atoms SET raw_signal_id = COALESCE(signal_id, 0) WHERE raw_signal_id IS NULL",
        )
        _ensure_column(conn, "problem_atoms", "source_quote", "TEXT DEFAULT ''")
        _ensure_column(
            conn,
            "problem_atoms",
            "confidence_score",
            "REAL DEFAULT 0",
            backfill_sql="UPDATE problem_atoms SET confidence_score = COALESCE(confidence, 0) WHERE confidence_score IS NULL",
        )
        _ensure_column(
            conn,
            "problem_atoms",
            "atom_json",
            "TEXT DEFAULT '{}'",
            backfill_sql="""
            UPDATE problem_atoms
            SET atom_json = COALESCE(metadata_json, '{}')
            WHERE atom_json IS NULL OR atom_json = ''
            """,
        )
        _ensure_column(
            conn,
            "problem_atoms",
            "score_json",
            "TEXT DEFAULT '{}'",
            backfill_sql="""
            UPDATE problem_atoms
            SET score_json = COALESCE(atom_json, '{}')
            WHERE score_json IS NULL OR score_json = ''
            """,
        )
        _ensure_column(conn, "problem_atoms", "metadata_json", "TEXT DEFAULT '{}'")
        _ensure_column(conn, "opportunity_clusters", "metadata_json", "TEXT DEFAULT '{}'")
        _ensure_column(conn, "discovery_feedback", "prototype_candidates", "INTEGER DEFAULT 0")
        _ensure_column(conn, "discovery_feedback", "build_briefs", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "idea_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "build_brief_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "opportunity_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "validation_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "location", "TEXT DEFAULT ''")
        _ensure_column(conn, "products", "build_brief_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "opportunity_id", "INTEGER DEFAULT 0")
        _ensure_column(conn, "products", "name", "TEXT")
        _ensure_column(conn, "products", "status", "TEXT DEFAULT 'proposed'")
        _ensure_column(conn, "products", "test_results", "TEXT")
        # Add missing JSON and timestamp columns to products table
        # SQLite doesn't support DEFAULT CURRENT_TIMESTAMP in ALTER TABLE
        try:
            columns = _table_columns(conn, "products")
            if "metadata_json" not in columns:
                conn.execute("ALTER TABLE products ADD COLUMN metadata_json TEXT DEFAULT '{}'")
            if "created_at" not in columns:
                conn.execute("ALTER TABLE products ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                conn.execute("UPDATE products SET created_at = COALESCE(built_at, CURRENT_TIMESTAMP) WHERE created_at IS NULL")
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE products ADD COLUMN updated_at TIMESTAMP")
                conn.execute(
                    "UPDATE products SET updated_at = COALESCE(built_at, CURRENT_TIMESTAMP) WHERE updated_at IS NULL"
                )
        except sqlite3.OperationalError:
            pass  # Columns may already exist
        _ensure_column(
            conn,
            "clusters",
            "summary_json",
            "TEXT DEFAULT '{}'",
            backfill_sql="""
            UPDATE clusters
            SET summary_json = COALESCE(summary, '{}')
            WHERE summary_json IS NULL OR summary_json = ''
            """,
        )
        _ensure_column(conn, "evidence_ledger", "entry_json", "TEXT DEFAULT '{}'")
        _dedupe_evidence_ledger(conn)
        _ensure_evidence_ledger_unique_index(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_products_build_brief
            ON products(build_brief_id, built_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_products_idea
            ON products(idea_id, built_at DESC)
            """
        )
        conn.commit()

    def get_latest_run_id(self) -> str:
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT run_id
            FROM (
                SELECT run_id, validated_at AS ts FROM validations WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM corroborations WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM market_enrichments WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM evidence_ledger WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM review_feedback WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM build_briefs WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM build_prep_outputs WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM experiments WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM validation_experiments WHERE NULLIF(run_id, '') IS NOT NULL
            )
            ORDER BY ts DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        return row["run_id"] if row else ""

    def get_recent_run_ids(self, limit: int = 2) -> list[str]:
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT run_id
            FROM (
                SELECT run_id, validated_at AS ts FROM validations WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM corroborations WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM market_enrichments WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM evidence_ledger WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM review_feedback WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM build_briefs WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, updated_at AS ts FROM build_prep_outputs WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM experiments WHERE NULLIF(run_id, '') IS NOT NULL
                UNION ALL
                SELECT run_id, created_at AS ts FROM validation_experiments WHERE NULLIF(run_id, '') IS NOT NULL
            )
            GROUP BY run_id
            ORDER BY MAX(ts) DESC, run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["run_id"] for row in rows]

    def set_active_run_id(self, run_id: str) -> None:
        self.active_run_id = run_id or ""

    def _row_to_finding(self, row: sqlite3.Row) -> Finding:
        return Finding(
            id=row["id"],
            source=row["source"],
            source_url=row["source_url"],
            entrepreneur=row["entrepreneur"] or "",
            tool_used=row["tool_used"] or "",
            product_built=row["product_built"] or "",
            monetization_method=row["monetization_method"] or "",
            outcome_summary=row["outcome_summary"] or "",
            content_hash=row["content_hash"] or "",
            status=row["status"] or "new",
            finding_kind=row["finding_kind"] or "",
            source_class=row["source_class"] or "",
            recurrence_key=row["recurrence_key"] or "",
            evidence=_json_loads(row["evidence_json"], {}),
            discovered_at=row["discovered_at"],
        )

    def _row_to_raw_signal(self, row: sqlite3.Row) -> RawSignal:
        return RawSignal(
            id=row["id"],
            finding_id=row["finding_id"] or 0,
            source_name=row["source_name"] or "",
            source_type=row["source_type"] or "",
            source_class=row["source_class"] or "pain_signal",
            source_url=row["source_url"] or "",
            title=row["title"] or "",
            body_excerpt=row["body_excerpt"] or "",
            quote_text=row["quote_text"] or "",
            role_hint=row["role_hint"] or "",
            published_at=row["published_at"] or "",
            timestamp_hint=row["timestamp_hint"] or "",
            content_hash=row["content_hash"] or "",
            metadata=_json_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
        )

    def _row_to_problem_atom(self, row: sqlite3.Row) -> ProblemAtom:
        score_json = _json_loads(row["score_json"], {})
        metadata = _json_loads(row["metadata_json"], {})
        atom_json = row["atom_json"] if row["atom_json"] not in (None, "") else "{}"
        return ProblemAtom(
            id=row["id"],
            finding_id=row["finding_id"] or 0,
            raw_signal_id=row["raw_signal_id"] or 0,
            signal_id=row["raw_signal_id"] or 0,
            cluster_key=metadata.get("cluster_key", ""),
            segment=metadata.get("segment", ""),
            user_role=row["user_role"] or "",
            job_to_be_done=row["job_to_be_done"] or "",
            pain_statement=row["pain_statement"] or "",
            trigger_event=row["trigger_event"] or "",
            failure_mode=row["failure_mode"] or "",
            current_workaround=row["current_workaround"] or "",
            current_tools=row["current_tools"] or "",
            source_quote=metadata.get("source_quote", ""),
            urgency_clues=metadata.get("urgency_clues", ""),
            frequency_clues=metadata.get("frequency_clues", ""),
            emotional_intensity=float(metadata.get("emotional_intensity", 0.0) or 0.0),
            cost_consequence_clues=metadata.get("cost_consequence_clues", ""),
            why_now_clues=metadata.get("why_now_clues", ""),
            confidence=float(score_json.get("confidence", row["confidence_score"] or 0.0) or 0.0),
            confidence_score=float(row["confidence_score"] or 0.0),
            atom_json=atom_json,
            score_json=score_json,
            metadata=metadata,
            created_at=row["created_at"],
        )

    def _row_to_cluster(self, row: sqlite3.Row) -> OpportunityCluster:
        summary = _json_loads(row["summary_json"], {})
        metadata = _json_loads(row["metadata_json"], {})
        return OpportunityCluster(
            id=row["id"],
            label=row["label"] or "",
            cluster_key=metadata.get("cluster_key", ""),
            status=row["status"] or "active",
            segment=metadata.get("segment", ""),
            summary=summary,
            user_role=row["user_role"] or "",
            job_to_be_done=row["job_to_be_done"] or "",
            trigger_summary=metadata.get("trigger_summary", ""),
            signal_count=int(metadata.get("signal_count", 0) or 0),
            atom_count=int(metadata.get("atom_count", 0) or 0),
            evidence_quality=float(metadata.get("evidence_quality", 0.0) or 0.0),
            metadata=metadata,
            updated_at=row["updated_at"],
        )

    def get_active_run_id(self) -> str:
        return self.active_run_id or self.get_latest_run_id()

    def insert_finding(self, finding: Finding) -> int:
        finding.__post_init__()
        conn = self._get_connection()
        cur = conn.execute(
            """
            INSERT INTO findings (
                source, source_url, entrepreneur, tool_used, product_built,
                monetization_method, outcome_summary, content_hash, status,
                finding_kind, source_class, recurrence_key, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.source,
                finding.source_url,
                finding.entrepreneur,
                finding.tool_used,
                finding.product_built,
                finding.monetization_method,
                finding.outcome_summary,
                finding.content_hash,
                finding.status,
                finding.finding_kind,
                finding.source_class,
                finding.recurrence_key,
                finding.evidence_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_finding_by_hash(self, content_hash: str) -> Optional[Finding]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM findings WHERE content_hash = ?", (content_hash,)).fetchone()
        return self._row_to_finding(row) if row else None

    def get_finding(self, finding_id: int) -> Optional[Finding]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        return self._row_to_finding(row) if row else None

    def _finding_matches_run(self, finding: Finding, run_id: Optional[str]) -> bool:
        if not run_id:
            return True
        evidence = finding.evidence or {}
        return evidence.get("run_id") == run_id

    def update_finding_status(self, finding_id: int, status: str) -> None:
        conn = self._get_connection()
        conn.execute("UPDATE findings SET status = ? WHERE id = ?", (status, finding_id))
        conn.commit()

    def insert_raw_signal(self, signal: RawSignal) -> int:
        signal.__post_init__()
        conn = self._get_connection()
        cur = conn.execute(
            """
            INSERT INTO raw_signals (
                finding_id, source_name, source_type, source_class, source_url,
                title, body_excerpt, quote_text, role_hint, published_at,
                timestamp_hint, content_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.finding_id,
                signal.source_name,
                signal.source_type,
                signal.source_class,
                signal.source_url,
                signal.title,
                signal.body_excerpt,
                signal.quote_text,
                signal.role_hint,
                signal.published_at,
                signal.timestamp_hint,
                signal.content_hash,
                signal.metadata_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_raw_signal(self, signal_id: int) -> Optional[RawSignal]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM raw_signals WHERE id = ?", (signal_id,)).fetchone()
        return self._row_to_raw_signal(row) if row else None

    def get_raw_signals_by_finding(self, finding_id: int) -> list[RawSignal]:
        return [self._row_to_raw_signal(row) for row in self._get_connection().execute(
            "SELECT * FROM raw_signals WHERE finding_id = ? ORDER BY id ASC", (finding_id,)
        ).fetchall()]

    def insert_problem_atom(self, atom: ProblemAtom) -> int:
        atom.__post_init__()
        metadata = dict(atom.metadata or {})
        metadata.update(
            {
                "cluster_key": atom.cluster_key,
                "segment": atom.segment,
                "source_quote": atom.source_quote,
                "urgency_clues": atom.urgency_clues,
                "frequency_clues": atom.frequency_clues,
                "emotional_intensity": atom.emotional_intensity,
                "cost_consequence_clues": atom.cost_consequence_clues,
                "why_now_clues": atom.why_now_clues,
            }
        )
        conn = self._get_connection()
        table_columns = _table_columns(conn, "problem_atoms")
        atom_payload = {
            "finding_id": atom.finding_id,
            "raw_signal_id": atom.raw_signal_id or atom.signal_id,
            "signal_id": atom.raw_signal_id or atom.signal_id,
            "cluster_key": atom.cluster_key,
            "segment": atom.segment,
            "user_role": atom.user_role,
            "job_to_be_done": atom.job_to_be_done,
            "pain_statement": atom.pain_statement,
            "trigger_event": atom.trigger_event,
            "failure_mode": atom.failure_mode,
            "current_workaround": atom.current_workaround,
            "current_tools": atom.current_tools,
            "source_quote": atom.source_quote,
            "urgency_clues": atom.urgency_clues,
            "frequency_clues": atom.frequency_clues,
            "emotional_intensity": atom.emotional_intensity,
            "cost_consequence_clues": atom.cost_consequence_clues,
            "why_now_clues": atom.why_now_clues,
            "confidence_score": atom.confidence or atom.confidence_score,
            "confidence": atom.confidence or atom.confidence_score,
            "score_json": _json_dumps(atom.score_json or {"confidence": atom.confidence or atom.confidence_score}),
            "atom_json": atom.atom_json
            or _json_dumps(
                {
                    "cluster_key": atom.cluster_key,
                    "segment": atom.segment,
                    "user_role": atom.user_role,
                    "job_to_be_done": atom.job_to_be_done,
                    "pain_statement": atom.pain_statement,
                    "trigger_event": atom.trigger_event,
                    "failure_mode": atom.failure_mode,
                    "current_workaround": atom.current_workaround,
                    "current_tools": atom.current_tools,
                    "source_quote": atom.source_quote,
                    "urgency_clues": atom.urgency_clues,
                    "frequency_clues": atom.frequency_clues,
                    "emotional_intensity": atom.emotional_intensity,
                    "cost_consequence_clues": atom.cost_consequence_clues,
                    "why_now_clues": atom.why_now_clues,
                    "confidence": atom.confidence or atom.confidence_score,
                }
            ),
            "metadata_json": _json_dumps(metadata),
        }
        insert_columns = [column for column in atom_payload if column in table_columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        values = tuple(atom_payload[column] for column in insert_columns)
        cur = conn.execute(
            f"INSERT INTO problem_atoms ({', '.join(insert_columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_problem_atoms_by_finding(self, finding_id: int) -> list[ProblemAtom]:
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM problem_atoms WHERE finding_id = ? ORDER BY id ASC", (finding_id,)).fetchall()
        return [self._row_to_problem_atom(row) for row in rows]

    def get_problem_atoms_by_cluster_key(self, cluster_key: str) -> list[ProblemAtom]:
        conn = self._get_connection()
        table_columns = _table_columns(conn, "problem_atoms")
        if "cluster_key" in table_columns:
            rows = conn.execute(
                "SELECT * FROM problem_atoms WHERE cluster_key = ? ORDER BY id ASC",
                (cluster_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM problem_atoms WHERE json_extract(metadata_json, '$.cluster_key') = ? ORDER BY id ASC",
                (cluster_key,),
            ).fetchall()
        return [self._row_to_problem_atom(row) for row in rows]

    def get_cluster(self, cluster_id: int) -> Optional[OpportunityCluster]:
        conn = self._get_connection()
        for table in ("clusters", "opportunity_clusters"):
            try:
                row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (cluster_id,)).fetchone()
            except sqlite3.OperationalError:
                row = None
            if row:
                return self._row_to_cluster(row)
        return None

    def get_cluster_record(self, cluster_id: int) -> Optional[OpportunityCluster]:
        return self.get_cluster(cluster_id)

    def upsert_cluster(self, cluster: OpportunityCluster) -> int:
        cluster.__post_init__()
        conn = self._get_connection()
        table_columns = _table_columns(conn, "opportunity_clusters")
        cluster_key = cluster.cluster_key or (cluster.metadata or {}).get("cluster_key", "")
        existing = None
        if cluster_key:
            if "cluster_key" in table_columns:
                row = conn.execute(
                    "SELECT * FROM opportunity_clusters WHERE cluster_key = ? ORDER BY id DESC LIMIT 1",
                    (cluster_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM opportunity_clusters WHERE json_extract(metadata_json, '$.cluster_key') = ? ORDER BY id DESC LIMIT 1",
                    (cluster_key,),
                ).fetchone()
            if row:
                existing = self._row_to_cluster(row)
        if existing is None:
            row = conn.execute(
                "SELECT * FROM opportunity_clusters WHERE label = ? AND job_to_be_done = ? ORDER BY id DESC LIMIT 1",
                (cluster.label, cluster.job_to_be_done),
            ).fetchone()
            if row:
                existing = self._row_to_cluster(row)
        payload = {
            "cluster_key": cluster_key or cluster.label,
            "label": cluster.label,
            "segment": cluster.segment,
            "status": cluster.status,
            "summary_json": cluster.summary_json,
            "user_role": cluster.user_role,
            "job_to_be_done": cluster.job_to_be_done,
            "trigger_summary": cluster.trigger_summary,
            "signal_count": cluster.signal_count,
            "atom_count": cluster.atom_count,
            "evidence_quality": cluster.evidence_quality,
            "metadata_json": cluster.metadata_json,
        }
        if existing:
            update_columns = [column for column in payload if column in table_columns]
            assignments = [f"{column} = ?" for column in update_columns]
            assignments.append("updated_at = CURRENT_TIMESTAMP")
            params = [payload[column] for column in update_columns]
            params.append(existing.id)
            conn.execute(
                f"UPDATE opportunity_clusters SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            cluster_id = int(existing.id)
        else:
            insert_columns = [column for column in payload if column in table_columns]
            insert_values = [payload[column] for column in insert_columns]
            cur = conn.execute(
                f"""
                INSERT INTO opportunity_clusters ({', '.join(insert_columns)}, updated_at)
                VALUES ({', '.join(['?'] * len(insert_columns))}, CURRENT_TIMESTAMP)
                """,
                insert_values,
            )
            cluster_id = int(cur.lastrowid)
        mirror_columns = _table_columns(conn, "clusters")
        mirror_payload = {
            "id": cluster_id,
            "cluster_key": payload["cluster_key"],
            "label": cluster.label,
            "status": cluster.status,
            "summary": cluster.summary_json,
            "summary_json": cluster.summary_json,
            "segment": cluster.segment,
            "user_role": cluster.user_role,
            "job_to_be_done": cluster.job_to_be_done,
            "trigger_pattern": cluster.trigger_summary,
            "trigger_summary": cluster.trigger_summary,
            "workaround_pattern": "",
            "failure_pattern": "",
            "source_count": cluster.signal_count,
            "signal_count": cluster.signal_count,
            "atom_count": cluster.atom_count,
            "evidence_quality": cluster.evidence_quality,
            "metadata_json": cluster.metadata_json,
        }
        mirror_insert_columns = [column for column in mirror_payload if column in mirror_columns]
        mirror_insert_values = [mirror_payload[column] for column in mirror_insert_columns]
        update_assignments = []
        for column in mirror_insert_columns:
            if column == "id":
                continue
            update_assignments.append(f"{column} = excluded.{column}")
        update_assignments.append("updated_at = CURRENT_TIMESTAMP")
        conn.execute(
            f"""
            INSERT INTO clusters ({', '.join(mirror_insert_columns)}, updated_at)
            VALUES ({', '.join(['?'] * len(mirror_insert_columns))}, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                {', '.join(update_assignments)}
            """,
            mirror_insert_values,
        )
        conn.commit()
        self._backfill_cluster_members()
        return cluster_id

    def _backfill_cluster_members(self) -> None:
        conn = self._get_connection()
        member_columns = _table_columns(conn, "cluster_members")
        atom_column = "problem_atom_id" if "problem_atom_id" in member_columns else "atom_id"
        atom_columns = _table_columns(conn, "problem_atoms")
        cluster_columns = _table_columns(conn, "opportunity_clusters")
        atom_cluster_expr = "pa.cluster_key" if "cluster_key" in atom_columns else "json_extract(pa.metadata_json, '$.cluster_key')"
        cluster_key_expr = "oc.cluster_key" if "cluster_key" in cluster_columns else "json_extract(oc.metadata_json, '$.cluster_key')"
        rows = conn.execute(
            f"""
            SELECT oc.id AS cluster_id, pa.id AS atom_id
            FROM problem_atoms pa
            JOIN opportunity_clusters oc ON {cluster_key_expr} = {atom_cluster_expr}
            LEFT JOIN cluster_members cm
              ON cm.cluster_id = oc.id AND cm.{atom_column} = pa.id
            WHERE {atom_cluster_expr} IS NOT NULL
              AND {atom_cluster_expr} != ''
              AND cm.cluster_id IS NULL
            ORDER BY pa.id ASC
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                f"INSERT INTO cluster_members (cluster_id, {atom_column}) VALUES (?, ?)",
                (row["cluster_id"], row["atom_id"]),
            )
        conn.commit()

    def get_cluster_members(self, cluster_id: int) -> list[int]:
        conn = self._get_connection()
        member_columns = _table_columns(conn, "cluster_members")
        atom_column = "problem_atom_id" if "problem_atom_id" in member_columns else "atom_id"
        rows = conn.execute(
            f"SELECT {atom_column} FROM cluster_members WHERE cluster_id = ? ORDER BY {atom_column} ASC",
            (cluster_id,),
        ).fetchall()
        return [int(row[atom_column]) for row in rows]

    def upsert_opportunity(self, opportunity: Opportunity) -> int:
        opportunity.__post_init__()
        conn = self._get_connection()
        existing = conn.execute("SELECT id FROM opportunities WHERE cluster_id = ?", (opportunity.cluster_id,)).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE opportunities
                SET title = ?, market_gap = ?, recommendation = ?, status = ?, pain_severity = ?, frequency_score = ?,
                    cost_of_inaction = ?, workaround_density = ?, urgency_score = ?, segment_concentration = ?,
                    reachability = ?, timing_shift = ?, buildability = ?, expansion_potential = ?, education_burden = ?,
                    dependency_risk = ?, adoption_friction = ?, evidence_quality = ?, composite_score = ?, confidence = ?,
                    scoring_version = ?, problem_truth_score = ?, revenue_readiness_score = ?, decision_score = ?,
                    problem_plausibility = ?, value_support = ?, corroboration_strength = ?, evidence_sufficiency = ?,
                    willingness_to_pay_proxy = ?, formula_version = ?, threshold_version = ?,
                    selection_status = ?, selection_reason = ?, notes_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    opportunity.title,
                    opportunity.market_gap,
                    opportunity.recommendation,
                    opportunity.status,
                    opportunity.pain_severity,
                    opportunity.frequency_score,
                    opportunity.cost_of_inaction,
                    opportunity.workaround_density,
                    opportunity.urgency_score,
                    opportunity.segment_concentration,
                    opportunity.reachability,
                    opportunity.timing_shift,
                    opportunity.buildability,
                    opportunity.expansion_potential,
                    opportunity.education_burden,
                    opportunity.dependency_risk,
                    opportunity.adoption_friction,
                    opportunity.evidence_quality,
                    opportunity.composite_score,
                    opportunity.confidence,
                    opportunity.scoring_version,
                    opportunity.problem_truth_score,
                    opportunity.revenue_readiness_score,
                    opportunity.decision_score,
                    opportunity.problem_plausibility,
                    opportunity.value_support,
                    opportunity.corroboration_strength,
                    opportunity.evidence_sufficiency,
                    opportunity.willingness_to_pay_proxy,
                    opportunity.formula_version,
                    opportunity.threshold_version,
                    opportunity.selection_status,
                    opportunity.selection_reason,
                    opportunity.notes_json,
                    existing["id"],
                ),
            )
            opportunity_id = int(existing["id"])
        else:
            cur = conn.execute(
                """
                INSERT INTO opportunities (
                    cluster_id, title, market_gap, recommendation, status, pain_severity, frequency_score,
                    cost_of_inaction, workaround_density, urgency_score, segment_concentration, reachability,
                    timing_shift, buildability, expansion_potential, education_burden, dependency_risk,
                    adoption_friction, evidence_quality, composite_score, confidence, scoring_version,
                    problem_truth_score, revenue_readiness_score, decision_score,
                    problem_plausibility, value_support, corroboration_strength, evidence_sufficiency,
                    willingness_to_pay_proxy, formula_version, threshold_version,
                    selection_status, selection_reason, notes_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    opportunity.cluster_id,
                    opportunity.title,
                    opportunity.market_gap,
                    opportunity.recommendation,
                    opportunity.status,
                    opportunity.pain_severity,
                    opportunity.frequency_score,
                    opportunity.cost_of_inaction,
                    opportunity.workaround_density,
                    opportunity.urgency_score,
                    opportunity.segment_concentration,
                    opportunity.reachability,
                    opportunity.timing_shift,
                    opportunity.buildability,
                    opportunity.expansion_potential,
                    opportunity.education_burden,
                    opportunity.dependency_risk,
                    opportunity.adoption_friction,
                    opportunity.evidence_quality,
                    opportunity.composite_score,
                    opportunity.confidence,
                    opportunity.scoring_version,
                    opportunity.problem_truth_score,
                    opportunity.revenue_readiness_score,
                    opportunity.decision_score,
                    opportunity.problem_plausibility,
                    opportunity.value_support,
                    opportunity.corroboration_strength,
                    opportunity.evidence_sufficiency,
                    opportunity.willingness_to_pay_proxy,
                    opportunity.formula_version,
                    opportunity.threshold_version,
                    opportunity.selection_status,
                    opportunity.selection_reason,
                    opportunity.notes_json,
                ),
            )
            opportunity_id = int(cur.lastrowid)
        conn.commit()
        return opportunity_id

    def insert_experiment(self, experiment: ValidationExperiment) -> int:
        experiment.__post_init__()
        conn = self._get_connection()
        run_id = experiment.run_id or self.get_active_run_id()
        cur = conn.execute(
            """
            INSERT INTO validation_experiments (
                run_id, opportunity_id, cluster_id, test_type, hypothesis, falsifier,
                smallest_test, success_signal, failure_signal, status, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                experiment.opportunity_id,
                experiment.cluster_id,
                experiment.test_type,
                experiment.hypothesis,
                experiment.falsifier,
                experiment.smallest_test,
                experiment.success_signal,
                experiment.failure_signal,
                experiment.status,
                experiment.result_json,
            ),
        )
        experiment_id = int(cur.lastrowid)
        mirror_columns = _table_columns(conn, "experiments")
        plan_json = _json_dumps(
            {
                "cluster_id": experiment.cluster_id,
                "test_type": experiment.test_type,
                "hypothesis": experiment.hypothesis,
                "falsifier": experiment.falsifier,
                "smallest_test": experiment.smallest_test,
                "success_signal": experiment.success_signal,
                "failure_signal": experiment.failure_signal,
                "status": experiment.status,
                "result": experiment.result or {},
            }
        )
        mirror_payload = {
            "id": experiment_id,
            "run_id": run_id,
            "opportunity_id": experiment.opportunity_id,
            "cluster_id": experiment.cluster_id,
            "test_type": experiment.test_type,
            "hypothesis": experiment.hypothesis,
            "falsifier": experiment.falsifier,
            "smallest_test": experiment.smallest_test,
            "success_signal": experiment.success_signal,
            "failure_signal": experiment.failure_signal,
            "falsification_condition": experiment.falsifier,
            "plan_hash": "",
            "status": experiment.status,
            "result_json": experiment.result_json,
            "plan_json": plan_json,
            "priority": 3,
        }
        mirror_insert_columns = [column for column in mirror_payload if column in mirror_columns]
        mirror_insert_values = [mirror_payload[column] for column in mirror_insert_columns]
        conn.execute(
            f"""
            INSERT INTO experiments ({', '.join(mirror_insert_columns)})
            VALUES ({', '.join(['?'] * len(mirror_insert_columns))})
            """,
            mirror_insert_values,
        )
        conn.commit()
        return experiment_id

    def insert_validation(self, validation: Validation) -> int:
        validation.__post_init__()
        conn = self._get_connection()
        cur = conn.execute(
            """
            INSERT INTO validations (run_id, finding_id, market_score, technical_score, distribution_score, overall_score, passed, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, finding_id) DO UPDATE SET
                market_score = excluded.market_score,
                technical_score = excluded.technical_score,
                distribution_score = excluded.distribution_score,
                overall_score = excluded.overall_score,
                passed = excluded.passed,
                evidence = excluded.evidence,
                validated_at = CURRENT_TIMESTAMP
            """,
            (
                validation.run_id or self.get_active_run_id(),
                validation.finding_id,
                validation.market_score,
                validation.technical_score,
                validation.distribution_score,
                validation.overall_score,
                1 if validation.passed else 0,
                validation.evidence_json,
            ),
        )
        row = conn.execute(
            "SELECT id FROM validations WHERE run_id = ? AND finding_id = ?",
            (validation.run_id or self.get_active_run_id(), validation.finding_id),
        ).fetchone()
        conn.commit()
        return int(row["id"] if row else cur.lastrowid)

    def upsert_validation(self, validation: Validation) -> int:
        return self.insert_validation(validation)

    def get_recent_validations(self, limit: int = 25) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT v.*, f.product_built, f.outcome_summary
            FROM validations v
            JOIN findings f ON f.id = v.finding_id
            ORDER BY v.validated_at DESC, v.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            evidence = _json_loads(row["evidence"], {})
            results.append(build_recent_validation_row(dict(row), evidence))
        return results

    def list_validation_evidence_payloads(
        self, *, run_id: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return parsed validation evidence JSON for gate diagnostics (``gate-diagnostics`` CLI)."""
        conn = self._get_connection()
        rid = run_id or self.get_latest_run_id()
        if not rid:
            return []
        rows = conn.execute(
            """
            SELECT id AS validation_id, finding_id, evidence
            FROM validations
            WHERE run_id = ?
            ORDER BY validated_at DESC, id DESC
            LIMIT ?
            """,
            (rid, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            evidence = _json_loads(row["evidence"], {})
            results.append(
                {
                    "validation_id": int(row["validation_id"]),
                    "finding_id": int(row["finding_id"]),
                    "evidence": evidence,
                }
            )
        return results

    def insert_ledger_entry(self, entry: EvidenceLedgerEntry) -> int:
        entry.__post_init__()
        conn = self._get_connection()
        run_id = entry.run_id or self.get_active_run_id()
        table_columns = _table_columns(conn, "evidence_ledger")
        entry_payload = dict(entry.entry_json or {})
        stance = entry.stance or str(entry_payload.get("stance") or "neutral")
        source_name = entry.source_name or str(entry_payload.get("source_name") or "")
        source_url = entry.source_url or str(entry_payload.get("source_url") or "")
        quote_text = entry.quote_text or str(entry_payload.get("quote_text") or "")
        summary = entry.summary or str(entry_payload.get("summary") or "")
        payload = {
            "run_id": run_id,
            "entity_type": entry.entity_type,
            "entity_id": entry.entity_id,
            "entry_kind": entry.entry_kind,
            "entry_json": _json_dumps(entry_payload),
            "stance": stance,
            "source_name": source_name,
            "source_url": source_url,
            "quote_text": quote_text,
            "summary": summary,
            "metadata_json": _json_dumps(entry_payload),
        }
        insert_columns = [column for column in payload if column in table_columns]
        insert_values = [payload[column] for column in insert_columns]
        conflict_keys = {"run_id", "entity_type", "entity_id", "entry_kind"}
        update_columns = [c for c in insert_columns if c not in conflict_keys]
        if not update_columns:
            placeholders = ", ".join(["?"] * len(insert_columns))
            cur = conn.execute(
                f"INSERT OR IGNORE INTO evidence_ledger ({', '.join(insert_columns)}) VALUES ({placeholders})",
                insert_values,
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM evidence_ledger WHERE run_id = ? AND entity_type = ? AND entity_id = ? AND entry_kind = ?",
                (run_id, entry.entity_type, entry.entity_id, entry.entry_kind),
            ).fetchone()
            return int(row["id"]) if row else int(cur.lastrowid)

        placeholders = ", ".join(["?"] * len(insert_columns))
        update_sql = ", ".join(f"{col} = excluded.{col}" for col in update_columns)
        conn.execute(
            f"""
            INSERT INTO evidence_ledger ({', '.join(insert_columns)})
            VALUES ({placeholders})
            ON CONFLICT(run_id, entity_type, entity_id, entry_kind) DO UPDATE SET {update_sql}
            """,
            insert_values,
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM evidence_ledger WHERE run_id = ? AND entity_type = ? AND entity_id = ? AND entry_kind = ?",
            (run_id, entry.entity_type, entry.entity_id, entry.entry_kind),
        ).fetchone()
        return int(row["id"]) if row else 0

    def get_evidence_ledger(self, entity_type: Optional[str] = None, entity_id: Optional[int] = None, *, limit: Optional[int] = None) -> list[EvidenceLedgerEntry]:
        return self.list_ledger_entries(entity_type=entity_type, entity_id=entity_id, limit=limit)

    def upsert_corroboration(self, record: CorroborationRecord) -> int:
        record.__post_init__()
        conn = self._get_connection()
        run_id = record.run_id or self.get_active_run_id()
        conn.execute(
            """
            INSERT INTO corroborations (
                run_id, finding_id, recurrence_state, recurrence_score, corroboration_score,
                evidence_sufficiency, query_coverage, independent_confirmations, source_diversity,
                query_set_hash, evidence_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, finding_id) DO UPDATE SET
                recurrence_state = excluded.recurrence_state,
                recurrence_score = excluded.recurrence_score,
                corroboration_score = excluded.corroboration_score,
                evidence_sufficiency = excluded.evidence_sufficiency,
                query_coverage = excluded.query_coverage,
                independent_confirmations = excluded.independent_confirmations,
                source_diversity = excluded.source_diversity,
                query_set_hash = excluded.query_set_hash,
                evidence_json = excluded.evidence_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                run_id,
                record.finding_id,
                record.recurrence_state,
                record.recurrence_score,
                record.corroboration_score,
                record.evidence_sufficiency,
                record.query_coverage,
                record.independent_confirmations,
                record.source_diversity,
                record.query_set_hash,
                record.evidence_json,
            ),
        )
        row = conn.execute(
            "SELECT id FROM corroborations WHERE run_id = ? AND finding_id = ?",
            (run_id, record.finding_id),
        ).fetchone()
        conn.commit()
        return int(row["id"])

    def get_corroboration(self, finding_id: int, run_id: Optional[str] = None) -> Optional[CorroborationRecord]:
        conn = self._get_connection()
        rid = run_id or self.get_active_run_id()
        row = conn.execute(
            "SELECT * FROM corroborations WHERE run_id = ? AND finding_id = ? ORDER BY id DESC LIMIT 1",
            (rid, finding_id),
        ).fetchone()
        return CorroborationRecord(**dict(row)) if row else None

    def upsert_market_enrichment(self, record: MarketEnrichment) -> int:
        record.__post_init__()
        conn = self._get_connection()
        run_id = record.run_id or self.get_active_run_id()
        conn.execute(
            """
            INSERT INTO market_enrichments (
                run_id, finding_id, demand_score, buyer_intent_score, competition_score,
                trend_score, review_signal_score, value_signal_score, evidence_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, finding_id) DO UPDATE SET
                demand_score = excluded.demand_score,
                buyer_intent_score = excluded.buyer_intent_score,
                competition_score = excluded.competition_score,
                trend_score = excluded.trend_score,
                review_signal_score = excluded.review_signal_score,
                value_signal_score = excluded.value_signal_score,
                evidence_json = excluded.evidence_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                run_id,
                record.finding_id,
                record.demand_score,
                record.buyer_intent_score,
                record.competition_score,
                record.trend_score,
                record.review_signal_score,
                record.value_signal_score,
                record.evidence_json,
            ),
        )
        row = conn.execute(
            "SELECT id FROM market_enrichments WHERE run_id = ? AND finding_id = ?",
            (run_id, record.finding_id),
        ).fetchone()
        conn.commit()
        return int(row["id"])

    def get_market_enrichment(self, finding_id: int, run_id: Optional[str] = None) -> Optional[MarketEnrichment]:
        conn = self._get_connection()
        rid = run_id or self.get_active_run_id()
        row = conn.execute(
            "SELECT * FROM market_enrichments WHERE run_id = ? AND finding_id = ? ORDER BY id DESC LIMIT 1",
            (rid, finding_id),
        ).fetchone()
        return MarketEnrichment(**dict(row)) if row else None

    def record_discovery_probe(
        self,
        source_name: str,
        query_text: str,
        *,
        docs_seen: int = 0,
        latency_ms: float = 0.0,
        status: str = "ok",
        error: str = "",
    ) -> None:
        conn = self._get_connection()
        last_status = error or status
        conn.execute(
            """
            INSERT INTO discovery_feedback (source_name, query_text, runs, docs_seen, last_latency_ms, last_status, updated_at)
            VALUES (?, ?, 1, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_name, query_text) DO UPDATE SET
                runs = discovery_feedback.runs + 1,
                docs_seen = discovery_feedback.docs_seen + excluded.docs_seen,
                last_latency_ms = excluded.last_latency_ms,
                last_status = excluded.last_status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (source_name, query_text, docs_seen, latency_ms, last_status),
        )
        conn.commit()

    def record_discovery_hit(self, source_name: str, query_text: str) -> None:
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO discovery_feedback (source_name, query_text, findings_emitted, updated_at)
            VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(source_name, query_text) DO UPDATE SET
                findings_emitted = discovery_feedback.findings_emitted + 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (source_name, query_text),
        )
        conn.commit()

    def record_validation_feedback(
        self,
        source_name: str,
        query_text: str,
        *,
        passed: bool,
        overall_score: float,
        selection_status: str = "",
        build_brief_created: bool = False,
    ) -> None:
        conn = self._get_connection()
        prototype_candidate = 1 if selection_status == "prototype_candidate" else 0
        build_brief_count = 1 if build_brief_created else 0
        conn.execute(
            """
            INSERT INTO discovery_feedback (
                source_name, query_text, validations, passes, prototype_candidates, build_briefs,
                avg_validation_score, updated_at
            )
            VALUES (?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_name, query_text) DO UPDATE SET
                validations = discovery_feedback.validations + 1,
                passes = discovery_feedback.passes + ?,
                prototype_candidates = discovery_feedback.prototype_candidates + ?,
                build_briefs = discovery_feedback.build_briefs + ?,
                avg_validation_score = (
                    (discovery_feedback.avg_validation_score * discovery_feedback.validations) + ?
                ) / NULLIF(discovery_feedback.validations + 1, 0),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                source_name,
                query_text,
                1 if passed else 0,
                prototype_candidate,
                build_brief_count,
                overall_score,
                1 if passed else 0,
                prototype_candidate,
                build_brief_count,
                overall_score,
            ),
        )
        conn.commit()

    def get_discovery_feedback(self, source_name: Optional[str] = None) -> list[dict[str, Any]]:
        conn = self._get_connection()
        sql = "SELECT * FROM discovery_feedback"
        params: list[Any] = []
        if source_name:
            sql += " WHERE source_name = ?"
            params.append(source_name)
        sql += " ORDER BY updated_at DESC, source_name ASC, query_text ASC"
        return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def upsert_discovery_theme(
        self,
        theme_key: str,
        *,
        label: str,
        query_seeds: list[str],
        source_signals: list[str],
        times_seen: int,
        yield_score: float = 0.0,
        run_id: Optional[str] = None,
    ) -> int:
        conn = self._get_connection()
        rid = run_id or self.get_active_run_id()
        conn.execute(
            """
            INSERT INTO discovery_themes (
                theme_key, label, query_seeds_json, source_signals_json,
                times_seen, yield_score, run_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(theme_key) DO UPDATE SET
                label = excluded.label,
                query_seeds_json = excluded.query_seeds_json,
                source_signals_json = excluded.source_signals_json,
                times_seen = excluded.times_seen,
                yield_score = excluded.yield_score,
                run_id = excluded.run_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                theme_key,
                label,
                _json_dumps(query_seeds),
                _json_dumps(source_signals),
                int(times_seen or 0),
                float(yield_score or 0.0),
                rid,
            ),
        )
        row = conn.execute("SELECT id FROM discovery_themes WHERE theme_key = ?", (theme_key,)).fetchone()
        conn.commit()
        return int(row["id"]) if row else 0

    def list_active_discovery_themes(self, limit: int = 25) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM discovery_themes
            ORDER BY times_seen DESC, yield_score DESC, updated_at DESC, theme_key ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["query_seeds"] = _json_loads(item.get("query_seeds_json"), [])
            item["source_signals"] = _json_loads(item.get("source_signals_json"), [])
            results.append(item)
        return results

    def get_findings(self, status: Optional[str] = None, finding_kind: Optional[str] = None, limit: Optional[int] = None) -> list[Finding]:
        conn = self._get_connection()
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if finding_kind:
            clauses.append("finding_kind = ?")
            params.append(finding_kind)
        sql = "SELECT * FROM findings"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_finding(row) for row in conn.execute(sql, params).fetchall()]

    def get_raw_signals(self, *, finding_id: Optional[int] = None, limit: Optional[int] = None) -> list[Any]:
        conn = self._get_connection()
        sql = "SELECT * FROM raw_signals"
        params: list[Any] = []
        if finding_id is not None:
            sql += " WHERE finding_id = ?"
            params.append(finding_id)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_raw_signal(row) for row in rows]

    def get_problem_atoms(self, *, finding_id: Optional[int] = None, limit: Optional[int] = None) -> list[Any]:
        conn = self._get_connection()
        sql = "SELECT * FROM problem_atoms"
        params: list[Any] = []
        if finding_id is not None:
            sql += " WHERE finding_id = ?"
            params.append(finding_id)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_problem_atom(row) for row in rows]

    def get_clusters(self, limit: int = 25) -> list[Any]:
        conn = self._get_connection()
        table = "clusters"
        try:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute("SELECT * FROM opportunity_clusters ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_cluster(row) for row in rows]

    def get_opportunity(self, opportunity_id: int) -> Optional[Opportunity]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
        if not row:
            return None
        return Opportunity(**dict(row))

    def get_opportunities(self, limit: int = 25, status: Optional[str] = None) -> list[Opportunity]:
        conn = self._get_connection()
        sql = "SELECT * FROM opportunities"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY composite_score DESC, evidence_quality DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        return [Opportunity(**dict(row)) for row in conn.execute(sql, params).fetchall()]

    def get_experiments(self, *, opportunity_id: Optional[int] = None, limit: Optional[int] = None) -> list[Any]:
        conn = self._get_connection()
        sql = "SELECT * FROM validation_experiments"
        params: list[Any] = []
        if opportunity_id is not None:
            sql += " WHERE opportunity_id = ?"
            params.append(opportunity_id)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [ValidationExperiment(**dict(row)) for row in conn.execute(sql, params).fetchall()]

    def list_ledger_entries(self, entity_type: Optional[str] = None, entity_id: Optional[int] = None, *, limit: Optional[int] = None) -> list[EvidenceLedgerEntry]:
        conn = self._get_connection()
        sql = "SELECT * FROM evidence_ledger"
        params: list[Any] = []
        clauses = []
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [
            EvidenceLedgerEntry(
                id=row["id"],
                run_id=row["run_id"] or "",
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                entry_kind=row["entry_kind"],
                entry_json=_json_loads(row["entry_json"], _json_loads(row["metadata_json"] if "metadata_json" in row.keys() else "{}", {})),
                stance=row["stance"] if "stance" in row.keys() else "neutral",
                source_name=row["source_name"] if "source_name" in row.keys() else "",
                source_url=row["source_url"] if "source_url" in row.keys() else "",
                quote_text=row["quote_text"] if "quote_text" in row.keys() else "",
                summary=row["summary"] if "summary" in row.keys() else "",
                metadata_json=row["metadata_json"] if "metadata_json" in row.keys() else "{}",
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_build_brief(self, build_brief_id: int) -> Optional[BuildBrief]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM build_briefs WHERE id = ?", (build_brief_id,)).fetchone()
        return BuildBrief(**dict(row)) if row else None

    def get_build_brief_for_opportunity(self, opportunity_id: int, run_id: Optional[str] = None) -> Optional[BuildBrief]:
        conn = self._get_connection()
        sql = "SELECT * FROM build_briefs WHERE opportunity_id = ?"
        params: list[Any] = [opportunity_id]
        if run_id:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY id DESC LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        return BuildBrief(**dict(row)) if row else None

    def list_build_briefs(self, *, run_id: Optional[str] = None, status: Optional[str] = None, limit: int = 25) -> list[BuildBrief]:
        conn = self._get_connection()
        sql = "SELECT * FROM build_briefs"
        params: list[Any] = []
        clauses = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return [BuildBrief(**dict(row)) for row in conn.execute(sql, params).fetchall()]

    def update_build_brief_status(self, build_brief_id: int, status: str) -> None:
        conn = self._get_connection()
        conn.execute("UPDATE build_briefs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, build_brief_id))
        conn.commit()

    def upsert_build_brief(self, brief: BuildBrief) -> int:
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO build_briefs (
                run_id, opportunity_id, validation_id, cluster_id, status,
                recommended_output_type, schema_version, brief_hash, brief_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, opportunity_id) DO UPDATE SET
                validation_id = excluded.validation_id,
                cluster_id = excluded.cluster_id,
                status = excluded.status,
                recommended_output_type = excluded.recommended_output_type,
                schema_version = excluded.schema_version,
                brief_hash = excluded.brief_hash,
                brief_json = excluded.brief_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                brief.run_id,
                brief.opportunity_id,
                brief.validation_id,
                brief.cluster_id,
                brief.status,
                brief.recommended_output_type,
                brief.schema_version,
                brief.brief_hash,
                brief.brief_json,
            ),
        )
        row = conn.execute("SELECT id FROM build_briefs WHERE run_id = ? AND opportunity_id = ?", (brief.run_id, brief.opportunity_id)).fetchone()
        conn.commit()
        return int(row["id"])

    def list_build_prep_outputs(self, *, run_id: Optional[str] = None, build_brief_id: Optional[int] = None, opportunity_id: Optional[int] = None, limit: int = 50) -> list[BuildPrepOutput]:
        conn = self._get_connection()
        sql = "SELECT * FROM build_prep_outputs"
        params: list[Any] = []
        clauses = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if build_brief_id is not None:
            clauses.append("build_brief_id = ?")
            params.append(build_brief_id)
        if opportunity_id is not None:
            clauses.append("opportunity_id = ?")
            params.append(opportunity_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return [BuildPrepOutput(**dict(row)) for row in conn.execute(sql, params).fetchall()]

    def upsert_build_prep_output(self, output: BuildPrepOutput) -> int:
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO build_prep_outputs (
                run_id, build_brief_id, opportunity_id, validation_id, agent_name,
                prep_stage, status, output_hash, output_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id, build_brief_id, agent_name) DO UPDATE SET
                opportunity_id = excluded.opportunity_id,
                validation_id = excluded.validation_id,
                prep_stage = excluded.prep_stage,
                status = excluded.status,
                output_hash = excluded.output_hash,
                output_json = excluded.output_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                output.run_id,
                output.build_brief_id,
                output.opportunity_id,
                output.validation_id,
                output.agent_name,
                output.prep_stage,
                output.status,
                output.output_hash,
                output.output_json,
            ),
        )
        row = conn.execute(
            "SELECT id FROM build_prep_outputs WHERE run_id = ? AND build_brief_id = ? AND agent_name = ?",
            (output.run_id, output.build_brief_id, output.agent_name),
        ).fetchone()
        conn.commit()
        return int(row["id"])

    def update_opportunity_selection(self, opportunity_id: int, *, selection_status: str, selection_reason: str) -> None:
        conn = self._get_connection()
        row = conn.execute("SELECT selection_status FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
        if row and not is_allowed_selection_transition(row["selection_status"] or "research_more", selection_status):
            raise ValueError(f"invalid selection transition: {row['selection_status']} -> {selection_status}")
        conn.execute(
            "UPDATE opportunities SET selection_status = ?, selection_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (selection_status, selection_reason, opportunity_id),
        )
        conn.commit()

    def insert_review_feedback(self, feedback: ReviewFeedback) -> int:
        conn = self._get_connection()
        cur = conn.execute(
            """
            INSERT INTO review_feedback (
                run_id, finding_id, cluster_id, opportunity_id, validation_id,
                review_label, note, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.run_id,
                feedback.finding_id,
                feedback.cluster_id,
                feedback.opportunity_id,
                feedback.validation_id,
                feedback.review_label,
                feedback.note,
                feedback.metadata_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_review_feedback_summary(
        self,
        *,
        finding_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
    ) -> dict[str, Any]:
        conn = self._get_connection()
        sql = "SELECT review_label, note, created_at FROM review_feedback"
        clauses: list[str] = []
        params: list[Any] = []
        if finding_id is not None:
            clauses.append("finding_id = ?")
            params.append(finding_id)
        if cluster_id is not None:
            clauses.append("cluster_id = ?")
            params.append(cluster_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = conn.execute(sql, params).fetchall()
        labels = [row["review_label"] for row in rows]
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        park_bias = 0.0
        kill_bias = 0.0
        for label, count in label_counts.items():
            if label in {"should_park", "needs_more_evidence"}:
                park_bias += 0.06 * count
            if label in {"should_kill", "false_positive"}:
                kill_bias += 0.0667 * count
        strongest_label = max(label_counts, key=label_counts.get) if label_counts else ""
        strongest_count = label_counts.get(strongest_label, 0)
        return {
            "count": len(rows),
            "labels": labels,
            "label_counts": label_counts,
            "strongest_label": strongest_label,
            "strongest_count": strongest_count,
            "consistency": (strongest_count / len(rows)) if rows else 0.0,
            "park_bias": round(min(park_bias, 0.2), 4),
            "kill_bias": round(min(kill_bias, 0.2), 4),
        }

    def get_validation_stats(self) -> dict[str, int]:
        conn = self._get_connection()
        validated = conn.execute("SELECT COUNT(*) AS n FROM validations WHERE passed = 1").fetchone()["n"]
        rejected = conn.execute("SELECT COUNT(*) AS n FROM validations WHERE passed = 0").fetchone()["n"]
        return {"validated": int(validated), "rejected": int(rejected)}

    def get_finding_status_counts(self, *, run_id: Optional[str] = None, actionable_only: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.get_findings(limit=1000):
            if not self._finding_matches_run(finding, run_id):
                continue
            if actionable_only and finding.status == "qualified":
                signal_rows = self.get_raw_signals_by_finding(finding.id or 0)
                atom_rows = self.get_problem_atoms_by_finding(finding.id or 0)
                if not (signal_rows and atom_rows):
                    continue
            counts[finding.status] = counts.get(finding.status, 0) + 1
        for status in ["new", "qualified", "screened_out", "parked", "killed", "promoted", "reviewed"]:
            counts.setdefault(status, 0)
        return counts

    def get_screening_summary(self, limit: int = 20, *, run_id: Optional[str] = None) -> dict[str, Any]:
        findings = self.get_findings(limit=500)
        by_source: dict[str, int] = {}
        by_source_class: dict[str, int] = {}
        by_policy_reason: dict[str, int] = {}
        by_negative_signal: dict[str, int] = {}
        items: list[dict[str, Any]] = []
        for finding in findings:
            if not self._finding_matches_run(finding, run_id):
                continue
            evidence = finding.evidence or {}
            source_policy = evidence.get("source_policy", {})
            source_classification = evidence.get("source_classification", {})
            by_source[finding.source] = by_source.get(finding.source, 0) + 1
            by_source_class[finding.source_class] = by_source_class.get(finding.source_class, 0) + 1
            reason = source_classification.get("policy_reason") or evidence.get("policy_reason")
            if reason:
                by_policy_reason[reason] = by_policy_reason.get(reason, 0) + 1
            negative = source_classification.get("negative_signal") or evidence.get("negative_signal")
            if negative:
                by_negative_signal[negative] = by_negative_signal.get(negative, 0) + 1
            if finding.status == "screened_out" and len(items) < limit:
                items.append({
                    "finding_id": finding.id,
                    "source": finding.source,
                    "title": evidence.get("title") or finding.outcome_summary or finding.product_built,
                    "source_class": finding.source_class,
                    "policy_reason": reason or "",
                    "negative_signal": negative or "",
                })
        return {
            "run_id": run_id or self.active_run_id or "legacy",
            "counts_by_source": by_source,
            "counts_by_source_class": by_source_class,
            "counts_by_policy_reason": by_policy_reason,
            "counts_by_negative_signal": by_negative_signal,
            "items": items,
        }

    def get_validation_review(self, limit: int = 25, *, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        conn = self._get_connection()
        sql = """
            SELECT v.*, f.source, f.source_url, f.source_class, f.product_built, f.outcome_summary, f.evidence_json
            FROM validations v
            JOIN findings f ON v.finding_id = f.id
        """
        params: list[Any] = []
        if run_id:
            sql += " WHERE v.run_id = ?"
            params.append(run_id)
        sql += " ORDER BY v.validated_at DESC, v.id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            evidence = _json_loads(row["evidence"], {})
            finding_evidence = _json_loads(row["evidence_json"], {})
            corroboration = conn.execute(
                "SELECT * FROM corroborations WHERE run_id = ? AND finding_id = ?",
                (row["run_id"], row["finding_id"]),
            ).fetchone()
            market = conn.execute(
                "SELECT * FROM market_enrichments WHERE run_id = ? AND finding_id = ?",
                (row["run_id"], row["finding_id"]),
            ).fetchone()
            result = build_validation_review_row(
                dict(row),
                evidence=evidence,
                finding_evidence=finding_evidence,
                corroboration_evidence=_json_loads(corroboration["evidence_json"], {}) if corroboration else None,
                corroboration_row=dict(corroboration) if corroboration else None,
                market_evidence=_json_loads(market["evidence_json"], {}) if market else None,
                market_row=dict(market) if market else None,
            )
            results.append(result)
        return results

    def get_candidate_workbench(self, limit: int = 25, *, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        rows = self.get_validation_review(limit=500, run_id=run_id)
        build_briefs = self.list_build_briefs(run_id=run_id, limit=500)
        brief_by_validation_id = {brief.validation_id: brief for brief in build_briefs}
        items: list[dict[str, Any]] = []
        for row in rows:
            brief = brief_by_validation_id.get(int(row.get("validation_id") or 0))
            items.append(build_candidate_workbench_item(row, brief))
        action_priority = {"prototype_now": 0, "gather_more_evidence": 1, "watchlist": 2, "archive": 3}
        items.sort(
            key=lambda item: (
                action_priority.get(str(item.get("next_recommended_action", "")), 9),
                -float(item.get("family_confirmation_count", 0) or 0),
                -float(item.get("validation_id", 0) or 0),
            )
        )
        return items[:limit]

    def get_review_queue(self, limit: int = 25, *, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        rows = self.get_validation_review(limit=500, run_id=run_id)
        queue = [row for row in rows if row.get("decision") in {"park", "kill"}]
        return queue[:limit]

    def get_validation_corroboration_digest(
        self,
        *,
        validation_id: Optional[int] = None,
        finding_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if validation_id is None and finding_id is None:
            return None
        rows = self.get_validation_review(limit=500, run_id=run_id)
        row = next(
            (
                item
                for item in rows
                if (validation_id is None or item.get("validation_id") == validation_id)
                and (finding_id is None or item.get("finding_id") == finding_id)
            ),
            None,
        )
        if row is None:
            return None
        return build_validation_corroboration_digest(row)

    def get_ideas(self, status: Optional[str] = None, limit: Optional[int] = None) -> list[Idea]:
        conn = self._get_connection()
        sql = "SELECT * FROM ideas"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY confidence_score DESC, created_at DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [Idea(**dict(row)) for row in conn.execute(sql, params).fetchall()]

    def get_idea(self, idea_id: int) -> Optional[Idea]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        return Idea(**dict(row)) if row else None

    def update_idea_status(self, idea_id: int, status: str) -> None:
        conn = self._get_connection()
        conn.execute("UPDATE ideas SET status = ? WHERE id = ?", (status, idea_id))
        conn.commit()

    def insert_product(self, product: Product) -> int:
        product.__post_init__()
        conn = self._get_connection()
        cur = conn.execute(
            """
            INSERT INTO products (
                idea_id, build_brief_id, opportunity_id, validation_id,
                name, location, status, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                product.idea_id,
                product.build_brief_id,
                product.opportunity_id,
                product.validation_id,
                product.name,
                product.location,
                product.status,
                product.metadata_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_product(self, product_id: int) -> Optional[dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        return dict(row) if row else None

    def get_product_for_idea(self, idea_id: int) -> Optional[dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM products WHERE idea_id = ? ORDER BY built_at DESC, id DESC LIMIT 1",
            (idea_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_product_for_build_brief(self, build_brief_id: int) -> Optional[dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM products WHERE build_brief_id = ? ORDER BY built_at DESC, id DESC LIMIT 1",
            (build_brief_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_product_status(self, product_id: int, status: str, metadata_json: Optional[str] = None) -> None:
        conn = self._get_connection()
        if metadata_json is None:
            conn.execute(
                "UPDATE products SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, product_id),
            )
        else:
            conn.execute(
                "UPDATE products SET status = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, metadata_json, product_id),
            )
        conn.commit()

    def upsert_product_for_build(
        self,
        *,
        build_brief_id: int,
        opportunity_id: int,
        validation_id: int,
        name: str,
        location: str,
        status: str,
        metadata: dict[str, Any],
    ) -> int:
        conn = self._get_connection()
        metadata_json = _json_dumps(metadata)
        existing = self.get_product_for_build_brief(build_brief_id)
        if existing is None:
            cur = conn.execute(
                """
                INSERT INTO products (
                    build_brief_id, opportunity_id, validation_id,
                    name, location, status, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    build_brief_id,
                    opportunity_id,
                    validation_id,
                    name,
                    location,
                    status,
                    metadata_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

        conn.execute(
            """
            UPDATE products
            SET opportunity_id = ?, validation_id = ?, name = ?, location = ?,
                status = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                opportunity_id,
                validation_id,
                name,
                location,
                status,
                metadata_json,
                existing["id"],
            ),
        )
        conn.commit()
        return int(existing["id"])

    def upsert_product_for_idea(
        self,
        *,
        idea_id: int,
        name: str,
        location: str,
        status: str,
        metadata: dict[str, Any],
    ) -> int:
        conn = self._get_connection()
        metadata_json = _json_dumps(metadata)
        existing = self.get_product_for_idea(idea_id)
        if existing is None:
            cur = conn.execute(
                """
                INSERT INTO products (
                    idea_id, name, location, status, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    idea_id,
                    name,
                    location,
                    status,
                    metadata_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

        conn.execute(
            """
            UPDATE products
            SET name = ?, location = ?, status = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                name,
                location,
                status,
                metadata_json,
                existing["id"],
            ),
        )
        conn.commit()
        return int(existing["id"])

    def get_products(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        conn = self._get_connection()
        sql = "SELECT * FROM products ORDER BY built_at DESC"
        params: list[Any] = []
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
