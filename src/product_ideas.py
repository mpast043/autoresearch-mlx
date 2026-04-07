"""Wedge detection pipeline - platform-explicit decision-grade wedge evaluation.

Stage-based pipeline:
- Discovery → Extraction → Validation → Sharpening → Wedge Gate → Product Mapping → Ranking

Each stage has one responsibility. Later stages must only narrow, not redefine.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.database import Database

logger = logging.getLogger(__name__)


class WedgeStage(Enum):
    """Pipeline stages in order."""
    DISCOVERY = "discovery"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    SHARPENING = "sharpening"
    WEDGE_GATE = "wedge_gate"
    PRODUCT_MAPPING = "product_mapping"
    RANKING = "ranking"


class PlatformConfidence(Enum):
    """Platform detection confidence levels."""
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class OutputBucket(Enum):
    """Final output categories."""
    ACCEPTED = "accepted"
    RESEARCH_NEEDED = "research_needed"
    REJECTED = "rejected"


@dataclass
class WedgeCandidate:
    """Unified candidate object carrying ALL relevant signals forward.

    No upstream signal may be discarded before wedge evaluation.
    """
    # Core extraction (from ProblemAtom / RawSignal)
    id: Optional[int] = None
    atom_id: Optional[int] = None
    opportunity_id: Optional[int] = None
    cluster_id: Optional[int] = None

    # Extraction signals
    user: str = ""
    workflow: str = ""
    trigger_moment: str = ""
    failure_mode: str = ""
    consequence: str = ""
    evidence_snippet: str = ""

    # Platform signals
    platform: str = "unknown"
    platform_confidence: str = "unknown"  # explicit, inferred, unknown

    # Term-level signals (from atom metadata if available)
    specificity_score: float = 0.0
    platform_native_score: float = 0.0
    plugin_fit_score: float = 0.0
    quality_score: float = 0.0

    # Opportunity signals (from ValidationAgent scoring)
    frequency_score: float = 0.0
    urgency_score: float = 0.0
    problem_score: float = 0.0
    value_score: float = 0.0
    feasibility_score: float = 0.0
    buyer_intent_score: float = 0.0
    corroboration_score: float = 0.0

    # Validation signals (from score_opportunity)
    buyer_clarity: float = 0.0  # problem_plausibility
    pain_severity: float = 0.0
    willingness_to_pay: float = 0.0
    market_size: float = 0.0
    buildability: float = 0.0
    distribution_access: float = 0.0

    # v4 composite scores
    problem_truth_score: float = 0.0
    revenue_readiness_score: float = 0.0
    decision_score: float = 0.0
    composite_score: float = 0.0

    # Evidence quality
    evidence_quality_score: float = 0.0

    # Governance context
    source_family: str = ""
    source_url: str = ""

    # Meta
    source: str = ""
    query_term: str = ""
    run_id: str = ""
    rejection_reasons: list[str] = field(default_factory=list)
    current_stage: str = "extraction"
    output_bucket: str = "rejected"

    # Output fields (filled in later stages)
    wedge_name: str = ""
    product_format: str = ""
    value_proposition: str = ""
    confidence: float = 0.0
    why_passed: str = ""


@dataclass
class ProductIdea:
    """Decision-grade product wedge - legacy output format."""
    name: str
    confidence: float
    evidence_quality: str  # high, medium, low
    user: str
    workflow: str
    trigger_moment: str
    failure_mode: str
    consequence: str
    host_platform: str
    platform_confidence: str  # explicit, inferred, unknown
    product_format: str
    value_proposition: str
    why_wedge_not_category: str
    evidence_snippets: list[str] = field(default_factory=list)
    suggested_mvp: str = ""
    why_passed: str = ""


# Platform formats
PLATFORM_FORMATS = {
    'excel': 'Excel add-in',
    'google_sheets': 'Google Sheets add-on',
    'quickbooks': 'QuickBooks integration',
    'xero': 'Xero integration',
    'shopify': 'Shopify app',
    'notion': 'Notion integration',
    'airtable': 'Airtable automation',
    'etsy': 'Etsy tool',
    'wordpress': 'WordPress plugin',
    'slack': 'Slack app',
    'gmail': 'Gmail add-on',
}

# Actions that make something a wedge
WEDGE_ACTIONS = [
    'detect', 'prevent', 'block', 'fix', 'stop', 'check', 'match', 'sync',
    'find', 'alert', 'remind', 'extract', 'convert', 'clean', 'notify',
    'reconcile', 'merge', 'dedupe', 'export', 'import', 'upload', 'download',
    'monitor', 'track', 'watch', 'scan', 'parse', 'format', 'transform',
    'reconciliation', 'packager', 'formatter',
]

# Blocked generic terms
GENERIC_BLOCKED = ['manual work', 'keep in sync', 'data handoff', 'automate manual']


def extract_platform_explicit(source: str, url: str, text: str) -> tuple[str, str]:
    """Extract platform explicitly from source/URL, return (platform, confidence)."""
    source_lower = (source or '').lower()
    url_lower = (url or '').lower()
    text_lower = (text or '').lower()

    # Map source names to platforms
    source_map = {
        'notion': 'notion',
        'shopify': 'shopify',
        'etsy': 'etsy',
        'airtable': 'airtable',
        'quickbooks': 'quickbooks',
        'xero': 'xero',
    }

    for src_key, platform in source_map.items():
        if src_key in source_lower:
            return platform, 'explicit'

    # Check URL
    for platform in ['shopify', 'notion', 'airtable', 'quickbooks', 'xero', 'etsy']:
        if platform in url_lower:
            return platform, 'explicit'

    return 'unknown', 'unknown'


def extract_platform_inferred(text: str) -> tuple[str, str]:
    """Infer platform from text content."""
    text_lower = (text or '').lower()

    # Must have explicit platform word - not just "spreadsheet"
    patterns = {
        'excel': [r'\bexcel\b', r'\bxlsx\b', r'\bmicrosoft excel\b'],
        'google_sheets': [r'\bgoogle sheets\b', r'\bsheets\b'],
        'quickbooks': [r'\bquickbooks\b', r'\bqb\b'],
        'xero': [r'\bxero\b'],
        'shopify': [r'\bshopify\b'],
        'notion': [r'\bnotion\b'],
        'airtable': [r'\bairtable\b'],
        'etsy': [r'\betsy\b'],
    }

    for platform, regexes in patterns.items():
        for regex in regexes:
            if re.search(regex, text_lower):
                return platform, 'inferred'

    return 'unknown', 'unknown'


def extract_pain_with_platform_context(title: str, body: str, platform: str = '') -> dict:
    """Extract pain components with better platform context awareness."""
    text = title + " " + body
    result = {'workflow': '', 'pain': '', 'trigger': '', 'consequence': ''}

    # Pattern 1: Spending X hours doing Y (HIGH CONFIDENCE)
    match = re.search(r'spend(?:ing|s)?\s+(\d+)\s+(?:hours?|hrs?)\s+(?:a\s+)?(?:day|week|month)?\s+(?:on|doing|just)\s+([^\.\?]+)', text, re.IGNORECASE)
    if match:
        hours = match.group(1)
        task = match.group(2).strip()
        result['pain'] = f"spending {hours} hours {task}"
        result['workflow'] = task[:50]
        result['consequence'] = f"{hours} hours lost per period"
        result['trigger'] = f"when {task[:30]}"
        return result

    # Pattern 2: have to manually X
    match = re.search(r'have to\s+manually\s+([^\.\?]+)', text, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        result['pain'] = f"manually {task}"
        result['workflow'] = f"manually {task}"[:50]
        result['consequence'] = "time waste and errors"
        result['trigger'] = f"when {task[:30]}"
        return result

    # Pattern 3: copy and paste pain
    if re.search(r'copy(ing|[- ])?paste', text, re.IGNORECASE):
        result['pain'] = "copying and pasting data manually"
        result['workflow'] = "copy-paste data entry"
        result['consequence'] = "time waste and data entry errors"
        result['trigger'] = "when copying data between systems"
        return result

    # Pattern 4: reconcile/match pain - VERY COMMON
    if re.search(r'reconcile|reconciliation|matching', text, re.IGNORECASE):
        # Try to find what is being matched
        match = re.search(r'(?:reconcil|match)ing?\s+(?:between\s+)?(?:the\s+)?([a-z\s]+?)(?:\s+and|\s+with|\s+to|\s+for|$)', text, re.IGNORECASE)
        if match:
            task = match.group(1).strip()
            result['pain'] = f"manually matching {task}"
            result['workflow'] = f"matching {task}"[:50]
        else:
            result['pain'] = "manually matching transactions"
            result['workflow'] = "transaction matching"
        result['consequence'] = "errors in financial records"
        result['trigger'] = "when reconciling records"
        return result

    # Pattern 5: X takes Y hours (time-focused pain)
    match = re.search(r'([a-z\s]+?)\s+takes?\s+(\d+)\s+hours?', text, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        hours = match.group(2)
        result['pain'] = f"{task} takes {hours} hours"
        result['workflow'] = task[:50]
        result['consequence'] = f"{hours} hours wasted"
        result['trigger'] = f"when {task[:30]}"
        return result

    # Pattern 6: tedious X, annoying X
    match = re.search(r'(?:tedious|annoying|manual|repetitive)\s+([a-z\s]{5,50})', text, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        result['pain'] = f"tedious {task}"
        result['workflow'] = task[:50]
        result['consequence'] = "time waste"
        result['trigger'] = f"when doing {task[:30]}"
        return result

    # Pattern 7: platform-specific complaints (look for platform + pain)
    if platform:
        # Check for common manual actions with platform
        platform_actions = {
            'shopify': ['updating inventory', 'creating labels', 'processing orders', 'syncing products'],
            'notion': ['updating databases', 'copying data', 'syncing pages', 'managing templates'],
            'quickbooks': ['entering invoices', 'matching transactions', 'sending reminders', 'reconciling accounts'],
            'etsy': ['processing orders', 'creating listings', 'packaging files', 'updating inventory'],
            'excel': ['copying formulas', 'updating ranges', 'merging cells', 'fixing errors'],
        }
        actions = platform_actions.get(platform, [])
        for action in actions:
            if action in text.lower():
                result['pain'] = f"manually {action}"
                result['workflow'] = action[:50]
                result['consequence'] = "time waste"
                result['trigger'] = f"when {action[:30]}"
                return result

    # Pattern 8: BROADER - any "manually X" with at least 3 chars
    match = re.search(r'manually\s+([a-z]{3,30})', text, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        result['pain'] = f"manually {task}"
        result['workflow'] = f"manual {task}"[:50]
        result['consequence'] = "time waste"
        result['trigger'] = f"when {task}"
        return result

    return result


def validate_workflow(workflow: str) -> tuple[bool, str]:
    """Validate workflow is specific."""
    if not workflow or len(workflow) < 5:
        return False, "too short"
    for blocked in GENERIC_BLOCKED:
        if blocked in workflow.lower():
            return False, f"blocked: {blocked}"
    return True, "ok"


def validate_user(user: str) -> tuple[bool, str]:
    """Validate user is specific or allow default."""
    if not user or user.strip() == '':
        return True, "default user"  # Allow empty - will default to "user"
    user_lower = user.lower().strip()
    if user_lower in ['business', 'person']:
        return False, "generic"
    # Allow specific roles + "business user" as it's a common valid descriptor
    return True, "ok"


def sharpen_wedge(platform: str, workflow: str, pain: str, trigger: str, consequence: str) -> dict | None:
    """Sharpen into specific wedge."""
    pain_lower = pain.lower()
    workflow_lower = workflow.lower()

    # Platform-specific sharpening
    wedges = {
        'shopify': [
            ('inventory', 'Shopify Inventory Alert', 'Notifies when inventory is running low', 'Shopify app'),
            ('order', 'Shopify Order Processor', 'Automates order processing', 'Shopify app'),
            ('label', 'Shopify Label Printer', 'Prints shipping labels in bulk', 'Shopify app'),
            ('match', 'Shopify Transaction Matcher', 'Matches orders with payments', 'Shopify app'),
        ],
        'notion': [
            ('copy', 'Notion Clipboard Sync', 'Syncs clipboard to Notion databases', 'Notion integration'),
            ('database', 'Notion Database Cleaner', 'Cleans and deduplicates Notion entries', 'Notion integration'),
            ('template', 'Notion Template Filler', 'Fills templates from external data', 'Notion integration'),
        ],
        'quickbooks': [
            ('invoice', 'Invoice Reminder', 'Sends overdue invoice alerts', 'QuickBooks integration'),
            ('match', 'Bank Transaction Matcher', 'Matches bank transactions to QB entries', 'QuickBooks integration'),
            ('receipt', 'Receipt Scanner', 'Extracts data from receipt photos', 'QuickBooks app'),
        ],
        'etsy': [
            ('file', 'Etsy File Packager', 'Auto-packages digital files for orders', 'Etsy tool'),
            ('order', 'Etsy Order Processor', 'Automates order processing', 'Etsy tool'),
            ('inventory', 'Etsy Inventory Sync', 'Keeps inventory in sync', 'Etsy tool'),
        ],
        'excel': [
            ('version', 'Spreadsheet Version Alert', 'Warns when spreadsheet was modified', 'Excel add-in'),
            ('formula', 'Formula Error Detector', 'Catches formula errors', 'Excel add-in'),
            ('match', 'Transaction Reconciliation', 'Matches transactions across sources', 'Excel macro'),
            ('paste', 'Paste Format Cleaner', 'Cleans formatting on paste', 'Excel add-in'),
        ],
    }

    # Get platform-specific wedges
    platform_wedges = wedges.get(platform, [])

    # Try to match
    for keyword, name, desc, fmt in platform_wedges:
        if keyword in pain_lower or keyword in workflow_lower:
            return {
                'name': name,
                'wedge': desc,
                'mvp': f'{fmt} that {desc.lower()}',
            }

    # Generic but specific patterns
    if 'match' in pain_lower or 'reconcil' in pain_lower:
        if platform != 'unknown':
            fmt = PLATFORM_FORMATS.get(platform, 'integration')
            return {
                'name': f'{platform.title()} Transaction Matcher',
                'wedge': f'Matches transactions in {platform.title()}',
                'mvp': f'{fmt} that auto-matches transactions',
            }
        return {
            'name': 'Transaction Matcher',
            'wedge': 'Matches transactions across platforms',
            'mvp': 'Script that matches CSV transactions by amount/date',
        }

    if 'copy' in pain_lower and 'paste' in pain_lower:
        return {
            'name': 'Data Paste Automator',
            'wedge': 'Transforms data when pasting',
            'mvp': 'Tool that transforms clipboard data on paste',
        }

    return None


# =============================================================================
# STAGE 1: EXTRACTION - Build WedgeCandidate from raw data
# =============================================================================

def _get_row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Safely get value from sqlite3.Row."""
    try:
        return row[key] if key in row.keys() else default
    except (KeyError, IndexError, TypeError):
        return default


def _build_wedge_query(db: Database) -> str:
    """Build a wedge query that matches the live SQLite storage model."""
    conn = db._get_connection()
    atom_columns = {row["name"] for row in conn.execute("PRAGMA table_info(problem_atoms)").fetchall()}
    cluster_columns = {row["name"] for row in conn.execute("PRAGMA table_info(opportunity_clusters)").fetchall()}

    atom_cluster_expr = "pa.cluster_key" if "cluster_key" in atom_columns else "json_extract(pa.metadata_json, '$.cluster_key')"
    cluster_key_expr = (
        "oc.cluster_key" if "cluster_key" in cluster_columns else "json_extract(oc.metadata_json, '$.cluster_key')"
    )
    atom_segment_expr = "pa.segment" if "segment" in atom_columns else "json_extract(pa.metadata_json, '$.segment')"
    cluster_segment_expr = "oc.segment" if "segment" in cluster_columns else "json_extract(oc.metadata_json, '$.segment')"
    atom_consequence_expr = (
        "pa.cost_consequence_clues"
        if "cost_consequence_clues" in atom_columns
        else "json_extract(pa.metadata_json, '$.cost_consequence_clues')"
    )
    atom_platform_expr = (
        "pa.platform"
        if "platform" in atom_columns
        else "COALESCE(json_extract(pa.metadata_json, '$.platform'), json_extract(pa.atom_json, '$.platform'))"
    )
    raw_signal_expr = "pa.raw_signal_id" if "raw_signal_id" in atom_columns else "pa.signal_id"

    return f"""
        SELECT
            o.id as opportunity_id,
            o.cluster_id,
            o.title,
            o.decision_score,
            o.problem_truth_score,
            o.revenue_readiness_score,
            o.composite_score,
            o.frequency_score,
            o.urgency_score,
            o.buildability,
            o.corroboration_strength,
            o.evidence_quality,
            o.pain_severity,
            o.willingness_to_pay_proxy,
            o.problem_plausibility,
            o.reachability,
            o.status as opportunity_status,
            pa.id as atom_id,
            pa.job_to_be_done,
            pa.pain_statement,
            pa.trigger_event,
            pa.failure_mode,
            {atom_consequence_expr} as cost_consequence_clues,
            {atom_segment_expr} as segment,
            pa.user_role,
            {atom_platform_expr} as atom_platform,
            rs.source_name,
            rs.source_url,
            rs.title as signal_title,
            rs.body_excerpt,
            rs.role_hint,
            rs.quote_text,
            {cluster_segment_expr} as cluster_segment
        FROM opportunities o
        JOIN opportunity_clusters oc ON oc.id = o.cluster_id
        LEFT JOIN problem_atoms pa ON {atom_cluster_expr} = {cluster_key_expr}
        LEFT JOIN raw_signals rs ON rs.id = {raw_signal_expr}
        WHERE o.status IN ('promoted', 'active')
            AND (o.decision_score > 0 OR o.composite_score > 0)
        ORDER BY o.decision_score DESC, o.problem_truth_score DESC
        LIMIT 500
    """


def extract_candidate_from_opportunity(row: sqlite3.Row) -> WedgeCandidate:
    """Extract WedgeCandidate from opportunity row with validation scores."""
    candidate = WedgeCandidate()

    # Core IDs
    candidate.opportunity_id = _get_row_value(row, 'opportunity_id')
    candidate.cluster_id = _get_row_value(row, 'cluster_id')
    candidate.atom_id = _get_row_value(row, 'atom_id')

    # Extraction signals from raw_signal and atom
    candidate.user = _get_row_value(row, 'role_hint', '') or "business user"
    candidate.evidence_snippet = _get_row_value(row, 'body_excerpt', '') or _get_row_value(row, 'quote_text', '')

    # Problem atom fields
    candidate.workflow = _get_row_value(row, 'job_to_be_done', '') or _get_row_value(row, 'segment', '')
    candidate.failure_mode = _get_row_value(row, 'pain_statement', '') or _get_row_value(row, 'failure_mode', '')
    candidate.trigger_moment = _get_row_value(row, 'trigger_event', '')
    candidate.consequence = _get_row_value(row, 'cost_consequence_clues', '')

    # Platform detection - prefer atom's extracted platform
    atom_platform = _get_row_value(row, 'atom_platform', '') or ''

    source = _get_row_value(row, 'source_name', '') or ''
    url = _get_row_value(row, 'source_url', '') or ''
    full_text = f"{_get_row_value(row, 'title', '')} {_get_row_value(row, 'body_excerpt', '')}"

    # If atom has platform, use it with explicit confidence
    if atom_platform:
        candidate.platform = atom_platform
        candidate.platform_confidence = 'explicit'
    else:
        # Fall back to text-based detection
        platform, conf = extract_platform_explicit(source, url, full_text)
        if conf == 'unknown':
            platform, conf = extract_platform_inferred(full_text)
        candidate.platform = platform
        candidate.platform_confidence = conf

    # Source info
    candidate.source = source
    candidate.source_url = url

    # Validation scores - THIS IS THE KEY DIFFERENCE
    # Use actual upstream scores instead of re-computing
    candidate.decision_score = float(_get_row_value(row, 'decision_score', 0.0) or 0.0)
    candidate.problem_truth_score = float(_get_row_value(row, 'problem_truth_score', 0.0) or 0.0)
    candidate.revenue_readiness_score = float(_get_row_value(row, 'revenue_readiness_score', 0.0) or 0.0)
    candidate.composite_score = float(_get_row_value(row, 'composite_score', 0.0) or 0.0)

    # Individual validation scores
    candidate.frequency_score = float(_get_row_value(row, 'frequency_score', 0.0) or 0.0)
    candidate.urgency_score = float(_get_row_value(row, 'urgency_score', 0.0) or 0.0)
    candidate.buildability = float(_get_row_value(row, 'buildability', 0.0) or 0.0)
    candidate.corroboration_score = float(_get_row_value(row, 'corroboration_strength', 0.0) or 0.0)
    candidate.evidence_quality_score = float(_get_row_value(row, 'evidence_quality', 0.0) or 0.0)
    candidate.pain_severity = float(_get_row_value(row, 'pain_severity', 0.0) or 0.0)
    candidate.willingness_to_pay = float(_get_row_value(row, 'willingness_to_pay_proxy', 0.0) or 0.0)

    # Legacy support
    candidate.problem_score = float(_get_row_value(row, 'problem_score', 0.0) or 0.0)
    candidate.value_score = float(_get_row_value(row, 'value_score', 0.0) or 0.0)
    candidate.feasibility_score = float(_get_row_value(row, 'feasibility_score', 0.0) or 0.0)
    candidate.buyer_clarity = float(_get_row_value(row, 'problem_plausibility', 0.0) or 0.0)
    candidate.distribution_access = float(_get_row_value(row, 'reachability', 0.0) or 0.0)

    candidate.current_stage = "extraction"
    return candidate


# =============================================================================
# STAGE 2: VALIDATION - Gate based on validation scores
# =============================================================================

def validate_candidate(candidate: WedgeCandidate) -> tuple[bool, str]:
    """Validate candidate using upstream validation scores.

    Rules:
    - No stage can override a previous rejection
    - No stage can re-score something already scored upstream
    - Later stages only narrow, not redefine
    Filters out extremely generic content early, but lets most through for sharpening.
    """
    candidate.current_stage = "validation"

    # Gate 1: Must have passed validation (decision_score > 0 or promoted status)
    if candidate.decision_score <= 0 and candidate.composite_score <= 0:
        return False, "no_validation_score"

    # Gate 2: Minimum problem truth score
    if candidate.problem_truth_score < 0.3 and candidate.problem_score < 0.4:
        return False, "weak_problem_truth"

    # Gate 3: Must have some evidence quality
    if candidate.evidence_quality_score < 0.25:
        return False, "insufficient_evidence_quality"

    # Gate 4: Buildability threshold
    if candidate.buildability < 0.3:
        return False, "low_buildability"

    # Gate 5: Only filter extremely broken atoms - let most through for sharpening
    workflow = (candidate.workflow or '').strip()
    failure = (candidate.failure_mode or '').strip()

    # Only reject if workflow is exactly a known generic label AND no failure
    GENERIC_CLUSTER_LABELS = [
        'keep operations data in sync without manual cleanup',
        'keep sync and data handoff workflows reliable',
    ]

    workflow_lower = workflow.lower()
    failure_lower = failure.lower()

    # If workflow is exactly a generic label AND failure is also empty/generic
    if workflow_lower in GENERIC_CLUSTER_LABELS and (not failure or len(failure) < 15):
        return False, "generic_cluster_label"

    return True, "passed_validation"


# =============================================================================
# STAGE 3: SHARPENING - Narrow into specific wedge
# =============================================================================

# SYNTHETIC TRANSFORMATION BLOCKLIST
# These patterns produce abstract wedges NOT grounded in evidence
SYNTHETIC_PATTERNS = [
    'time saver', 'speed optimizer', 'error prevention', 'error detector',
    'deduplication', 'reminder', 'alert', 'sync automation', 'data pipeline',
    'spreadsheet automation', 'automation tool', 'helper', 'assistant',
    'workflow speed', 'smart reminder', 'data sync',
]


def _verify_evidence_alignment(
    wedge_name: str,
    workflow: str,
    failure_mode: str,
    consequence: str,
    evidence_snippet: str,
) -> tuple[bool, str]:
    """Verify wedge is grounded in extracted evidence.

    Returns (is_aligned, reason_if_not).
    """
    wedge_lower = wedge_name.lower()
    source_text = f"{workflow} {failure_mode} {consequence} {evidence_snippet}".lower()

    # Check 1: Wedge name must appear in source text (or be traceable)
    # Allow for some transformation but core terms must be present
    wedge_terms = wedge_lower.split()

    # Filter out generic suffixes
    filtered_terms = [
        t for t in wedge_terms
        if t not in ('tool', 'automator', 'system', 'solution', 'app', 'software', 'service')
    ]

    # At least one significant term must trace to source
    has_trace = any(
        term in source_text
        for term in filtered_terms
        if len(term) > 3
    )

    if not has_trace:
        return False, "wedge_not_in_evidence"

    # Check 2: If failure mode exists, it must relate to evidence snippet
    if failure_mode and evidence_snippet:
        failure_words = set(failure_mode.lower().split())
        evidence_words = set(evidence_snippet.lower().split())

        # At least 20% overlap or failure is in evidence
        overlap = len(failure_words & evidence_words)
        min_needed = max(3, len(failure_words) // 5)

        if overlap < min_needed:
            # Allow if failure describes a concrete event in evidence
            failure_has_event = any(
                kw in evidence_snippet.lower()
                for kw in ('error', 'miss', 'fail', 'wrong', 'duplicate', 'lost', 'late', 'break')
            )
            if not failure_has_event:
                return False, "failure_not_in_evidence"

    # Check 3: Consequence should appear in source if present
    if consequence:
        consequence_terms = consequence.lower().split()
        has_consequence_trace = any(
            term in source_text
            for term in consequence_terms
            if len(term) > 3
        )
        if not has_consequence_trace:
            return False, "consequence_not_in_evidence"

    # Check 4: Reject synthetic patterns entirely
    for pattern in SYNTHETIC_PATTERNS:
        if pattern in wedge_lower:
            return False, f"synthetic_pattern: {pattern}"

    return True, "aligned"


def _try_dynamic_sharpening(workflow: str, failure_mode: str, consequence: str, platform: str) -> dict | None:
    """Narrow vague descriptions into specific wedges - ONLY narrowing, no abstraction.

    STRICT RULE: Only allow transformations that extract specific nouns/verbs from source.
    Reject any transformation that introduces a new abstract concept.

    Returns dict with name, wedge, mvp or None if no sharpening possible.
    """
    pain_text = f"{workflow} {failure_mode} {consequence}".lower()

    # ALLOWED: Extract specific task from "manually [task]"
    # This extracts the actual task mentioned, not inventing a new one
    match = re.search(r'manually\s+([a-z]{3,30})', pain_text)
    if match:
        task = match.group(1).strip()
        # Only use if task is specific (not generic)
        if task and task not in ('work', 'task', 'job', 'process', 'entry', 'data'):
            if platform != 'unknown':
                return {
                    'name': f'{platform.title()} {task.title()} Fix',
                    'wedge': f'Fixes manual {task} problem',
                    'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} for {task}',
                }
            return {
                'name': f'{task.title()} Problem Fix',
                'wedge': f'Fixes manual {task}',
                'mvp': f'Script that fixes {task}',
            }

    # ALLOWED: Extract specific noun from failure mode
    # "duplicate X" → extract X
    match = re.search(r'duplicate[s]?\s+([a-z]{3,30})', pain_text)
    if match:
        target = match.group(1).strip()
        if target and target not in ('data', 'entry', 'record', 'item'):
            return {
                'name': f'{target.title()} Deduplicator',
                'wedge': f'Removes duplicate {target}',
                'mvp': f'Tool that deduplicates {target}',
            }

    # ALLOWED: Extract specific noun from "error in X"
    match = re.search(r'error[s]?\s+(?:in|with|on)\s+([a-z]{3,30})', pain_text)
    if match:
        target = match.group(1).strip()
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} {target.title()} Error Fix',
                'wedge': f'Fixes {target} errors',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that fixes {target} errors',
            }

    # ALLOWED: Extract specific thing being missed/forgotten
    match = re.search(r'(?:missed?|forgot(?:ten)?)\s+([a-z]{3,30})', pain_text)
    if match:
        target = match.group(1).strip()
        if target and target not in ('deadline', 'payment', 'order'):
            return {
                'name': f'{target.title()} Tracker',
                'wedge': f'Tracks missed {target}',
                'mvp': f'Tool that tracks {target}',
            }

    # ALLOWED: Extract specific workflow from explicit mentions
    # "X reconciliation" → extract X
    match = re.search(r'(\w+)\s+reconcil', pain_text)
    if match:
        domain = match.group(1).strip()
        return {
            'name': f'{domain.title()} Reconciliation',
            'wedge': f'{domain.title()} transaction reconciliation',
            'mvp': f'Tool that reconciles {domain} transactions',
        }

    # ALLOWED: Extract specific platform operation
    if 'paste' in pain_text or 'copy' in pain_text:
        match = re.search(r'(?:copy|paste)\s+(\w+)', pain_text)
        if match:
            target = match.group(1).strip()
            return {
                'name': f'{target.title()} Paste Fix',
                'wedge': f'Fixes {target} paste issues',
                'mvp': f'Tool that fixes {target} pasting',
            }

    # NO synthetic abstractions allowed - reject everything else
    return None


def sharpen_candidate(candidate: WedgeCandidate) -> tuple[bool, str]:
    """Sharpen candidate into platform-specific wedge.

    Uses workflow and failure_mode to determine specific wedge.
    Does NOT use synthetic generation - relies on extracted signals.
    STRICT: Wedge must be evidence-grounded - no abstract transformations.
    """
    candidate.current_stage = "sharpening"

    workflow = (candidate.workflow or '').lower()
    failure_mode = (candidate.failure_mode or '').lower()
    consequence = (candidate.consequence or '').lower()
    platform = candidate.platform

    # Must have meaningful workflow
    if not workflow or len(workflow) < 5:
        candidate.rejection_reasons.append("generic_workflow")
        return False, "generic_workflow"

    # Blocked terms check
    for blocked in GENERIC_BLOCKED:
        if blocked in workflow:
            candidate.rejection_reasons.append("blocked_workflow")
            return False, f"blocked: {blocked}"

    # Try platform-specific sharpening if platform known (exact match)
    if platform != 'unknown':
        sharpened = sharpen_wedge(platform, workflow, failure_mode,
                                  candidate.trigger_moment, consequence)
        if sharpened:
            candidate.wedge_name = sharpened['name']
            candidate.value_proposition = sharpened['wedge']
            candidate.product_format = PLATFORM_FORMATS.get(platform, 'integration')

            # Verify evidence alignment before accepting
            aligned, reason = _verify_evidence_alignment(
                candidate.wedge_name,
                candidate.workflow or '',
                candidate.failure_mode or '',
                candidate.consequence or '',
                candidate.evidence_snippet or '',
            )
            if aligned:
                return True, "sharpened"
            # If not aligned, fall through to try other options

    # Try dynamic sharpening - ONLY narrowing transformations
    dynamic = _try_dynamic_sharpening(workflow, failure_mode, consequence, platform)
    if dynamic:
        candidate.wedge_name = dynamic['name']
        candidate.value_proposition = dynamic['wedge']
        candidate.product_format = PLATFORM_FORMATS.get(platform, 'integration') if platform != 'unknown' else dynamic['mvp'].split(' ')[0]

        # Verify evidence alignment before accepting
        aligned, reason = _verify_evidence_alignment(
            candidate.wedge_name,
            candidate.workflow or '',
            candidate.failure_mode or '',
            candidate.consequence or '',
            candidate.evidence_snippet or '',
        )
        if aligned:
            return True, "dynamic_sharpened"
        # If not aligned, fall through to reject

    # NO generic fallbacks - all wedges must be evidence-grounded
    # This is stricter: reject rather than create synthetic wedges
    candidate.rejection_reasons.append("no_evidence_grounded_wedge")
    return False, "no_evidence_grounded_wedge"


# =============================================================================
# STAGE 4: WEDGE GATE - Final pass/fail decision
# =============================================================================

# Category-level terms that indicate a wedge is too generic
CATEGORY_TERMS = [
    'sync', 'automation', 'integration', 'reminder', 'tool', 'system', 'manager',
    'workflow', 'platform', 'solution', 'app', 'software', 'service', 'processor',
    'connector', 'bridge', 'importer', 'exporter', 'helper', 'assistant',
]

# Generic workflow phrases that indicate low specificity
GENERIC_WORKFLOW_PHRASES = [
    'keep in sync',
    'manual cleanup',
    'data cleanup',
    'keep operations data',
    'keep sync',
    'data handoff',
    'manual work',
    'automate manual',
]

# Failure-mode keywords that indicate specific breaks
FAILURE_KEYWORDS = [
    'duplicate', 'mismatch', 'error', 'mistake', 'break', 'broken',
    'missed', 'lost', 'late', 'failed', 'conflict', 'wrong',
    'incorrect', 'inconsistent', 'corrupt', 'overdue', 'forgot',
]


def _is_category_wedge(wedge_name: str, failure_mode: str) -> bool:
    """Check if wedge is category-level rather than failure-specific."""
    name_lower = wedge_name.lower()
    failure_lower = failure_mode.lower()

    # If name contains category term without failure qualifier
    for term in CATEGORY_TERMS:
        if term in name_lower:
            # Check if failure mode provides specificity
            has_failure_qualifier = any(kw in failure_lower for kw in FAILURE_KEYWORDS)
            if not has_failure_qualifier:
                return True

    # If wedge name is just "Platform + Category"
    if len(wedge_name.split()) <= 2:
        return True

    return False


def _is_generic_workflow(workflow: str) -> bool:
    """Check if workflow is too generic to be a wedge."""
    workflow_lower = workflow.lower()
    for phrase in GENERIC_WORKFLOW_PHRASES:
        if phrase in workflow_lower:
            return True
    return False


def _has_specific_failure(failure_mode: str) -> bool:
    """Check if failure mode describes a specific break."""
    if not failure_mode or len(failure_mode) < 15:
        return False

    failure_lower = failure_mode.lower()

    # Must contain at least one failure keyword
    has_failure_keyword = any(kw in failure_lower for kw in FAILURE_KEYWORDS)

    # Must describe a concrete problem (not just "issues with X")
    is_specific = (
        has_failure_keyword and
        not failure_lower.startswith('keep') and
        not failure_lower.startswith('need') and
        not failure_lower.startswith('want')
    )

    return is_specific


def gate_wedge(candidate: WedgeCandidate) -> tuple[bool, str]:
    """Gate wedge based on specificity and failure-mode requirements.

    Rules:
    - Must have specific failure mode (not category-level)
    - Must not have generic workflow
    - Must have trigger moment or meaningful consequence
    - Explicit platform → eligible for accepted
    - Inferred platform → research-needed
    - Unknown platform → rejected
    """
    candidate.current_stage = "wedge_gate"

    platform_conf = candidate.platform_confidence
    decision_score = candidate.decision_score
    problem_truth = candidate.problem_truth_score
    workflow = candidate.workflow or ''
    failure_mode = candidate.failure_mode or ''
    wedge_name = candidate.wedge_name or ''
    consequence = candidate.consequence or ''
    trigger = candidate.trigger_moment or ''

    # SPECIFICITY CHECKS - route to appropriate bucket based on severity

    # Check 1: Category-level wedge
    is_category = _is_category_wedge(wedge_name, failure_mode)
    is_generic = _is_generic_workflow(workflow)
    weak_failure = not _has_specific_failure(failure_mode)
    missing_tc = not trigger and not consequence

    # Check 2: Generic workflow rejection
    is_generic_workflow = _is_generic_workflow(workflow)

    # Check 3: Weak failure mode rejection
    weak_failure_mode = not _has_specific_failure(failure_mode)

    # STRICT MODE: For explicit platform, enforce specificity
    if platform_conf == 'explicit':
        if is_category:
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append("category_label")
            return False, "category_label"

        if is_generic_workflow:
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append("generic_workflow")
            return False, "generic_workflow"

        if weak_failure_mode:
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append("weak_failure_mode")
            return False, "weak_failure_mode"

    # For inferred/unknown, route to research-needed instead of rejection
    else:
        if is_category:
            candidate.rejection_reasons.append("category_label")
            # Will route to research_needed below based on platform

        if is_generic_workflow:
            candidate.rejection_reasons.append("generic_workflow")

        if weak_failure_mode:
            candidate.rejection_reasons.append("weak_failure_mode")

    # Check 4: Missing trigger AND consequence
    if not trigger and not consequence:
        if platform_conf == 'explicit':
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append("missing_trigger")
            return False, "missing_trigger"
        # For inferred/unknown, we'll allow through to research-needed

    # Check 5: EVIDENCE ALIGNMENT - verify wedge is grounded in extracted signals
    # This must pass before accepting ANY wedge
    if wedge_name:
        aligned, reason = _verify_evidence_alignment(
            wedge_name,
            workflow,
            failure_mode,
            consequence,
            candidate.evidence_snippet or '',
        )
        if not aligned:
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append(f"evidence_mismatch: {reason}")
            return False, reason

    # PLATFORM CHECKS (existing logic)

    # STRICT: Category-level wedges are ALWAYS rejected - no exceptions
    # This applies regardless of platform confidence
    if is_category:
        candidate.output_bucket = "rejected"
        candidate.rejection_reasons.append("category_label")
        return False, "category_label"

    # Explicit platform: strong gate with specific wedge
    if platform_conf == 'explicit':
        if decision_score >= 0.5 or problem_truth >= 0.5:
            candidate.output_bucket = "accepted"
            candidate.why_passed = "explicit_platform_strong_validation_specific"
            return True, "accepted"
        elif decision_score >= 0.3 or problem_truth >= 0.4:
            candidate.output_bucket = "accepted"
            candidate.why_passed = "explicit_platform_adequate_validation_specific"
            return True, "accepted"

    # Inferred platform: needs stronger validation, goes to research-needed
    elif platform_conf == 'inferred':
        if decision_score >= 0.3 or problem_truth >= 0.4:
            candidate.output_bucket = "research_needed"
            candidate.why_passed = "inferred_platform_research_needed"
            return True, "research_needed"

    # Unknown platform: needs exceptional validation OR goes to research
    else:  # unknown
        if decision_score >= 0.6 and problem_truth >= 0.6 and candidate.corroboration_score >= 0.5:
            candidate.output_bucket = "research_needed"
            candidate.why_passed = "unknown_platform_exceptional"
            return True, "research_needed"
        else:
            candidate.output_bucket = "rejected"
            candidate.rejection_reasons.append("missing_platform")
            return False, "missing_platform"

    # Default rejection
    candidate.output_bucket = "rejected"
    candidate.rejection_reasons.append("wedge_gate_failed")
    return False, "wedge_gate_failed"


# =============================================================================
# STAGE 5: PRODUCT MAPPING - Assign product format
# =============================================================================

def map_product_format(candidate: WedgeCandidate) -> None:
    """Map wedge to product format based on platform."""
    candidate.current_stage = "product_mapping"

    if candidate.platform != 'unknown':
        candidate.product_format = PLATFORM_FORMATS.get(candidate.platform, 'microSaaS')
    else:
        candidate.product_format = 'microSaaS'


# =============================================================================
# STAGE 6: RANKING - Prioritize based on upstream signals
# =============================================================================

def rank_wedge(candidate: WedgeCandidate) -> float:
    """Rank wedge using upstream signals.

    Ranking prioritizes:
    - platform explicitness
    - user specificity
    - workflow specificity
    - trigger clarity
    - consequence severity
    - corroboration (multi-signal evidence)
    - buyer clarity
    - buildability
    - distribution access
    """
    # Platform explicitness is primary (0-0.3)
    platform_score = 0.0
    if candidate.platform_confidence == 'explicit':
        platform_score = 0.3
    elif candidate.platform_confidence == 'inferred':
        platform_score = 0.15

    # Validation scores (0-0.5)
    validation_score = min(0.5, candidate.decision_score * 0.5)

    # Evidence quality (0-0.2)
    evidence_score = min(0.2, candidate.evidence_quality_score * 0.2)

    # Buildability bonus (0-0.1)
    buildability_score = min(0.1, candidate.buildability * 0.1)

    # Corroboration bonus (0-0.1)
    corroboration_score = min(0.1, candidate.corroboration_score * 0.1)

    total = platform_score + validation_score + evidence_score + buildability_score + corroboration_score
    candidate.confidence = min(1.0, total)

    # Set evidence quality label
    if candidate.confidence >= 0.7 and candidate.platform_confidence == 'explicit':
        candidate.evidence_quality = 'high'
    elif candidate.confidence >= 0.5:
        candidate.evidence_quality = 'medium'
    else:
        candidate.evidence_quality = 'low'

    return candidate.confidence


def run_pipeline_stage(candidate: WedgeCandidate) -> WedgeCandidate:
    """Run full pipeline on a candidate."""
    # Stage 2: Validation
    valid, reason = validate_candidate(candidate)
    if not valid:
        candidate.rejection_reasons.append(reason)
        candidate.output_bucket = "rejected"
        return candidate

    # Stage 3: Sharpening
    sharp, reason = sharpen_candidate(candidate)
    if not sharp:
        candidate.rejection_reasons.append(reason)
        if candidate.platform_confidence == 'unknown':
            candidate.output_bucket = "rejected"
        return candidate

    # Stage 4: Gate
    gated, reason = gate_wedge(candidate)
    if not gated:
        candidate.rejection_reasons.append(reason)
        return candidate

    # Stage 5: Product mapping
    map_product_format(candidate)

    # Stage 6: Ranking
    rank_wedge(candidate)

    return candidate


def generate_wedges(db: Database, limit: int = 10) -> list[ProductIdea]:
    """Generate platform-explicit decision-grade wedges.

    This is now a STRICT EVALUATOR, not a generator.
    Uses ALL upstream validation signals.
    """
    conn = db._get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute(_build_wedge_query(db)).fetchall()

    candidates: list[WedgeCandidate] = []
    rejected_counts: dict[str, int] = {}
    stats = {'explicit': 0, 'inferred': 0, 'unknown': 0}

    for r in rows:
        # Extract candidate with validation scores
        candidate = extract_candidate_from_opportunity(r)

        # Track platform stats
        if candidate.platform_confidence == 'explicit':
            stats['explicit'] += 1
        elif candidate.platform_confidence == 'inferred':
            stats['inferred'] += 1
        else:
            stats['unknown'] += 1

        # Run pipeline stages
        candidate = run_pipeline_stage(candidate)

        # Collect rejection reasons
        for reason in candidate.rejection_reasons:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        if candidate.output_bucket != "rejected":
            candidates.append(candidate)

    # Sort by confidence
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    # Convert to ProductIdea for backwards compatibility
    accepted: list[ProductIdea] = []
    seen_names = set()

    for c in candidates:
        if c.wedge_name in seen_names:
            rejected_counts['duplicate'] = rejected_counts.get('duplicate', 0) + 1
            continue

        # Skip if not accepted bucket
        if c.output_bucket != "accepted":
            continue

        seen_names.add(c.wedge_name)

        idea = ProductIdea(
            name=c.wedge_name,
            confidence=round(c.confidence, 2),
            evidence_quality=c.evidence_quality,
            user=c.user,
            workflow=c.workflow,
            trigger_moment=c.trigger_moment,
            failure_mode=c.failure_mode,
            consequence=c.consequence,
            host_platform=c.platform,
            platform_confidence=c.platform_confidence,
            product_format=c.product_format,
            value_proposition=c.value_proposition,
            why_wedge_not_category=c.value_proposition,
            evidence_snippets=[c.evidence_snippet[:200]] if c.evidence_snippet else [],
            suggested_mvp=f"{c.product_format} that {c.value_proposition.lower()}",
            why_passed=c.why_passed,
        )
        accepted.append(idea)

        if len(accepted) >= limit:
            break

    logger.info(f"Wedges: {len(accepted)} accepted. Stats: {stats}. Rejections: {rejected_counts}")
    return accepted


def save_wedges(
    ideas: list[ProductIdea],
    output_path: str = "data/product_ideas.json",
    candidates: list[WedgeCandidate] = None,
    stats: dict = None,
    rejected_counts: dict = None
) -> dict:
    """Save decision-grade wedges with full pipeline output."""
    output = {
        "accepted_wedges": [],
        "research_needed_wedges": [],
        "funnel_metrics": {},
        "platform_breakdown": {},
        "rejected_reasons": {},
    }

    # Accepted wedges
    for idea in ideas:
        output["accepted_wedges"].append({
            "name": idea.name,
            "confidence": round(idea.confidence, 2),
            "evidence_quality": idea.evidence_quality,
            "user": idea.user,
            "workflow": idea.workflow,
            "trigger_moment": idea.trigger_moment,
            "failure_mode": idea.failure_mode,
            "consequence": idea.consequence,
            "host_platform": idea.host_platform,
            "platform_confidence": idea.platform_confidence,
            "product_format": idea.product_format,
            "value_proposition": idea.value_proposition,
            "why_wedge_not_category": idea.why_wedge_not_category,
            "evidence_snippets": idea.evidence_snippets,
            "suggested_mvp": idea.suggested_mvp,
            "why_passed": idea.why_passed,
        })

    # Research-needed wedges (from candidates)
    if candidates:
        research_needed = [c for c in candidates if c.output_bucket == "research_needed"]
        for c in research_needed:
            output["research_needed_wedges"].append({
                "name": c.wedge_name or c.workflow,
                "confidence": round(c.confidence, 2),
                "user": c.user,
                "workflow": c.workflow,
                "host_platform": c.platform,
                "platform_confidence": c.platform_confidence,
                "decision_score": round(c.decision_score, 3),
                "problem_truth_score": round(c.problem_truth_score, 3),
                "why_passed": c.why_passed,
            })

    # Funnel metrics
    if stats and candidates:
        total_opportunities = len(candidates)
        validated = len([c for c in candidates if c.current_stage != "extraction"])
        sharpened = len([c for c in candidates if c.current_stage in ("sharpening", "wedge_gate", "product_mapping", "ranking")])
        gated = len([c for c in candidates if c.output_bucket in ("accepted", "research_needed")])
        accepted = len(ideas)

        output["funnel_metrics"] = {
            "total_opportunities": total_opportunities,
            "validated_candidates": validated,
            "sharpened_candidates": sharpened,
            "gate_passes": gated,
            "accepted_wedges": accepted,
            "research_needed": len(research_needed),
        }

    # Platform breakdown
    if stats:
        output["platform_breakdown"] = stats

    # Rejection reasons
    if rejected_counts:
        output["rejected_reasons"] = rejected_counts

    Path(output_path).write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(ideas)} accepted wedges to {output_path}")
    return output


def run_product_idea_generation(db: Database, limit: int = 10) -> tuple[list[ProductIdea], dict]:
    """Run product idea generation with full pipeline output."""
    ideas, candidates, stats, rejected_counts = generate_wedges_with_metrics(db, limit)
    save_wedges(ideas, candidates=candidates, stats=stats, rejected_counts=rejected_counts)
    return ideas, {
        "candidates": candidates,
        "stats": stats,
        "rejected_counts": rejected_counts,
    }


def generate_wedges_with_metrics(db: Database, limit: int = 10) -> tuple[list[ProductIdea], list[WedgeCandidate], dict, dict]:
    """Generate wedges with full metrics for output."""
    conn = db._get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute(_build_wedge_query(db)).fetchall()

    candidates: list[WedgeCandidate] = []
    rejected_counts: dict[str, int] = {}
    stats = {'explicit': 0, 'inferred': 0, 'unknown': 0}

    for r in rows:
        candidate = extract_candidate_from_opportunity(r)

        if candidate.platform_confidence == 'explicit':
            stats['explicit'] += 1
        elif candidate.platform_confidence == 'inferred':
            stats['inferred'] += 1
        else:
            stats['unknown'] += 1

        candidate = run_pipeline_stage(candidate)

        for reason in candidate.rejection_reasons:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        if candidate.output_bucket != "rejected":
            candidates.append(candidate)

    candidates.sort(key=lambda c: c.confidence, reverse=True)

    accepted: list[ProductIdea] = []
    seen_names = set()

    for c in candidates:
        if c.wedge_name in seen_names:
            rejected_counts['duplicate'] = rejected_counts.get('duplicate', 0) + 1
            continue

        if c.output_bucket != "accepted":
            continue

        seen_names.add(c.wedge_name)

        idea = ProductIdea(
            name=c.wedge_name,
            confidence=round(c.confidence, 2),
            evidence_quality=c.evidence_quality,
            user=c.user,
            workflow=c.workflow,
            trigger_moment=c.trigger_moment,
            failure_mode=c.failure_mode,
            consequence=c.consequence,
            host_platform=c.platform,
            platform_confidence=c.platform_confidence,
            product_format=c.product_format,
            value_proposition=c.value_proposition,
            why_wedge_not_category=c.value_proposition,
            evidence_snippets=[c.evidence_snippet[:200]] if c.evidence_snippet else [],
            suggested_mvp=f"{c.product_format} that {c.value_proposition.lower()}",
            why_passed=c.why_passed,
        )
        accepted.append(idea)

        if len(accepted) >= limit:
            break

    logger.info(f"Wedges: {len(accepted)} accepted. Stats: {stats}. Rejections: {rejected_counts}")
    return accepted, candidates, stats, rejected_counts


# Add Path import
from pathlib import Path
