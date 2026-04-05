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

    # Platform detection
    source = _get_row_value(row, 'source_name', '') or ''
    url = _get_row_value(row, 'source_url', '') or ''
    full_text = f"{_get_row_value(row, 'title', '')} {_get_row_value(row, 'body_excerpt', '')}"

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

def _try_dynamic_sharpening(workflow: str, failure_mode: str, consequence: str, platform: str) -> dict | None:
    """Attempt to extract a wedge from vague descriptions using pattern matching.

    Returns dict with name, wedge, mvp or None if no sharpening possible.
    """
    pain_text = f"{workflow} {failure_mode} {consequence}".lower()

    # Pattern: "manually X" → "Automate X"
    match = re.search(r'manually\s+([a-z\s]{3,30})', pain_text)
    if match:
        task = match.group(1).strip()
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} {task.title()} Automator',
                'wedge': f'Automates manual {task}',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that automates {task}',
            }
        return {
            'name': f'{task.title()} Automator',
            'wedge': f'Automates manual {task}',
            'mvp': f'Script that automates {task}',
        }

    # Pattern: "spending X hours" → "Time Saver"
    match = re.search(r'spend(?:ing|s)?\s+(\d+)\s+hours?', pain_text)
    if match:
        hours = match.group(1)
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Time Saver',
                'wedge': f'Saves {hours} hours of manual work',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that saves {hours} hours',
            }
        return {
            'name': 'Time Saver',
            'wedge': f'Saves {hours} hours of manual work',
            'mvp': f'Automation that saves {hours} hours per week',
        }

    # Pattern: "error" / "mistake" → "Error Prevention"
    if 'error' in pain_text or 'mistake' in pain_text:
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Error Detector',
                'wedge': 'Prevents errors before they happen',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that catches errors',
            }
        return {
            'name': 'Error Prevention Tool',
            'wedge': 'Prevents common mistakes',
            'mvp': 'Tool that detects and prevents errors',
        }

    # Pattern: "duplicate" → "Deduplication"
    if 'duplicate' in pain_text:
        return {
            'name': 'Deduplication Tool',
            'wedge': 'Removes duplicate entries',
            'mvp': 'Tool that identifies and removes duplicates',
        }

    # Pattern: "missed" / "forgot" → "Reminder/Alert"
    if 'missed' in pain_text or 'forgot' in pain_text or 'forget' in pain_text:
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Reminder',
                'wedge': 'Prevents missed items',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that sends reminders',
            }
        return {
            'name': 'Smart Reminder',
            'wedge': 'Prevents missed tasks',
            'mvp': 'System that reminds about forgotten items',
        }

    # Pattern: "slow" → "Speed Optimizer"
    if 'slow' in pain_text or 'takes too long' in pain_text:
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Speed Optimizer',
                'wedge': 'Speeds up workflow',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that speeds up work',
            }
        return {
            'name': 'Workflow Speed Tool',
            'wedge': 'Speeds up manual processes',
            'mvp': 'Tool that accelerates workflow',
        }

    # Pattern: "sync" / "in sync" → "Sync Automation"
    if 'sync' in pain_text and 'manual' in pain_text:
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Sync',
                'wedge': 'Keeps data in sync automatically',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that syncs data',
            }
        return {
            'name': 'Data Sync Automation',
            'wedge': 'Keeps data synchronized',
            'mvp': 'Tool that syncs data between systems',
        }

    # Pattern: "export" / "import" → "Data Pipeline"
    if 'export' in pain_text or 'import' in pain_text:
        if platform != 'unknown':
            return {
                'name': f'{platform.title()} Data Pipeline',
                'wedge': 'Automates data import/export',
                'mvp': f'{PLATFORM_FORMATS.get(platform, "integration")} that moves data',
            }
        return {
            'name': 'Data Pipeline',
            'wedge': 'Automates data movement',
            'mvp': 'Tool that automates exports/imports',
        }

    # Pattern: "spreadsheet" → "Spreadsheet Automation"
    if 'spreadsheet' in pain_text or 'excel' in pain_text or 'sheet' in pain_text:
        if platform == 'excel':
            return {
                'name': 'Excel Automation',
                'wedge': 'Automates spreadsheet tasks',
                'mvp': 'Script that automates Excel workflows',
            }
        elif platform == 'google_sheets':
            return {
                'name': 'Google Sheets Automation',
                'wedge': 'Automates spreadsheet tasks',
                'mvp': 'Script that automates Sheets workflows',
            }
        return {
            'name': 'Spreadsheet Automation',
            'wedge': 'Automates spreadsheet tasks',
            'mvp': 'Tool that automates spreadsheet workflows',
        }

    return None


def sharpen_candidate(candidate: WedgeCandidate) -> tuple[bool, str]:
    """Sharpen candidate into platform-specific wedge.

    Uses workflow and failure_mode to determine specific wedge.
    Does NOT use synthetic generation - relies on extracted signals.
    Now attempts dynamic sharpening before giving up.
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
            return True, "sharpened"

    # Try dynamic sharpening - extract from vague descriptions
    dynamic = _try_dynamic_sharpening(workflow, failure_mode, consequence, platform)
    if dynamic:
        candidate.wedge_name = dynamic['name']
        candidate.value_proposition = dynamic['wedge']
        candidate.product_format = PLATFORM_FORMATS.get(platform, 'integration') if platform != 'unknown' else dynamic['mvp'].split(' ')[0]
        return True, "dynamic_sharpened"

    # Generic patterns for unknown platform (but with good validation scores)
    if 'match' in failure_mode or 'reconcil' in failure_mode:
        candidate.wedge_name = "Transaction Matcher"
        candidate.value_proposition = "Matches transactions across sources"
        candidate.product_format = "integration"
        return True, "generic_match"

    if 'copy' in failure_mode and 'paste' in failure_mode:
        candidate.wedge_name = "Data Paste Automator"
        candidate.value_proposition = "Transforms data when pasting"
        candidate.product_format = "tool"
        return True, "generic_copy_paste"

    # No sharpening possible
    candidate.rejection_reasons.append("no_sharpen")
    return False, "no_sharpen"


# =============================================================================
# STAGE 4: WEDGE GATE - Final pass/fail decision
# =============================================================================

def gate_wedge(candidate: WedgeCandidate) -> tuple[bool, str]:
    """Gate wedge based on platform explicitness and validation scores.

    Rules:
    - explicit platform → eligible for accepted
    - inferred platform → accepted only with strong evidence
    - unknown platform → goes to research-needed unless exceptional
    """
    candidate.current_stage = "wedge_gate"

    platform_conf = candidate.platform_confidence
    decision_score = candidate.decision_score
    problem_truth = candidate.problem_truth_score

    # Explicit platform: strong gate
    if platform_conf == 'explicit':
        if decision_score >= 0.5 or problem_truth >= 0.5:
            candidate.output_bucket = "accepted"
            candidate.why_passed = "explicit_platform_strong_validation"
            return True, "accepted"
        elif decision_score >= 0.3 or problem_truth >= 0.4:
            candidate.output_bucket = "accepted"
            candidate.why_passed = "explicit_platform_adequate_validation"
            return True, "accepted"

    # Inferred platform: needs stronger validation
    elif platform_conf == 'inferred':
        if decision_score >= 0.5 and problem_truth >= 0.5:
            candidate.output_bucket = "accepted"
            candidate.why_passed = "inferred_platform_strong_validation"
            return True, "accepted"
        elif decision_score >= 0.35 or problem_truth >= 0.45:
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

    # Query opportunities WITH validation scores - THIS IS THE KEY CHANGE
    # Join with problem_atoms and raw_signals to get extraction data
    rows = conn.execute('''
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
            pa.cost_consequence_clues,
            pa.segment,
            pa.user_role,
            rs.source_name,
            rs.source_url,
            rs.title as signal_title,
            rs.body_excerpt,
            rs.role_hint,
            rs.quote_text,
            c.segment as cluster_segment
        FROM opportunities o
        JOIN clusters c ON c.id = o.cluster_id
        LEFT JOIN problem_atoms pa ON pa.cluster_id = o.cluster_id
        LEFT JOIN raw_signals rs ON rs.id = pa.signal_id
        WHERE o.status IN ('promoted', 'active')
            AND (o.decision_score > 0 OR o.composite_score > 0)
        ORDER BY o.decision_score DESC, o.problem_truth_score DESC
        LIMIT 500
    ''').fetchall()

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

    rows = conn.execute('''
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
            pa.cost_consequence_clues,
            pa.segment,
            pa.user_role,
            rs.source_name,
            rs.source_url,
            rs.title as signal_title,
            rs.body_excerpt,
            rs.role_hint,
            rs.quote_text,
            c.segment as cluster_segment
        FROM opportunities o
        JOIN clusters c ON c.id = o.cluster_id
        LEFT JOIN problem_atoms pa ON pa.cluster_key = c.cluster_key
        LEFT JOIN raw_signals rs ON rs.id = pa.signal_id
        WHERE o.status IN ('promoted', 'active')
            AND (o.decision_score > 0 OR o.composite_score > 0)
        ORDER BY o.decision_score DESC, o.problem_truth_score DESC
        LIMIT 500
    ''').fetchall()

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