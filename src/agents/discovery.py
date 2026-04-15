"""Discovery agent for recurring pain signals and evidence-first qualification."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from collections import defaultdict
from typing import Any, Dict, Optional

from src.agents.base import AgentStatus, BaseAgent
from src.database import Database, Finding, ProblemAtom, RawSignal
from src.discovery_queries import (
    reddit_discovery_subreddits,
    reddit_problem_keywords,
    reddit_success_keywords,
)
from src.messaging import MessageQueue, MessageType
from src.opportunity_engine import (
    build_problem_atom,
    build_raw_signal_payload,
    classify_source_signal,
    qualify_problem_signal,
)
from src.reddit_seed import RedditSeeder
from src.research_tools import DiscoveryQueryPlan, ResearchToolkit
from src.discovery_expander import run_expansion, get_expanded_config
from src.discovery_term_lifecycle import (
    TermLifecycleManager,
    TermMetrics,
    calculate_consequence_score,
    calculate_platform_native_score,
    calculate_plugin_fit_score,
    calculate_quality_score,
    calculate_specificity_score,
    calculate_wedge_quality_score,
    compute_next_state,
    recompute_wedge_quality_score,
)
from src.high_leverage import score_high_leverage_finding
from src.llm_discovery_expander import LLMDiscoveryExpander
from src.problem_space import EXPLORING, VALIDATED
from src.problem_space_lifecycle import ProblemSpaceLifecycleManager
from src.source_patterns import (
    MANUAL_WORKFLOW_HINTS as CANONICAL_MANUAL_WORKFLOW_HINTS,
    MANUAL_WORKFLOW_STAKES_HINTS as CANONICAL_MANUAL_WORKFLOW_STAKES_HINTS,
    contains_any_phrase,
)
from src.utils.hashing import generate_content_hash


# =============================================================================
# PRE-ATOM SIGNAL FILTER - Reject weak signals before atom creation
# =============================================================================

CONCRETE_OBJECTS = {
    'invoice', 'invoices', 'payment', 'payments', 'order', 'orders',
    'row', 'rows', 'record', 'records', 'entry', 'entries',
    'file', 'files', 'transaction', 'transactions',
    'sheet', 'sheets', 'cell', 'cells', 'formula', 'formulas',
    'item', 'items', 'product', 'products', 'customer', 'customers',
    'email', 'emails', 'data', 'import', 'export',
    'spreadsheet', 'csv', 'report', 'reports',
}

FAILURE_VERBS = {
    'missing', 'missed', 'mismatch', 'mismatched',
    'duplicate', 'duplicated', 'duplicates',
    'not matching', 'does not match', 'dont match', "don't match",
    'incorrect', 'wrong', 'error', 'errors', 'failed', 'fail',
    'break', 'broken', 'out of sync', 'not updating',
    'lost', 'late', 'inconsistent', 'corrupt',
}

BUSINESS_RISK_PATTERNS = {
    "biggest client",
    "major client",
    "client concentration",
    "revenue concentration",
    "owes me",
    "good fit for me",
    "review my resume",
    "resume review",
    "resume roast",
    "career advice",
    "first sales",
    "hiring first sales",
    "se fue un cliente",
}

OPERATIONAL_WEDGE_HINTS = {
    "reconciliation",
    "reconcile",
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
    "supplier data",
    "returns",
}

GENERIC_TASK_BUCKETS = {
    "manual tasks",
    "repetitive tasks",
    "routine business tasks",
    "daily tasks",
    "basic follow-ups",
    "follow-ups",
    "data entry",
    "moving info between tools",
    "moving information between tools",
    "reporting",
    "scheduling",
    "admin work",
    "administrative work",
    "operations work",
    "day-to-day operations",
    "every single day",
    "wasting hours",
    "hours every week",
    "hours every day",
}

SPECIFIC_CONTEXT_HINTS = {
    "csv",
    "invoice",
    "invoices",
    "receipt",
    "receipts",
    "stripe",
    "quickbooks",
    "shopify",
    "google reviews",
    "google review",
    "order received",
    "label printed",
    "bank payment",
    "bank payments",
    "bank deposit",
    "reconciliation",
    "expense",
    "expenses",
    "vendor",
    "vendors",
    "payout",
    "fulfillment",
    "fulfilment",
    "returns",
    "chargeback",
    "duplicate",
    "mismatch",
    "missing",
    "import",
    "export",
}

GENERIC_PROMPT_PATTERNS = [
    r"what'?s the one task in your business",
    r"if you'?re still doing this manually in your business",
    r"if your team is still doing this manually",
    r"most of these aren'?t hard problems",
]

FINANCE_BROAD_PROMPT_PATTERNS = [
    r"looking for the best virtual credit card",
    r"looking for (?:a|the) corporate card solution",
    r"corporate card solution that can automate this",
    r"we(?:'ve| have) looked at standard options",
    r"will this finally fix vendor payments",
    r"\bjust bought\b.*\bmelio\b",
    r"unified cash position across multiple payment channels",
    r"single up-to-date view of (?:what'?s|what is) been received",
]

SPECIFIC_FINANCE_FAILURE_HINTS = {
    "reconcile",
    "reconciliation",
    "mismatch",
    "match",
    "matching",
    "duplicate",
    "duplicates",
    "csv",
    "import",
    "export",
    "refund",
    "refunded",
    "wrong dates",
    "fees not separated",
    "shared ledger",
}

PRACTITIONER_RECONCILIATION_PROMPTS = {
    "is anyone doing this",
    "anyone doing this",
    "curious how people handle",
    "how are people handling",
    "how do you handle",
}

MANUAL_WORKFLOW_HINTS = list(CANONICAL_MANUAL_WORKFLOW_HINTS)
MANUAL_WORKFLOW_STAKES_HINTS = list(CANONICAL_MANUAL_WORKFLOW_STAKES_HINTS)

META_PATTERNS = [
    r'^what (is|are) ', r'^how do i ', r'^how to ',
    r'looking to build', r'want to build', r'building a',
    r'best tool', r'recommend', r'any suggestions',
    r'i need a tool', r'looking for software',
    r'automation idea', r'automate this',
    r'productivity hack', r'workflow tip',
]


def is_wedge_ready_signal(finding_data: Dict[str, Any]) -> tuple[bool, str]:
    """Check if a finding contains a wedge-ready signal.

    Returns (is_ready, rejection_reason).
    A signal is wedge-ready if it contains:
    - At least one concrete object
    - A failure/mismatch pattern
    - A workflow context (import, export, copy, sync, reconcile)

    Rejects:
    - Meta discussion posts
    - Generic productivity talk
    - Too short (< 20 words)
    - No concrete objects
    - No failure patterns
    """
    # Support multiple field names - product_built/title for title, outcome_summary/body_excerpt for body
    title = (finding_data.get('product_built') or finding_data.get('title') or '').lower()
    body = (finding_data.get('outcome_summary') or finding_data.get('body_excerpt') or '').lower()
    text = f'{title} {body}'.replace("’", "'")

    # Check 1: Too short
    if len(text.strip()) < 30:
        return False, "too_short"

    # Check 2: Meta discussion pattern
    for pattern in META_PATTERNS:
        if re.search(pattern, text):
            return False, "meta_post"

    if any(phrase in text for phrase in BUSINESS_RISK_PATTERNS):
        has_operational_anchor = any(hint in text for hint in OPERATIONAL_WEDGE_HINTS)
        if not has_operational_anchor:
            return False, "business_risk_or_career_post"

    generic_bucket_hits = sum(1 for phrase in GENERIC_TASK_BUCKETS if phrase in text)
    has_specific_context = any(hint in text for hint in SPECIFIC_CONTEXT_HINTS)
    has_failure = any(verb in text for verb in FAILURE_VERBS)
    has_manual_workflow = contains_any_phrase(text, MANUAL_WORKFLOW_HINTS)
    has_manual_stakes = contains_any_phrase(text, MANUAL_WORKFLOW_STAKES_HINTS)
    strong_manual_workflow_slice = has_specific_context and has_manual_workflow and has_manual_stakes
    practitioner_reconciliation_question = (
        contains_any_phrase(text, PRACTITIONER_RECONCILIATION_PROMPTS)
        and has_specific_context
        and has_manual_workflow
        and any(hint in text for hint in SPECIFIC_FINANCE_FAILURE_HINTS)
    )
    narrow_operator_slice = strong_manual_workflow_slice or practitioner_reconciliation_question

    # Check 2b: Broad productivity sermons and multi-workflow bundles
    if any(re.search(pattern, text) for pattern in GENERIC_PROMPT_PATTERNS):
        return False, "generic_prompt"
    if any(re.search(pattern, text) for pattern in FINANCE_BROAD_PROMPT_PATTERNS):
        has_specific_finance_failure = any(hint in text for hint in SPECIFIC_FINANCE_FAILURE_HINTS)
        if not has_specific_finance_failure:
            return False, "broad_finance_prompt"
    if generic_bucket_hits >= 2 and not has_specific_context:
        return False, "generic_task_bundle"
    if generic_bucket_hits >= 2 and not has_failure and ("," in text or "etc" in text or " and " in text):
        return False, "multi_workflow_bundle"

    # Check 3: Contains concrete object
    has_object = any(obj in text for obj in CONCRETE_OBJECTS)
    if not has_object and not narrow_operator_slice:
        return False, "no_concrete_object"

    # Check 4: Contains failure verb/pattern
    if not has_failure and not narrow_operator_slice:
        return False, "no_failure_pattern"

    # Check 5: Generic productivity talk (without specific failure)
    generic_productivity = ['productivity', 'efficiency', 'workflow', 'automation']
    if all(word in text for word in generic_productivity) and not has_failure:
        return False, "generic_productivity"

    return True, "passed"


# Continue with original imports and rest of file...

logger = logging.getLogger(__name__)


def _serialize_atom_json(atom_payload: dict[str, Any]) -> str:
    atom_json = atom_payload.get("atom_json")
    if isinstance(atom_json, str):
        return atom_json or "{}"
    if atom_json is None:
        atom_json = {
            key: atom_payload.get(key)
            for key in (
                "cluster_key",
                "segment",
                "user_role",
                "job_to_be_done",
                "trigger_event",
                "pain_statement",
                "failure_mode",
                "current_workaround",
                "current_tools",
                "urgency_clues",
                "frequency_clues",
                "emotional_intensity",
                "cost_consequence_clues",
                "why_now_clues",
                "confidence",
                "platform",
                "specificity_score",
                "consequence_score",
                "atom_extraction_method",
            )
        }
    return json.dumps(atom_json)


def _normalize_term_list(value: Any, *, subreddit: bool = False) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        items = [piece for piece in re.split(r"[\n,]+", value) if piece]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if subreddit:
            text = re.sub(r"^/?r/", "", text, flags=re.IGNORECASE).strip()
            if len(text) < 2:
                continue
        if text not in normalized:
            normalized.append(text)
    return normalized


DISCOVERY_THEME_RULES = [
    {
        "theme_key": "workflow_fragility",
        "label": "Workflow fragility and spreadsheet glue",
        "terms": [
            "duct tape",
            "spreadsheet",
            "spreadsheets",
            "latest version",
            "latest file",
            "out of sync",
            "handoff",
            "missed step",
            "copy paste",
        ],
        "query_seeds": [
            "duct tape spreadsheets",
            "manual handoff workflow",
            "latest spreadsheet version confusion",
            "outgrown spreadsheets operations",
        ],
    },
    {
        "theme_key": "manual_reconciliation",
        "label": "Manual reconciliation and reporting burden",
        "terms": [
            "manual reconciliation",
            "reconciliation",
            "csv import",
            "manual entry",
            "spreadsheet cleanup",
            "merging reports manually",
        ],
        "query_seeds": [
            "manual reconciliation workflow",
            "spreadsheet reconciliation process",
            "csv import cleanup workflow",
        ],
    },
    {
        "theme_key": "finance_close_ops",
        "label": "Finance close and bank matching operations",
        "terms": [
            "month end close",
            "bank deposit",
            "bank deposits",
            "payout export",
            "channel profitability",
            "sales channel",
            "close checklist",
        ],
        "query_seeds": [
            "month end close spreadsheet",
            "bank deposit reconciliation spreadsheet",
            "sales channel reconciliation spreadsheet",
            "spreadsheet close checklist",
        ],
    },
    {
        "theme_key": "ecommerce_ops_handoffs",
        "label": "Ecommerce fulfillment and returns handoffs",
        "terms": [
            "label printed",
            "order received",
            "returns workflow",
            "manual label generation",
            "supplier data",
            "product data workflow",
            "fulfilment",
            "fulfillment",
        ],
        "query_seeds": [
            '"order received" "label printed" whatsapp spreadsheet',
            "returns workflow spreadsheet",
            "supplier data spreadsheet workflow",
            "manual label generation returns",
        ],
    },
    {
        "theme_key": "audit_export_ops",
        "label": "Manual audit/export operations",
        "terms": [
            "audit exports",
            "merge manually",
            "compliance data",
            "m365",
            "export manually",
            "slow and messy",
        ],
        "query_seeds": [
            "manual audit exports workflow",
            "merge manually compliance data",
            "m365 audit export manual process",
            "manual audit evidence collection",
            "compliance export evidence workflow",
        ],
    },
]


class DiscoveryAgent(BaseAgent):
    """Discovers pain signals, screens them, and persists evidence-first artifacts."""

    def __init__(
        self,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        sources: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        status_tracker: Optional[Any] = None,
        bypass_cache: bool = False,
    ):
        super().__init__("discovery", message_queue)
        self.db = db
        self.base_config = config or {}
        # Apply expanded config at init
        self.config = get_expanded_config(self.base_config)
        self.sources = sources if sources is not None else ["youtube", "reddit", "github"]
        self.check_interval = self.config.get("discovery", {}).get("check_interval", 300)
        self.toolkit = ResearchToolkit(self.config)
        self._seen_hashes: set[str] = set()
        self.status_tracker = status_tracker
        self._cycle_health: dict[tuple[str, str], dict[str, Any]] = {}
        self._cycle_strategy: dict[str, dict[str, Any]] = {}
        self._cycle_counts: dict[str, int] = defaultdict(int)
        self._last_reddit_seed_summary: dict[str, Any] = {}
        self.bypass_cache = bypass_cache
        # Term lifecycle manager for forward+reverse search space control
        self.term_lifecycle = TermLifecycleManager(db, self.config)
        # LLM-driven discovery expansion
        self._llm_expansion_cycle = 0
        self._llm_expander: LLMDiscoveryExpander | None = None
        self._space_lifecycle: ProblemSpaceLifecycleManager | None = None
        llm_config = self.config.get("discovery", {}).get("llm_expansion", {})
        if llm_config.get("enabled", False):
            self._llm_expander = LLMDiscoveryExpander(db, self.config)
            self._space_lifecycle = ProblemSpaceLifecycleManager(db, self.config)
        # Rate-limiting semaphore: caps concurrent outbound HTTP calls
        _sem_limit = int(self.config.get("discovery", {}).get("api_concurrency", 6))
        self._api_semaphore = asyncio.Semaphore(max(1, _sem_limit))
        self._sync_sources_from_config()

    def _sync_sources_from_config(self) -> None:
        configured_sources = [
            str(source).lower()
            for source in (self.config.get("discovery", {}).get("sources", []) or [])
            if str(source).strip()
        ]
        if configured_sources:
            self.sources = configured_sources

    async def _refresh_toolkit(self, new_config: dict[str, Any]) -> None:
        old_toolkit = self.toolkit
        self.config = new_config
        self.check_interval = self.config.get("discovery", {}).get("check_interval", 300)
        self._sync_sources_from_config()
        self.toolkit = ResearchToolkit(self.config)
        close_old = getattr(old_toolkit, "close", None)
        if callable(close_old):
            try:
                await close_old()
            except Exception as exc:
                logger.warning("error closing previous discovery toolkit during refresh: %s", exc)

    def _screened_out_retention_limit(self) -> int:
        retention = self.config.get("discovery", {}).get("screened_out_retention", {}) or {}
        try:
            return max(0, int(retention.get("max_findings", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _enforce_screened_out_retention(self) -> int:
        keep_limit = self._screened_out_retention_limit()
        if keep_limit <= 0:
            return 0
        trimmed = self.db.trim_screened_out_findings(keep_limit)
        if trimmed:
            logger.info("trimmed %s screened_out findings to enforce retention limit=%s", trimmed, keep_limit)
        return trimmed

    async def _run_loop(self) -> None:
        while self.status in (AgentStatus.RUNNING, AgentStatus.PAUSED):
            try:
                await self._pause_event.wait()
                if self.status == AgentStatus.STOPPED:
                    break

                findings = await self._discover_once()
                logger.info("discovery cycle produced %s findings", len(findings))
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("discovery cycle failed: %s", exc)
                self._error_count += 1
                if self._error_count >= self._max_errors:
                    self.status = AgentStatus.ERROR
                    break

    async def _discover_once(self) -> list[int]:
        # Clear seen hashes from previous cycles to allow dedup within a single cycle only
        self._seen_hashes.clear()
        self._sync_sources_from_config()
        # Run expansion to add new keywords/subreddits based on previous wave's feedback
        try:
            expansion_result = run_expansion(self.db, self.base_config)
            if expansion_result.get("expanded"):
                logger.info(f"Discovery expanded: +{len(expansion_result.get('added_keywords', []))} keywords, "
                            f"+{len(expansion_result.get('added_subreddits', []))} subreddits")
                # Refresh toolkit with expanded config so the new scope is used immediately.
                await self._refresh_toolkit(get_expanded_config(self.base_config))
        except Exception as e:
            logger.warning(f"Expansion failed: {e}")

        self._load_learning_feedback()
        self._cycle_health = {}
        self._cycle_strategy = {}
        planned_sources = self._planned_sources_for_cycle()
        if self.status_tracker:
            skipped_sources = [source for source in self.sources if source.lower() not in planned_sources]
            self.status_tracker.log(
                f"source_selection active={','.join(planned_sources)} skipped={','.join(skipped_sources)}"
            )
        prime_task = asyncio.create_task(self._prime_reddit_relay())

        async def _gated_check(source: str) -> list[dict[str, Any]]:
            async with self._api_semaphore:
                return await self._check_source(source)

        grouped_results = await asyncio.gather(
            *(_gated_check(source) for source in planned_sources),
            return_exceptions=True,
        )
        finding_ids: list[int] = []
        for result in grouped_results:
            if isinstance(result, Exception):
                continue
            for finding_data in result:
                finding_id = await self._process_finding(finding_data)
                stored_finding = self.db.get_finding(finding_id) if finding_id is not None else None
                if finding_id is not None and stored_finding and stored_finding.status != "screened_out":
                    finding_ids.append(finding_id)
        if not prime_task.done():
            prime_task.cancel()
        try:
            await prime_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("reddit relay priming task failed: %s", exc)
        self._persist_cycle_health()
        self._publish_cycle_health()
        # Update term lifecycle states based on this wave's results
        self._update_term_lifecycle()
        self._enforce_screened_out_retention()
        # LLM-driven problem space expansion (after validation cycle)
        await self._run_llm_expansion()
        return finding_ids

    def _feedback_source_names(self, normalized_source: str) -> list[str]:
        mapping = {
            "reddit": ["reddit-problem"],
            "web": ["web-problem", "market-problem", "web-success"],
            "github": ["github-problem"],
            "wordpress_reviews": ["wordpress-reviews"],
            "shopify_reviews": ["shopify-reviews"],
            "youtube": ["youtube-success"],
            "youtube-comments": ["youtube-comments"],
        }
        return mapping.get(normalized_source, [normalized_source])

    def _feedback_totals_for_source(self, normalized_source: str) -> dict[str, int]:
        totals = {
            "runs": 0,
            "findings_emitted": 0,
            "validations": 0,
            "prototype_candidates": 0,
            "build_briefs": 0,
        }
        for source_name in self._feedback_source_names(normalized_source):
            for row in (self.toolkit._discovery_feedback.get(source_name, {}) or {}).values():
                for key in totals:
                    totals[key] += int(row.get(key, 0) or 0)
        return totals

    def _source_selection_settings(self) -> tuple[set[str], int, int]:
        selection = self.config.get("discovery", {}).get("source_selection", {}) or {}
        always_run = {
            str(source).lower()
            for source in selection.get("always_run", ["reddit", "web"])
            if str(source).strip()
        }
        exploratory_slots = max(0, int(selection.get("exploratory_low_yield_sources_per_cycle", 2)))
        min_runs = max(1, int(selection.get("low_yield_min_runs", 50)))
        return always_run, exploratory_slots, min_runs

    def _rotation_offset(self, key: str, size: int) -> int:
        if size <= 0:
            return 0
        token = f"{self.db.get_active_run_id() or datetime.now(UTC).isoformat()}:{key}"
        return sum(ord(char) for char in token) % size

    def _planned_sources_for_cycle(self) -> list[str]:
        always_run, exploratory_slots, min_runs = self._source_selection_settings()
        selected: list[str] = []
        exploratory: list[str] = []

        for source in [str(item).lower() for item in self.sources if str(item).strip()]:
            if source in always_run:
                selected.append(source)
                continue
            totals = self._feedback_totals_for_source(source)
            low_yield = (
                totals["runs"] >= min_runs
                and totals["validations"] == 0
                and totals["prototype_candidates"] == 0
                and totals["build_briefs"] == 0
            )
            if low_yield:
                exploratory.append(source)
            else:
                selected.append(source)

        if exploratory and exploratory_slots > 0:
            offset = self._rotation_offset("discovery-source-exploration", len(exploratory))
            rotated = exploratory[offset:] + exploratory[:offset]
            selected.extend(rotated[: min(exploratory_slots, len(rotated))])

        deduped: list[str] = []
        for source in selected:
            if source not in deduped:
                deduped.append(source)
        return deduped

    async def _check_source(self, source: str) -> list[dict[str, Any]]:
        normalized = source.lower()
        observer = self._make_probe_observer()
        if normalized == "youtube":
            queries = self._plan_queries(
                "youtube-success",
                self.config.get("discovery", {}).get("youtube", {}).get(
                    "keywords",
                    ["AI business", "AI startup revenue", "make money with AI"],
                ),
                default_limit=4,
            )
            return await self.toolkit._discover_youtube_successes(keywords=queries, observer=observer)
        if normalized == "youtube-comments":
            queries = self._plan_queries(
                "youtube-comments",
                self.config.get("discovery", {}).get("youtube_comments", {}).get(
                    "keywords",
                    ["shopify app review", "shopify problems", "ecommerce tools"],
                ),
                default_limit=3,
            )
            return await self.toolkit._discover_youtube_comments(keywords=queries, observer=observer)
        if normalized == "reddit":
            reddit_subreddits = reddit_discovery_subreddits(self.config)
            success_candidates = reddit_success_keywords(self.config)
            success_queries = (
                self._plan_queries("reddit-success", success_candidates, default_limit=3)
                if success_candidates
                else []
            )
            problem_queries = self._plan_queries(
                "reddit-problem",
                reddit_problem_keywords(self.config),
                default_limit=4,
            )
            if success_queries:
                success_findings, problem_findings = await asyncio.gather(
                    self.toolkit._discover_reddit_successes(
                        subreddits=reddit_subreddits,
                        keywords=success_queries,
                        observer=observer,
                    ),
                    self.toolkit._discover_reddit_problem_threads(
                        subreddits=reddit_subreddits,
                        queries=problem_queries,
                        observer=observer,
                    ),
                )
            else:
                success_findings = []
                problem_findings = await self.toolkit._discover_reddit_problem_threads(
                    subreddits=reddit_subreddits,
                    queries=problem_queries,
                    observer=observer,
                )
            return success_findings + problem_findings
        if normalized == "github":
            if self._should_skip_github_discovery():
                if observer:
                    observer(
                        {
                            "source_name": "github-problem",
                            "query_text": "[source-skipped]",
                            "docs_seen": 0,
                            "latency_ms": 0.0,
                            "status": "ok",
                        }
                    )
                logger.info("github discovery skipped after repeated zero-yield feedback")
                return []
            queries = self._plan_queries(
                "github-problem",
                self.config.get("discovery", {}).get("github", {}).get(
                    "problem_keywords",
                    [
                        '"csv import" issue workflow',
                        '"manual reconciliation" issue',
                        '"spreadsheet workflow" issue',
                        '"copy paste" automation issue',
                        '"data cleanup" import issue',
                    ],
                ),
                default_limit=4,
            )
            github_timeout = self._adaptive_github_timeout_seconds()
            try:
                return await asyncio.wait_for(
                    self.toolkit._discover_github_problem_threads(queries=queries, observer=observer),
                    timeout=github_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("github discovery timed out after %ss; continuing without GitHub findings", github_timeout)
                if observer:
                    observer(
                        {
                            "source_name": "github-problem",
                            "query_text": "[source-timeout]",
                            "docs_seen": 0,
                            "latency_ms": round(github_timeout * 1000, 2),
                            "status": "error",
                            "error": "github discovery timeout",
                        }
                    )
                return []
        if normalized == "web":
            web_cfg = self.config.get("discovery", {}).get("web", {}) or {}
            focused_problem_only = bool(self.config.get("discovery", {}).get("focused_problem_only", False))
            success_queries = self._plan_queries(
                "web-success",
                web_cfg.get(
                    "success_keywords",
                    web_cfg.get(
                        "keywords",
                        [
                            "AI startup success story revenue",
                            "GPT business revenue story",
                            "AI side project customers",
                        ],
                    ),
                ),
                default_limit=4,
            ) if not focused_problem_only else []
            problem_queries = self._plan_queries(
                "web-problem",
                web_cfg.get(
                    "problem_keywords",
                    [
                        "spreadsheet version confusion forum",
                        "manual reconciliation forum",
                        "manual handoff workflow forum",
                        "workflow handoff tool too expensive",
                    ],
                ),
                default_limit=4,
            )
            market_queries = self._plan_queries(
                "market-problem",
                web_cfg.get(
                    "market_keywords",
                    [
                        '"etsy seller" "wish there was" automation',
                        '"google reviews" "too expensive" tool',
                        '"youtube comments" "need a way" automate',
                    ],
                ),
                default_limit=3,
            ) if not focused_problem_only else []
            success_timeout = self._web_timeout_seconds("success", default=10.0)
            market_timeout = self._web_timeout_seconds("market", default=8.0)
            problem_timeout = self._web_timeout_seconds("problem", default=15.0)
            success_findings: list[dict[str, Any]] = []
            problem_findings: list[dict[str, Any]] = []
            if not focused_problem_only:
                success_findings, problem_findings = await asyncio.gather(
                    self._run_source_with_timeout(
                        "web-success",
                        self.toolkit._discover_success_stories_on_web(
                            queries=success_queries,
                            observer=observer,
                        ),
                        timeout_seconds=success_timeout,
                        observer=observer,
                    ),
                    self._run_source_with_timeout(
                        "market-problem",
                        self.toolkit._discover_marketplace_problem_threads(
                            queries=market_queries,
                            observer=observer,
                        ),
                        timeout_seconds=market_timeout,
                        observer=observer,
                    ),
                )
            web_problem_findings = await self._run_source_with_timeout(
                "web-problem",
                self.toolkit._discover_web_problem_threads(
                    queries=problem_queries,
                    observer=observer,
                ),
                timeout_seconds=problem_timeout,
                observer=observer,
            )
            return success_findings + problem_findings + web_problem_findings
        if normalized == "wordpress_reviews":
            return await self.toolkit._discover_wordpress_review_threads(observer=observer)
        if normalized == "shopify_reviews":
            return await self.toolkit._discover_shopify_review_threads(observer=observer)
        return []

    async def _prime_reddit_relay(self) -> None:
        relay_config = self.config.get("reddit_relay", {})
        bridge_config = self.config.get("reddit_bridge", {})
        if "reddit" not in self.sources:
            return
        if not bridge_config.get("enabled", False):
            return
        if not relay_config.get("auto_seed_on_discovery", True):
            return

        subreddits = reddit_discovery_subreddits(self.config)
        reddit_config = self.config.get("discovery", {}).get("reddit", {}) or {}
        try:
            max_subs_raw = int(reddit_config.get("max_subreddits_per_wave", len(subreddits) or 1))
        except (TypeError, ValueError):
            max_subs_raw = len(subreddits) or 1
        subreddit_limit = len(subreddits) if max_subs_raw <= 0 else min(len(subreddits), max(1, max_subs_raw))
        subreddit_cycle_key = "reddit-relay-seed-subreddits"
        subreddit_cycle_index = self._cycle_counts.get(subreddit_cycle_key, 0)
        subreddit_plan = self.toolkit.build_discovery_query_plan(
            subreddit_cycle_key,
            list(subreddits),
            limit=subreddit_limit,
            cycle_index=subreddit_cycle_index,
        )
        subreddits = list(subreddit_plan.queries)
        if subreddits:
            self._cycle_counts[subreddit_cycle_key] = subreddit_cycle_index + 1
        queries = self._plan_queries(
            "reddit-relay-seed",
            reddit_problem_keywords(self.config),
            default_limit=max(4, len(reddit_problem_keywords(self.config))),
        )
        learned_queries, _theme_keys = self._learned_theme_queries("reddit-problem")
        seed_limit = max(
            1,
            int(
                reddit_config.get(
                    "reddit_seed_query_limit",
                    self.config.get("discovery", {}).get("reddit_seed_query_limit", 8),
                )
            ),
        )
        reserved_learned = max(0, int(self.config.get("discovery", {}).get("theme_query_limit_per_cycle", 2)))
        learned_seed_queries: list[str] = []
        for query in learned_queries:
            if query not in learned_seed_queries:
                learned_seed_queries.append(query)
            if len(learned_seed_queries) >= min(seed_limit, reserved_learned):
                break
        remaining_seed_slots = max(0, seed_limit - len(learned_seed_queries))
        seed_plan = self.toolkit.build_discovery_query_plan(
            "reddit-relay-seed",
            list(queries),
            limit=min(remaining_seed_slots, len(queries)),
            cycle_index=self._cycle_counts.get("reddit-relay-seed", 0),
        )
        merged_queries: list[str] = []
        for query in [*learned_seed_queries, *seed_plan.queries]:
            if query not in merged_queries:
                merged_queries.append(query)
        queries = merged_queries[:seed_limit]
        self._cycle_counts["reddit-relay-seed"] = self._cycle_counts.get("reddit-relay-seed", 0) + 1
        try:
            seeder = RedditSeeder(self.config, bypass_cache=self.bypass_cache)
            summary = await seeder.seed(subreddits=subreddits, queries=queries)
            self._last_reddit_seed_summary = {
                "seeded_total_pairs": summary.total_pairs,
                "seeded_pairs_searched": summary.searched_pairs,
                "seeded_pairs_fresh": summary.skipped_fresh_pairs,
                "seeded_pairs_existing_cache": summary.existing_cached_pairs,
                "seeded_pairs_uncovered": summary.uncovered_pairs,
                "seeded_cached_searches": summary.cached_searches,
                "seeded_cached_threads": summary.cached_threads,
                "seeded_thread_cache_hits": summary.thread_cache_hits,
                "seeded_unique_urls": summary.unique_urls,
            }
            logger.info(
                "reddit relay primed total_pairs=%s searched_pairs=%s cached_searches=%s cached_threads=%s uncovered_pairs=%s",
                summary.total_pairs,
                summary.searched_pairs,
                summary.cached_searches,
                summary.cached_threads,
                summary.uncovered_pairs,
            )
            if self.status_tracker:
                self.status_tracker.log(
                    f"reddit_relay_seed total_pairs={summary.total_pairs} searched_pairs={summary.searched_pairs} cached_threads={summary.cached_threads} uncovered_pairs={summary.uncovered_pairs}"
                )
        except Exception as exc:
            logger.warning("reddit relay auto-seed failed: %s", exc)
            if self.status_tracker:
                self.status_tracker.log(f"reddit_relay_seed_failed error={exc}")

    def reddit_runtime_summary(self) -> dict[str, Any]:
        return {
            **self.toolkit.get_reddit_runtime_metrics(),
            **self._last_reddit_seed_summary,
        }

    async def _process_finding(self, finding_data: Dict[str, Any]) -> Optional[int]:
        content_hash = self._generate_content_hash(finding_data)
        if content_hash in self._seen_hashes:
            return None

        existing = self.db.get_finding_by_hash(content_hash)
        if existing:
            self._seen_hashes.add(content_hash)
            return None

        evidence = dict(finding_data.get("evidence", {}) or {})
        evidence.setdefault("run_id", self.db.get_active_run_id())
        discovery_query = evidence.get("discovery_query")
        source_plan = evidence.get("source_plan")

        # PART 5: PRE-ATOM FILTER - Reject weak signals before atom creation
        is_ready, reject_reason = is_wedge_ready_signal(finding_data)
        if not is_ready:
            evidence["pre_atom_filter"] = {"accepted": False, "reason": reject_reason}
            evidence["screening"] = {
                "accepted": False,
                "score": 0.0,
                "positive_signals": [],
                "negative_signals": [f"pre_atom_filter:{reject_reason}"],
                "source_class": "low_signal_summary",
            }
            finding = Finding(
                source=finding_data.get("source", "unknown"),
                source_url=finding_data.get("source_url", ""),
                entrepreneur=finding_data.get("entrepreneur"),
                tool_used=finding_data.get("tool_used"),
                product_built=finding_data.get("product_built"),
                monetization_method=finding_data.get("monetization_method"),
                outcome_summary=finding_data.get("outcome_summary"),
                content_hash=content_hash,
                status="screened_out",
                finding_kind=finding_data.get("finding_kind", "problem_signal"),
                source_class="low_signal_summary",
                recurrence_key=finding_data.get("recurrence_key"),
                evidence=evidence,
            )
            finding_id = self.db.insert_finding(finding)
            self._seen_hashes.add(content_hash)
            if source_plan and discovery_query:
                self.db.record_discovery_screening(
                    source_plan,
                    discovery_query,
                    accepted=False,
                    source_class="low_signal_summary",
                    screening_score=0.0,
                )
            self._enforce_screened_out_retention()
            logger.debug("filtered weak signal %s as screened_out finding %s", reject_reason, finding_id)
            return finding_id

        signal_payload = build_raw_signal_payload(finding_data)
        atom_payload = build_problem_atom(signal_payload, finding_data)
        source_classification = classify_source_signal(finding_data, signal_payload, atom_payload)
        finding_data["source_class"] = source_classification["source_class"]
        signal_payload.setdefault("metadata_json", {})["source_class"] = source_classification["source_class"]
        screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)
        screening["source_class"] = source_classification["source_class"]
        evidence["screening"] = screening
        evidence["source_classification"] = source_classification

        finding = Finding(
            source=finding_data.get("source", "unknown"),
            source_url=finding_data.get("source_url", ""),
            entrepreneur=finding_data.get("entrepreneur"),
            tool_used=finding_data.get("tool_used"),
            product_built=finding_data.get("product_built"),
            monetization_method=finding_data.get("monetization_method"),
            outcome_summary=finding_data.get("outcome_summary"),
            content_hash=content_hash,
            status="qualified" if screening["accepted"] else "screened_out",
            finding_kind=finding_data.get("finding_kind", "problem_signal"),
            source_class=source_classification["source_class"],
            recurrence_key=finding_data.get("recurrence_key"),
            evidence=evidence,
        )

        temp_signal = RawSignal(
            finding_id=0,
            source_name=signal_payload["source_name"],
            source_type=signal_payload["source_type"],
            source_class=source_classification["source_class"],
            source_url=signal_payload["source_url"],
            title=signal_payload["title"],
            body_excerpt=signal_payload["body_excerpt"],
            quote_text=signal_payload["quote_text"],
            role_hint=signal_payload["role_hint"],
            published_at=signal_payload["published_at"],
            timestamp_hint=signal_payload["timestamp_hint"],
            content_hash=content_hash,
            metadata=signal_payload["metadata_json"],
        )
        temp_atom = ProblemAtom(
            signal_id=0,
            finding_id=0,
            cluster_key=atom_payload["cluster_key"],
            segment=atom_payload["segment"],
            user_role=atom_payload["user_role"],
            job_to_be_done=atom_payload["job_to_be_done"],
            trigger_event=atom_payload["trigger_event"],
            pain_statement=atom_payload["pain_statement"],
            failure_mode=atom_payload["failure_mode"],
            current_workaround=atom_payload["current_workaround"],
            current_tools=atom_payload["current_tools"],
            urgency_clues=atom_payload["urgency_clues"],
            frequency_clues=atom_payload["frequency_clues"],
            emotional_intensity=atom_payload["emotional_intensity"],
            cost_consequence_clues=atom_payload["cost_consequence_clues"],
            why_now_clues=atom_payload["why_now_clues"],
            confidence=atom_payload["confidence"],
            platform=atom_payload.get("platform", ""),
            specificity_score=atom_payload.get("specificity_score", 0.0),
            consequence_score=atom_payload.get("consequence_score", 0.0),
            atom_extraction_method=atom_payload.get("atom_extraction_method", "heuristic"),
            atom_json=_serialize_atom_json(atom_payload),
        )
        high_leverage = score_high_leverage_finding(finding, temp_signal, temp_atom, evidence)
        evidence["high_leverage"] = high_leverage
        finding.evidence = evidence
        finding.evidence_json = json.dumps(evidence)

        finding_id = self.db.insert_finding(finding)
        self._seen_hashes.add(content_hash)
        if source_plan and discovery_query:
            self.db.record_discovery_screening(
                source_plan,
                discovery_query,
                accepted=bool(screening["accepted"]),
                source_class=source_classification["source_class"],
                screening_score=float(screening.get("score", 0.0) or 0.0),
            )
        if not screening["accepted"]:
            logger.info(
                "screened out finding %s (%s): score=%s negatives=%s",
                finding_id,
                finding.finding_kind,
                screening["score"],
                ",".join(screening["negative_signals"]),
            )
            if self.status_tracker:
                self.status_tracker.log(
                    f"screened_out finding={finding_id} score={screening['score']} negatives={','.join(screening['negative_signals']) or 'none'}"
                )
            self._enforce_screened_out_retention()
            return finding_id

        signal_payload.setdefault("metadata_json", {})["high_leverage"] = high_leverage

        signal = RawSignal(
            finding_id=finding_id,
            source_name=signal_payload["source_name"],
            source_type=signal_payload["source_type"],
            source_class=source_classification["source_class"],
            source_url=signal_payload["source_url"],
            title=signal_payload["title"],
            body_excerpt=signal_payload["body_excerpt"],
            quote_text=signal_payload["quote_text"],
            role_hint=signal_payload["role_hint"],
            published_at=signal_payload["published_at"],
            timestamp_hint=signal_payload["timestamp_hint"],
            content_hash=content_hash,
            metadata=signal_payload["metadata_json"],
        )
        signal_id = self.db.insert_raw_signal(signal)

        atom = ProblemAtom(
            signal_id=signal_id,
            finding_id=finding_id,
            cluster_key=atom_payload["cluster_key"],
            segment=atom_payload["segment"],
            user_role=atom_payload["user_role"],
            job_to_be_done=atom_payload["job_to_be_done"],
            trigger_event=atom_payload["trigger_event"],
            pain_statement=atom_payload["pain_statement"],
            failure_mode=atom_payload["failure_mode"],
            current_workaround=atom_payload["current_workaround"],
            current_tools=atom_payload["current_tools"],
            urgency_clues=atom_payload["urgency_clues"],
            frequency_clues=atom_payload["frequency_clues"],
            emotional_intensity=atom_payload["emotional_intensity"],
            cost_consequence_clues=atom_payload["cost_consequence_clues"],
            why_now_clues=atom_payload["why_now_clues"],
            confidence=atom_payload["confidence"],
            platform=atom_payload.get("platform", ""),
            specificity_score=atom_payload.get("specificity_score", 0.0),
            consequence_score=atom_payload.get("consequence_score", 0.0),
            atom_extraction_method=atom_payload.get("atom_extraction_method", "heuristic"),
            atom_json=_serialize_atom_json(atom_payload),
        )
        atom_id = self.db.insert_problem_atom(atom)

        if source_plan and discovery_query:
            self.db.record_discovery_hit(source_plan, discovery_query)
            key = (source_plan, discovery_query)
            bucket = self._cycle_health.setdefault(
                key,
                {
                    "source": source_plan,
                    "query": discovery_query,
                    "docs_seen": 0,
                    "findings_emitted": 0,
                    "latency_ms": 0.0,
                    "errors": 0,
                    "status": "ok",
                },
            )
            bucket["findings_emitted"] += 1

        logger.info(
            "stored finding %s (%s): %s",
            finding_id,
            finding.finding_kind,
            (finding.product_built or "")[:80],
        )
        if self.status_tracker:
            self.status_tracker.log(
                f"qualified finding={finding_id} signal={signal_id} atom={atom_id} score={screening['score']}"
            )

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.FINDING,
            payload={
                "finding_id": finding_id,
                "source": finding.source,
                "source_url": finding.source_url,
                "content_hash": content_hash,
                "finding_kind": finding.finding_kind,
                "source_class": finding.source_class,
                "title": finding.product_built,
                "summary": finding.outcome_summary,
                "signal_id": signal_id,
                "problem_atom_ids": [atom_id],
            },
            priority=2,
        )
        return finding_id

    def _load_learning_feedback(self) -> None:
        rows = self.db.get_discovery_feedback()
        self._refresh_query_cooldowns(rows)
        self.toolkit.set_discovery_feedback(self.db.get_discovery_feedback())
        self._refresh_learned_themes()

    def _refresh_query_cooldowns(self, rows: list[dict[str, Any]]) -> None:
        discovery_config = self.config.get("discovery", {}) or {}
        cooldown_hours = max(1, int(discovery_config.get("query_cooldown_hours", 12)))
        min_runs = max(2, int(discovery_config.get("query_quarantine_min_runs", 3)))
        now = datetime.now(UTC)
        family_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            source_name = str(row.get("source_name", "") or "")
            query_text = str(row.get("query_text", "") or "")
            if not source_name or not query_text:
                continue
            family_key = self.toolkit.discovery_query_family_key(query_text)
            family_rows[(source_name, family_key)].append(row)
            runs = int(row.get("runs", 0) or 0)
            findings = int(row.get("findings_emitted", 0) or 0)
            validations = int(row.get("validations", 0) or 0)
            passes = int(row.get("passes", 0) or 0)
            prototype_candidates = int(row.get("prototype_candidates", 0) or 0)
            build_briefs = int(row.get("build_briefs", 0) or 0)
            screened_out = int(row.get("screened_out", 0) or 0)
            low_signal_count = int(row.get("low_signal_count", 0) or 0)
            thin_recurrence_count = int(row.get("thin_recurrence_count", 0) or 0)
            single_source_only_count = int(row.get("single_source_only_count", 0) or 0)
            cooldown_until = str(row.get("cooldown_until", "") or "").strip()

            low_yield_noise = (
                runs >= min_runs
                and findings == 0
                and validations == 0
                and screened_out >= min_runs
            )
            thin_validation_trap = (
                runs >= min_runs
                and validations >= min_runs
                and passes == 0
                and prototype_candidates == 0
                and build_briefs == 0
                and (thin_recurrence_count >= min_runs or single_source_only_count >= min_runs or low_signal_count >= min_runs)
            )
            should_cooldown = low_yield_noise or thin_validation_trap

            if should_cooldown:
                active_cooldown = False
                if cooldown_until:
                    try:
                        cooldown_dt = datetime.fromisoformat(cooldown_until)
                        if cooldown_dt.tzinfo is None:
                            cooldown_dt = cooldown_dt.replace(tzinfo=UTC)
                        active_cooldown = cooldown_dt > now
                    except (ValueError, TypeError):
                        active_cooldown = False
                if not active_cooldown:
                    self.db.set_discovery_query_cooldown(
                        source_name,
                        query_text,
                        (now + timedelta(hours=cooldown_hours)).isoformat(),
                    )

        family_decay_hours = max(cooldown_hours + 6, int(discovery_config.get("query_family_decay_hours", cooldown_hours * 2)))
        family_min_queries = max(2, int(discovery_config.get("query_family_decay_min_queries", 2)))
        for (source_name, family_key), family in family_rows.items():
            if not family_key or len(family) < family_min_queries:
                continue
            total_runs = sum(int(row.get("runs", 0) or 0) for row in family)
            total_passes = sum(int(row.get("passes", 0) or 0) for row in family)
            total_prototypes = sum(int(row.get("prototype_candidates", 0) or 0) for row in family)
            total_build_briefs = sum(int(row.get("build_briefs", 0) or 0) for row in family)
            total_promotes = sum(int(row.get("promotes", 0) or 0) for row in family)
            total_thin = sum(int(row.get("thin_recurrence_count", 0) or 0) for row in family)
            total_single_source = sum(int(row.get("single_source_only_count", 0) or 0) for row in family)
            total_parks = sum(int(row.get("parks", 0) or 0) for row in family)
            if (
                total_runs < max(min_runs + 1, family_min_queries + 1)
                or total_passes > 0
                or total_prototypes > 0
                or total_build_briefs > 0
                or total_promotes > 0
                or (total_thin + total_single_source + total_parks) < max(min_runs + 1, 4)
            ):
                continue
            for row in family:
                query_text = str(row.get("query_text", "") or "")
                cooldown_until = str(row.get("cooldown_until", "") or "").strip()
                active_cooldown = False
                if cooldown_until:
                    try:
                        cooldown_dt = datetime.fromisoformat(cooldown_until)
                        if cooldown_dt.tzinfo is None:
                            cooldown_dt = cooldown_dt.replace(tzinfo=UTC)
                        active_cooldown = cooldown_dt > now
                    except (ValueError, TypeError):
                        active_cooldown = False
                if active_cooldown:
                    continue
                self.db.set_discovery_query_cooldown(
                    source_name,
                    query_text,
                    (now + timedelta(hours=family_decay_hours)).isoformat(),
                )

    def _refresh_learned_themes(self) -> None:
        min_hits = max(1, int(self.config.get("discovery", {}).get("theme_min_hits", 2)))
        corpus = self._theme_learning_corpus()
        for rule in DISCOVERY_THEME_RULES:
            matched = [
                item
                for item in corpus
                if any(term in item["text"] for term in rule["terms"])
            ]
            weighted_hits = sum(int(item.get("weight", 1) or 1) for item in matched)
            if weighted_hits < min_hits:
                continue
            source_signals = [item["title"] for item in matched[:5] if item["title"]]
            query_seeds = self._theme_query_seeds_from_signals(
                rule["theme_key"],
                base_seeds=list(rule["query_seeds"]),
                source_signals=source_signals,
            )
            self.db.upsert_discovery_theme(
                rule["theme_key"],
                label=rule["label"],
                query_seeds=query_seeds,
                source_signals=source_signals,
                times_seen=weighted_hits,
                yield_score=round(min(1.0, weighted_hits / max(min_hits, 1)), 3),
                run_id=self.db.get_active_run_id(),
            )

    def _theme_query_seeds_from_signals(
        self,
        theme_key: str,
        *,
        base_seeds: list[str],
        source_signals: list[str],
    ) -> list[str]:
        deduped: list[str] = []

        def add(seed: str) -> None:
            normalized = " ".join(str(seed).strip().split()).lower()
            if normalized and normalized not in deduped:
                deduped.append(normalized)

        for seed in base_seeds:
            add(seed)

        signal_text = " ".join(source_signals).lower()
        if theme_key == "workflow_fragility":
            if any(term in signal_text for term in ["shared workbook", "edit conflict", "saving conflict"]):
                add("excel shared workbook conflict")
                add("shared spreadsheet saving conflicts")
            if any(term in signal_text for term in ["latest version", "wrong file", "version confusion", "changes not showing"]):
                add("latest spreadsheet version confusion")
                add("google sheets collaborator changes not showing")
            if any(term in signal_text for term in ["handoff", "copy", "manual", "out of sync"]):
                add("manual handoff workflow")
                add("copy paste workflow handoff")
        elif theme_key == "manual_reconciliation":
            if any(term in signal_text for term in ["reconciliation", "cleanup", "csv import"]):
                add("spreadsheet reconciliation errors")
                add("csv import cleanup workflow")
        elif theme_key == "audit_export_ops":
            if any(term in signal_text for term in ["audit", "evidence", "export", "compliance"]):
                add("manual audit evidence collection")
                add("compliance export evidence workflow")
        return deduped[:8]

    def _theme_learning_corpus(self) -> list[dict[str, Any]]:
        corpus: list[dict[str, Any]] = []
        findings = self.db.get_findings(limit=200)
        for finding in findings:
            if finding.status == "screened_out":
                continue
            text = " ".join(
                [
                    finding.product_built or "",
                    finding.outcome_summary or "",
                    str((finding.evidence or {}).get("source_classification", {})),
                ]
            ).lower()
            if not text.strip():
                continue
            corpus.append(
                {
                    "title": finding.product_built or finding.outcome_summary or "",
                    "text": text,
                    "weight": 1,
                }
            )
        for row in self.db.get_validation_review(limit=80):
            decision = str(row.get("decision", "") or "")
            selection_status = str(row.get("selection_status", "") or "")
            recurrence_state = str(row.get("recurrence_state", "") or "")
            if decision == "kill":
                continue
            if selection_status not in {"prototype_candidate", "research_more"}:
                continue
            if recurrence_state not in {"thin", "supported", "strong"}:
                continue
            matched_titles = [
                record.get("title", "")
                for records in (row.get("reviewable_recurrence_matches_by_source", {}) or {}).values()
                for record in records[:2]
                if record.get("title")
            ]
            text = " ".join(
                [
                    row.get("title", "") or "",
                    row.get("decision_reason", "") or "",
                    row.get("selection_reason", "") or "",
                    row.get("best_surfaced_evidence", {}).get("title", "") if isinstance(row.get("best_surfaced_evidence", {}), dict) else "",
                    " ".join(matched_titles),
                ]
            ).lower()
            if not text.strip():
                continue
            weight = 3 if selection_status == "prototype_candidate" else 2
            corpus.append(
                {
                    "title": row.get("title", "") or "",
                    "text": text,
                    "weight": weight,
                }
            )
        return corpus

    def _learned_theme_queries(self, source_name: str) -> tuple[list[str], list[str]]:
        if source_name not in {"reddit-problem", "web-problem", "github-problem", "market-problem"}:
            return [], []
        learned_limit = max(0, int(self.config.get("discovery", {}).get("theme_query_limit_per_cycle", 2)))
        if learned_limit <= 0:
            return [], []
        themes = self.db.list_active_discovery_themes(limit=10)
        query_seeds: list[str] = []
        theme_keys: list[str] = []
        for theme in themes:
            seeds = [str(seed) for seed in (theme.get("query_seeds") or []) if str(seed).strip()]
            if not seeds:
                continue
            theme_keys.append(str(theme.get("theme_key", "")))
            for seed in seeds:
                if source_name == "github-problem":
                    query_seeds.append(f"{seed} issue")
                elif source_name == "web-problem":
                    query_seeds.extend(self._web_shaped_theme_queries(str(theme.get("theme_key", "")), seed))
                elif source_name == "market-problem":
                    query_seeds.extend(self._market_shaped_theme_queries(str(theme.get("theme_key", "")), seed))
                else:
                    query_seeds.append(seed)
        deduped: list[str] = []
        for query in query_seeds:
            if query not in deduped:
                deduped.append(query)
        return deduped[: max(learned_limit * 4, learned_limit)], [key for key in theme_keys if key]

    def _adaptive_github_timeout_seconds(self) -> float:
        configured = float(self.config.get("discovery", {}).get("github", {}).get("timeout_seconds", 20))
        feedback_rows = list((self.toolkit._discovery_feedback.get("github-problem", {}) or {}).values())
        if len(feedback_rows) < 3:
            return configured
        low_yield_rows = [
            row
            for row in feedback_rows
            if int(row.get("runs", 0) or 0) >= 1
            and int(row.get("findings_emitted", 0) or 0) == 0
            and int(row.get("validations", 0) or 0) == 0
            and int(row.get("prototype_candidates", 0) or 0) == 0
            and int(row.get("build_briefs", 0) or 0) == 0
        ]
        if len(low_yield_rows) == len(feedback_rows):
            return min(configured, 6.0)
        return configured

    def _web_timeout_seconds(self, lane: str, *, default: float) -> float:
        web_config = self.config.get("discovery", {}).get("web", {}) or {}
        return float(web_config.get(f"{lane}_timeout_seconds", default))

    async def _run_source_with_timeout(
        self,
        source_name: str,
        coro: asyncio.Future,
        *,
        timeout_seconds: float,
        observer=None,
    ) -> list[dict[str, Any]]:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "%s discovery timed out after %ss; continuing without findings",
                source_name,
                timeout_seconds,
            )
            if observer:
                observer(
                    {
                        "source_name": source_name,
                        "query_text": "[source-timeout]",
                        "docs_seen": 0,
                        "latency_ms": round(timeout_seconds * 1000, 2),
                        "status": "error",
                        "error": f"{source_name} discovery timeout",
                    }
                )
            return []

    def _should_skip_github_discovery(self) -> bool:
        github_config = self.config.get("discovery", {}).get("github", {}) or {}
        if not bool(github_config.get("hard_skip_after_zero_yield", False)):
            return False
        feedback_rows = list((self.toolkit._discovery_feedback.get("github-problem", {}) or {}).values())
        if len(feedback_rows) < 4:
            return False
        return all(
            int(row.get("findings_emitted", 0) or 0) == 0
            and int(row.get("validations", 0) or 0) == 0
            and int(row.get("prototype_candidates", 0) or 0) == 0
            and int(row.get("build_briefs", 0) or 0) == 0
            for row in feedback_rows
        )

    def _web_shaped_theme_queries(self, theme_key: str, seed: str) -> list[str]:
        seed = str(seed).strip()
        if not seed:
            return []
        source_specific = {
            "workflow_fragility": [
                "spreadsheet version confusion forum",
                "manual handoff workflow forum",
                "outgrown spreadsheets operations",
            ],
            "manual_reconciliation": [
                "manual reconciliation forum",
                "csv import cleanup workflow",
            ],
            "audit_export_ops": [
                "manual audit exports forum",
                "compliance evidence collection workflow",
            ],
        }
        return [*source_specific.get(theme_key, []), seed]

    def _market_shaped_theme_queries(self, theme_key: str, seed: str) -> list[str]:
        seed = str(seed).strip()
        if not seed:
            return []
        source_specific = {
            "workflow_fragility": [
                "spreadsheet workflow software",
                "manual handoff workflow software",
                "spreadsheet version control for operations",
                "workflow handoff tool too expensive",
            ],
            "manual_reconciliation": [
                "reconciliation workflow software",
                "csv import cleanup software",
                "manual reconciliation tool too expensive",
            ],
            "audit_export_ops": [
                "audit evidence collection software",
                "compliance export workflow tool",
                "manual audit exports software",
            ],
        }
        return [*source_specific.get(theme_key, []), seed]

    def _discovery_slice_limit(self, source_name: str, default_limit: int) -> int:
        query_limits = self.config.get("discovery", {}).get("query_limits", {}) or {}
        raw_limit = query_limits.get(source_name)
        if raw_limit is None:
            return default_limit
        try:
            return max(0, int(raw_limit))
        except (TypeError, ValueError):
            return default_limit

    def _plan_queries(self, source_name: str, candidates: list[str], *, default_limit: int) -> list[str]:
        limit = self._discovery_slice_limit(source_name, default_limit)
        cycle_index = self._cycle_counts.get(source_name, 0)
        learned_queries, learned_theme_keys = self._learned_theme_queries(source_name)
        configured_learned_limit = max(0, int(self.config.get("discovery", {}).get("theme_query_limit_per_cycle", 2)))
        learned_limit = min(limit, configured_learned_limit)
        if cycle_index > 0 and learned_queries:
            learned_limit = min(limit, max(learned_limit, min(3, len(learned_queries))))
        theme_cycle_key = f"{source_name}:learned_theme"
        theme_cycle_index = self._cycle_counts.get(theme_cycle_key, 0)

        learned_plan = self.toolkit.build_discovery_query_plan(
            source_name,
            list(learned_queries),
            limit=learned_limit,
            cycle_index=theme_cycle_index,
        ) if learned_queries and learned_limit > 0 else DiscoveryQueryPlan(source_name=source_name, queries=[], slice_size=0)
        base_limit = max(0, limit - len(learned_plan.queries))
        plan = self.toolkit.build_discovery_query_plan(
            source_name,
            list(candidates),
            limit=base_limit if learned_plan.queries else limit,
            cycle_index=cycle_index,
        )
        merged: list[str] = []
        for query in [*learned_plan.queries, *plan.queries]:
            if query not in merged:
                merged.append(query)
        if candidates:
            self._cycle_counts[source_name] = cycle_index + 1
        if learned_queries:
            self._cycle_counts[theme_cycle_key] = theme_cycle_index + 1
        self._record_strategy(
            source_name,
            plan,
            learned_theme_keys=learned_theme_keys,
            learned_theme_queries=list(learned_plan.queries),
        )
        return merged[:limit]

    def _record_strategy(
        self,
        source_name: str,
        plan: DiscoveryQueryPlan,
        *,
        learned_theme_keys: Optional[list[str]] = None,
        learned_theme_queries: Optional[list[str]] = None,
    ) -> None:
        self._cycle_strategy[source_name] = {
            "source": source_name,
            "queries": list(plan.queries),
            "discovery_cycle_query_offset": plan.query_offset,
            "discovery_slice_size": plan.slice_size,
            "discovery_rotation_applied": plan.rotation_applied,
            "rotated_queries_used": list(plan.rotated_queries_used),
            "learned_theme_keys": list(learned_theme_keys or []),
            "learned_theme_queries": list(learned_theme_queries or []),
        }

    def _make_probe_observer(self):
        def _observer(payload: dict[str, Any]) -> None:
            source_name = payload.get("source_name", "")
            query_text = payload.get("query_text", "")
            if not source_name or not query_text:
                return
            key = (source_name, query_text)
            bucket = self._cycle_health.setdefault(
                key,
                {
                    "source": source_name,
                    "query": query_text,
                    "docs_seen": 0,
                    "findings_emitted": 0,
                    "latency_ms": 0.0,
                    "errors": 0,
                    "status": "ok",
                },
            )
            bucket["docs_seen"] += int(payload.get("docs_seen", 0) or 0)
            bucket["latency_ms"] += float(payload.get("latency_ms", 0.0) or 0.0)
            if payload.get("status") == "error":
                bucket["errors"] += 1
                bucket["status"] = "error"

        return _observer

    def _persist_cycle_health(self) -> None:
        for metric in self._cycle_health.values():
            self.db.record_discovery_probe(
                metric["source"],
                metric["query"],
                docs_seen=metric["docs_seen"],
                latency_ms=metric["latency_ms"],
                status=metric["status"],
                error="source probe error" if metric["errors"] else "",
            )

    def _publish_cycle_health(self) -> None:
        if not self.status_tracker:
            return
        by_source: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "source": "",
                "queries": 0,
                "docs_seen": 0,
                "findings_emitted": 0,
                "errors": 0,
                "latency_ms": 0.0,
            }
        )
        for metric in self._cycle_health.values():
            source_bucket = by_source[metric["source"]]
            source_bucket["source"] = metric["source"]
            source_bucket["queries"] += 1
            source_bucket["docs_seen"] += metric["docs_seen"]
            source_bucket["findings_emitted"] += metric["findings_emitted"]
            source_bucket["errors"] += metric["errors"]
            source_bucket["latency_ms"] += metric["latency_ms"]

        source_health = sorted(
            [
                {
                    **bucket,
                    "avg_latency_ms": round(bucket["latency_ms"] / max(bucket["queries"], 1), 2),
                }
                for bucket in by_source.values()
            ],
            key=lambda item: (-item["findings_emitted"], -item["docs_seen"], item["source"]),
        )
        strategy_entries = list(self._cycle_strategy.values())
        insights = self.toolkit.source_learning_insights([entry["source"] for entry in strategy_entries])
        self.status_tracker.update(
            sourceHealth=source_health,
            discoveryStrategy=strategy_entries,
            learningInsights=insights,
        )

    def _update_term_lifecycle(self) -> dict[str, Any]:
        """Update term lifecycle states based on cycle results.

        Synchronizes cumulative query feedback into term lifecycle state exactly once.
        """
        feedback_rows = self.db.get_discovery_feedback()

        keyword_metrics: dict[str, dict[str, float]] = defaultdict(lambda: {
            "runs": 0,
            "findings_emitted": 0,
            "validations": 0,
            "passes": 0,
            "prototype_candidates": 0,
            "build_briefs": 0,
            "screened_out": 0,
            "validation_score_total": 0.0,
            "screening_score_total": 0.0,
            "screening_weight": 0,
        })

        for row in feedback_rows:
            source_name = row.get("source_name", "")
            query_text = row.get("query_text", "")
            if not query_text or query_text == "[source-skipped]":
                continue

            if not source_name.startswith(("reddit", "web", "github", "market")):
                continue

            metrics = keyword_metrics[query_text]
            findings_emitted = int(row.get("findings_emitted", 0) or 0)
            validations = int(row.get("validations", 0) or 0)
            screened_out = int(row.get("screened_out", 0) or 0)
            metrics["runs"] += int(row.get("runs", 0) or 0)
            metrics["findings_emitted"] += findings_emitted
            metrics["validations"] += validations
            metrics["passes"] += int(row.get("passes", 0) or 0)
            metrics["prototype_candidates"] += int(row.get("prototype_candidates", 0) or 0)
            metrics["build_briefs"] += int(row.get("build_briefs", 0) or 0)
            metrics["screened_out"] += screened_out
            metrics["validation_score_total"] += float(row.get("avg_validation_score", 0.0) or 0.0) * validations
            metrics["screening_score_total"] += float(row.get("avg_screening_score", 0.0) or 0.0) * (
                findings_emitted + screened_out
            )
            metrics["screening_weight"] += findings_emitted + screened_out

        existing_terms = self.db.list_search_terms(limit=500)
        term_state_map = {(t["term_type"], t["term_value"]): t for t in existing_terms}

        transitions = {"keywords": [], "subreddits": []}
        for keyword, metrics in keyword_metrics.items():
            existing = term_state_map.get(("keyword", keyword), {})
            current_state = existing.get("state", "new") if existing else "new"
            term_metrics = TermMetrics(
                times_searched=int(metrics["runs"] or 0),
                findings_emitted=int(metrics["findings_emitted"] or 0),
                validations=int(metrics["validations"] or 0),
                passes=int(metrics["passes"] or 0),
                prototype_candidates=int(metrics["prototype_candidates"] or 0),
                build_briefs=int(metrics["build_briefs"] or 0),
                screened_out=int(metrics["screened_out"] or 0),
                low_yield_count=(
                    max(
                        int(existing.get("low_yield_count", 0) or 0),
                        int(metrics["runs"] or 0),
                    )
                    if int(metrics["findings_emitted"] or 0) == 0
                    else int(existing.get("low_yield_count", 0) or 0)
                ),
                noisy_count=int(existing.get("noisy_count", 0) or 0),
                thin_validation_count=int(existing.get("thin_validation_count", 0) or 0),
                avg_validation_score=(
                    float(metrics["validation_score_total"]) / float(metrics["validations"])
                    if metrics["validations"]
                    else 0.0
                ),
                avg_screening_score=(
                    float(metrics["screening_score_total"]) / float(metrics["screening_weight"])
                    if metrics["screening_weight"]
                    else 0.0
                ),
                waves_since_state_change=0,
            )
            term_metrics.quality_score = calculate_quality_score(term_metrics)

            specificity = calculate_specificity_score(keyword)
            consequence = calculate_consequence_score(keyword)
            platform_native = calculate_platform_native_score(keyword)
            plugin_fit = calculate_plugin_fit_score(keyword)
            heuristic_wedge = calculate_wedge_quality_score(
                keyword,
                specificity,
                consequence,
                platform_native,
                plugin_fit,
            )

            # Blend heuristic with actual outcome data
            buildable_count = int(existing.get("buildable_opportunity_count", 0) or 0)
            vague_count = int(existing.get("vague_bucket_count", 0) or 0)
            total_outcomes = buildable_count + vague_count + int(existing.get("screened_out", 0) or 0)
            wedge_quality = recompute_wedge_quality_score(
                heuristic_score=heuristic_wedge,
                buildable_opportunity_count=buildable_count,
                vague_bucket_count=vague_count,
                total_opportunities=total_outcomes,
            )

            self.term_lifecycle.ensure_term_exists("keyword", keyword)
            new_state, reason = compute_next_state(
                current_state,
                term_metrics,
                self.term_lifecycle.config,
                term_value=keyword,
            )
            self.db.update_search_term_state("keyword", keyword, new_state, notes=reason)
            self.db.update_search_term_metrics(
                "keyword",
                keyword,
                times_searched=term_metrics.times_searched,
                findings_emitted=term_metrics.findings_emitted,
                validations=term_metrics.validations,
                passes=term_metrics.passes,
                prototype_candidates=term_metrics.prototype_candidates,
                build_briefs=term_metrics.build_briefs,
                screened_out=term_metrics.screened_out,
                low_yield_count=term_metrics.low_yield_count,
                noisy_count=term_metrics.noisy_count,
                thin_validation_count=term_metrics.thin_validation_count,
                avg_validation_score=term_metrics.avg_validation_score,
                avg_screening_score=term_metrics.avg_screening_score,
                quality_score=term_metrics.quality_score,
                specificity_score=specificity,
                consequence_score=consequence,
                platform_native_score=platform_native,
                plugin_fit_score=plugin_fit,
                wedge_quality_score=wedge_quality,
                vague_bucket_count=int(existing.get("vague_bucket_count", 0) or 0),
                abstraction_collapse_count=int(existing.get("abstraction_collapse_count", 0) or 0),
                buildable_opportunity_count=int(existing.get("buildable_opportunity_count", 0) or 0),
                platform_native_count=int(existing.get("platform_native_count", 0) or 0),
            )
            if current_state != new_state:
                transitions["keywords"].append(
                    {
                        "term_type": "keyword",
                        "term_value": keyword,
                        "old_state": current_state,
                        "new_state": new_state,
                        "reason": reason,
                    }
                )

        logger.info(f"Term lifecycle: {len(transitions['keywords'])} keyword state changes")
        return transitions

    # =========================================================================
    # LLM-DRIVEN DISCOVERY EXPANSION
    # =========================================================================

    async def _run_llm_expansion(self) -> list:
        """Run LLM-driven problem space expansion if conditions are met.

        Called at the end of each discovery cycle. Tracks cycle count to
        respect trigger_after_cycles and trigger_interval_cycles.
        Returns the list of newly created ProblemSpace objects (empty if skipped).
        """
        self._llm_expansion_cycle += 1
        if not self._llm_expander:
            return []

        expansion_config = self.config.get("discovery", {}).get("llm_expansion", {})
        trigger_after = int(expansion_config.get("trigger_after_cycles", 2))
        trigger_interval = int(expansion_config.get("trigger_interval_cycles", 3))

        if self._llm_expansion_cycle < trigger_after:
            return []

        # Check if it's time for another expansion
        cycles_since_trigger = self._llm_expansion_cycle - trigger_after
        if cycles_since_trigger % trigger_interval != 0:
            return []

        try:
            new_spaces = await self._llm_expander.expand_after_validation()
            if new_spaces:
                logger.info(
                    "LLM expansion created %d new problem spaces: %s",
                    len(new_spaces),
                    ", ".join(s.space_key for s in new_spaces),
                )
                # Merge derived queries into discovery config for next cycle
                await self._merge_problem_space_queries(new_spaces)
                # Update space lifecycle metrics
                self._update_problem_space_lifecycle()
            return new_spaces
        except Exception as exc:
            logger.warning("LLM expansion failed: %s", exc)
            return []

    async def _merge_problem_space_queries(self, spaces: list) -> None:
        """Merge problem space derived queries into the discovery config.

        This makes the LLM-derived keywords, subreddits, and queries available
        to the next discovery wave through the existing expansion mechanism.
        """
        if not spaces:
            return

        discovery_cfg = self.config.get("discovery", {})
        reddit_cfg = discovery_cfg.get("reddit", {})
        base_discovery_cfg = self.base_config.setdefault("discovery", {})
        base_reddit_cfg = base_discovery_cfg.setdefault("reddit", {})

        # Collect all derived terms from new spaces
        new_keywords: list[str] = []
        new_subreddits: list[str] = []
        new_web_queries: list[str] = []
        new_github_queries: list[str] = []

        for space in spaces:
            if space.keywords:
                new_keywords.extend(_normalize_term_list(space.keywords))
            if space.subreddits:
                new_subreddits.extend(_normalize_term_list(space.subreddits, subreddit=True))
            if space.web_queries:
                new_web_queries.extend(_normalize_term_list(space.web_queries))
            if space.github_queries:
                new_github_queries.extend(_normalize_term_list(space.github_queries))

        # Merge into config (dedup)
        existing_keywords = _normalize_term_list(reddit_cfg.get("problem_keywords", []) or [])
        existing_subreddits = _normalize_term_list(reddit_cfg.get("problem_subreddits", []) or [], subreddit=True)

        merged_keywords = list(dict.fromkeys(existing_keywords + new_keywords))
        merged_subreddits = list(dict.fromkeys(existing_subreddits + new_subreddits))

        # Update config in-place so get_expanded_config picks them up
        reddit_cfg["problem_keywords"] = merged_keywords
        reddit_cfg["problem_subreddits"] = merged_subreddits
        base_reddit_cfg["problem_keywords"] = list(merged_keywords)
        base_reddit_cfg["problem_subreddits"] = list(merged_subreddits)

        # Also add web and github queries to their respective config sections
        web_cfg = discovery_cfg.get("web", {})
        base_web_cfg = base_discovery_cfg.setdefault("web", {})
        existing_web_keywords = _normalize_term_list(web_cfg.get("keywords", []) or [])
        merged_web = list(dict.fromkeys(existing_web_keywords + new_web_queries))
        web_cfg["keywords"] = merged_web
        base_web_cfg["keywords"] = list(merged_web)

        github_cfg = discovery_cfg.get("github", {})
        base_github_cfg = base_discovery_cfg.setdefault("github", {})
        existing_gh_keywords = _normalize_term_list(github_cfg.get("problem_keywords", []) or [])
        merged_gh = list(dict.fromkeys(existing_gh_keywords + new_github_queries))
        github_cfg["problem_keywords"] = merged_gh
        base_github_cfg["problem_keywords"] = list(merged_gh)

        logger.info(
            "Merged problem space queries: +%d keywords, +%d subreddits, +%d web, +%d github",
            len(new_keywords), len(new_subreddits),
            len(new_web_queries), len(new_github_queries),
        )

        # Push expanded subreddit list to relay so Devvit cron can seed them
        if new_subreddits and hasattr(self, 'toolkit') and hasattr(self.toolkit, 'reddit_bridge'):
            try:
                await self.toolkit.reddit_bridge.push_seed_subreddits(merged_subreddits)
            except Exception:
                pass  # non-critical; Devvit falls back to hardcoded list

        # Refresh toolkit with updated config
        await self._refresh_toolkit(get_expanded_config(self.base_config))

    def _update_problem_space_lifecycle(self) -> None:
        """Update problem space lifecycle states and metrics.

        Runs after each expansion cycle. Updates metrics for all active
        spaces and transitions their lifecycle state as needed.
        """
        if not self._space_lifecycle:
            return

        try:
            spaces = self._space_lifecycle.get_active_spaces(limit=100)
            for space in spaces:
                self._space_lifecycle.update_space_metrics(space.space_key)

                # Count idle cycles based on findings in recent runs
                space = self.db.get_problem_space(space.space_key)
                if not space:
                    continue

                # Determine idle cycles from metrics
                recent_findings = space.total_findings
                previous = getattr(space, "_prev_findings", recent_findings)
                idle_cycles = 0 if recent_findings > previous else getattr(space, "_idle_cycles", 0) + 1

                new_state = self._space_lifecycle.compute_next_state(space, idle_cycles=idle_cycles)
                if new_state != space.status:
                    self._space_lifecycle.transition_space(space, new_state)
                    self.db.update_problem_space_status(space.space_key, new_state)
                    logger.info(
                        "Problem space '%s' transitioned: %s → %s",
                        space.space_key, space.status, new_state,
                    )
        except Exception as exc:
            logger.warning("Problem space lifecycle update failed: %s", exc)

    def _generate_content_hash(self, finding_data: Dict[str, Any]) -> str:
        title = str(finding_data.get("product_built", "") or "")
        summary = str(finding_data.get("outcome_summary", "") or "")
        kind = str(finding_data.get("finding_kind", "") or "")
        source_url = str(finding_data.get("source_url", "") or "")
        content_basis = " ".join(part for part in [kind, title, summary] if part).strip()
        if len(content_basis) < 40:
            content_basis = f"{content_basis} {source_url}".strip()
        return generate_content_hash(content_basis)

    async def process(self, message) -> Dict[str, Any]:
        payload = message.payload
        command = payload.get("command")

        if command == "discover_now":
            finding_ids = await self._discover_once()
            return {"success": True, "finding_ids": finding_ids}

        if command == "add_source":
            source = payload.get("source")
            if not source:
                return {"success": False, "error": "source is required"}
            if source in self.sources:
                return {"success": False, "error": f"source already exists: {source}"}
            self.sources.append(source)
            return {"success": True, "sources": self.sources}

        if command == "remove_source":
            source = payload.get("source")
            if not source:
                return {"success": False, "error": "source is required"}
            if source not in self.sources:
                return {"success": False, "error": f"source not found: {source}"}
            self.sources.remove(source)
            return {"success": True, "sources": self.sources}

        if command == "set_interval":
            interval = payload.get("interval")
            if isinstance(interval, (int, float)) and interval > 0:
                self.check_interval = int(interval)
                return {"success": True, "interval": self.check_interval}
            return {"success": False, "error": "invalid interval"}

        if command == "get_sources":
            return {"sources": self.sources}

        return {"processed": True, "unknown_command": command}
