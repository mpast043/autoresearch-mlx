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
from src.discovery_term_lifecycle import TermLifecycleManager


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
    text = f'{title} {body}'

    # Check 1: Too short
    if len(text.strip()) < 30:
        return False, "too_short"

    # Check 2: Meta discussion pattern
    for pattern in META_PATTERNS:
        if re.search(pattern, text):
            return False, "meta_post"

    # Check 3: Contains concrete object
    has_object = any(obj in text for obj in CONCRETE_OBJECTS)
    if not has_object:
        return False, "no_concrete_object"

    # Check 4: Contains failure verb/pattern
    has_failure = any(verb in text for verb in FAILURE_VERBS)
    if not has_failure:
        return False, "no_failure_pattern"

    # Check 5: Generic productivity talk (without specific failure)
    generic_productivity = ['productivity', 'efficiency', 'workflow', 'automation']
    if all(word in text for word in generic_productivity) and not has_failure:
        return False, "generic_productivity"

    return True, "passed"


# Continue with original imports and rest of file...

logger = logging.getLogger(__name__)


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
            "which spreadsheet is latest",
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
        self.term_lifecycle = TermLifecycleManager(db)

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
        # Run expansion to add new keywords/subreddits based on previous wave's feedback
        try:
            expansion_result = run_expansion(self.db, self.base_config)
            if expansion_result.get("expanded"):
                logger.info(f"Discovery expanded: +{len(expansion_result.get('added_keywords', []))} keywords, "
                            f"+{len(expansion_result.get('added_subreddits', []))} subreddits")
                # Refresh toolkit with expanded config so the new scope is used immediately.
                self.config = get_expanded_config(self.base_config)
                self.check_interval = self.config.get("discovery", {}).get("check_interval", 300)
                self.toolkit = ResearchToolkit(self.config)
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
        grouped_results = await asyncio.gather(
            *(self._check_source(source) for source in planned_sources),
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
                [
                    '"csv import" issue workflow',
                    '"manual reconciliation" issue',
                    '"spreadsheet workflow" issue',
                    '"copy paste" automation issue',
                    '"data cleanup" import issue',
                ],
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
            success_queries = self._plan_queries(
                "web-success",
                self.config.get("discovery", {}).get("web", {}).get(
                    "keywords",
                    [
                        "AI startup success story revenue",
                        "GPT business revenue story",
                        "AI side project customers",
                    ],
                ),
                default_limit=4,
            )
            problem_queries = self._plan_queries(
                "web-problem",
                [
                    "spreadsheet version confusion forum",
                    "manual reconciliation forum",
                    "manual handoff workflow forum",
                    "which spreadsheet is latest",
                    "workflow handoff tool too expensive",
                ],
                default_limit=4,
            )
            market_queries = self._plan_queries(
                "market-problem",
                [
                    '"etsy seller" "wish there was" automation',
                    '"google reviews" "too expensive" tool',
                    '"youtube comments" "need a way" automate',
                ],
                default_limit=3,
            )
            success_timeout = self._web_timeout_seconds("success", default=10.0)
            market_timeout = self._web_timeout_seconds("market", default=8.0)
            problem_timeout = self._web_timeout_seconds("problem", default=15.0)
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
        if learned_queries:
            merged_queries: list[str] = []
            for query in [*learned_queries, *queries]:
                if query not in merged_queries:
                    merged_queries.append(query)
            queries = merged_queries
        seed_limit = max(
            1,
            int(
                reddit_config.get(
                    "reddit_seed_query_limit",
                    self.config.get("discovery", {}).get("reddit_seed_query_limit", 8),
                )
            ),
        )
        seed_plan = self.toolkit.build_discovery_query_plan(
            "reddit-relay-seed",
            list(queries),
            limit=min(seed_limit, len(queries)),
            cycle_index=self._cycle_counts.get("reddit-relay-seed", 0),
        )
        queries = list(seed_plan.queries)
        self._cycle_counts["reddit-relay-seed"] = self._cycle_counts.get("reddit-relay-seed", 0) + 1
        try:
            seeder = RedditSeeder(self.config, bypass_cache=self.bypass_cache)
            baseline_coverage = seeder.coverage_report(subreddits=subreddits, queries=queries)
            self._last_reddit_seed_summary = {
                "seeded_total_pairs": baseline_coverage.total_pairs,
                "seeded_pairs_searched": 0,
                "seeded_pairs_fresh": baseline_coverage.skipped_fresh_pairs,
                "seeded_pairs_existing_cache": baseline_coverage.existing_cached_pairs,
                "seeded_pairs_uncovered": baseline_coverage.uncovered_pairs,
                "seeded_cached_searches": 0,
                "seeded_cached_threads": 0,
                "seeded_thread_cache_hits": 0,
                "seeded_unique_urls": 0,
            }
            summary = await seeder.seed(subreddits=subreddits, queries=queries)
            coverage = seeder.coverage_report(subreddits=subreddits, queries=queries)
            self._last_reddit_seed_summary = {
                "seeded_total_pairs": summary.total_pairs,
                "seeded_pairs_searched": summary.searched_pairs,
                "seeded_pairs_fresh": summary.skipped_fresh_pairs,
                "seeded_pairs_existing_cache": summary.existing_cached_pairs,
                "seeded_pairs_uncovered": coverage.uncovered_pairs,
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
                coverage.uncovered_pairs,
            )
            if self.status_tracker:
                self.status_tracker.log(
                    f"reddit_relay_seed total_pairs={summary.total_pairs} searched_pairs={summary.searched_pairs} cached_threads={summary.cached_threads} uncovered_pairs={coverage.uncovered_pairs}"
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

        # PART 5: PRE-ATOM FILTER - Reject weak signals before atom creation
        is_ready, reject_reason = is_wedge_ready_signal(finding_data)
        if not is_ready:
            logger.debug(f"filtering weak signal: {reject_reason}")
            self._seen_hashes.add(content_hash)
            return None

        evidence = dict(finding_data.get("evidence", {}) or {})
        evidence.setdefault("run_id", self.db.get_active_run_id())
        discovery_query = evidence.get("discovery_query")
        source_plan = evidence.get("source_plan")
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
            return finding_id

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
            atom_json=json.dumps(atom_payload["atom_json"]),
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

        Aggregates query-level feedback into term-level state transitions.
        """
        # Get feedback aggregated by keyword/subreddit
        feedback_rows = self.db.get_discovery_feedback()

        # Track which terms we've seen to update
        keyword_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {
            "findings_emitted": 0, "validations": 0, "passes": 0,
            "prototype_candidates": 0, "build_briefs": 0, "screened_out": 0,
            "low_yield": 0, "noisy": 0, "thin_validation": 0,
        })
        subreddit_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {
            "findings_emitted": 0, "validations": 0, "passes": 0,
            "prototype_candidates": 0, "build_briefs": 0, "screened_out": 0,
            "low_yield": 0, "noisy": 0, "thin_validation": 0,
        })

        for row in feedback_rows:
            source_name = row.get("source_name", "")
            query_text = row.get("query_text", "")

            # Determine source type
            if source_name.startswith("reddit"):
                # Try to extract subreddit from query or source
                # For now, track keywords
                keyword_metrics[query_text] = {
                    "findings_emitted": keyword_metrics[query_text]["findings_emitted"] + int(row.get("findings_emitted", 0)),
                    "validations": keyword_metrics[query_text]["validations"] + int(row.get("validations", 0)),
                    "passes": keyword_metrics[query_text]["passes"] + int(row.get("passes", 0)),
                    "prototype_candidates": keyword_metrics[query_text]["prototype_candidates"] + int(row.get("prototype_candidates", 0)),
                    "build_briefs": keyword_metrics[query_text]["build_briefs"] + int(row.get("build_briefs", 0)),
                    "screened_out": keyword_metrics[query_text]["screened_out"] + int(row.get("screened_out", 0)),
                    "low_yield": keyword_metrics[query_text]["low_yield"],
                    "noisy": keyword_metrics[query_text]["noisy"],
                    "thin_validation": keyword_metrics[query_text]["thin_validation"],
                }
            elif source_name.startswith("web"):
                keyword_metrics[query_text] = {
                    "findings_emitted": keyword_metrics[query_text]["findings_emitted"] + int(row.get("findings_emitted", 0)),
                    "validations": keyword_metrics[query_text]["validations"] + int(row.get("validations", 0)),
                    "passes": keyword_metrics[query_text]["passes"] + int(row.get("passes", 0)),
                    "prototype_candidates": keyword_metrics[query_text]["prototype_candidates"] + int(row.get("prototype_candidates", 0)),
                    "build_briefs": keyword_metrics[query_text]["build_briefs"] + int(row.get("build_briefs", 0)),
                    "screened_out": keyword_metrics[query_text]["screened_out"] + int(row.get("screened_out", 0)),
                    "low_yield": keyword_metrics[query_text]["low_yield"],
                    "noisy": keyword_metrics[query_text]["noisy"],
                    "thin_validation": keyword_metrics[query_text]["thin_validation"],
                }
            elif source_name.startswith("github"):
                keyword_metrics[query_text] = {
                    "findings_emitted": keyword_metrics[query_text]["findings_emitted"] + int(row.get("findings_emitted", 0)),
                    "validations": keyword_metrics[query_text]["validations"] + int(row.get("validations", 0)),
                    "passes": keyword_metrics[query_text]["passes"] + int(row.get("passes", 0)),
                    "prototype_candidates": keyword_metrics[query_text]["prototype_candidates"] + int(row.get("prototype_candidates", 0)),
                    "build_briefs": keyword_metrics[query_text]["build_briefs"] + int(row.get("build_briefs", 0)),
                    "screened_out": keyword_metrics[query_text]["screened_out"] + int(row.get("screened_out", 0)),
                    "low_yield": keyword_metrics[query_text]["low_yield"],
                    "noisy": keyword_metrics[query_text]["noisy"],
                    "thin_validation": keyword_metrics[query_text]["thin_validation"],
                }

        # Also look at discovery_search_terms to get low_yield, noisy, thin_validation counts
        existing_terms = self.db.list_search_terms(limit=500)
        term_state_map = {(t["term_type"], t["term_value"]): t for t in existing_terms}

        # Update keywords
        transitions = {"keywords": [], "subreddits": []}
        for keyword, metrics in keyword_metrics.items():
            if not keyword or keyword == "[source-skipped]":
                continue

            # Get existing state for quality flags
            existing = term_state_map.get(("keyword", keyword), {})
            low_yield = existing.get("low_yield_count", 0)
            noisy = existing.get("noisy_count", 0)
            thin_val = existing.get("thin_validation_count", 0)

            result = self.term_lifecycle.record_search_run(
                "keyword",
                keyword,
                findings_emitted=metrics["findings_emitted"],
                validations=metrics["validations"],
                passes=metrics["passes"],
                prototype_candidates=metrics["prototype_candidates"],
                build_briefs=metrics["build_briefs"],
                screened_out=metrics["screened_out"],
                low_yield=low_yield > 0,
                noisy=noisy > 0,
                thin_validation=thin_val > 0,
            )
            if result["old_state"] != result["new_state"]:
                transitions["keywords"].append(result)

        logger.info(f"Term lifecycle: {len(transitions['keywords'])} keyword state changes")
        return transitions

    def _generate_content_hash(self, finding_data: Dict[str, Any]) -> str:
        normalized = json.dumps(
            {
                "source_url": finding_data.get("source_url", ""),
                "title": finding_data.get("product_built", ""),
                "kind": finding_data.get("finding_kind", ""),
                "summary": finding_data.get("outcome_summary", ""),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        import hashlib

        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

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
