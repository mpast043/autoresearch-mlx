"""Unified runtime for the evidence-first discovery and validation pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import signal
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import yaml

from src.runtime.env import load_local_env  # noqa: E402
from src.runtime.paths import DEFAULT_CONFIG_PATH, build_runtime_paths, resolve_project_path  # noqa: E402
from src.utils.logging_utils import set_run_id, StructuredJsonFormatter, StructuredTextFormatter
from src.database import Database  # noqa: E402
from src.high_leverage import build_high_leverage_report  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.messaging import MessageType  # noqa: E402
from src.agents.discovery import DiscoveryAgent  # noqa: E402
from src.agents.evidence import EvidenceAgent  # noqa: E402
from src.agents.build_prep import ExperimentDesignAgent, SolutionFramingAgent, SpecGenerationAgent  # noqa: E402
from src.agents.validation import ValidationAgent  # noqa: E402
from src.agents.ideation import IdeationAgent  # noqa: E402
from src.agents.builder import BuilderAgent  # noqa: E402
from src.status_tracker import StatusTracker  # noqa: E402


logger = logging.getLogger(__name__)
DEFAULT_STALLED_IDLE_CYCLES_THRESHOLD = 50


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


class AutoResearcher:
    """Executable application wrapper for the orchestrated pipeline."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        load_local_env()
        self.config = self._load_config(config_path)
        self._runtime_paths = build_runtime_paths(self.config)
        self.output_dir = self._runtime_paths["output_dir"]
        self.log_path = self._configure_logging()
        self.db_path = self._runtime_paths["db_path"]
        self.db: Database | None = None
        self.orchestrator: Orchestrator | None = None
        self.agents: dict[str, Any] = {}
        self.shutdown_event = asyncio.Event()
        self.status_tracker = StatusTracker(str(self.output_dir))
        self.sources_db = self._load_sources_db()
        self.current_run_id = ""
        self.discovery_bypass_cache = False
        self._ignore_backlog_for_completion = False
        self._sigint_count = 0

    def _load_config(self, path: str | Path) -> dict[str, Any]:
        config_file = resolve_project_path(path, default=DEFAULT_CONFIG_PATH)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with config_file.open() as handle:
            return yaml.safe_load(handle) or {}

    def _load_sources_db(self) -> dict[str, Any]:
        sources_path = self._runtime_paths["sources_db_path"]
        if not sources_path.exists():
            return {"sources": {}}
        try:
            return json.loads(sources_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s; using empty sources db", sources_path)
            return {"sources": {}}

    def _configure_logging(self) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(self._runtime_paths["log_path"].resolve())

        # Use RotatingFileHandler for log rotation (10MB per file, keep 5 backups)
        rotating_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        rotating_handler.setLevel(logging.INFO)
        use_json = self.config.get("logging", {}).get("json_format", False)
        formatter = StructuredJsonFormatter() if use_json else StructuredTextFormatter()
        rotating_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                rotating_handler,
                stream_handler,
            ],
            force=True,
        )
        if self.config.get("logging", {}).get("debug_research_tools", False):
            logging.getLogger("src.research_tools").setLevel(logging.DEBUG)
            logging.getLogger("research_tools").setLevel(logging.DEBUG)
        return log_path

    def runtime_paths(self) -> dict[str, str]:
        return {
            "db_path": str(self.db_path.resolve()),
            "status_path": str(self._runtime_paths["status_path"].resolve()),
            "log_path": str(self.log_path),
        }

    async def initialize(self, start_new_run: bool = True) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(str(self.db_path))
        self.db.init_schema()

        if start_new_run:
            self.current_run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        else:
            self.current_run_id = self.db.get_latest_run_id()
        self.db.set_active_run_id(self.current_run_id)
        set_run_id(self.current_run_id)

        # Bootstrap checks before any pipeline work
        failures = await self._bootstrap_checks()
        if failures:
            for f in failures:
                logger.warning("BOOTSTRAP FAIL: %s", f)
            logger.error("Bootstrap checks failed (%d). Fix before running.", len(failures))
            raise RuntimeError(f"Bootstrap checks failed: {failures}")

        # Pass config to opportunity_engine for LLM atom extraction
        from src.opportunity_engine import configure_opportunity_engine
        from src.build_prep import configure_build_prep
        configure_opportunity_engine(self.config)
        configure_build_prep(self.config)

        builder_config = self.config.get("builder", {})
        orchestration_config = self.config.get("orchestration", {})
        auto_build = builder_config.get("auto_build", False)
        auto_ideate = orchestration_config.get("auto_ideate_after_validation", False)

        self.orchestrator = Orchestrator(
            self.db,
            self.status_tracker,
            auto_build=auto_build,
            auto_ideate=auto_ideate,
            shutdown_event=self.shutdown_event,
            stop_on_hit_config=orchestration_config.get("stop_on_hit") or {},
        )

        await self._create_agents()
        self._setup_signals()

    async def _bootstrap_checks(self) -> list[str]:
        """Pre-flight checks: verify required services are reachable before pipeline starts."""
        failures: list[str] = []

        # 1. Ollama (LLM) — needed for atom extraction, ideation, build-prep
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama")
        base_url = llm_config.get("base_url", "http://localhost:11434")
        if provider in ("ollama", "auto"):
            try:
                import urllib.request
                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status != 200:
                        failures.append(f"Ollama returned HTTP {resp.status} at {base_url}")
                    else:
                        models = json.loads(resp.read()).get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        logger.info("Bootstrap OK: Ollama at %s (%d models: %s)", base_url, len(model_names), ", ".join(model_names[:5]))
            except Exception as e:
                failures.append(f"Ollama unreachable at {base_url}: {e}")

        # 2. ddgs (web search) — needed for validation recurrence evidence
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = ddgs.text("test search", backend="auto", max_results=1)
            if not results:
                failures.append("ddgs 'auto' backend returned 0 results (may be rate-limited)")
            else:
                logger.info("Bootstrap OK: ddgs search returned results")
        except Exception as e:
            failures.append(f"ddgs search failed: {e}")

        # 3. Reddit bridge (if enabled) — warn if unreachable but don't fail (fallback exists)
        reddit_bridge = self.config.get("reddit_bridge", {})
        if reddit_bridge.get("enabled", False):
            client = None
            try:
                from src.reddit_bridge import RedditBridgeClient

                client = RedditBridgeClient(reddit_bridge)
                try:
                    if not client.enabled:
                        logger.warning("Bootstrap WARN: Reddit bridge enabled but base_url resolved empty (will use fallback)")
                    else:
                        health = await client.health()
                        logger.info(
                            "Bootstrap OK: Reddit bridge at %s service=%s",
                            client.base_url,
                            health.get("service", "unknown"),
                        )
                finally:
                    await client.close()
            except Exception as e:
                logger.warning(
                    "Bootstrap WARN: Reddit bridge unreachable at %s: %s (will use fallback)",
                    getattr(client, "base_url", ""),
                    e,
                )

        # 4. Database writable
        try:
            self.db.set_active_run_id(self.current_run_id)
            logger.info("Bootstrap OK: SQLite database writable")
        except Exception as e:
            failures.append(f"Database not writable: {e}")

        return failures

    async def _create_agents(self) -> None:
        assert self.db is not None
        assert self.orchestrator is not None

        discovery_sources = self.config.get("discovery", {}).get("sources")
        message_queue = self.orchestrator._message_queue

        self.agents = {
            "discovery": DiscoveryAgent(
                self.db,
                message_queue=message_queue,
                sources=discovery_sources,
                config=self.config,
                status_tracker=self.status_tracker,
                bypass_cache=self.discovery_bypass_cache,
            ),
            "evidence": EvidenceAgent(self.db, message_queue=message_queue, config=self.config),
            "validation": ValidationAgent(self.db, message_queue=message_queue, config=self.config),
            "solution_framing": SolutionFramingAgent(self.db, message_queue=message_queue, config=self.config),
            "experiment_design": ExperimentDesignAgent(self.db, message_queue=message_queue, config=self.config),
            "spec_generation": SpecGenerationAgent(self.db, message_queue=message_queue, config=self.config),
        }

        if self.config.get("orchestration", {}).get("auto_ideate_after_validation", False):
            self.agents["ideation"] = IdeationAgent(self.db, message_queue=message_queue, config=self.config)
        if self.config.get("builder", {}).get("auto_build", False):
            from src.agents.builder_v2 import BuilderV2Agent
            self.agents["builder"] = BuilderV2Agent(self.config, db=self.db)
        if self.config.get("security", {}).get("enabled", False):
            from src.agents.security import SecurityAgent
            self.agents["security"] = SecurityAgent(db=self.db, message_queue=message_queue, config=self.config.get("security", {}))
        if self.config.get("technical_writer", {}).get("enabled", False):
            from src.agents.technical_writer import TechnicalWriterAgent
            self.agents["technical_writer"] = TechnicalWriterAgent(db=self.db, message_queue=message_queue, config=self.config.get("technical_writer", {}))
        if self.config.get("sre", {}).get("enabled", False):
            from src.agents.sre import SREAgent
            self.agents["sre"] = SREAgent(db=self.db, message_queue=message_queue, config=self.config.get("sre", {}))

        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)

    def _setup_signals(self) -> None:
        def _handle_sigint(*_args: Any) -> None:
            self._sigint_count += 1
            if self._sigint_count == 1:
                logger.info("SIGINT received: initiating graceful shutdown (press Ctrl+C again to force quit)")
                self.shutdown_event.set()
            else:
                logger.warning("SIGINT received again: force quitting immediately")
                sys.exit(1)

        def _handle_sigterm(*_args: Any) -> None:
            logger.info("SIGTERM received: initiating graceful shutdown")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigterm)

    async def _wait_for_pipeline_drained(self, max_wait_seconds: float) -> bool:
        """Block until queue/agents idle or deadline. Returns True if drained, False on timeout."""
        deadline = asyncio.get_running_loop().time() + max(5.0, max_wait_seconds)
        quiet_cycles = 0
        stalled_idle_cycles = 0
        last_log_at = 0.0
        last_state: dict[str, Any] | None = None
        last_idle_signature: tuple[Any, ...] | None = None
        stalled_idle_threshold = int(
            (self.config.get("orchestration", {}) or {}).get(
                "stalled_idle_cycles_threshold",
                DEFAULT_STALLED_IDLE_CYCLES_THRESHOLD,
            )
            or DEFAULT_STALLED_IDLE_CYCLES_THRESHOLD
        )
        while asyncio.get_running_loop().time() < deadline and not self.shutdown_event.is_set():
            state = self.completion_state()
            loop_now = asyncio.get_running_loop().time()
            if last_state != state or (loop_now - last_log_at) >= 5.0:
                logger.info("pipeline drain wait state: %s", state)
                last_log_at = loop_now
                last_state = dict(state)
            if state.get("drained"):
                quiet_cycles += 1
                if quiet_cycles >= 5:
                    return True
            else:
                quiet_cycles = 0

            is_idle_but_stuck = (
                not state.get("queue_empty")
                and state.get("open_qualified") == 0
                and state.get("total_busy") == 0
            )
            idle_signature = (
                state.get("queue_size"),
                state.get("open_qualified"),
                state.get("total_busy"),
                tuple(sorted((state.get("queue_by_agent") or {}).items())),
                tuple(sorted((state.get("agent_statuses") or {}).items())),
            )

            if is_idle_but_stuck and idle_signature == last_idle_signature:
                stalled_idle_cycles += 1
                if stalled_idle_cycles >= stalled_idle_threshold:
                    logger.warning("treating stable idle queue as drained: %s", state)
                    return True
            else:
                stalled_idle_cycles = 0
            last_idle_signature = idle_signature if is_idle_but_stuck else None
            await asyncio.sleep(0.1)
        return bool(self.completion_state().get("drained"))

    async def run(self) -> None:
        await self.initialize()
        assert self.orchestrator is not None
        await self.orchestrator.start()
        self.status_tracker.reset()
        self.status_tracker.start_run(self.current_run_id)
        self.status_tracker.set_stage("discovery")
        self.status_tracker.update(status="running")

        orch = self.config.get("orchestration", {})
        stop_cfg = orch.get("stop_on_hit") or {}
        stop_enabled = bool(stop_cfg.get("enabled", False))
        continuous_waves = bool(orch.get("continuous_waves", False))
        wave_loop_enabled = stop_enabled or continuous_waves
        max_wait = float(orch.get("run_once_max_wait_seconds", 180))
        retry_interval = float(stop_cfg.get("retry_interval_seconds", 60))

        wave_index = 0
        while not self.shutdown_event.is_set():
            wave_index += 1
            logger.info(
                "run: discovery wave %s starting (continuous_waves=%s, retry_interval_seconds=%s)",
                wave_index,
                continuous_waves,
                retry_interval,
            )
            await self.agents["discovery"]._discover_once()
            drained_ok = await self._wait_for_pipeline_drained(max_wait)
            if not drained_ok and not self.shutdown_event.is_set():
                logger.warning(
                    "run: pipeline not fully drained after wait: %s",
                    self.completion_state(),
                )
            if self.shutdown_event.is_set():
                break
            if not wave_loop_enabled:
                logger.info("run: single discovery wave complete; waiting for SIGINT/SIGTERM (stop_on_hit disabled)")
                await self.shutdown_event.wait()
                break
            if retry_interval <= 0:
                if continuous_waves and not stop_enabled:
                    logger.info(
                        "continuous_waves: wave %s done; no sleep before next wave (retry_interval_seconds=0)",
                        wave_index,
                    )
                else:
                    logger.info(
                        "stop_on_hit loop: wave %s done; no sleep (retry_interval_seconds=0)",
                        wave_index,
                    )
                await asyncio.sleep(0)
            else:
                if continuous_waves and not stop_enabled:
                    logger.info(
                        "continuous_waves: sleeping %ss before next discovery wave (Ctrl+C to exit)",
                        retry_interval,
                    )
                else:
                    logger.info(
                        "stop_on_hit: no matching hit yet; sleeping %ss before next discovery wave (Ctrl+C to exit)",
                        retry_interval,
                    )
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=retry_interval)
                except asyncio.TimeoutError:
                    pass

        self.status_tracker.complete()
        await self.shutdown()

    async def run_once(self, *, skip_backlog: bool = False) -> dict[str, Any]:
        await self.initialize()
        assert self.orchestrator is not None
        assert self.db is not None

        await self.orchestrator.start(skip_agents={"discovery"})
        self.status_tracker.reset()
        self.status_tracker.start_run(self.current_run_id)
        self.status_tracker.set_stage("discovery")

        self._ignore_backlog_for_completion = bool(skip_backlog)
        try:
            backlog_ids: list[int] = []
            if skip_backlog:
                logger.info("run_once: discovery-only mode enabled; skipping qualified backlog requeue")
                self.status_tracker.log("discovery_only skip_backlog=true")
            else:
                backlog_ids = await self._dispatch_open_qualified_findings()
            discovery_ids = await self.agents["discovery"]._discover_once()
            max_wait_seconds = float(self.config.get("orchestration", {}).get("run_once_max_wait_seconds", 60))
            drained_ok = await self._wait_for_pipeline_drained(max_wait_seconds)
            if not drained_ok:
                logger.warning(
                    "run_once reached max wait with pipeline not drained: %s",
                    self.completion_state(),
                )
                self.status_tracker.log(f"drain_timeout state={self.completion_state()}")

            await asyncio.sleep(0.2)
            summary = self.snapshot()
            summary["backlog_ids"] = backlog_ids
            summary["discovery_ids"] = discovery_ids
            summary["skip_backlog"] = bool(skip_backlog)
            summary["drained"] = bool(drained_ok)
            self.status_tracker.complete()
            await self.shutdown()
            return summary
        finally:
            self._ignore_backlog_for_completion = False

    async def _dispatch_open_qualified_findings(self) -> list[int]:
        assert self.db is not None
        assert self.orchestrator is not None

        backlog_ids: list[int] = []
        backlog_items = (
            self.db.get_backlog_workbench(limit=500)
            if hasattr(self.db, "get_backlog_workbench")
            else []
        )
        for item in backlog_items:
            backlog_ids.append(int(item["finding_id"]))
            await self.orchestrator.send_message(
                to_agent="orchestrator",
                msg_type=MessageType.FINDING,
                payload={
                    "finding_id": int(item["finding_id"]),
                    "source": item["source"],
                    "source_url": item["source_url"],
                    "content_hash": item.get("content_hash", ""),
                    "finding_kind": item.get("finding_kind", "problem_signal"),
                    "source_class": item["current_source_class"],
                    "title": item["title"],
                    "summary": item["summary"],
                    "signal_id": int(item["signal_id"]),
                    "problem_atom_ids": list(item["problem_atom_ids"]),
                    "backlog_requeue": True,
                    "backlog_priority_score": item.get("backlog_priority_score", 0.0),
                },
                priority=2,
            )
        if backlog_ids:
            logger.info("requeued %s qualified findings for evidence", len(backlog_ids))
            self.status_tracker.log(f"requeued_qualified findings={len(backlog_ids)}")
        return backlog_ids

    def _count_actionable_qualified_findings(self) -> int:
        assert self.db is not None
        if hasattr(self.db, "get_backlog_workbench"):
            return len(self.db.get_backlog_workbench(limit=500))
        return 0

    async def run_unseeded(
        self,
        vertical: str = "devtools",
        max_findings: int = 20,
    ) -> dict[str, Any]:
        """
        Cold-start discovery loop: no DB seeding required.

        Searches Reddit, GitHub, and the web for weak signals in the given
        vertical, creates finding records, and routes them directly to
        ValidationAgent (skipping the EvidenceAgent stage).
        """
        from src.unseeded_loop import run_unseeded

        assert self.db is not None
        summary = await run_unseeded(
            vertical=vertical,
            config=self.config,
            db=self.db,
            max_findings=max_findings,
        )
        return summary

    def completion_state(self) -> dict[str, Any]:
        queue_empty = True
        queue_size = 0
        queue_by_agent: dict[str, int] = {}
        if self.orchestrator is not None:
            queue_empty = self.orchestrator._message_queue.empty()
            queue_size = self.orchestrator._message_queue.qsize()
            queue_registry = getattr(self.orchestrator._message_queue, "registered_agents", None)
            if callable(queue_registry):
                for agent_name in queue_registry():
                    agent_queue_size = int(self.orchestrator._message_queue.qsize(agent_name))
                    if agent_queue_size > 0:
                        queue_by_agent[agent_name] = agent_queue_size
        actual_open_qualified = self._count_actionable_qualified_findings() if self.db is not None else 0
        open_qualified = 0 if self._ignore_backlog_for_completion else actual_open_qualified

        evidence_busy = 0
        validation_busy = 0
        total_busy = 0
        busy_agents: dict[str, int] = {}
        agent_statuses: dict[str, str] = {}
        if self.agents:
            for agent_name, agent in self.agents.items():
                agent_status = getattr(getattr(agent, "status", None), "value", "")
                if agent_status:
                    agent_statuses[agent_name] = str(agent_status)
                if agent is None or not hasattr(agent, "busy_count"):
                    continue
                agent_busy = int(agent.busy_count())
                busy_agents[agent_name] = agent_busy
                total_busy += agent_busy
            evidence_busy = busy_agents.get("evidence", 0)
            validation_busy = busy_agents.get("validation", 0)

        drained = queue_empty and open_qualified == 0 and total_busy == 0
        return {
            "queue_empty": queue_empty,
            "queue_size": queue_size,
            "queue_by_agent": queue_by_agent,
            "open_qualified": open_qualified,
            "actual_open_qualified": actual_open_qualified,
            "evidence_busy": evidence_busy,
            "validation_busy": validation_busy,
            "total_busy": total_busy,
            "busy_agents": busy_agents,
            "agent_statuses": agent_statuses,
            "drained": drained,
        }

    def reddit_runtime_summary(self) -> dict[str, Any]:
        phases: dict[str, dict[str, Any]] = {}
        summed_keys = {
            "reddit_bridge_hits",
            "reddit_bridge_misses",
            "reddit_fallback_queries",
            "reddit_public_direct_queries",
            "bridge_search_count",
            "bridge_search_result_count",
            "public_json_search_count",
            "bridge_thread_count",
            "bridge_thread_no_cached_count",
            "public_json_thread_count",
            "degraded_fallback_findings_count",
            "degraded_fallback_docs_count",
            "degraded_fallback_validation_count",
            "bridge_upstream_failure_count",
            "bridge_search_retry_count",
            "bridge_search_seed_recovery_count",
            "devvit_hydration_configured_count",
            "devvit_hydration_unconfigured_count",
            "devvit_hydration_attempt_count",
            "devvit_hydration_success_count",
            "devvit_hydration_failure_count",
            "devvit_hydration_retry_success_count",
            "devvit_hydration_retry_failure_count",
            "reddit_validation_seed_runs",
            "reddit_validation_seeded_pairs",
            "reddit_validation_seed_searches",
            "reddit_validation_seed_uncovered_before",
            "reddit_validation_seed_uncovered_after",
            "reddit_validation_seed_bridge_covered_pairs",
            "reddit_validation_seed_degraded_covered_pairs",
            "reddit_validation_seed_pairs_with_usable_bridge_docs",
            "reddit_validation_seed_pairs_with_only_degraded_docs",
            "reddit_validation_seed_failed_pairs",
            "reddit_validation_seed_truly_uncovered_pairs",
            "seeded_bridge_covered_pairs",
            "seeded_degraded_covered_pairs",
            "seeded_pairs_with_usable_bridge_docs",
            "seeded_pairs_with_only_degraded_docs",
            "seeded_failed_pairs",
            "seeded_truly_uncovered_pairs",
        }
        passthrough_keys = {
            "seeded_total_pairs",
            "seeded_pairs_searched",
            "seeded_pairs_fresh",
            "seeded_pairs_existing_cache",
            "seeded_pairs_uncovered",
            "seeded_cached_searches",
            "seeded_cached_threads",
            "seeded_thread_cache_hits",
            "seeded_unique_urls",
            "seeded_degraded_covered_pairs",
            "seeded_pairs_with_only_degraded_docs",
            "seeded_truly_uncovered_pairs",
        }

        aggregate: dict[str, Any] = {key: 0 for key in summed_keys}
        modes: list[str] = []

        for name, agent in (self.agents or {}).items():
            metrics: dict[str, Any] = {}
            if hasattr(agent, "reddit_runtime_summary"):
                metrics = agent.reddit_runtime_summary() or {}
            elif hasattr(agent, "toolkit") and hasattr(agent.toolkit, "get_reddit_runtime_metrics"):
                metrics = agent.toolkit.get_reddit_runtime_metrics() or {}
            if not metrics:
                continue
            phases[name] = dict(metrics)
            mode = str(metrics.get("reddit_mode", "") or "").strip()
            if mode and mode not in modes:
                modes.append(mode)
            for key in summed_keys:
                aggregate[key] += int(metrics.get(key, 0) or 0)
            for key in passthrough_keys:
                if key in metrics and metrics.get(key) is not None:
                    aggregate[key] = metrics.get(key)

        if not phases:
            return {}

        discovery_phase = phases.get("discovery", {})
        aggregate["degraded_fallback_findings_count"] = int(
            discovery_phase.get("degraded_fallback_findings_count", 0) or 0
        )
        aggregate["degraded_fallback_validation_count"] = sum(
            int((phases.get(name, {}) or {}).get("public_json_search_count", 0) or 0)
            + int((phases.get(name, {}) or {}).get("public_json_thread_count", 0) or 0)
            for name in ("evidence", "validation")
        )
        if int(aggregate.get("reddit_validation_seed_runs", 0) or 0) == 0:
            for key in (
                "reddit_validation_seed_uncovered_before",
                "reddit_validation_seed_uncovered_after",
            ):
                aggregate[key] = None
        aggregate["reddit_mode"] = modes[0] if len(modes) == 1 else modes
        aggregate["phases"] = phases
        return aggregate

    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_paths(),
            "counts": self.summary_counts(),
            "decisions": self.decision_summary(),
            "reddit_runtime": self.reddit_runtime_summary(),
            "screening": self.db.get_finding_status_counts(run_id=self.current_run_id) if self.db else {},
            "screening_all_time": self.db.get_finding_status_counts() if self.db else {},
            "actionable_screening": self.db.get_finding_status_counts(run_id=self.current_run_id, actionable_only=True) if self.db else {},
            "screening_summary": self.db.get_screening_summary(limit=10, run_id=self.current_run_id) if self.db else {},
            "validation": self.validation_report(limit=10),
            "run_diff": self.run_diff(limit=10),
            "build_prep": {
                "build_briefs": [item.__dict__ for item in self.db.list_build_briefs(run_id=self.current_run_id, limit=10)],
                "outputs": [item.__dict__ for item in self.db.list_build_prep_outputs(run_id=self.current_run_id, limit=20)],
            } if self.db and hasattr(self.db, "list_build_briefs") and hasattr(self.db, "list_build_prep_outputs") else {},
            "review": self.review_report(limit=10),
            "recent_logs": self.status_tracker.status.get("logs", [])[-12:],
        }

    def summary_counts(self) -> dict[str, Any]:
        if not self.db:
            return {}
        # Most entity totals are DB-wide; validations / build artifacts below use current_run_id when set.
        run_id = self.current_run_id or ""
        return {
            "findings": len(self.db.get_findings(limit=1000)),
            "raw_signals": len(self.db.get_raw_signals(limit=1000)),
            "problem_atoms": len(self.db.get_problem_atoms(limit=1000)),
            "clusters": len(self.db.get_clusters(limit=1000)),
            "opportunities": len(self.db.get_opportunities(limit=1000)),
            "experiments": len(self.db.get_experiments(limit=1000)),
            "validations": len(self.validation_report(limit=1000)),
            "build_briefs": len(self.db.list_build_briefs(run_id=self.current_run_id, limit=1000)) if hasattr(self.db, "list_build_briefs") else 0,
            "build_prep_outputs": len(self.db.list_build_prep_outputs(run_id=self.current_run_id, limit=1000)) if hasattr(self.db, "list_build_prep_outputs") else 0,
            "current_run_id": run_id,
            "count_semantics": {
                "db_wide_totals": [
                    "findings",
                    "raw_signals",
                    "problem_atoms",
                    "clusters",
                    "opportunities",
                    "experiments",
                ],
                "scoped_to_current_run": ["validations", "build_briefs", "build_prep_outputs"],
            },
        }

    def review_report(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.db:
            return []
        try:
            return self.db.get_validation_review(limit=limit, run_id=self.current_run_id)
        except TypeError:
            return self.db.get_validation_review(limit=limit)

    def decision_summary(self) -> dict[str, Any]:
        rows = self.validation_report(limit=1000)
        decisions = Counter((row.get("decision") or "unknown") for row in rows)
        reasons = Counter((row.get("decision_reason") or "unknown") for row in rows)
        park_subreasons = Counter((row.get("park_subreason") or "") for row in rows if row.get("decision") == "park")
        recurrence_states = Counter((row.get("recurrence_state") or "unknown") for row in rows)
        recurrence_gap_reasons = Counter((row.get("recurrence_gap_reason") or "") for row in rows if row.get("recurrence_gap_reason"))
        recurrence_failure_classes = Counter(
            (row.get("recurrence_failure_class") or "")
            for row in rows
            if row.get("recurrence_failure_class")
        )
        score_values = {
            "composite_score": _distribution([float(row.get("composite_score") or 0.0) for row in rows]),
            "corroboration_score": _distribution([float(row.get("corroboration_score") or 0.0) for row in rows]),
            "value_support": _distribution([float(row.get("value_support") or 0.0) for row in rows]),
            "overall_score": _distribution([float(row.get("overall_score") or 0.0) for row in rows]),
        }
        return {
            "decision_mix": dict(decisions),
            "decision_reason_mix": dict(reasons),
            "park_subreason_mix": {k: v for k, v in park_subreasons.items() if k},
            "recurrence_state_mix": dict(recurrence_states),
            "recurrence_gap_reason_mix": dict(recurrence_gap_reasons),
            "recurrence_failure_class_mix": dict(recurrence_failure_classes),
            "score_distributions": score_values,
        }

    def validation_report(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.db:
            return []
        rows = self.db.get_validation_review(limit=limit, run_id=self.current_run_id)
        return rows

    def high_leverage_report(self, limit: int = 25) -> dict[str, Any]:
        if not self.db:
            return {"run_id": self.current_run_id, "count": 0, "band_mix": {}, "status_mix": {}, "findings": []}
        return build_high_leverage_report(self.db, run_id=self.current_run_id, limit=limit)

    def run_diff(self, limit: int = 25) -> dict[str, Any]:
        if not self.db:
            return {}
        run_ids = self.db.get_recent_run_ids(limit=2)
        if len(run_ids) < 2:
            return {"available": False, "current_run_id": self.current_run_id, "previous_run_id": ""}

        current_run_id, previous_run_id = run_ids[0], run_ids[1]
        current_rows = {row["finding_id"]: row for row in self.db.get_validation_review(limit=500, run_id=current_run_id)}
        previous_rows = {row["finding_id"]: row for row in self.db.get_validation_review(limit=500, run_id=previous_run_id)}
        changed: list[dict[str, Any]] = []

        for finding_id, row in current_rows.items():
            prev = previous_rows.get(finding_id)
            if not prev:
                changed.append({
                    "finding_id": finding_id,
                    "title": row.get("title"),
                    "previous_decision": None,
                    "current_decision": row.get("decision"),
                    "previous_reason": None,
                    "current_reason": row.get("decision_reason"),
                    "previous_composite_score": None,
                    "current_composite_score": row.get("composite_score"),
                    "composite_delta": round(float(row.get("composite_score") or 0.0), 4),
                    "previous_recurrence_state": None,
                    "current_recurrence_state": row.get("recurrence_state"),
                })
                continue

            current_score = float(row.get("composite_score") or 0.0)
            previous_score = float(prev.get("composite_score") or 0.0)
            if (
                row.get("decision") != prev.get("decision")
                or row.get("decision_reason") != prev.get("decision_reason")
                or abs(current_score - previous_score) >= 0.03
            ):
                changed.append({
                    "finding_id": finding_id,
                    "title": row.get("title"),
                    "previous_decision": prev.get("decision"),
                    "current_decision": row.get("decision"),
                    "previous_reason": prev.get("decision_reason"),
                    "current_reason": row.get("decision_reason"),
                    "previous_composite_score": prev.get("composite_score"),
                    "current_composite_score": row.get("composite_score"),
                    "composite_delta": round(current_score - previous_score, 4),
                    "previous_recurrence_state": prev.get("recurrence_state"),
                    "current_recurrence_state": row.get("recurrence_state"),
                })

        return {
            "current_run_id": current_run_id,
            "previous_run_id": previous_run_id,
            "available": True,
            "changed_items": changed[:limit],
            "changed_count": len(changed),
        }

    async def shutdown(self) -> None:
        if self.orchestrator:
            await self.orchestrator.stop()
        for agent in self.agents.values():
            toolkit = getattr(agent, "toolkit", None)
            if toolkit and hasattr(toolkit, "close"):
                await toolkit.close()
        if self.db:
            self.db.close()

    async def _show_stats(self) -> None:
        if not self.db:
            return
        findings = self.db.get_findings(limit=200)
        logger.info("Findings in database: %s", len(findings))
        stats = self.db.get_validation_stats()
        logger.info("Validated: %s | Rejected: %s", stats["validated"], stats["rejected"])

    def search_opportunities(self, keyword: str) -> list[dict[str, Any]]:
        keyword_lower = keyword.lower()
        matches: list[dict[str, Any]] = []

        for category, items in self.sources_db.get("sources", {}).items():
            for item in items:
                if keyword_lower not in json.dumps(item).lower():
                    continue
                matches.append({"category": category, **item})

        if self.db:
            for finding in self.db.get_findings(limit=100):
                haystack = json.dumps(finding.__dict__).lower()
                if keyword_lower in haystack:
                    matches.append({
                        "category": "live-findings",
                        "name": finding.product_built,
                        "revenue": finding.monetization_method,
                        "founder": finding.entrepreneur,
                    })

            for cluster in self.db.get_clusters(limit=100):
                haystack = json.dumps(cluster.__dict__).lower()
                if keyword_lower in haystack:
                    matches.append({
                        "category": "clusters",
                        "name": cluster.label,
                        "status": cluster.status,
                        "cluster_id": cluster.id,
                    })

            for opportunity in self.db.get_opportunities(limit=100):
                haystack = json.dumps(opportunity.__dict__).lower()
                if keyword_lower in haystack:
                    matches.append({
                        "category": "opportunities",
                        "name": opportunity.title,
                        "status": opportunity.status,
                        "recommendation": opportunity.recommendation,
                        "opportunity_id": opportunity.id,
                    })

        return matches


async def main() -> None:
    app = AutoResearcher()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
