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
from src.database import Database  # noqa: E402
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
        rotating_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[
                rotating_handler,
                logging.StreamHandler(),
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
            "solution_framing": SolutionFramingAgent(self.db, message_queue=message_queue),
            "experiment_design": ExperimentDesignAgent(self.db, message_queue=message_queue),
            "spec_generation": SpecGenerationAgent(self.db, message_queue=message_queue),
        }

        if self.config.get("orchestration", {}).get("auto_ideate_after_validation", False):
            self.agents["ideation"] = IdeationAgent(self.db, message_queue=message_queue, config=self.config)
        if self.config.get("builder", {}).get("auto_build", False):
            from src.agents.builder_v2 import BuilderV2Agent
            self.agents["builder"] = BuilderV2Agent(self.config, db=self.db)

        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)

    def _setup_signals(self) -> None:
        def _handle_shutdown(*_args: Any) -> None:
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

    async def _wait_for_pipeline_drained(self, max_wait_seconds: float) -> bool:
        """Block until queue/agents idle or deadline. Returns True if drained, False on timeout."""
        deadline = asyncio.get_running_loop().time() + max(5.0, max_wait_seconds)
        quiet_cycles = 0
        while asyncio.get_running_loop().time() < deadline and not self.shutdown_event.is_set():
            if self.completion_state().get("drained"):
                quiet_cycles += 1
                if quiet_cycles >= 5:
                    return True
            else:
                quiet_cycles = 0
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

        await self.shutdown()

    async def run_once(self) -> dict[str, Any]:
        await self.initialize()
        assert self.orchestrator is not None
        assert self.db is not None

        await self.orchestrator.start(skip_agents={"discovery"})
        self.status_tracker.reset()
        self.status_tracker.start_run(self.current_run_id)
        self.status_tracker.set_stage("discovery")

        backlog_ids = await self._dispatch_open_qualified_findings()
        discovery_ids = await self.agents["discovery"]._discover_once()
        max_wait_seconds = float(self.config.get("orchestration", {}).get("run_once_max_wait_seconds", 60))
        if not await self._wait_for_pipeline_drained(max_wait_seconds):
            logger.warning(
                "run_once reached max wait with pipeline not drained: %s",
                self.completion_state(),
            )

        await asyncio.sleep(0.2)
        summary = self.snapshot()
        summary["backlog_ids"] = backlog_ids
        summary["discovery_ids"] = discovery_ids
        await self.shutdown()
        return summary

    async def _dispatch_open_qualified_findings(self) -> list[int]:
        assert self.db is not None
        assert self.orchestrator is not None

        backlog_ids: list[int] = []
        for finding in self.db.get_findings(limit=500):
            if finding.status != "qualified" or finding.source_class != "pain_signal":
                continue
            signal_rows = self.db.get_raw_signals_by_finding(finding.id or 0)
            atom_rows = self.db.get_problem_atoms_by_finding(finding.id or 0)
            if not signal_rows or not atom_rows:
                continue
            backlog_ids.append(int(finding.id))
            await self.orchestrator.send_message(
                to_agent="orchestrator",
                msg_type=MessageType.FINDING,
                payload={
                    "finding_id": int(finding.id),
                    "source": finding.source,
                    "source_url": finding.source_url,
                    "content_hash": finding.content_hash,
                    "finding_kind": finding.finding_kind,
                    "source_class": finding.source_class,
                    "title": finding.product_built,
                    "summary": finding.outcome_summary,
                    "signal_id": int(signal_rows[0].id),
                    "problem_atom_ids": [int(atom.id) for atom in atom_rows if atom.id is not None],
                    "backlog_requeue": True,
                },
                priority=2,
            )
        if backlog_ids:
            logger.info("requeued %s qualified findings for evidence", len(backlog_ids))
            self.status_tracker.log(f"requeued_qualified findings={len(backlog_ids)}")
        return backlog_ids

    def _count_actionable_qualified_findings(self) -> int:
        assert self.db is not None
        count = 0
        for finding in self.db.get_findings(limit=500):
            if finding.status != "qualified" or finding.source_class != "pain_signal":
                continue
            signal_rows = self.db.get_raw_signals_by_finding(finding.id or 0)
            atom_rows = self.db.get_problem_atoms_by_finding(finding.id or 0)
            if signal_rows and atom_rows:
                count += 1
        return count

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
        assert self.db is not None
        assert self.orchestrator is not None

        backlog_ids: list[int] = []
        for finding in self.db.get_findings(limit=500):
            if finding.status != "qualified" or finding.source_class != "pain_signal":
                continue
            signal_rows = self.db.get_raw_signals_by_finding(finding.id or 0)
            atom_rows = self.db.get_problem_atoms_by_finding(finding.id or 0)
            if not signal_rows or not atom_rows:
                continue
            backlog_ids.append(int(finding.id))
            await self.orchestrator.send_message(
                to_agent="orchestrator",
                msg_type=MessageType.FINDING,
                payload={
                    "finding_id": int(finding.id),
                    "source": finding.source,
                    "source_url": finding.source_url,
                    "content_hash": finding.content_hash,
                    "finding_kind": finding.finding_kind,
                    "source_class": finding.source_class,
                    "title": finding.product_built,
                    "summary": finding.outcome_summary,
                    "signal_id": int(signal_rows[0].id),
                    "problem_atom_ids": [int(atom.id) for atom in atom_rows if atom.id is not None],
                    "backlog_requeue": True,
                },
                priority=2,
            )
        if backlog_ids:
            logger.info("requeued %s qualified findings for evidence", len(backlog_ids))
            self.status_tracker.log(f"requeued_qualified findings={len(backlog_ids)}")
        return backlog_ids

    def _count_actionable_qualified_findings(self) -> int:
        assert self.db is not None
        count = 0
        for finding in self.db.get_findings(limit=500):
            if finding.status != "qualified" or finding.source_class != "pain_signal":
                continue
            signal_rows = self.db.get_raw_signals_by_finding(finding.id or 0)
            atom_rows = self.db.get_problem_atoms_by_finding(finding.id or 0)
            if signal_rows and atom_rows:
                count += 1
        return count

    def completion_state(self) -> dict[str, Any]:
        queue_empty = True
        queue_size = 0
        if self.orchestrator is not None:
            queue_empty = self.orchestrator._message_queue.empty()
            queue_size = self.orchestrator._message_queue.qsize()
        open_qualified = self._count_actionable_qualified_findings() if self.db is not None else 0

        evidence_busy = 0
        validation_busy = 0
        if self.agents:
            evidence_agent = self.agents.get("evidence")
            validation_agent = self.agents.get("validation")
            if evidence_agent is not None and hasattr(evidence_agent, "busy_count"):
                evidence_busy = int(evidence_agent.busy_count())
            if validation_agent is not None and hasattr(validation_agent, "busy_count"):
                validation_busy = int(validation_agent.busy_count())

        drained = queue_empty and open_qualified == 0 and evidence_busy == 0 and validation_busy == 0
        return {
            "queue_empty": queue_empty,
            "queue_size": queue_size,
            "open_qualified": open_qualified,
            "evidence_busy": evidence_busy,
            "validation_busy": validation_busy,
            "drained": drained,
        }

    def reddit_runtime_summary(self) -> dict[str, Any]:
        phases: dict[str, dict[str, Any]] = {}
        summed_keys = {
            "reddit_bridge_hits",
            "reddit_bridge_misses",
            "reddit_fallback_queries",
            "reddit_public_direct_queries",
            "reddit_validation_seed_runs",
            "reddit_validation_seeded_pairs",
            "reddit_validation_seed_searches",
            "reddit_validation_seed_uncovered_before",
            "reddit_validation_seed_uncovered_after",
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
