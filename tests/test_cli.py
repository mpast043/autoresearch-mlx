"""Tests for CLI operator-facing helpers."""

import asyncio
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cli
from cli import (
    build_builder_jobs_view,
    build_discovery_sort_diagnostics,
    build_operator_report,
    build_verbose_report,
    resolve_database_path_from_config,
    render_watch_snapshot,
)


async def _fake_deep_research_run(self, max_signals_per_source=1):
    return {"ok": True, "command": "deep-research"}


class DummyStatusTracker:
    def __init__(self):
        self.status = {"logs": ["12:00:00 stage=discovery", "12:00:01 qualified finding=4"]}


class DummyDB:
    def get_finding_status_counts(self, run_id=None, actionable_only=False):
        if actionable_only:
            return {"screened_out": 0, "parked": 1, "killed": 0, "qualified": 0, "new": 0, "promoted": 0, "reviewed": 0}
        if run_id:
            return {"screened_out": 1, "parked": 0, "killed": 0, "qualified": 0, "new": 0, "promoted": 0, "reviewed": 0}
        return {"screened_out": 1, "parked": 2, "killed": 0, "qualified": 3, "new": 0, "promoted": 0, "reviewed": 0}

    def get_screening_summary(self, limit=10, run_id=""):
        return {"status_counts": {"screened_out": 1}, "reason_counts": {"generic_review_title": 1}}

    def list_build_briefs(self, run_id="", limit=10):
        return [
            SimpleNamespace(
                id=11,
                run_id="run-1",
                opportunity_id=6,
                validation_id=5,
                status="build_ready",
                recommended_output_type="workflow_reliability_console",
                updated_at="2026-04-01T10:00:00",
            )
        ]

    def list_build_prep_outputs(self, run_id="", limit=20):
        return [
            SimpleNamespace(build_brief_id=11, prep_stage="solution_framing", status="ready"),
            SimpleNamespace(build_brief_id=11, prep_stage="experiment_design", status="ready"),
            SimpleNamespace(build_brief_id=11, prep_stage="spec_generation", status="ready"),
        ]

    def get_candidate_workbench(self, limit=10, run_id=""):
        return [
            {
                "title": "Duct tape and spreadsheets ops pain",
                "state": "prototype_candidate",
                "selection_reason": "prototype_candidate_gate",
                "confidence_posture": "prototype_checkpoint",
                "repeatability_posture": "single_family_volatile",
                "family_confirmation_count": 1,
                "build_brief_present": True,
                "next_recommended_action": "prototype_now",
            }
        ]

    def get_backlog_workbench(self, limit=10):
        return [
            {
                "finding_id": 77,
                "title": "Spreadsheet-heavy reconciliation backlog",
                "backlog_priority_score": 8.4,
            }
        ]

    def get_discovery_feedback(self, source_name=None):
        rows = [
            {
                "source_name": "reddit",
                "query_text": "spreadsheet handoff failures",
                "runs": 4,
                "docs_seen": 12,
                "findings_emitted": 3,
                "validations": 2,
                "passes": 1,
                "prototype_candidates": 1,
                "build_briefs": 1,
                "last_latency_ms": 340,
                "last_status": "ok",
            }
        ]
        if source_name:
            return [row for row in rows if row["source_name"] == source_name]
        return rows

    def get_findings(self, limit=100):
        class _F:
            def __init__(self, finding_id, source, status, evidence):
                self.id = finding_id
                self.source = source
                self.status = status
                self.evidence = evidence
                self.source_class = "pain_signal"

        return [
            _F(5, "reddit-problem/smallbusiness", "qualified", {"discovery_sort": "new", "run_id": "run-1"}),
            _F(6, "reddit-problem/smallbusiness", "parked", {"discovery_sort": "top", "run_id": "run-1"}),
            _F(7, "reddit-problem/webdev", "qualified", {"discovery_sort": "new", "run_id": "run-1"}),
            _F(8, "reddit-problem/webdev", "killed", {"discovery_sort": "comments", "run_id": "run-2"}),
            _F(9, "github-problem", "qualified", {"run_id": "run-1"}),
        ]

    def get_raw_signals_by_finding(self, finding_id):
        return [{"id": 1}] if finding_id == 5 else []

    def get_problem_atoms_by_finding(self, finding_id):
        return [{"id": 1}] if finding_id == 5 else []

    def get_validation_review(self, limit=25, run_id=None):
        return [{"id": 5, "finding_id": 5}]


class DummyApp:
    def __init__(self):
        self.status_tracker = DummyStatusTracker()
        self.db = DummyDB()
        self.current_run_id = "run-1"
        self.agents = {}

    def runtime_paths(self):
        return {
            "db_path": "/tmp/autoresearch.db",
            "status_path": "/tmp/pipeline_status.json",
            "log_path": "/tmp/autoresearcher.log",
        }

    def summary_counts(self):
        return {"raw_signals": 2, "ledger_entries": 1}

    def decision_summary(self):
        return {"decision_mix": {"promote": 1, "park": 2}}

    def reddit_runtime_summary(self):
        return {
            "reddit_mode": "bridge_only",
            "reddit_bridge_hits": 4,
            "reddit_bridge_misses": 1,
            "reddit_fallback_queries": 0,
            "phases": {
                "discovery": {"reddit_bridge_hits": 2},
                "evidence": {"reddit_bridge_hits": 2},
            },
        }

    def validation_report(self, limit=10):
        return [{"finding_id": 5, "decision": "park", "recurrence_state": "thin", "recurrence_score": 0.22}]

    def run_diff(self, limit=10):
        return {"available": False}

    def review_report(self, limit=10):
        return [{"finding_id": 5, "decision": "park", "recurrence_state": "thin", "recurrence_score": 0.22}]

    def high_leverage_report(self, limit=10):
        return {
            "run_id": "run-1",
            "count": 1,
            "band_mix": {"strong": 1},
            "status_mix": {"candidate": 1},
            "findings": [
                {
                    "finding_id": 5,
                    "title": "Stripe payouts do not match QuickBooks invoices",
                    "high_leverage_score": 0.71,
                    "high_leverage_status": "candidate",
                    "high_leverage_band": "strong",
                    "evidence_tier": "one_family_strong",
                }
            ],
        }

    def snapshot(self):
        return {"raw_signals": [1, 2], "ledger_entries": [1], "validations": []}


def test_build_verbose_report_includes_runtime_paths_counts_and_logs():
    app = DummyApp()
    report = build_verbose_report(app, app.snapshot())

    assert report["runtime"]["db_path"] == "/tmp/autoresearch.db"
    assert report["counts"]["raw_signals"] == 2
    assert report["decisions"]["decision_mix"]["promote"] == 1
    assert report["reddit_runtime"]["reddit_mode"] == "bridge_only"
    assert report["reddit_runtime"]["reddit_bridge_hits"] == 4
    assert report["reddit_runtime"]["phases"]["evidence"]["reddit_bridge_hits"] == 2
    assert report["screening"]["screened_out"] == 1
    assert report["screening_all_time"]["parked"] == 2
    assert report["actionable_screening"]["parked"] == 1
    assert report["candidate_workbench"][0]["next_recommended_action"] == "prototype_now"
    assert report["decision_surface"][0]["next_recommended_action"] == "prototype_now"
    assert report["builder_jobs"][0]["builder_status"] == "ready_to_build"
    assert report["high_leverage"]["findings"][0]["finding_id"] == 5
    assert report["operator_report"]["money_surface"]["prototype_now_count"] == 1
    assert report["review"][0]["finding_id"] == 5
    assert "stage=discovery" in report["recent_logs"][0]


def test_build_builder_jobs_view_derives_queue_from_briefs_and_prep_outputs():
    rows = build_builder_jobs_view(DummyDB(), run_id="run-1", limit=10)

    assert rows[0]["build_brief_id"] == 11
    assert rows[0]["builder_status"] == "ready_to_build"
    assert rows[0]["prep_output_count"] == 3
    assert rows[0]["ready_for_build"] is True


def test_build_operator_report_combines_health_sources_and_build_queue():
    report = build_operator_report(DummyApp(), limit=5)

    assert report["pipeline_health"]["actionable_qualified_for_pipeline"] == 1
    assert report["source_health"]["top_sources"][0]["source_name"] == "reddit"
    assert report["money_surface"]["prototype_now_count"] == 1
    assert report["money_surface"]["build_ready_count"] == 1
    assert report["money_surface"]["builder_job_status_mix"]["ready_to_build"] == 1
    assert report["high_leverage"]["findings"][0]["high_leverage_status"] == "candidate"
    assert report["operator_focus"]["recommended_focus"] == "prototype_now"


def test_cli_run_applies_pattern_and_fresh_before_dispatch(monkeypatch):
    captured = {}

    class DummyRunApp:
        def __init__(self, config_path=None):
            self.config = {
                "discovery": {
                    "sources": ["reddit", "web", "github", "shopify_reviews"],
                    "web": {"keywords": ["original web"]},
                    "reddit": {"problem_keywords": ["original reddit"], "theme_keywords": {}},
                    "github": {"problem_keywords": ["original github"]},
                }
            }
            self.discovery_bypass_cache = False

        async def run(self):
            captured["sources"] = list(self.config["discovery"]["sources"])
            captured["focused_problem_only"] = self.config["discovery"].get("focused_problem_only")
            captured["web_keywords"] = list(self.config["discovery"]["web"]["keywords"])
            captured["web_problem_keywords"] = list(self.config["discovery"]["web"]["problem_keywords"])
            captured["web_success_keywords"] = list(self.config["discovery"]["web"]["success_keywords"])
            captured["web_market_keywords"] = list(self.config["discovery"]["web"]["market_keywords"])
            captured["reddit_keywords"] = list(self.config["discovery"]["reddit"]["problem_keywords"])
            captured["github_keywords"] = list(self.config["discovery"]["github"]["problem_keywords"])
            captured["bypass_cache"] = self.discovery_bypass_cache

    monkeypatch.setattr(cli, "AutoResearcher", DummyRunApp)
    monkeypatch.setattr(sys, "argv", ["cli.py", "run", "--pattern", "bank_reconciliation", "--fresh"])

    asyncio.run(cli.main())

    assert captured["bypass_cache"] is True
    assert captured["sources"] == ["reddit", "web", "github"]
    assert captured["focused_problem_only"] is True
    assert captured["web_keywords"][0] == "bank reconciliation manual process"
    assert captured["web_problem_keywords"][0] == "bank reconciliation manual process"
    assert captured["web_success_keywords"] == []
    assert captured["web_market_keywords"] == []
    assert captured["reddit_keywords"][0] == "bank reconciliation manual process"
    assert captured["github_keywords"][0] == "bank reconciliation manual process"


def test_cli_run_once_restores_pattern_overrides_after_dispatch(monkeypatch):
    captured = {}

    class DummyRunOnceApp:
        last_instance = None

        def __init__(self, config_path=None):
            self.config = {
                "discovery": {
                    "sources": ["reddit", "web", "github", "youtube"],
                    "web": {
                        "keywords": ["original web"],
                        "problem_keywords": ["original web problem"],
                        "success_keywords": ["original web success"],
                        "market_keywords": ["original web market"],
                    },
                    "reddit": {"problem_keywords": ["original reddit"], "theme_keywords": {}},
                    "github": {"problem_keywords": ["original github"]},
                }
            }
            self.discovery_bypass_cache = False
            DummyRunOnceApp.last_instance = self

        async def run_once(self, *, skip_backlog=False):
            captured["sources_during_run"] = list(self.config["discovery"]["sources"])
            captured["focused_problem_only_during_run"] = self.config["discovery"].get("focused_problem_only")
            captured["web_keywords_during_run"] = list(self.config["discovery"]["web"]["keywords"])
            captured["web_problem_keywords_during_run"] = list(self.config["discovery"]["web"]["problem_keywords"])
            captured["web_success_keywords_during_run"] = list(self.config["discovery"]["web"]["success_keywords"])
            captured["web_market_keywords_during_run"] = list(self.config["discovery"]["web"]["market_keywords"])
            captured["reddit_keywords_during_run"] = list(self.config["discovery"]["reddit"]["problem_keywords"])
            captured["github_keywords_during_run"] = list(self.config["discovery"]["github"]["problem_keywords"])
            captured["bypass_cache_during_run"] = self.discovery_bypass_cache
            captured["skip_backlog_during_run"] = skip_backlog
            return {"ok": True}

    monkeypatch.setattr(cli, "AutoResearcher", DummyRunOnceApp)
    monkeypatch.setattr(sys, "argv", ["cli.py", "run-once", "--pattern", "bank_reconciliation", "--fresh"])

    asyncio.run(cli.main())

    app = DummyRunOnceApp.last_instance
    assert captured["bypass_cache_during_run"] is True
    assert captured["skip_backlog_during_run"] is False
    assert captured["sources_during_run"] == ["reddit", "web", "github"]
    assert captured["focused_problem_only_during_run"] is True
    assert captured["web_keywords_during_run"][0] == "bank reconciliation manual process"
    assert captured["web_problem_keywords_during_run"][0] == "bank reconciliation manual process"
    assert captured["web_success_keywords_during_run"] == []
    assert captured["web_market_keywords_during_run"] == []
    assert captured["reddit_keywords_during_run"][0] == "bank reconciliation manual process"
    assert captured["github_keywords_during_run"][0] == "bank reconciliation manual process"
    assert app.config["discovery"]["sources"] == ["reddit", "web", "github", "youtube"]
    assert app.config["discovery"]["focused_problem_only"] is False
    assert app.config["discovery"]["web"]["keywords"] == ["original web"]
    assert app.config["discovery"]["web"]["problem_keywords"] == ["original web problem"]
    assert app.config["discovery"]["web"]["success_keywords"] == ["original web success"]
    assert app.config["discovery"]["web"]["market_keywords"] == ["original web market"]
    assert app.config["discovery"]["reddit"]["problem_keywords"] == ["original reddit"]
    assert app.config["discovery"]["github"]["problem_keywords"] == ["original github"]


def test_cli_run_once_passes_skip_backlog_flag(monkeypatch):
    captured = {}

    class DummyRunOnceApp:
        def __init__(self, config_path=None):
            self.config = {"discovery": {"web": {"keywords": []}, "reddit": {"problem_keywords": [], "theme_keywords": {}}}}
            self.discovery_bypass_cache = False
            self.shutdown_calls = 0

        async def run_once(self, *, skip_backlog=False):
            captured["skip_backlog"] = skip_backlog
            return {"ok": True}

        async def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(cli, "AutoResearcher", DummyRunOnceApp)
    monkeypatch.setattr(sys, "argv", ["cli.py", "run-once", "--skip-backlog"])

    asyncio.run(cli.main())

    assert captured["skip_backlog"] is True


def test_cli_run_once_shuts_down_app_when_run_fails(monkeypatch):
    captured = {}

    class DummyRunOnceApp:
        last_instance = None

        def __init__(self, config_path=None):
            self.config = {"discovery": {"web": {"keywords": []}, "reddit": {"problem_keywords": [], "theme_keywords": {}}}}
            self.discovery_bypass_cache = False
            self.shutdown_calls = 0
            DummyRunOnceApp.last_instance = self

        async def run_once(self, *, skip_backlog=False):
            raise RuntimeError("boom")

        async def shutdown(self):
            self.shutdown_calls += 1
            captured["shutdown_calls"] = self.shutdown_calls

    monkeypatch.setattr(cli, "AutoResearcher", DummyRunOnceApp)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(cli.cmd_run_once(SimpleNamespace(pattern=None, fresh=False, skip_backlog=False, verbose=False, config="config.yaml"), None))

    assert captured["shutdown_calls"] == 1


def test_render_watch_snapshot_shows_runtime_and_live_counts():
    rendered = render_watch_snapshot(
        {
            "stage": "validation",
            "status": "running",
            "discoveries": 3,
            "rawSignals": 2,
            "problemAtoms": 2,
            "clusters": 1,
            "experiments": 1,
            "ledgerEntries": 4,
            "validated": 1,
            "logs": ["12:00:00 stage=validation", "12:00:01 decision=park"],
        },
        {
            "db_path": "/tmp/autoresearch.db",
            "status_path": "/tmp/pipeline_status.json",
            "log_path": "/tmp/autoresearcher.log",
        },
    )

    assert "stage=validation" in rendered
    assert "status=running" in rendered
    assert "raw_signals=2" in rendered
    assert "db_path" in rendered
    assert "decision=park" in rendered


def test_build_discovery_sort_diagnostics_groups_by_sort_and_status():
    db = DummyDB()
    report = build_discovery_sort_diagnostics(db, run_id="run-1")

    assert report["rows_examined"] == 3
    assert report["sort_counts"]["new"] == 2
    assert report["sort_counts"]["top"] == 1
    assert report["sort_status_counts"]["new"]["qualified"] == 2
    assert report["top_subreddits"][0]["subreddit"] == "smallbusiness"


def test_cli_eval_runs_from_another_cwd_via_absolute_path():
    temp_dir = tempfile.mkdtemp()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [sys.executable, str(repo_root / "cli.py"), "eval"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        os.rmdir(temp_dir)

    assert '"passed_cases": 15' in result.stdout


def test_resolve_database_path_from_config_uses_configured_database_location(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  path: data/custom.db\n",
        encoding="utf-8",
    )

    db_path = resolve_database_path_from_config(config_path)

    assert db_path == Path(__file__).resolve().parents[1] / "data" / "custom.db"


@pytest.mark.parametrize(
    ("argv", "module_name", "module_attr", "module_value", "expected_fragment"),
    [
        (
            ["cli.py", "suggest-discovery", "--limit", "1"],
            "src.discovery_suggestions",
            "build_discovery_suggestions",
            lambda db, **kwargs: {"ok": True, "command": "suggest-discovery"},
            '"command": "suggest-discovery"',
        ),
        (
            ["cli.py", "deep-research", "--max-findings", "1"],
            "src.agents.deep_research",
            "DeepResearchAgent",
            type(
                "DeepResearchAgent",
                (),
                {
                    "__init__": lambda self, name, db, vertical="devtools": None,
                    "run_deep_research": _fake_deep_research_run,
                },
            ),
            '"command": "deep-research"',
        ),
        (
            ["cli.py", "gate-diagnostics"],
            "src.gate_diagnostics",
            "build_gate_diagnostics_report",
            lambda db, config, run_id=None, limit=25, finding_id=None: {"ok": True, "command": "gate-diagnostics"},
            '"command": "gate-diagnostics"',
        ),
        (
            ["cli.py", "pipeline-health"],
            "src.pipeline_health",
            "compute_pipeline_health",
            lambda db: {"ok": True, "command": "pipeline-health"},
            '"command": "pipeline-health"',
        ),
    ],
)
def test_cli_command_import_paths_use_src_modules(monkeypatch, capsys, argv, module_name, module_attr, module_value, expected_fragment):
    class DummyAppForCommands:
        def __init__(self, config_path=None):
            self.config = {"discovery": {"reddit": {"theme_keywords": {}}}}
            self.db = DummyDB()
            self.current_run_id = "run-1"
            self.status_tracker = DummyStatusTracker()

        async def initialize(self, start_new_run=False):
            return None

        async def shutdown(self):
            return None

    module = types.ModuleType(module_name)
    setattr(module, module_attr, module_value)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(cli, "AutoResearcher", DummyAppForCommands)
    monkeypatch.setattr(sys, "argv", argv)

    asyncio.run(cli.main())

    output = capsys.readouterr().out
    assert expected_fragment in output
