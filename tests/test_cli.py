"""Tests for CLI operator-facing helpers."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli import build_verbose_report, render_watch_snapshot


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
        return []

    def list_build_prep_outputs(self, run_id="", limit=20):
        return []

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
    assert report["review"][0]["finding_id"] == 5
    assert "stage=discovery" in report["recent_logs"][0]


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
