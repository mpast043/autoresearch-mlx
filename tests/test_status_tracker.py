"""Tests for StatusTracker."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from status_tracker import StatusTracker


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "output")


class TestStatusTrackerInit:
    """Tests for StatusTracker initialization."""

    def test_creates_output_directory(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        assert Path(tmp_output_dir).exists()

    def test_default_status_fields(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        assert tracker.status["stage"] == "idle"
        assert tracker.status["status"] == "idle"
        assert tracker.status["runId"] == ""
        assert tracker.status["startTime"] == ""
        assert tracker.status["discoveries"] == 0
        assert tracker.status["rawSignals"] == 0
        assert tracker.status["problemAtoms"] == 0
        assert tracker.status["clusters"] == 0
        assert tracker.status["experiments"] == 0
        assert tracker.status["ledgerEntries"] == 0
        assert tracker.status["validated"] == 0
        assert tracker.status["ideas"] == 0
        assert tracker.status["built"] == 0

    def test_writes_json_file_on_init(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["stage"] == "idle"


class TestStatusTrackerStartRun:
    """Tests for start_run method."""

    def test_sets_run_id(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.start_run("run-123")
        assert tracker.status["runId"] == "run-123"

    def test_persists_run_id(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.start_run("run-abc")
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        data = json.loads(status_file.read_text())
        assert data["runId"] == "run-abc"


class TestStatusTrackerSetStage:
    """Tests for set_stage method."""

    def test_sets_stage(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("discovery")
        assert tracker.status["stage"] == "discovery"
        assert tracker.status["status"] == "running"

    def test_sets_start_time_on_first_stage(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("discovery")
        assert tracker.status["startTime"] != ""

    def test_does_not_override_start_time_on_subsequent_stages(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("discovery")
        first_time = tracker.status["startTime"]
        tracker.set_stage("validation")
        assert tracker.status["startTime"] == first_time

    def test_logs_stage_change(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("discovery")
        log_entries = tracker.status["logs"]
        assert any("stage=discovery" in entry for entry in log_entries)


class TestStatusTrackerUpdate:
    """Tests for update method."""

    def test_updates_single_field(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.update(discoveries=5)
        assert tracker.status["discoveries"] == 5

    def test_updates_multiple_fields(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.update(discoveries=10, rawSignals=3, problemAtoms=1)
        assert tracker.status["discoveries"] == 10
        assert tracker.status["rawSignals"] == 3
        assert tracker.status["problemAtoms"] == 1

    def test_persists_update(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.update(discoveries=7)
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        data = json.loads(status_file.read_text())
        assert data["discoveries"] == 7


class TestStatusTrackerLog:
    """Tests for log method."""

    def test_adds_log_entry(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.log("test message")
        assert len(tracker.status["logs"]) >= 1
        assert any("test message" in entry for entry in tracker.status["logs"])

    def test_log_entry_has_timestamp(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.log("test message")
        # Timestamp format: HH:MM:SS
        latest_log = tracker.status["logs"][-1]
        # The log entry format is "HH:MM:SS message"
        assert " " in latest_log

    def test_log_rotation_max_200(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        for i in range(250):
            tracker.log(f"message {i}")
        assert len(tracker.status["logs"]) == 200

    def test_log_rotation_keeps_recent(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        for i in range(250):
            tracker.log(f"message {i}")
        # Should keep the latest 200 messages
        first_kept = tracker.status["logs"][0]
        assert "message 50" in first_kept
        last_kept = tracker.status["logs"][-1]
        assert "message 249" in last_kept


class TestStatusTrackerComplete:
    """Tests for complete method."""

    def test_sets_completed_status(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("validation")
        tracker.complete()
        assert tracker.status["status"] == "completed"
        assert tracker.status["stage"] == "completed"

    def test_logs_completion(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.complete()
        assert any("pipeline completed" in entry for entry in tracker.status["logs"])


class TestStatusTrackerFail:
    """Tests for fail method."""

    def test_sets_failed_status(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.set_stage("discovery")
        tracker.fail("connection timeout")
        assert tracker.status["status"] == "failed"

    def test_logs_error_message(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.fail("connection timeout")
        assert any("error=connection timeout" in entry for entry in tracker.status["logs"])


class TestStatusTrackerReset:
    """Tests for reset method."""

    def test_resets_all_fields(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.start_run("run-123")
        tracker.set_stage("discovery")
        tracker.update(discoveries=10, rawSignals=5)
        tracker.reset()
        assert tracker.status["runId"] == ""
        assert tracker.status["stage"] == "idle"
        assert tracker.status["status"] == "idle"
        assert tracker.status["discoveries"] == 0
        assert tracker.status["rawSignals"] == 0

    def test_persists_reset(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.start_run("run-123")
        tracker.reset()
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        data = json.loads(status_file.read_text())
        assert data["runId"] == ""
        assert data["stage"] == "idle"


class TestStatusTrackerJsonPersistence:
    """Tests for JSON file persistence."""

    def test_json_file_is_valid(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        data = json.loads(status_file.read_text())
        assert isinstance(data, dict)
        assert "stage" in data

    def test_json_file_updates_on_each_operation(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        tracker.start_run("run-1")
        status_file = Path(tmp_output_dir) / "pipeline_status.json"

        data1 = json.loads(status_file.read_text())
        assert data1["runId"] == "run-1"

        tracker.update(discoveries=5)
        data2 = json.loads(status_file.read_text())
        assert data2["discoveries"] == 5

    def test_json_is_indented(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)
        status_file = Path(tmp_output_dir) / "pipeline_status.json"
        content = status_file.read_text()
        # Indented JSON has newlines
        assert "\n" in content


class TestStatusTrackerStateTransitions:
    """Tests for full state transition sequences."""

    def test_full_pipeline_lifecycle(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)

        # Start
        tracker.start_run("run-001")
        assert tracker.status["runId"] == "run-001"

        # Discovery stage
        tracker.set_stage("discovery")
        assert tracker.status["stage"] == "discovery"
        assert tracker.status["status"] == "running"

        # Update counts
        tracker.update(discoveries=10, rawSignals=8, problemAtoms=3)
        assert tracker.status["discoveries"] == 10

        # Validation stage
        tracker.set_stage("validation")
        assert tracker.status["stage"] == "validation"

        # Complete
        tracker.complete()
        assert tracker.status["status"] == "completed"
        assert tracker.status["stage"] == "completed"

    def test_failure_lifecycle(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)

        tracker.start_run("run-002")
        tracker.set_stage("discovery")
        tracker.fail("API rate limit exceeded")

        assert tracker.status["status"] == "failed"
        assert any("API rate limit exceeded" in entry for entry in tracker.status["logs"])

    def test_reset_and_restart(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)

        tracker.start_run("run-003")
        tracker.set_stage("discovery")
        tracker.update(discoveries=5)

        tracker.reset()
        assert tracker.status["runId"] == ""
        assert tracker.status["stage"] == "idle"

        # Restart
        tracker.start_run("run-004")
        assert tracker.status["runId"] == "run-004"

    def test_log_rotation_during_lifecycle(self, tmp_output_dir):
        tracker = StatusTracker(output_dir=tmp_output_dir)

        tracker.start_run("run-005")
        for i in range(50):
            tracker.log(f"progress {i}")

        assert len(tracker.status["logs"]) == 50  # start_run does not add a log entry