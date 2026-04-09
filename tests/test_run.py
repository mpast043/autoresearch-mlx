"""Tests for runtime snapshot surfaces."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.messaging import MessageType

from src.database import (
    CorroborationRecord,
    Database,
    EvidenceLedgerEntry,
    Finding,
    MarketEnrichment,
    Opportunity,
    OpportunityCluster,
    ProblemAtom,
    RawSignal,
    Validation,
    ValidationExperiment,
)
from run import AutoResearcher
from src.runtime.paths import DEFAULT_CONFIG_PATH, DEFAULT_EVAL_PATH


def test_autoresearcher_resolves_runtime_paths_from_project_root_outside_cwd():
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        app = AutoResearcher(config_path=str(DEFAULT_CONFIG_PATH))

        repo_root = Path(__file__).resolve().parents[1]
        assert app.db_path == repo_root / "data" / "autoresearch.db"
        assert app.output_dir == repo_root / "output"
        assert app.runtime_paths()["status_path"] == str((repo_root / "output" / "pipeline_status.json").resolve())
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_packaged_default_runtime_resources_exist():
    assert DEFAULT_CONFIG_PATH.exists()
    assert DEFAULT_EVAL_PATH.exists()


def test_snapshot_includes_evidence_layers_and_ledger():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("test-run")

        finding_id = db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/1",
                entrepreneur="Ops Lead",
                product_built="Manual reconciliation workflow",
                content_hash="snapshot-finding",
                finding_kind="problem_signal",
                evidence={"discovery_query": "manual reconciliation", "run_id": "test-run"},
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                source_name="reddit",
                source_type="forum",
                source_url="https://reddit.com/r/ops/comments/1",
                title="Manual reconciliation workflow",
                body_excerpt="Teams still use spreadsheets every week.",
                quote_text="Teams still use spreadsheets every week.",
                role_hint="operations manager",
                content_hash="snapshot-signal",
                finding_id=finding_id,
            )
        )
        atom_id = db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-manual-reconciliation",
                segment="small business operations",
                user_role="operations manager",
                job_to_be_done="reconcile payouts",
                trigger_event="after payout runs",
                pain_statement="Teams still use spreadsheets every week.",
                failure_mode="exceptions break the current workflow",
                current_workaround="spreadsheets",
                current_tools="Excel",
                urgency_clues="every week",
                frequency_clues="every week",
                emotional_intensity=0.6,
                cost_consequence_clues="time",
                why_now_clues="pricing change",
                confidence=0.8,
                atom_json="{}",
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                cluster_key="ops-manual-reconciliation",
                label="Operations - reconcile payouts",
                segment="small business operations",
                user_role="operations manager",
                job_to_be_done="reconcile payouts",
                trigger_summary="after payout runs",
                signal_count=1,
                atom_count=1,
                evidence_quality=0.72,
                summary={"sample_atoms": [atom_id]},
            )
        )
        opportunity_id = db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="Payout reconciliation wedge",
                market_gap="underserved_edge_case",
                recommendation="promote",
                status="promoted",
                pain_severity=0.8,
                frequency_score=0.7,
                cost_of_inaction=0.75,
                workaround_density=0.9,
                urgency_score=0.7,
                segment_concentration=0.8,
                reachability=0.65,
                timing_shift=0.55,
                buildability=0.7,
                expansion_potential=0.6,
                education_burden=0.2,
                dependency_risk=0.25,
                adoption_friction=0.3,
                evidence_quality=0.72,
                composite_score=0.74,
                confidence=0.76,
                notes={"rationale": "Strong workaround behavior."},
            )
        )
        db.insert_experiment(
            ValidationExperiment(
                opportunity_id=opportunity_id,
                cluster_id=cluster_id,
                test_type="workflow_walkthrough",
                hypothesis="Users will share the spreadsheet if the pain is real.",
                falsifier="Nobody shares the workflow.",
                smallest_test="Run 5 walkthroughs.",
                success_signal="Users show the workflow.",
                failure_signal="Users refuse to show the workflow.",
            )
        )
        db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="opportunity",
                entity_id=opportunity_id,
                entry_kind="decision",
                summary="Promoted for validation follow-up.",
                entry_json={"decision": "promote"},
                run_id="test-run",
            )
        )
        db.insert_validation(
            Validation(
                finding_id=finding_id,
                market_score=0.8,
                technical_score=0.75,
                distribution_score=0.7,
                overall_score=0.76,
                passed=True,
                evidence={"opportunity_id": opportunity_id},
                run_id="test-run",
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.current_run_id = "test-run"
        app.agents = {
            "discovery": type("DiscoveryStub", (), {"reddit_runtime_summary": lambda self: {
                "reddit_mode": "bridge_with_fallback",
                "reddit_bridge_hits": 3,
                "reddit_bridge_misses": 1,
                "reddit_fallback_queries": 1,
                "seeded_total_pairs": 24,
                "seeded_pairs_uncovered": 2,
            }})(),
            "evidence": type("EvidenceStub", (), {"reddit_runtime_summary": lambda self: {
                "reddit_mode": "bridge_with_fallback",
                "reddit_bridge_hits": 4,
                "reddit_bridge_misses": 2,
                "reddit_fallback_queries": 0,
                "reddit_validation_seed_runs": 1,
                "reddit_validation_seeded_pairs": 6,
            }})(),
            "validation": type("ValidationStub", (), {"reddit_runtime_summary": lambda self: {
                "reddit_mode": "bridge_with_fallback",
                "reddit_bridge_hits": 1,
                "reddit_bridge_misses": 1,
                "reddit_fallback_queries": 0,
            }})(),
        }
        snapshot = app.snapshot()

        assert snapshot["counts"]["raw_signals"] == 1
        assert snapshot["counts"]["problem_atoms"] == 1
        assert snapshot["counts"]["clusters"] == 1
        assert snapshot["counts"]["opportunities"] == 1
        assert snapshot["counts"]["experiments"] == 1
        assert snapshot["validation"][0]["finding_id"] == finding_id
        assert snapshot["reddit_runtime"]["reddit_mode"] == "bridge_with_fallback"
        assert snapshot["reddit_runtime"]["reddit_bridge_hits"] == 8
        assert snapshot["reddit_runtime"]["reddit_bridge_misses"] == 4
        assert snapshot["reddit_runtime"]["reddit_validation_seed_runs"] == 1
        assert snapshot["reddit_runtime"]["seeded_total_pairs"] == 24
        assert snapshot["reddit_runtime"]["phases"]["evidence"]["reddit_validation_seeded_pairs"] == 6
        assert snapshot["screening_summary"]["run_id"] == "test-run"
        assert snapshot["screening"]["new"] == 1
        assert snapshot["screening_all_time"]["new"] == 1
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_validation_report_exposes_recurrence_and_enrichment_fields():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("test-run")
        finding_id = db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/sysadmin/comments/1",
                entrepreneur="Ops Lead",
                product_built="Manual device setup thread",
                content_hash="report-finding",
                finding_kind="problem_signal",
                evidence={"discovery_query": "manual process"},
            )
        )
        db.insert_validation(
            Validation(
                finding_id=finding_id,
                market_score=0.3,
                technical_score=0.7,
                distribution_score=0.6,
                overall_score=0.28,
                passed=False,
                evidence={
                    "decision": "park",
                    "decision_reason": "park_recurrence",
                    "market_gap_state": "needs_more_recurrence_evidence",
                    "evidence": {
                        "recurrence_state": "thin",
                        "recurrence_query_coverage": 0.25,
                        "recurrence_doc_count": 2,
                    },
                    "opportunity_scorecard": {
                        "problem_plausibility": 0.52,
                        "evidence_sufficiency": 0.31,
                        "value_support": 0.29,
                        "composite_score": 0.28,
                    },
                },
                run_id="test-run",
            )
        )
        db.upsert_corroboration(
            CorroborationRecord(
                finding_id=finding_id,
                recurrence_state="thin",
                recurrence_score=0.22,
                corroboration_score=0.34,
                evidence_sufficiency=0.31,
                query_coverage=0.25,
                independent_confirmations=2,
                source_diversity=1,
                query_set_hash="abc",
                evidence={"corroboration_score": 0.34, "independent_confirmations": 2, "source_diversity": 1},
            )
        )
        db.upsert_market_enrichment(
            MarketEnrichment(
                finding_id=finding_id,
                demand_score=0.18,
                buyer_intent_score=0.21,
                competition_score=0.11,
                trend_score=0.14,
                evidence={"review_count": 9},
            )
        )
        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.current_run_id = "test-run"
        report = app.validation_report(limit=10)
        assert report[0]["decision"] == "park"
        assert report[0]["recurrence_state"] == "thin"
        assert report[0]["corroboration_score"] == 0.34
        assert report[0]["value_support"] == 0.29
        assert report[0]["demand_score"] == 0.18
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_completion_state_reports_queue_and_worker_activity():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("test-run")
        finding_id = db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/2",
                entrepreneur="Ops Lead",
                product_built="Manual handoff",
                content_hash="completion-state-finding",
                finding_kind="problem_signal",
                source_class="pain_signal",
                evidence={"run_id": "test-run"},
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                source_name="reddit",
                source_type="forum",
                source_url="https://reddit.com/r/ops/comments/2",
                title="Manual handoff",
                body_excerpt="Teams still re-enter data by hand.",
                quote_text="Teams still re-enter data by hand.",
                role_hint="ops lead",
                content_hash="completion-state-signal",
                finding_id=finding_id,
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-manual-handoff",
                segment="small business operations",
                user_role="ops lead",
                job_to_be_done="hand off work cleanly",
                trigger_event="after intake",
                pain_statement="Teams still re-enter data by hand.",
                failure_mode="handoff breaks and requires copy-paste",
                current_workaround="copying between systems",
                current_tools="email sheets",
                urgency_clues="daily",
                frequency_clues="daily",
                emotional_intensity=0.5,
                cost_consequence_clues="time",
                why_now_clues="growth",
                confidence=0.7,
                atom_json="{}",
            )
        )
        db.update_finding_status(finding_id, "qualified")

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.orchestrator = type(
            "OrchestratorStub",
            (),
            {"_message_queue": type("QueueStub", (), {"empty": lambda self: False, "qsize": lambda self: 2})()},
        )()
        app.agents = {
            "evidence": type("EvidenceStub", (), {"busy_count": lambda self: 1})(),
            "validation": type("ValidationStub", (), {"busy_count": lambda self: 0})(),
        }

        state = app.completion_state()

        assert state["queue_empty"] is False
        assert state["queue_size"] == 2
        assert state["open_qualified"] == 1
        assert state["evidence_busy"] == 1
        assert state["validation_busy"] == 0
        assert state["total_busy"] == 1
        assert state["busy_agents"]["evidence"] == 1
        assert state["drained"] is False
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_count_actionable_qualified_findings_ignores_non_actionable_backlog():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        actionable_id = db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/1",
                product_built="Actionable workflow issue",
                content_hash="actionable-qualified",
                status="qualified",
                source_class="pain_signal",
            )
        )
        inert_id = db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/2",
                product_built="Thin generic complaint",
                content_hash="inert-qualified",
                status="qualified",
                source_class="pain_signal",
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=actionable_id,
                source_name="reddit",
                source_type="forum",
                source_url="https://reddit.com/r/ops/comments/1",
                title="Actionable workflow issue",
                body_excerpt="Manual reconciliation keeps breaking.",
                content_hash="actionable-qualified-signal",
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                finding_id=actionable_id,
                signal_id=signal_id,
                cluster_key="ops|reconcile",
                user_role="operator",
                job_to_be_done="keep reconciliation reliable",
                pain_statement="Manual reconciliation keeps breaking.",
                trigger_event="after imports",
                failure_mode="manual reconciliation keeps breaking",
                current_workaround="spreadsheets",
                confidence=0.7,
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db

        assert app._count_actionable_qualified_findings() == 1
        assert actionable_id != inert_id
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_dispatch_open_qualified_findings_requeues_backlog():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("test-run")

        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/smallbusiness",
                source_url="https://reddit.com/r/smallbusiness/comments/1",
                entrepreneur="Ops Lead",
                product_built="Spreadsheet-heavy operations",
                outcome_summary="Still reconciling everything manually.",
                content_hash="qualified-finding",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=finding_id,
                source_name="reddit",
                source_type="forum",
                source_url="https://reddit.com/r/smallbusiness/comments/1",
                title="Spreadsheet-heavy operations",
                body_excerpt="Still reconciling everything manually.",
                content_hash="qualified-signal",
            )
        )
        atom_id = db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-spreadsheet",
                segment="small business ops",
                user_role="ops lead",
                job_to_be_done="keep operations data in sync",
                pain_statement="Still reconciling everything manually.",
                trigger_event="weekly close",
                failure_mode="manual reconciliation piles up",
                current_workaround="spreadsheets",
                current_tools="Excel",
                confidence=0.8,
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.current_run_id = "test-run"

        class StubOrchestrator:
            def __init__(self):
                self.sent = []

            async def send_message(self, **kwargs):
                self.sent.append(kwargs)

        app.orchestrator = StubOrchestrator()
        backlog_ids = asyncio.run(app._dispatch_open_qualified_findings())

        assert backlog_ids == [finding_id]
        assert len(app.orchestrator.sent) == 1
        sent = app.orchestrator.sent[0]
        assert sent["to_agent"] == "orchestrator"
        assert sent["msg_type"] == MessageType.FINDING
        assert sent["payload"]["finding_id"] == finding_id
        assert sent["payload"]["signal_id"] == signal_id
        assert sent["payload"]["problem_atom_ids"] == [atom_id]
        assert sent["payload"]["backlog_requeue"] is True
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_dispatch_open_qualified_findings_skips_stale_backlog_under_current_gates():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("test-run")

        valid_finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/ecommerce",
                source_url="https://reddit.com/r/ecommerce/comments/1",
                entrepreneur="Ops Lead",
                product_built='How are u guys managing fulfilment between "order received" and "label printed"?',
                outcome_summary="The team coordinates through WhatsApp and spreadsheets.",
                content_hash="qualified-valid",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
                evidence={"discovery_query": '"order received" "label printed" whatsapp spreadsheet'},
            )
        )
        valid_signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=valid_finding_id,
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://reddit.com/r/ecommerce/comments/1",
                title='How are u guys managing fulfilment between "order received" and "label printed"?',
                body_excerpt="Received -> Picking -> Processing -> Packed -> Shipped. We coordinate through WhatsApp and spreadsheets.",
                content_hash="qualified-valid-signal",
            )
        )
        valid_atom_id = db.insert_problem_atom(
            ProblemAtom(
                signal_id=valid_signal_id,
                finding_id=valid_finding_id,
                cluster_key="ops-fulfillment-gap",
                segment="shopify merchants",
                user_role="operator",
                job_to_be_done="keep operations data in sync without manual cleanup",
                pain_statement="The team coordinates through WhatsApp and spreadsheets.",
                trigger_event="Received -> Picking -> Processing -> Packed -> Shipped",
                failure_mode="manual coordination between order received and label printed",
                current_workaround="spreadsheets, manual work",
                current_tools="Shopify, WhatsApp",
                confidence=0.8,
                atom_json=json.dumps(
                    {
                        "is_specific_problem": True,
                        "specific_patterns": [{"pattern": "fulfillment_gap", "confidence": 0.9}],
                    }
                ),
            )
        )

        stale_finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/accounting",
                source_url="https://reddit.com/r/accounting/comments/2",
                entrepreneur="Analyst",
                product_built="Resume roast",
                outcome_summary="Feeling lost with career, please roast my resume.",
                content_hash="qualified-stale",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
                evidence={"discovery_query": "which spreadsheet is latest"},
            )
        )
        stale_signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=stale_finding_id,
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://reddit.com/r/accounting/comments/2",
                title="Feeling lost with career, please roast my resume.",
                body_excerpt="Please dissect my resume and help me transition to a larger firm.",
                content_hash="qualified-stale-signal",
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                signal_id=stale_signal_id,
                finding_id=stale_finding_id,
                cluster_key="resume-thread",
                segment="operators with recurring workflow pain",
                user_role="finance lead",
                job_to_be_done="improve my resume",
                pain_statement="Please dissect my resume",
                trigger_event="trying to transition to a larger firm",
                failure_mode="resume is too wordy",
                current_workaround="",
                confidence=0.4,
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.current_run_id = "test-run"

        class StubOrchestrator:
            def __init__(self):
                self.sent = []

            async def send_message(self, **kwargs):
                self.sent.append(kwargs)

        app.orchestrator = StubOrchestrator()
        backlog_ids = asyncio.run(app._dispatch_open_qualified_findings())

        assert backlog_ids == [valid_finding_id]
        assert len(app.orchestrator.sent) == 1
        sent = app.orchestrator.sent[0]
        assert sent["payload"]["finding_id"] == valid_finding_id
        assert sent["payload"]["signal_id"] == valid_signal_id
        assert sent["payload"]["problem_atom_ids"] == [valid_atom_id]
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_run_once_can_skip_backlog_requeue(monkeypatch):
    app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    app.current_run_id = "test-run"

    async def fake_initialize(start_new_run=True):
        app.db = object()
        app.orchestrator = type("OrchestratorStub", (), {"start": _start})()
        app.agents = {"discovery": type("DiscoveryStub", (), {"_discover_once": _discover_once})()}

    async def _start(self, skip_agents=None):
        return None

    async def _discover_once(self):
        return [101]

    async def fake_wait(max_wait_seconds):
        return True

    async def fake_shutdown():
        return None

    async def fake_dispatch():
        raise AssertionError("backlog should not be dispatched in discovery-only mode")

    monkeypatch.setattr(app, "initialize", fake_initialize)
    monkeypatch.setattr(app, "_dispatch_open_qualified_findings", fake_dispatch)
    monkeypatch.setattr(app, "_wait_for_pipeline_drained", fake_wait)
    monkeypatch.setattr(app, "shutdown", fake_shutdown)
    monkeypatch.setattr(app, "snapshot", lambda: {"ok": True})

    summary = asyncio.run(app.run_once(skip_backlog=True))

    assert summary["backlog_ids"] == []
    assert summary["discovery_ids"] == [101]
    assert summary["skip_backlog"] is True


def test_completion_state_ignores_backlog_in_discovery_only_mode():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()

        finding_id = db.insert_finding(
            Finding(
                source="reddit-problem/smallbusiness",
                source_url="https://reddit.com/r/smallbusiness/comments/1",
                entrepreneur="Ops Lead",
                product_built="Spreadsheet-heavy operations",
                outcome_summary="Still reconciling everything manually.",
                content_hash="qualified-finding-discovery-only",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
            )
        )
        signal_id = db.insert_raw_signal(
            RawSignal(
                finding_id=finding_id,
                source_name="reddit",
                source_type="forum",
                source_url="https://reddit.com/r/smallbusiness/comments/1",
                title="Spreadsheet-heavy operations",
                body_excerpt="Still reconciling everything manually.",
                content_hash="qualified-signal-discovery-only",
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-spreadsheet-discovery-only",
                segment="small business ops",
                user_role="ops lead",
                job_to_be_done="keep operations data in sync",
                pain_statement="Still reconciling everything manually.",
                trigger_event="weekly close",
                failure_mode="manual reconciliation piles up",
                current_workaround="spreadsheets",
                current_tools="Excel",
                confidence=0.8,
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app._ignore_backlog_for_completion = True
        app.orchestrator = type("OrchestratorStub", (), {"_message_queue": asyncio.Queue()})()
        app.agents = {
            "evidence": type("EvidenceStub", (), {"busy_count": lambda self: 0})(),
            "validation": type("ValidationStub", (), {"busy_count": lambda self: 0})(),
        }

        state = app.completion_state()

        assert state["actual_open_qualified"] == 1
        assert state["open_qualified"] == 0
        assert state["total_busy"] == 0
        assert state["drained"] is True
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)


def test_wait_for_pipeline_drained_treats_stable_idle_queue_as_drained():
    app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    app.shutdown_event = asyncio.Event()
    states = [
        {
            "queue_empty": False,
            "queue_size": 3,
            "open_qualified": 0,
            "actual_open_qualified": 0,
            "evidence_busy": 0,
            "validation_busy": 0,
            "total_busy": 0,
            "busy_agents": {},
            "drained": False,
        }
    ]
    app.completion_state = lambda: states[0]

    drained = asyncio.run(app._wait_for_pipeline_drained(6))

    assert drained is True


def test_snapshot_screening_uses_current_run_counts():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        db.init_schema()
        db.set_active_run_id("current-run")

        db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/old",
                product_built="Older incomplete item",
                content_hash="old-qualified",
                status="qualified",
                source_class="pain_signal",
                evidence={"run_id": "older-run"},
            )
        )
        db.insert_finding(
            Finding(
                source="reddit",
                source_url="https://reddit.com/r/ops/comments/new",
                product_built="Current run parked item",
                content_hash="current-parked",
                status="parked",
                source_class="pain_signal",
                evidence={"run_id": "current-run"},
            )
        )

        app = AutoResearcher(config_path=str(Path(__file__).resolve().parents[1] / "config.yaml"))
        app.db = db
        app.current_run_id = "current-run"
        snapshot = app.snapshot()

        assert snapshot["screening"]["parked"] == 1
        assert snapshot["screening"]["qualified"] == 0
        assert snapshot["screening_all_time"]["qualified"] == 1
        assert snapshot["screening_all_time"]["parked"] == 1
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)
