"""Tests for database schema and recovered compatibility operations."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import unittest

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import (  # noqa: E402
    BuildBrief,
    Database,
    EvidenceLedgerEntry,
    Finding,
    Opportunity,
    OpportunityCluster,
    ProblemAtom,
    RawSignal,
    Validation,
    ValidationExperiment,
)


class TestDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_schema()
        self.db.set_active_run_id("test-run")

    def tearDown(self) -> None:
        self.db.close()
        # WAL mode creates -wal and -shm files; clean them up before removing the directory
        for ext in ("-wal", "-shm"):
            wal_path = self.db_path + ext
            if os.path.exists(wal_path):
                os.remove(wal_path)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_foreign_keys_pragma_enabled(self) -> None:
        conn = self.db._get_connection()
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_evidence_ledger_upsert_idempotent(self) -> None:
        finding_id = self.db.insert_finding(
            Finding(source="t", source_url="https://example.com/l", content_hash="ledger-upsert")
        )
        eid1 = self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="finding",
                entity_id=finding_id,
                entry_kind="note",
                summary="first",
                run_id="test-run",
            )
        )
        eid2 = self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="finding",
                entity_id=finding_id,
                entry_kind="note",
                summary="second",
                run_id="test-run",
            )
        )
        self.assertEqual(eid1, eid2)
        rows = self.db.list_ledger_entries(entity_type="finding", entity_id=finding_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].entry_json or {}).get("summary"), "second")

    def test_init_schema_creates_expected_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {
            "findings",
            "validations",
            "ideas",
            "products",
            "resources",
            "discovery_feedback",
            "raw_signals",
            "problem_atoms",
            "opportunity_clusters",
            "clusters",
            "cluster_members",
            "opportunities",
            "validation_experiments",
            "experiments",
            "evidence_ledger",
            "corroborations",
            "market_enrichments",
            "review_feedback",
            "build_briefs",
            "build_prep_outputs",
        }
        self.assertTrue(expected_tables.issubset(tables))
        conn.close()

    def test_insert_and_retrieve_finding(self) -> None:
        finding = Finding(
            source="test_source",
            source_url="https://example.com/test",
            entrepreneur="Test Entrepreneur",
            tool_used="Test Tool",
            product_built="Test Product",
            monetization_method="Subscription",
            outcome_summary="Success",
            content_hash="abc123hash",
        )
        finding_id = self.db.insert_finding(finding)
        retrieved = self.db.get_finding_by_hash("abc123hash")
        self.assertEqual(finding_id, retrieved.id)
        self.assertEqual(retrieved.status, "new")
        self.assertEqual(retrieved.product_built, "Test Product")

    def test_init_schema_migrates_legacy_opportunities_scoring_columns(self) -> None:
        legacy_db_path = os.path.join(self.temp_dir, "legacy.db")
        conn = sqlite3.connect(legacy_db_path)
        conn.execute(
            """
            CREATE TABLE opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                market_gap TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                status TEXT NOT NULL,
                pain_severity REAL DEFAULT 0,
                frequency_score REAL DEFAULT 0,
                cost_of_inaction REAL DEFAULT 0,
                workaround_density REAL DEFAULT 0,
                urgency_score REAL DEFAULT 0,
                segment_concentration REAL DEFAULT 0,
                reachability REAL DEFAULT 0,
                timing_shift REAL DEFAULT 0,
                buildability REAL DEFAULT 0,
                expansion_potential REAL DEFAULT 0,
                education_burden REAL DEFAULT 0,
                dependency_risk REAL DEFAULT 0,
                adoption_friction REAL DEFAULT 0,
                evidence_quality REAL DEFAULT 0,
                composite_score REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                selection_status TEXT DEFAULT 'research_more',
                selection_reason TEXT DEFAULT '',
                notes_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

        legacy_db = Database(legacy_db_path)
        legacy_db.init_schema()
        columns = {
            row[1]
            for row in sqlite3.connect(legacy_db_path).execute("PRAGMA table_info(opportunities)").fetchall()
        }

        self.assertIn("scoring_version", columns)
        self.assertIn("decision_score", columns)
        self.assertIn("problem_truth_score", columns)
        self.assertIn("formula_version", columns)
        self.assertIn("threshold_version", columns)
        legacy_db.close()
        os.remove(legacy_db_path)

    def test_duplicate_detection_via_hash(self) -> None:
        finding = Finding(source="source1", source_url="https://example.com/1", content_hash="duplicate_hash")
        self.db.insert_finding(finding)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_finding(Finding(source="source2", source_url="https://example.com/2", content_hash="duplicate_hash"))

    def test_update_finding_status(self) -> None:
        finding_id = self.db.insert_finding(Finding(source="test", source_url="https://example.com", content_hash="status_hash"))
        self.db.update_finding_status(finding_id, "validated")
        self.assertEqual(self.db.get_finding_by_hash("status_hash").status, "validated")

    def test_review_requalification_revives_parked_pain_signal(self) -> None:
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem/accounting",
                source_url="https://example.com/review-revive",
                content_hash="review-revive-hash",
                status="parked",
                source_class="pain_signal",
                evidence={"run_id": "test-run"},
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                finding_id=finding_id,
                source_name="reddit",
                source_type="thread",
                source_class="pain_signal",
                source_url="https://example.com/review-revive",
                title="Manual reconciliation keeps breaking",
                body_excerpt="The spreadsheet and payout totals never match cleanly.",
                quote_text="We keep fixing this manually every month.",
                content_hash="review-revive-signal",
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                finding_id=finding_id,
                raw_signal_id=signal_id,
                signal_id=signal_id,
                pain_statement="Manual reconciliation keeps breaking",
                user_role="finance team",
                job_to_be_done="close the books",
                failure_mode="payment totals do not match invoice totals",
                current_workaround="manual spreadsheet reconciliation",
                atom_json=json.dumps({"is_specific_problem": True}),
            )
        )

        changed = self.db.requalify_finding_for_evidence(
            finding_id,
            review_label="needs_more_evidence",
            note="Borderline but real operator pain.",
        )

        refreshed = self.db.get_finding(finding_id)
        self.assertTrue(changed)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.status, "qualified")
        self.assertEqual((refreshed.evidence or {}).get("review_requalification", {}).get("review_label"), "needs_more_evidence")

    def test_review_requalification_skips_screened_out_without_atoms(self) -> None:
        finding_id = self.db.insert_finding(
            Finding(
                source="web",
                source_url="https://example.com/not-pain",
                content_hash="review-skip-hash",
                status="screened_out",
                source_class="pain_signal",
            )
        )
        changed = self.db.requalify_finding_for_evidence(finding_id, review_label="needs_more_evidence")
        self.assertFalse(changed)
        self.assertEqual(self.db.get_finding(finding_id).status, "screened_out")

    def test_insert_validation(self) -> None:
        finding_id = self.db.insert_finding(Finding(source="test", source_url="https://example.com", content_hash="validation_hash"))
        validation_id = self.db.insert_validation(
            Validation(
                finding_id=finding_id,
                market_score=8.5,
                technical_score=7.0,
                distribution_score=9.0,
                overall_score=8.2,
                passed=True,
                evidence={"key_insight": "Strong market demand"},
                run_id="test-run",
            )
        )
        self.assertGreater(validation_id, 0)
        self.assertEqual(len(self.db.get_recent_validations(limit=10)), 1)

    def test_problem_atom_persists_atom_json(self) -> None:
        finding_id = self.db.insert_finding(
            Finding(source="reddit", source_url="https://example.com/thread", content_hash="atom-json-hash")
        )
        atom_json = json.dumps({"pain_statement": "handoff breaks", "segment": "ops"})
        atom_id = self.db.insert_problem_atom(
            ProblemAtom(
                finding_id=finding_id,
                raw_signal_id=12,
                signal_id=12,
                pain_statement="handoff breaks",
                atom_json=atom_json,
                metadata={"segment": "ops"},
            )
        )

        atom = self.db.get_problem_atoms(finding_id=finding_id, limit=1)[0]
        conn = self.db._get_connection()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(problem_atoms)").fetchall()}
        stored = conn.execute("SELECT atom_json FROM problem_atoms WHERE id = ?", (atom_id,)).fetchone()

        self.assertIn("atom_json", columns)
        self.assertEqual(stored["atom_json"], atom_json)
        self.assertEqual(atom.atom_json, atom_json)

    def test_validation_review_preserves_recurrence_match_records(self) -> None:
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://example.com/thread",
                content_hash="validation-review-hash",
                product_built="Duct tape and spreadsheets ops pain",
                outcome_summary="Spreadsheet handoffs break follow-up and teams lose track of the latest file.",
            )
        )
        evidence = {
            "decision": "park",
            "matched_results_by_source": {"reddit": 1, "web": 1},
            "partial_results_by_source": {"reddit": 2, "web": 1},
            "matched_docs_by_source": {
                "web": [
                    {
                        "source_family": "web",
                        "source": "web",
                        "query_text": "which spreadsheet is latest operations",
                        "normalized_url": "https://ops.example.com/latest-spreadsheet-confusion",
                        "title": "Which spreadsheet is the latest? Teams keep getting out of sync",
                        "snippet": "Operators keep copying updates across files and miss follow-up steps.",
                        "match_class": "strong",
                    }
                ]
            },
            "partial_docs_by_source": {
                "web": [
                    {
                        "source_family": "web",
                        "source": "web",
                        "query_text": "manual handoff workflow",
                        "normalized_url": "https://ops.example.com/manual-handoff-workflow",
                        "title": "Manual handoff workflow creates follow-up misses",
                        "snippet": "Teams copy updates between files and lose track of ownership.",
                        "match_class": "partial",
                    }
                ]
            },
            "source_yield": {
                "web": {
                    "attempts": 1,
                    "docs_retrieved": 2,
                    "docs_partial_match": 1,
                    "docs_strong_match": 1,
                    "confirmed": True,
                }
            },
        }
        self.db.insert_validation(
            Validation(
                finding_id=finding_id,
                market_score=7.1,
                technical_score=6.9,
                distribution_score=6.7,
                overall_score=6.9,
                passed=False,
                evidence=evidence,
                run_id="test-run",
            )
        )

        recent = self.db.get_recent_validations(limit=10)
        review = self.db.get_validation_review(run_id="test-run")

        self.assertEqual(recent[0]["matched_docs_by_source"]["web"][0]["match_class"], "strong")
        self.assertEqual(
            recent[0]["matched_docs_by_source"]["web"][0]["normalized_url"],
            "https://ops.example.com/latest-spreadsheet-confusion",
        )
        self.assertEqual(review[0]["matched_docs_by_source"]["web"][0]["source_family"], "web")
        self.assertEqual(review[0]["matched_docs_by_source"]["web"][0]["query_text"], "which spreadsheet is latest operations")
        self.assertEqual(review[0]["source_yield"]["web"]["docs_strong_match"], 1)
        self.assertEqual(
            review[0]["reviewable_recurrence_matches_by_source"]["web"][0]["match_class"],
            "strong",
        )
        self.assertEqual(
            review[0]["reviewable_recurrence_matches_by_source"]["web"][1]["match_class"],
            "partial",
        )
        self.assertEqual(
            review[0]["reviewable_recurrence_matches_by_source"]["web"][0]["normalized_url"],
            "https://ops.example.com/latest-spreadsheet-confusion",
        )
        digest = self.db.get_validation_corroboration_digest(finding_id=finding_id, run_id="test-run")
        self.assertIsNotNone(digest)
        self.assertEqual(digest["recurrence_state"], None)
        self.assertTrue(digest["web_contributed"])
        self.assertEqual(digest["web_matched_count"], 1)
        self.assertEqual(digest["web_partial_count"], 1)
        self.assertEqual(digest["web_source_yield"]["docs_strong_match"], 1)
        self.assertEqual(digest["web_matches"][0]["match_class"], "strong")
        self.assertEqual(digest["web_matches"][1]["match_class"], "partial")
        self.assertEqual(
            digest["web_matches"][0]["normalized_url"],
            "https://ops.example.com/latest-spreadsheet-confusion",
        )

    def test_candidate_workbench_surfaces_posture_actions_and_best_evidence(self) -> None:
        prototype_finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://example.com/duct-tape-excel",
                content_hash="workbench-prototype",
                outcome_summary="Ops are held together by duct tape and spreadsheets.",
            )
        )
        prototype_validation_id = self.db.insert_validation(
            Validation(
                finding_id=prototype_finding_id,
                market_score=7.2,
                technical_score=6.8,
                distribution_score=6.6,
                overall_score=6.9,
                passed=False,
                evidence={
                    "decision": "park",
                    "decision_reason": "plausible_but_unproven",
                    "selection_status": "prototype_candidate",
                    "selection_reason": "prototype_candidate_gate",
                    "family_confirmation_count": 1,
                    "recurrence_state": "supported",
                    "recurrence_failure_class": "single_source_only",
                    "matched_docs_by_source": {
                        "web": [
                            {
                                "source_family": "web",
                                "source": "web",
                                "query_text": "duct tape spreadsheets operations",
                                "normalized_url": "https://ops.example.com/duct-tape-spreadsheets",
                                "title": "Teams run operations with duct tape and spreadsheets",
                                "snippet": "Manual handoffs and version confusion break the workflow.",
                                "match_class": "strong",
                            }
                        ]
                    },
                    "matched_results_by_source": {"web": 1},
                    "partial_results_by_source": {"web": 0},
                    "source_yield": {"web": {"docs_retrieved": 1, "docs_strong_match": 1}},
                },
                run_id="test-run",
            )
        )
        cluster_id = self.db.upsert_cluster(
            OpportunityCluster(
                cluster_key="workbench-prototype-cluster",
                label="Workbench prototype cluster",
                segment="ops",
                user_role="ops",
                job_to_be_done="keep spreadsheets from breaking",
                trigger_summary="daily ops",
                signal_count=1,
                atom_count=1,
                evidence_quality=0.7,
                summary={},
            )
        )
        opportunity_id = self.db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="Workbench prototype opportunity",
                market_gap="underserved",
                recommendation="park",
                status="parked",
                pain_severity=0.6,
                frequency_score=0.6,
                cost_of_inaction=0.5,
                workaround_density=0.7,
                urgency_score=0.5,
                segment_concentration=0.5,
                reachability=0.5,
                timing_shift=0.5,
                buildability=0.5,
                expansion_potential=0.5,
                education_burden=0.3,
                dependency_risk=0.3,
                adoption_friction=0.3,
                evidence_quality=0.7,
                composite_score=0.65,
                confidence=0.65,
                notes={},
            )
        )
        self.db.upsert_build_brief(
            BuildBrief(
                opportunity_id=opportunity_id,
                validation_id=prototype_validation_id,
                cluster_id=cluster_id,
                run_id="test-run",
                status="prototype_candidate",
                recommended_output_type="workflow_reliability_assistant",
                brief_hash="prototype-brief-hash",
                brief_json=json.dumps(
                    {
                        "recommended_narrow_output_type": "workflow_reliability_assistant",
                        "prototype_gate": {
                            "market_confidence_level": "prototype_checkpoint",
                        },
                    }
                ),
            )
        )

        watch_finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://example.com/timeout-case",
                content_hash="workbench-watch",
                outcome_summary="A promising case ran out of budget before recurrence completed.",
            )
        )
        self.db.insert_validation(
            Validation(
                finding_id=watch_finding_id,
                market_score=6.4,
                technical_score=6.0,
                distribution_score=6.0,
                overall_score=6.1,
                passed=False,
                evidence={
                    "decision": "park",
                    "decision_reason": "park_recurrence",
                    "selection_status": "research_more",
                    "selection_reason": "selection_gate_not_met",
                    "family_confirmation_count": 0,
                    "recurrence_state": "timeout",
                    "recurrence_failure_class": "budget_exhausted",
                },
                run_id="test-run",
            )
        )

        archive_finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://example.com/archive-case",
                content_hash="workbench-archive",
                outcome_summary="Economically weak pain that should not be pursued.",
            )
        )
        self.db.insert_validation(
            Validation(
                finding_id=archive_finding_id,
                market_score=2.0,
                technical_score=3.0,
                distribution_score=2.5,
                overall_score=2.4,
                passed=False,
                evidence={
                    "decision": "kill",
                    "decision_reason": "unlikely_or_economically_weak",
                    "selection_status": "archive",
                    "selection_reason": "validation_kill",
                    "family_confirmation_count": 0,
                    "recurrence_state": "weak",
                    "recurrence_failure_class": "no_corroboration_found",
                },
                run_id="test-run",
            )
        )

        workbench = self.db.get_candidate_workbench(run_id="test-run")
        self.assertEqual(workbench[0]["title"], "Ops are held together by duct tape and spreadsheets.")
        self.assertEqual(workbench[0]["next_recommended_action"], "prototype_now")
        self.assertEqual(workbench[0]["confidence_posture"], "prototype_checkpoint")
        self.assertEqual(workbench[0]["repeatability_posture"], "cross_family_signal")
        self.assertTrue(workbench[0]["build_brief_present"])
        self.assertEqual(workbench[0]["recommended_output_type"], "workflow_reliability_assistant")
        self.assertEqual(
            workbench[0]["best_surfaced_evidence"]["normalized_url"],
            "https://ops.example.com/duct-tape-spreadsheets",
        )
        self.assertEqual(workbench[1]["next_recommended_action"], "watchlist")
        self.assertEqual(workbench[2]["next_recommended_action"], "archive")

    def test_backlog_workbench_requalifies_and_prioritizes_operator_grade_items(self) -> None:
        valid_finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem/ecommerce",
                source_url="https://reddit.com/r/ecommerce/comments/1",
                product_built='How are u guys managing fulfilment between "order received" and "label printed"?',
                outcome_summary="The team coordinates through WhatsApp and spreadsheets.",
                content_hash="backlog-valid",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
                evidence={"discovery_query": '"order received" "label printed" whatsapp spreadsheet'},
            )
        )
        valid_signal_id = self.db.insert_raw_signal(
            RawSignal(
                finding_id=valid_finding_id,
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://reddit.com/r/ecommerce/comments/1",
                title='How are u guys managing fulfilment between "order received" and "label printed"?',
                body_excerpt="Received -> Picking -> Processing -> Packed -> Shipped. We coordinate through WhatsApp and spreadsheets.",
                quote_text="We coordinate through WhatsApp and spreadsheets.",
                role_hint="operator",
                content_hash="backlog-valid-signal",
                metadata={"evidence": {"discovery_query": '"order received" "label printed" whatsapp spreadsheet'}},
            )
        )
        self.db.insert_problem_atom(
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

        stale_finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem/accounting",
                source_url="https://reddit.com/r/accounting/comments/2",
                product_built="Resume roast",
                outcome_summary="Feeling lost with career, please roast my resume.",
                content_hash="backlog-stale",
                status="qualified",
                finding_kind="problem_signal",
                source_class="pain_signal",
                evidence={"discovery_query": "which spreadsheet is latest"},
            )
        )
        stale_signal_id = self.db.insert_raw_signal(
            RawSignal(
                finding_id=stale_finding_id,
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://reddit.com/r/accounting/comments/2",
                title="Feeling lost with career, please roast my resume.",
                body_excerpt="Please review my resume and help me transition to a larger firm.",
                content_hash="backlog-stale-signal",
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=stale_signal_id,
                finding_id=stale_finding_id,
                cluster_key="resume-thread",
                segment="operators with recurring workflow pain",
                user_role="finance lead",
                job_to_be_done="improve my resume",
                pain_statement="Please review my resume.",
                trigger_event="trying to transition to a larger firm",
                failure_mode="resume is too wordy",
                current_workaround="",
                confidence=0.4,
            )
        )

        workbench = self.db.get_backlog_workbench(limit=10)

        self.assertEqual(len(workbench), 1)
        self.assertEqual(workbench[0]["finding_id"], valid_finding_id)
        self.assertIn("operator_context", workbench[0]["priority_reasons"])
        self.assertGreater(workbench[0]["backlog_priority_score"], 4.0)

    def test_threading_local_connections(self) -> None:
        results: list[int] = []

        def insert_in_thread() -> None:
            fid = self.db.insert_finding(
                Finding(
                    source="thread_test",
                    source_url="https://example.com/thread",
                    content_hash=f"thread_hash_{threading.current_thread().ident}",
                )
            )
            results.append(fid)

        threads = [threading.Thread(target=insert_in_thread) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 3)
        self.assertEqual(len(set(results)), 3)

    def test_discovery_feedback_records_and_updates(self) -> None:
        self.db.record_discovery_probe("reddit-problem", "manual process", docs_seen=3, latency_ms=125.0, status="ok")
        self.db.record_discovery_screening(
            "reddit-problem",
            "manual process",
            accepted=False,
            source_class="low_signal_summary",
            screening_score=0.22,
        )
        self.db.record_discovery_hit("reddit-problem", "manual process")
        self.db.record_validation_feedback(
            "reddit-problem",
            "manual process",
            passed=True,
            overall_score=0.81,
            selection_status="prototype_candidate",
            build_brief_created=True,
            decision="promote",
            recurrence_state="thin",
            recurrence_failure_class="single_source_only",
        )
        rows = self.db.get_discovery_feedback("reddit-problem")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["query_text"], "manual process")
        self.assertEqual(row["runs"], 1)
        self.assertEqual(row["docs_seen"], 3)
        self.assertEqual(row["findings_emitted"], 1)
        self.assertEqual(row["screened_out"], 1)
        self.assertEqual(row["low_signal_count"], 1)
        self.assertEqual(row["pain_signal_count"], 0)
        self.assertEqual(row["validations"], 1)
        self.assertEqual(row["passes"], 1)
        self.assertEqual(row["promotes"], 1)
        self.assertEqual(row["thin_recurrence_count"], 1)
        self.assertEqual(row["single_source_only_count"], 1)
        self.assertEqual(row["prototype_candidates"], 1)
        self.assertEqual(row["build_briefs"], 1)
        self.assertGreater(row["avg_screening_score"], 0)

    def test_discovery_feedback_can_store_query_cooldown(self) -> None:
        self.db.set_discovery_query_cooldown("reddit-problem", "manual process", "2099-01-01T00:00:00+00:00")
        rows = self.db.get_discovery_feedback("reddit-problem")
        self.assertEqual(rows[0]["cooldown_until"], "2099-01-01T00:00:00+00:00")

    def test_discovery_themes_round_trip(self) -> None:
        theme_id = self.db.upsert_discovery_theme(
            "workflow_fragility",
            label="Workflow fragility and spreadsheet glue",
            query_seeds=["duct tape spreadsheets", "manual handoff workflow"],
            source_signals=["Ops are held together by duct tape and spreadsheets"],
            times_seen=3,
            yield_score=0.75,
            run_id="test-run",
        )
        self.assertGreater(theme_id, 0)
        rows = self.db.list_active_discovery_themes(limit=10)
        self.assertEqual(rows[0]["theme_key"], "workflow_fragility")
        self.assertEqual(rows[0]["query_seeds"][0], "duct tape spreadsheets")
        self.assertEqual(rows[0]["source_signals"][0], "Ops are held together by duct tape and spreadsheets")
        self.assertEqual(rows[0]["times_seen"], 3)

    def test_raw_signal_and_problem_atom_round_trip(self) -> None:
        finding_id = self.db.insert_finding(Finding(source="reddit-problem", source_url="https://example.com/thread", content_hash="signal-round-trip"))
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                finding_id=finding_id,
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://example.com/thread",
                title="Manual reconciliation workflow",
                body_excerpt="Teams still use spreadsheets every week.",
                content_hash="signal-round-trip-signal",
                metadata={"role_hint": "operations lead"},
            )
        )
        atom_id = self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-manual-reconciliation",
                segment="operations teams",
                user_role="operations lead",
                job_to_be_done="reconcile payouts and exceptions without manual cleanup",
                trigger_event="after payout runs",
                pain_statement="Teams still use spreadsheets every week.",
                failure_mode="exceptions break the workflow",
                current_workaround="spreadsheets",
                current_tools="Excel",
                urgency_clues="daily",
                frequency_clues="every week",
                emotional_intensity=0.7,
                cost_consequence_clues="hours lost",
                why_now_clues="pricing change",
                confidence=0.8,
            )
        )

        signals = self.db.get_raw_signals(finding_id=finding_id)
        atoms = self.db.get_problem_atoms(finding_id=finding_id)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].id, signal_id)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].id, atom_id)
        self.assertEqual(atoms[0].current_workaround, "spreadsheets")

    def test_cluster_opportunity_experiment_and_ledger_round_trip(self) -> None:
        cluster_id = self.db.upsert_cluster(
            OpportunityCluster(
                cluster_key="finance|reconcile|manual",
                label="Reconcile payouts without manual exceptions",
                segment="small_business",
                user_role="finance",
                job_to_be_done="reconcile payouts",
                trigger_summary="after payout runs",
                signal_count=3,
                atom_count=3,
                evidence_quality=0.72,
                summary={"sample_pains": ["Finance teams keep cleaning payout exceptions by hand."]},
            )
        )
        opportunity_id = self.db.upsert_opportunity(
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
                notes={"rationale": "Strong workaround behavior and repeated exceptions."},
            )
        )
        experiment_id = self.db.insert_experiment(
            ValidationExperiment(
                opportunity_id=opportunity_id,
                cluster_id=cluster_id,
                test_type="workflow_walkthrough",
                hypothesis="Teams will share current spreadsheets if the pain is urgent.",
                falsifier="If no one shares the workflow, park it.",
                smallest_test="Run 5 walkthroughs.",
                success_signal="Operators share their actual process.",
                failure_signal="Operators refuse to show the workflow.",
                run_id="test-run",
            )
        )
        ledger_id = self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="opportunity",
                entity_id=opportunity_id,
                entry_kind="decision",
                summary="Promoted for interview follow-up.",
                run_id="test-run",
            )
        )

        cluster = self.db.get_cluster(cluster_id)
        opportunity = self.db.get_opportunity(opportunity_id)
        experiments = self.db.get_experiments(opportunity_id=opportunity_id)
        ledger = self.db.get_evidence_ledger(entity_type="opportunity", entity_id=opportunity_id)

        self.assertEqual(cluster.id, cluster_id)
        self.assertEqual(opportunity.id, opportunity_id)
        self.assertEqual(experiments[0].id, experiment_id)
        self.assertEqual(ledger[0].id, ledger_id)

    def test_cluster_member_backfill_links_existing_atoms_to_clusters(self) -> None:
        finding_id = self.db.insert_finding(Finding(source="reddit-problem", source_url="https://example.com/thread", content_hash="cluster-member-backfill"))
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://example.com/thread",
                title="Manual reconciliation workflow",
                body_excerpt="Teams still use spreadsheets every week.",
                content_hash="cluster-member-signal",
                finding_id=finding_id,
            )
        )
        atom_id = self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-manual-reconciliation",
                segment="operations teams",
                user_role="operations lead",
                job_to_be_done="reconcile payouts and exceptions without manual cleanup",
                trigger_event="after payout runs",
                pain_statement="Teams still use spreadsheets every week.",
                failure_mode="exceptions break the current workflow",
                current_workaround="spreadsheets",
                current_tools="Excel",
                urgency_clues="need",
                frequency_clues="every week",
                emotional_intensity=0.6,
                cost_consequence_clues="time",
                why_now_clues="pricing change",
                confidence=0.8,
            )
        )
        cluster_id = self.db.upsert_cluster(
            OpportunityCluster(
                cluster_key="ops-manual-reconciliation",
                label="operations lead - reconcile payouts",
                segment="operations teams",
                user_role="operations lead",
                job_to_be_done="reconcile payouts and exceptions without manual cleanup",
                trigger_summary="after payout runs",
                signal_count=1,
                atom_count=1,
                evidence_quality=0.7,
                summary={},
            )
        )
        self.db._backfill_cluster_members()
        self.assertEqual(self.db.get_cluster_members(cluster_id), [atom_id])

    def test_insert_ledger_entry_supports_legacy_schema(self) -> None:
        legacy_path = os.path.join(self.temp_dir, "legacy.db")
        conn = sqlite3.connect(legacy_path)
        conn.executescript(
            """
            CREATE TABLE evidence_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT DEFAULT '',
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                entry_kind TEXT NOT NULL,
                stance TEXT NOT NULL,
                source_name TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                quote_text TEXT DEFAULT '',
                summary TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX idx_evidence_ledger_unique_kind
            ON evidence_ledger(run_id, entity_type, entity_id, entry_kind);
            """
        )
        conn.commit()
        conn.close()

        legacy_db = Database(legacy_path)
        legacy_db.set_active_run_id("legacy-run")
        ledger_id = legacy_db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="finding",
                entity_id=7,
                entry_kind="evidence_enrichment",
                stance="supporting",
                source_name="evidence",
                source_url="https://example.com/evidence",
                quote_text="Restore fails repeatedly after migration.",
                summary="Legacy ledger compatibility write.",
                entry_json={"recurrence_state": "supported"},
            )
        )
        legacy_db.close()

        row = sqlite3.connect(legacy_path).execute(
            "SELECT run_id, stance, source_name, summary, metadata_json FROM evidence_ledger WHERE id = ?",
            (ledger_id,),
        ).fetchone()
        self.assertEqual(row[0], "legacy-run")
        self.assertEqual(row[1], "supporting")
        self.assertEqual(row[2], "evidence")
        self.assertEqual(row[3], "Legacy ledger compatibility write.")
        self.assertEqual(json.loads(row[4])["recurrence_state"], "supported")
        os.remove(legacy_path)

    def test_upsert_cluster_supports_legacy_schema(self) -> None:
        legacy_path = os.path.join(self.temp_dir, "legacy-clusters.db")
        conn = sqlite3.connect(legacy_path)
        conn.executescript(
            """
            CREATE TABLE opportunity_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                segment TEXT NOT NULL,
                user_role TEXT NOT NULL,
                job_to_be_done TEXT NOT NULL,
                trigger_summary TEXT DEFAULT '',
                signal_count INTEGER DEFAULT 0,
                atom_count INTEGER DEFAULT 0,
                evidence_quality REAL DEFAULT 0,
                status TEXT DEFAULT 'candidate',
                summary_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT DEFAULT '{}'
            );
            CREATE TABLE clusters (
                id INTEGER PRIMARY KEY,
                cluster_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                segment TEXT NOT NULL,
                trigger_pattern TEXT DEFAULT '',
                workaround_pattern TEXT DEFAULT '',
                failure_pattern TEXT DEFAULT '',
                source_count INTEGER DEFAULT 0,
                signal_count INTEGER DEFAULT 0,
                atom_count INTEGER DEFAULT 0,
                evidence_quality REAL DEFAULT 0,
                status TEXT DEFAULT 'active',
                summary_json TEXT DEFAULT '{}',
                user_role TEXT DEFAULT '',
                job_to_be_done TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE cluster_members (
                cluster_id INTEGER NOT NULL,
                atom_id INTEGER NOT NULL,
                UNIQUE(cluster_id, atom_id)
            );
            CREATE TABLE problem_atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id INTEGER DEFAULT 0,
                raw_signal_id INTEGER DEFAULT 0,
                signal_id INTEGER DEFAULT 0,
                cluster_key TEXT DEFAULT ''
            );
            """
        )
        conn.commit()
        conn.close()

        legacy_db = Database(legacy_path)
        cluster_id = legacy_db.upsert_cluster(
            OpportunityCluster(
                cluster_key="backup|restore|reliability",
                label="operator - keep backup restore and recovery reliable",
                segment="ops",
                user_role="operator",
                job_to_be_done="keep backup restore and recovery reliable",
                trigger_summary="after restore attempts",
                signal_count=3,
                atom_count=2,
                evidence_quality=0.8,
                summary={"sample_pains": ["restore fails repeatedly"]},
            )
        )
        legacy_db.close()

        row = sqlite3.connect(legacy_path).execute(
            "SELECT cluster_key, segment, trigger_summary, signal_count, atom_count, evidence_quality FROM opportunity_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        self.assertEqual(row[0], "backup|restore|reliability")
        self.assertEqual(row[1], "ops")
        self.assertEqual(row[2], "after restore attempts")
        self.assertEqual(row[3], 3)
        self.assertEqual(row[4], 2)
        self.assertEqual(row[5], 0.8)
        mirror_conn = sqlite3.connect(legacy_path)
        mirror = mirror_conn.execute(
            "SELECT cluster_key, summary, segment, trigger_pattern, source_count FROM clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        self.assertEqual(mirror[0], "backup|restore|reliability")
        self.assertIn("restore fails repeatedly", mirror[1])
        self.assertEqual(mirror[2], "ops")
        self.assertEqual(mirror[3], "after restore attempts")
        self.assertEqual(mirror[4], 3)
        mirror_conn.close()
        os.remove(legacy_path)

    def test_batch_commit_defers_and_commits(self) -> None:
        """Batch context manager defers commits and commits once on exit."""
        finding1 = Finding(
            source="batch_test", source_url="https://example.com/1",
            entrepreneur="", tool_used="", product_built="Test 1",
            monetization_method="", outcome_summary="Summary 1",
            content_hash="hash_batch_1", status="new", finding_kind="pain_signal",
        )
        finding2 = Finding(
            source="batch_test", source_url="https://example.com/2",
            entrepreneur="", tool_used="", product_built="Test 2",
            monetization_method="", outcome_summary="Summary 2",
            content_hash="hash_batch_2", status="new", finding_kind="pain_signal",
        )
        with self.db.batch():
            id1 = self.db.insert_finding(finding1)
            id2 = self.db.insert_finding(finding2)
            # Verify they exist in the current connection (uncommitted but visible)
            result = self.db.get_finding(id1)
            self.assertIsNotNone(result)

        # After batch exit, both should be committed
        result = self.db.get_finding(id1)
        self.assertIsNotNone(result)
        self.assertEqual(result.product_built, "Test 1")
        result = self.db.get_finding(id2)
        self.assertIsNotNone(result)
        self.assertEqual(result.product_built, "Test 2")

    def test_batch_rollback_on_exception(self) -> None:
        """Batch context manager rolls back on exception."""
        finding = Finding(
            source="rollback_test", source_url="https://example.com/rb",
            entrepreneur="", tool_used="", product_built="Rollback Test",
            monetization_method="", outcome_summary="Rollback",
            content_hash="hash_rollback", status="new", finding_kind="pain_signal",
        )
        try:
            with self.db.batch():
                self.db.insert_finding(finding)
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Finding should not be persisted after rollback
        result = self.db.get_finding_by_hash("hash_rollback")
        self.assertIsNone(result)

    def test_batch_active_flag_is_thread_local(self) -> None:
        """batch_depth counter is thread-local and doesn't leak across threads."""
        finding = Finding(
            source="thread_test", source_url="https://example.com/thr",
            entrepreneur="", tool_used="", product_built="Thread Test",
            monetization_method="", outcome_summary="Thread",
            content_hash="hash_thread", status="new", finding_kind="pain_signal",
        )
        results = []

        def insert_in_batch():
            db2 = Database(self.db_path)
            db2.init_schema()
            with db2.batch():
                db2.insert_finding(finding)
                # Inside batch, batch_depth should be >= 1
                results.append(getattr(db2._local, "batch_depth", 0))
            db2.close()

        t = threading.Thread(target=insert_in_batch)
        t.start()
        t.join()

        # The other thread saw batch_depth >= 1
        self.assertGreater(results[0], 0)
        # Main thread should NOT have batch_depth set (or it should be 0)
        self.assertEqual(getattr(self.db._local, "batch_depth", 0), 0)

    def test_batch_nested_success_commits_on_outer_exit(self) -> None:
        """Nested batch blocks only commit when the outermost block exits."""
        finding1 = Finding(
            source="nest_ok", source_url="https://example.com/n1",
            entrepreneur="", tool_used="", product_built="Nested OK 1",
            monetization_method="", outcome_summary="Nested",
            content_hash="hash_nest_ok_1", status="new", finding_kind="pain_signal",
        )
        finding2 = Finding(
            source="nest_ok", source_url="https://example.com/n2",
            entrepreneur="", tool_used="", product_built="Nested OK 2",
            monetization_method="", outcome_summary="Nested",
            content_hash="hash_nest_ok_2", status="new", finding_kind="pain_signal",
        )
        with self.db.batch():
            id1 = self.db.insert_finding(finding1)
            with self.db.batch():
                id2 = self.db.insert_finding(finding2)
                # Depth should be 2 inside the inner block
                self.assertEqual(getattr(self.db._local, "batch_depth", 0), 2)
            # After inner exit, depth should be 1, no commit yet
            self.assertEqual(getattr(self.db._local, "batch_depth", 0), 1)

        # After outer exit, both should be committed
        self.assertIsNotNone(self.db.get_finding(id1))
        self.assertIsNotNone(self.db.get_finding(id2))

    def test_batch_nested_outer_failure_rolls_back_all(self) -> None:
        """If the outer batch block fails, inner writes are also rolled back."""
        finding = Finding(
            source="nest_fail", source_url="https://example.com/nf",
            entrepreneur="", tool_used="", product_built="Nested Fail",
            monetization_method="", outcome_summary="Nested Fail",
            content_hash="hash_nest_fail", status="new", finding_kind="pain_signal",
        )
        try:
            with self.db.batch():
                with self.db.batch():
                    self.db.insert_finding(finding)
                raise RuntimeError("Outer failure")
        except RuntimeError:
            pass

        # Inner writes should be rolled back because the outer block failed
        self.assertIsNone(self.db.get_finding_by_hash("hash_nest_fail"))

    def test_batch_nested_inner_failure_does_not_commit(self) -> None:
        """Inner batch failure propagates; outer rollback undoes everything."""
        finding = Finding(
            source="nest_inner_fail", source_url="https://example.com/nif",
            entrepreneur="", tool_used="", product_built="Inner Fail",
            monetization_method="", outcome_summary="Inner Fail",
            content_hash="hash_nest_inner_fail", status="new", finding_kind="pain_signal",
        )
        try:
            with self.db.batch():
                with self.db.batch():
                    self.db.insert_finding(finding)
                    raise RuntimeError("Inner failure")
        except RuntimeError:
            pass

        # Both inner and outer roll back, nothing is committed
        self.assertIsNone(self.db.get_finding_by_hash("hash_nest_inner_fail"))


if __name__ == "__main__":
    unittest.main()
