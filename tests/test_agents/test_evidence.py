"""Tests for the evidence enrichment agent."""

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.agents.evidence import EvidenceAgent
from src.database import Database, Finding, ProblemAtom, RawSignal
from src.messaging import MessageQueue, MessageType


class TestEvidenceAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_db_path = tempfile.mktemp(suffix=".db")
        self.db = Database(self.temp_db_path)
        self.db.init_schema()
        self.db.set_active_run_id("test-run")
        self.queue = MessageQueue()
        self.agent = EvidenceAgent(self.db, self.queue, config={})

    async def asyncTearDown(self):
        self.db.close()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_enrichment_collects_scores_writes_ledger_and_routes_to_orchestrator(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.33,
                "value_score": 0.41,
                "feasibility_score": 0.62,
                "solution_gap_score": 0.58,
                "saturation_score": 0.44,
                "evidence": {
                    "query": '"manual data entry" spreadsheet',
                    "competitor_query": '"manual data entry" spreadsheet software tool alternative',
                    "recurrence_docs": [{"url": "https://forum.example.com/thread"}],
                    "competitor_docs": [{"url": "https://tool.example.com"}],
                    "recurrence_results_by_source": {"reddit": 2, "web": 1, "github": 0},
                    "matched_results_by_source": {"reddit": 2, "web": 1, "github": 0},
                    "partial_results_by_source": {"reddit": 0, "web": 1, "github": 0},
                    "family_confirmation_count": 2,
                    "source_yield": {
                        "reddit": {"attempts": 1, "docs_retrieved": 2, "docs_strong_match": 2, "confirmed": True},
                        "web": {"attempts": 1, "docs_retrieved": 1, "docs_strong_match": 1, "confirmed": True},
                    },
                    "reshaped_query_history": [{"source": "github", "attempt": 2, "reason": "workaround_missing"}],
                    "last_action": "GATHER_CORROBORATION",
                    "last_transition_reason": "highest_information_gain:operator_workflow_fit",
                    "chosen_family": "web",
                    "expected_gain_class": "high",
                    "source_attempts_snapshot": {"web": {"attempts": 1}},
                    "skipped_families": {"github": "low_public_issue_fit"},
                    "controller_actions": [{"action": "GATHER_CORROBORATION", "target_family": "web"}],
                    "budget_snapshot": {"remaining_beta": 2},
                    "fallback_strategy_used": "",
                    "decomposed_atom_queries": ["small business reporting"],
                    "routing_override_reason": "operator_surface_queries_first",
                    "cohort_query_pack_used": True,
                    "cohort_query_pack_name": "spreadsheet_operator_admin",
                    "web_query_strategy_path": ["atom_shaped", "cohort_pack", "specialized_surface_targeting"],
                    "specialized_surface_targeting_used": True,
                    "promotion_gap_class": "corroboration_gap",
                    "near_miss_enrichment_action": "GATHER_CORROBORATION",
                    "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
                    "value_enrichment_used": False,
                    "value_enrichment_queries": [],
                    "value_enrichment_docs": [],
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://reddit.com/test",
                product_built="Spreadsheet cleanup every week",
                outcome_summary="Ops still handle spreadsheet cleanup by hand every week.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )

        result = await self.agent._enrich_finding({"finding_id": finding_id})

        self.assertTrue(result["success"])
        entries = self.db.list_ledger_entries(entity_type="finding", entity_id=finding_id, limit=5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].entry_kind, "evidence_enrichment")
        ledger_payload = json.loads(entries[0].metadata_json)
        self.assertIn("source_yield", ledger_payload)
        self.assertIn("matched_results_by_source", ledger_payload)
        self.assertIn("partial_results_by_source", ledger_payload)
        self.assertEqual(ledger_payload["family_confirmation_count"], 2)
        self.assertEqual(ledger_payload["last_action"], "GATHER_CORROBORATION")
        self.assertEqual(ledger_payload["chosen_family"], "web")
        self.assertIn("controller_actions", ledger_payload)
        self.assertIn("skipped_families", ledger_payload)
        self.assertTrue(ledger_payload["cohort_query_pack_used"])
        self.assertEqual(ledger_payload["cohort_query_pack_name"], "spreadsheet_operator_admin")
        self.assertEqual(
            ledger_payload["web_query_strategy_path"],
            ["atom_shaped", "cohort_pack", "specialized_surface_targeting"],
        )
        self.assertTrue(ledger_payload["specialized_surface_targeting_used"])
        self.assertEqual(ledger_payload["promotion_gap_class"], "corroboration_gap")
        self.assertEqual(ledger_payload["near_miss_enrichment_action"], "GATHER_CORROBORATION")
        self.assertEqual(
            ledger_payload["sufficiency_priority_reason"],
            "single_or_thin_family_support_blocks_selection",
        )
        self.assertFalse(ledger_payload["value_enrichment_used"])

        queued = await self.queue.get()
        self.assertEqual(queued.to_agent, "orchestrator")
        self.assertEqual(queued.msg_type, MessageType.EVIDENCE)
        self.assertEqual(queued.payload["finding_id"], finding_id)
        self.assertIn("evidence_scores", queued.payload)
        self.assertIn("corroboration", queued.payload)
        self.assertIn("market_enrichment", queued.payload)
        json.dumps(queued.payload["evidence_scores"]["evidence"])

    async def test_busy_count_does_not_double_count_inflight_work(self):
        gate = asyncio.Event()

        async def blocked():
            await gate.wait()

        task = asyncio.create_task(blocked())
        self.agent._inflight.add(task)
        self.agent._processing_count = 1

        try:
            self.assertEqual(self.agent.busy_count(), 1)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_enrichment_passes_structured_atom_into_recurrence_gathering(self):
        observed = {}

        async def fake_validate_problem(**kwargs):
            observed.update(kwargs)
            return {
                "problem_score": 0.47,
                "value_score": 0.41,
                "feasibility_score": 0.62,
                "solution_gap_score": 0.58,
                "saturation_score": 0.44,
                "evidence": {
                    "query": '"manual data entry" spreadsheet',
                    "recurrence_queries": ['"manual data entry" operations'],
                    "recurrence_state": "thin",
                    "recurrence_query_coverage": 0.5,
                    "recurrence_docs": [{"url": "https://forum.example.com/thread"}],
                    "competitor_docs": [{"url": "https://tool.example.com"}],
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://reddit.com/test",
                product_built="Spreadsheet cleanup every week",
                outcome_summary="Ops still handle spreadsheet cleanup by hand every week.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="reddit-problem",
                source_type="forum",
                source_url="https://reddit.com/test",
                title="Spreadsheet cleanup every week",
                body_excerpt="Ops still handle spreadsheet cleanup by hand every week.",
                quote_text="Ops still handle spreadsheet cleanup by hand every week.",
                role_hint="operations lead",
                content_hash="evidence-signal",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="ops-weekly-cleanup",
                segment="small business operators",
                user_role="operations lead",
                job_to_be_done="keep supplier contracts and client follow-up on track",
                trigger_event="email follow-up stays manual",
                pain_statement="Manual data entry piles up.",
                failure_mode="manual data entry piles up",
                current_workaround="spreadsheets, manual work",
                current_tools="Excel",
                urgency_clues="",
                frequency_clues="weekly",
                emotional_intensity=0.5,
                cost_consequence_clues="time, consequence",
                why_now_clues="",
                confidence=0.8,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        self.assertIsNotNone(observed.get("atom"))
        self.assertEqual(observed["atom"].finding_id, finding_id)

    async def test_enrichment_persists_corroboration_and_market_enrichment_objects(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.47,
                "value_score": 0.52,
                "feasibility_score": 0.62,
                "solution_gap_score": 0.58,
                "saturation_score": 0.44,
                "evidence": {
                    "query": '"manual data entry" spreadsheet',
                    "competitor_query": '"manual data entry" spreadsheet software tool alternative',
                    "recurrence_queries": ['"manual data entry" operations'],
                    "recurrence_state": "thin",
                    "recurrence_query_coverage": 0.5,
                    "recurrence_doc_count": 2,
                    "recurrence_domain_count": 1,
                    "recurrence_results_by_source": {"reddit": 2, "web": 0},
                    "recurrence_docs": [{"url": "https://forum.example.com/thread"}],
                    "competitor_docs": [{"url": "https://tool.example.com"}],
                    "competitor_domains": ["tool.example.com"],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://reddit.com/test",
                product_built="Spreadsheet cleanup every week",
                outcome_summary="Ops still handle spreadsheet cleanup by hand every week.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        corroboration = self.db.get_corroboration(finding_id)
        market = self.db.get_market_enrichment(finding_id)
        self.assertIsNotNone(corroboration)
        self.assertEqual(corroboration.run_id, "test-run")
        self.assertEqual(corroboration.recurrence_state, "thin")
        self.assertGreater(corroboration.corroboration_score, 0.0)
        self.assertIn("confirmation_depth_score", corroboration.evidence)
        self.assertIn("source_concentration", corroboration.evidence)
        self.assertIsNotNone(market)
        self.assertEqual(market.run_id, "test-run")
        self.assertGreater(market.demand_score, 0.0)
        self.assertIn("cost_pressure_score", market.evidence)
        self.assertIn("willingness_to_pay_signal", market.evidence)

    async def test_review_metadata_flows_into_market_enrichment_only(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.39,
                "value_score": 0.44,
                "feasibility_score": 0.62,
                "solution_gap_score": 0.58,
                "saturation_score": 0.44,
                "evidence": {
                    "query": '"cart abandonment" analytics',
                    "competitor_query": '"cart abandonment" analytics software tool alternative',
                    "recurrence_queries": ['"cart abandonment" analytics'],
                    "recurrence_state": "thin",
                    "recurrence_query_coverage": 0.5,
                    "recurrence_doc_count": 2,
                    "recurrence_domain_count": 1,
                    "recurrence_results_by_source": {"reddit": 2, "web": 0},
                    "recurrence_docs": [{"url": "https://forum.example.com/thread"}],
                    "competitor_docs": [{"url": "https://tool.example.com"}],
                    "competitor_domains": ["tool.example.com"],
                    "pain_hits": 2,
                    "value_hits": 3,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="wordpress-review/woocommerce",
                source_url="https://wordpress.org/support/topic/slow-development-dated-features-far-behind-shopify/",
                product_built="Slow development, dated features, far behind shopify",
                tool_used="WooCommerce",
                outcome_summary="We still rely on paid third party analytics and manual GA4 work.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps(
                    {
                        "source_classification": {
                            "review_issue_type": "reusable_workflow_pain",
                            "review_generalizability_score": 0.72,
                            "review_generalizability_reasons": [
                                "workflow_relevance",
                                "manual_fallback_or_recovery",
                            ],
                        },
                        "review_metadata": {
                            "review_source": "wordpress_plugin_directory",
                            "product_name": "WooCommerce",
                            "review_rating": 1.0,
                            "aggregate_rating": 4.4,
                            "review_count": 4582,
                            "active_installs": 7000000,
                            "pricing": "free",
                            "version": "10.6.1",
                            "last_updated": "6 days ago",
                        }
                    }
                ),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="wordpress-review/woocommerce",
                source_type="review",
                source_url="https://wordpress.org/support/topic/slow-development-dated-features-far-behind-shopify/",
                title="Slow development, dated features, far behind shopify",
                body_excerpt="We still rely on paid third party analytics and manual GA4 work.",
                quote_text="We still rely on paid third party analytics and manual GA4 work.",
                role_hint="operator",
                content_hash="review-evidence-signal",
                finding_id=finding_id,
                metadata_json=json.dumps({"evidence": {"review_metadata": {"active_installs": 7000000}}}),
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="woocommerce-review-analytics",
                segment="commerce operators",
                user_role="operator",
                job_to_be_done="understand conversion analytics without manual reporting",
                trigger_event="after analytics gaps show up",
                pain_statement="Analytics gaps still require manual work.",
                failure_mode="conversion analytics stay manual",
                current_workaround="paid plugins, ga4 setup",
                current_tools="WooCommerce",
                urgency_clues="",
                frequency_clues="weekly",
                emotional_intensity=0.5,
                cost_consequence_clues="time, cost",
                why_now_clues="",
                confidence=0.8,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        market = self.db.get_market_enrichment(finding_id)
        atom = self.db.get_problem_atoms_by_finding(finding_id)[0]

        self.assertIsNotNone(market)
        self.assertEqual(market.evidence["review_product_name"], "WooCommerce")
        self.assertEqual(market.evidence["review_issue_type"], "reusable_workflow_pain")
        self.assertEqual(market.evidence["review_generalizability_score"], 0.72)
        self.assertEqual(market.evidence["review_count"], 4582)
        self.assertEqual(market.evidence["active_installs"], 7000000)
        self.assertEqual(market.evidence["pricing"], "free")
        self.assertEqual(atom.current_tools, "WooCommerce")
        self.assertNotIn("7000000", atom.pain_statement)

    async def test_enrichment_times_out_into_thin_evidence_package(self):
        async def slow_validate_problem(**_kwargs):
            await asyncio.sleep(0.05)
            return {
                "problem_score": 0.9,
                "value_score": 0.9,
                "feasibility_score": 0.9,
                "solution_gap_score": 0.9,
                "saturation_score": 0.1,
                "evidence": {"recurrence_docs": [{"url": "https://example.com"}]},
            }

        self.agent.evidence_timeout_seconds = 0.01
        self.agent.toolkit.validate_problem = slow_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem",
                source_url="https://reddit.com/test",
                product_built="Spreadsheet cleanup every week",
                outcome_summary="Ops still handle spreadsheet cleanup by hand every week.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )

        result = await self.agent._enrich_finding({"finding_id": finding_id})

        self.assertTrue(result["success"])
        self.assertEqual(result["evidence_scores"]["evidence"]["recurrence_state"], "timeout")
        self.assertTrue(result["evidence_scores"]["evidence"]["timeout"])
        self.assertTrue(result["evidence_scores"]["evidence"]["recurrence_timeout"])
        self.assertEqual(result["evidence_scores"]["evidence"]["recurrence_gap_reason"], "evidence_agent_timeout")

    async def test_shopify_review_metadata_flows_into_market_enrichment_only(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.41,
                "value_score": 0.49,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"backup restore" shopify review',
                    "competitor_query": '"backup restore" shopify app alternative',
                    "recurrence_queries": ['"backup restore" shopify'],
                    "recurrence_state": "thin",
                    "recurrence_query_coverage": 0.5,
                    "recurrence_doc_count": 2,
                    "recurrence_domain_count": 1,
                    "recurrence_results_by_source": {"reddit": 1, "web": 1},
                    "recurrence_docs": [{"url": "https://forum.example.com/thread"}],
                    "competitor_docs": [{"url": "https://apps.shopify.com/backup-1"}],
                    "competitor_domains": ["apps.shopify.com"],
                    "pain_hits": 2,
                    "value_hits": 3,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="shopify-review/backup-and-sync",
                source_url="https://apps.shopify.com/backup-and-sync/reviews/101",
                product_built="Restore jobs fail every time",
                tool_used="AppsByB: Backup & Sync",
                outcome_summary="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps(
                    {
                        "source_classification": {
                            "review_issue_type": "reusable_workflow_pain",
                            "review_generalizability_score": 0.76,
                            "review_generalizability_reasons": [
                                "workflow_relevance",
                                "manual_fallback_or_recovery",
                            ],
                        },
                        "review_metadata": {
                            "review_source": "shopify_app_store",
                            "record_origin": "review_text",
                            "product_name": "AppsByB: Backup & Sync",
                            "listing_url": "https://apps.shopify.com/backup-and-sync",
                            "category": "Store management Backups",
                            "review_rating": 1.0,
                            "aggregate_rating": 3.7,
                            "review_count": 118,
                            "pricing": "$19/month. Free trial available.",
                            "popularity_proxy": 118,
                            "developer_name": "AppsByB",
                            "launched_at": "January 14, 2021",
                            "built_for_shopify": True,
                        },
                    }
                ),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="shopify-review/backup-and-sync",
                source_type="review",
                source_url="https://apps.shopify.com/backup-and-sync/reviews/101",
                title="Restore jobs fail every time",
                body_excerpt="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                quote_text="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                role_hint="operator",
                content_hash="shopify-review-evidence-signal",
                finding_id=finding_id,
                metadata_json=json.dumps({"evidence": {"review_metadata": {"popularity_proxy": 118}}}),
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="shopify-backup-review",
                segment="commerce operators",
                user_role="operator",
                job_to_be_done="keep backup restore and recovery reliable",
                trigger_event="after restore jobs fail",
                pain_statement="Restore jobs fail every time, so teams fall back to manual recovery.",
                failure_mode="backup restore fails repeatedly",
                current_workaround="manual recovery checklists",
                current_tools="AppsByB: Backup & Sync",
                urgency_clues="",
                frequency_clues="every time",
                emotional_intensity=0.5,
                cost_consequence_clues="time, risk",
                why_now_clues="",
                confidence=0.82,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        market = self.db.get_market_enrichment(finding_id)
        atom = self.db.get_problem_atoms_by_finding(finding_id)[0]

        self.assertIsNotNone(market)
        self.assertEqual(market.evidence["review_product_name"], "AppsByB: Backup & Sync")
        self.assertEqual(market.evidence["category"], "Store management Backups")
        self.assertEqual(market.evidence["developer_name"], "AppsByB")
        self.assertEqual(market.evidence["popularity_proxy"], 118)
        self.assertTrue(market.evidence["built_for_shopify"])
        self.assertNotIn("Store management Backups", atom.pain_statement)
        self.assertNotIn("January 14, 2021", atom.pain_statement)

    async def test_backup_restore_wedge_enrichment_boosts_value_context(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.58,
                "value_score": 0.34,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"backup restore" recovery reliability',
                    "competitor_query": '"backup restore" software tool alternative',
                    "recurrence_queries": ['"backup restore" recovery reliability'],
                    "recurrence_state": "supported",
                    "recurrence_query_coverage": 1.0,
                    "recurrence_doc_count": 6,
                    "recurrence_domain_count": 2,
                    "recurrence_results_by_source": {"reddit": 3, "github": 3},
                    "recurrence_results_by_query": {'"backup restore" recovery reliability': 6},
                    "recurrence_docs": [
                        {
                            "title": "Backup restore fails every time",
                            "url": "https://www.reddit.com/r/sysadmin/comments/restore",
                            "snippet": "Restore jobs fail and teams repeat manual recovery steps.",
                            "source": "reddit.com",
                        }
                    ],
                    "competitor_docs": [
                        {
                            "title": "Veeam Backup & Replication",
                            "url": "https://www.veeam.com/",
                            "snippet": "Backup and recovery platform for restore operations.",
                            "source": "veeam.com",
                        }
                    ],
                    "competitor_domains": ["veeam.com"],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="github-issue/org/repo",
                source_url="https://github.com/org/repo/issues/123",
                product_built="Backup restore fails after migration",
                outcome_summary="After restore, environments stay unreachable and teams rerun recovery manually during cutovers.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="github-issue/org/repo",
                source_type="github_issue",
                source_url="https://github.com/org/repo/issues/123",
                title="Backup restore fails after migration",
                body_excerpt="After restore, environments stay unreachable and teams rerun recovery manually during cutovers.",
                quote_text="After restore, environments stay unreachable and teams rerun recovery manually during cutovers.",
                role_hint="platform engineer",
                content_hash="backup-restore-wedge",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="backup-restore-wedge",
                segment="infrastructure teams",
                user_role="platform engineer",
                job_to_be_done="keep backup restore and recovery reliable",
                trigger_event="after restore jobs run",
                pain_statement="Restored environments stay unreachable.",
                failure_mode="restored environments stay unreachable",
                current_workaround="manual recovery, custom scripts",
                current_tools="Veeam",
                urgency_clues="blocked",
                frequency_clues="every time",
                emotional_intensity=0.5,
                cost_consequence_clues="time, consequence",
                why_now_clues="",
                confidence=0.82,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        market = self.db.get_market_enrichment(finding_id)
        self.assertIsNotNone(market)
        self.assertEqual(market.evidence["wedge_name"], "backup_restore_reliability")
        self.assertGreater(market.evidence["wedge_fit_score"], 0.7)
        self.assertGreater(market.evidence["wedge_operational_risk_score"], 0.3)
        self.assertGreater(market.evidence["wedge_buyer_ownership_score"], 0.25)
        self.assertGreater(market.evidence["wedge_value_lift"], 0.4)
        self.assertTrue(market.evidence["wedge_active"])
        self.assertEqual(market.evidence["wedge_rule_version"], "backup_restore_v2")
        self.assertIn("explicit_backup_restore_job", market.evidence["wedge_activation_reasons"])
        self.assertEqual(market.evidence["wedge_block_reasons"], [])

    async def test_non_backup_case_does_not_receive_backup_restore_wedge(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.51,
                "value_score": 0.36,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"etsy p&l" spreadsheet',
                    "competitor_query": '"etsy p&l" spreadsheet software',
                    "recurrence_queries": ['"etsy p&l" spreadsheet'],
                    "recurrence_state": "supported",
                    "recurrence_query_coverage": 1.0,
                    "recurrence_doc_count": 5,
                    "recurrence_domain_count": 2,
                    "recurrence_results_by_source": {"reddit": 3, "web": 2},
                    "recurrence_results_by_query": {'"etsy p&l" spreadsheet': 5},
                    "recurrence_docs": [
                        {
                            "title": "How do you track your P&L accurately?",
                            "url": "https://www.reddit.com/r/EtsySellers/comments/pnl",
                            "snippet": "We still rely on spreadsheets to track pricing and costs.",
                            "source": "reddit.com",
                        }
                    ],
                    "competitor_docs": [
                        {
                            "title": "Profit analytics tools for Etsy",
                            "url": "https://example.com/etsy-analytics",
                            "snippet": "Pricing and profitability analytics for sellers.",
                            "source": "example.com",
                        }
                    ],
                    "competitor_domains": ["example.com"],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="reddit-problem/EtsySellers",
                source_url="https://reddit.com/r/EtsySellers/comments/123",
                product_built="How do you track your P&L accurately?",
                outcome_summary="We still rely on spreadsheets to track pricing, costs, and profitability accurately.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps({}),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="reddit-problem/EtsySellers",
                source_type="forum",
                source_url="https://reddit.com/r/EtsySellers/comments/123",
                title="How do you track your P&L accurately?",
                body_excerpt="We still rely on spreadsheets to track pricing, costs, and profitability accurately.",
                quote_text="We still rely on spreadsheets to track pricing, costs, and profitability accurately.",
                role_hint="seller",
                content_hash="etsy-pnl-no-wedge",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="etsy-profit-tracking",
                segment="commerce sellers",
                user_role="seller",
                job_to_be_done="track pricing, costs, and profitability accurately",
                trigger_event="after sales and fees land",
                pain_statement="Teams fall back to spreadsheets.",
                failure_mode="teams fall back to spreadsheets",
                current_workaround="spreadsheets, csv exports",
                current_tools="Etsy, Excel",
                urgency_clues="",
                frequency_clues="weekly",
                emotional_intensity=0.4,
                cost_consequence_clues="time",
                why_now_clues="",
                confidence=0.78,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        market = self.db.get_market_enrichment(finding_id)
        self.assertIsNotNone(market)
        self.assertEqual(market.evidence["wedge_name"], "")
        self.assertLess(market.evidence["wedge_fit_score"], 0.45)
        self.assertEqual(market.evidence["wedge_value_lift"], 0.0)
        self.assertFalse(market.evidence["wedge_active"])
        self.assertIn("excluded_non_backup_domain_signal", market.evidence["wedge_block_reasons"])
        self.assertEqual(market.evidence["wedge_rule_version"], "backup_restore_v2")

    async def test_backup_restore_wedge_false_positives_are_blocked_across_source_families(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.48,
                "value_score": 0.35,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"generic workflow" manual',
                    "competitor_query": '"generic workflow" software',
                    "recurrence_queries": ['"generic workflow" manual'],
                    "recurrence_state": "supported",
                    "recurrence_query_coverage": 1.0,
                    "recurrence_doc_count": 4,
                    "recurrence_domain_count": 2,
                    "recurrence_results_by_source": {"reddit": 2, "web": 2},
                    "recurrence_results_by_query": {'"generic workflow" manual': 4},
                    "recurrence_docs": [{"url": "https://example.com/thread"}],
                    "competitor_docs": [{"url": "https://example.com/tool", "title": "Workflow tool"}],
                    "competitor_domains": ["example.com"],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        cases = [
            ("reddit-problem/smallbusiness", "Track pricing and P&L accurately", "commerce operators", "seller"),
            ("wordpress-review/woocommerce", "Slow development, dated features", "commerce operators", "operator"),
            ("shopify-review/parcel-intelligence", "The app doesn't work how it should", "commerce operators", "operator"),
            ("github-issue/org/repo", "[Bug]: switching session.dmScope back to main", "software teams", "developer"),
        ]
        for idx, (source, title, segment, role) in enumerate(cases, start=1):
            finding_id = self.db.insert_finding(
                Finding(
                    source=source,
                    source_url=f"https://example.com/{idx}",
                    product_built=title,
                    outcome_summary="Teams fall back to spreadsheets and manual work.",
                    content_hash=f"no-wedge-finding-{idx}",
                    finding_kind="pain_point",
                    source_class="pain_signal",
                    evidence_json=json.dumps({}),
                )
            )
            signal_id = self.db.insert_raw_signal(
                RawSignal(
                    source_name=source,
                    source_type="review" if "review" in source else ("github_issue" if "github" in source else "forum"),
                    source_url=f"https://example.com/{idx}",
                    title=title,
                    body_excerpt="Teams fall back to spreadsheets and manual work.",
                    quote_text="Teams fall back to spreadsheets and manual work.",
                    role_hint=role,
                    content_hash=f"no-wedge-{idx}",
                    finding_id=finding_id,
                )
            )
            self.db.insert_problem_atom(
                ProblemAtom(
                    signal_id=signal_id,
                    finding_id=finding_id,
                    cluster_key=f"no-wedge-{idx}",
                    segment=segment,
                    user_role=role,
                    job_to_be_done="keep operations data and follow-up in sync without manual cleanup",
                    trigger_event="after weekly reconciliation",
                    pain_statement="Teams fall back to spreadsheets.",
                    failure_mode="teams fall back to spreadsheets",
                    current_workaround="spreadsheets, manual work",
                    current_tools="Excel",
                    urgency_clues="",
                    frequency_clues="weekly",
                    emotional_intensity=0.4,
                    cost_consequence_clues="time",
                    why_now_clues="",
                    confidence=0.7,
                    atom_json="{}",
                )
            )
            await self.agent._enrich_finding({"finding_id": finding_id})
            market = self.db.get_market_enrichment(finding_id)
            self.assertEqual(market.evidence["wedge_name"], "")
            self.assertFalse(market.evidence["wedge_active"])
            self.assertTrue(market.evidence["wedge_block_reasons"])

    async def test_cross_source_corroboration_counts_reddit_and_review_families(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.52,
                "value_score": 0.51,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"backup restore" manual recovery',
                    "competitor_query": '"backup restore" manual recovery software',
                    "recurrence_queries": ['"backup restore" manual recovery'],
                    "recurrence_state": "supported",
                    "recurrence_query_coverage": 1.0,
                    "recurrence_doc_count": 4,
                    "recurrence_domain_count": 2,
                    "recurrence_results_by_source": {"reddit": 2, "web": 1, "github": 0},
                    "recurrence_results_by_query": {'"backup restore" manual recovery': 4},
                    "recurrence_docs": [
                        {
                            "title": "Restore jobs fail every time",
                            "url": "https://www.reddit.com/r/sysadmin/comments/restore",
                            "snippet": "Manual recovery checklist after every restore failure.",
                            "source": "reddit.com",
                        },
                        {
                            "title": "Backup and restore failures",
                            "url": "https://wordpress.org/support/topic/backup-restore-fails/",
                            "snippet": "Teams keep manual recovery steps when restore jobs fail.",
                            "source": "wordpress.org",
                        },
                    ],
                    "competitor_docs": [{"url": "https://tool.example.com"}],
                    "competitor_domains": ["tool.example.com"],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="shopify-review/backup-and-sync",
                source_url="https://apps.shopify.com/backup-and-sync/reviews/101",
                product_built="Restore jobs fail every time",
                outcome_summary="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps(
                    {
                        "source_classification": {
                            "review_issue_type": "reusable_workflow_pain",
                            "review_generalizability_score": 0.8,
                            "review_generalizability_reasons": ["workflow_relevance"],
                        }
                    }
                ),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="shopify-review/backup-and-sync",
                source_type="review",
                source_url="https://apps.shopify.com/backup-and-sync/reviews/101",
                title="Restore jobs fail every time",
                body_excerpt="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                quote_text="Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.",
                role_hint="operator",
                content_hash="cross-source-review-signal",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="cross-source-restore",
                segment="commerce operators",
                user_role="operator",
                job_to_be_done="keep backup restore and recovery reliable",
                trigger_event="after restore jobs fail",
                pain_statement="Restore jobs fail every time.",
                failure_mode="backup restore fails repeatedly",
                current_workaround="manual recovery checklists",
                current_tools="AppsByB: Backup & Sync",
                urgency_clues="",
                frequency_clues="every time",
                emotional_intensity=0.6,
                cost_consequence_clues="time, risk",
                why_now_clues="",
                confidence=0.84,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        corroboration = self.db.get_corroboration(finding_id)
        market = self.db.get_market_enrichment(finding_id)
        self.assertEqual(corroboration.evidence["source_family_diversity"], 3)
        self.assertEqual(corroboration.evidence["source_family_match_counts"]["reddit"], 1)
        self.assertEqual(corroboration.evidence["source_family_match_counts"]["wordpress_review"], 1)
        self.assertEqual(corroboration.evidence["core_source_family_diversity"], 3)
        self.assertEqual(corroboration.evidence["source_group_diversity"], 2)
        self.assertGreater(corroboration.evidence["cross_source_match_score"], 0.3)
        self.assertGreater(corroboration.evidence["core_source_family_bonus"], 0.0)
        self.assertEqual(corroboration.evidence["generalizability_class"], "reusable_workflow_pain")
        self.assertGreater(market.evidence["multi_source_value_lift"], 0.2)

    async def test_cross_source_corroboration_counts_reddit_and_github_families(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.58,
                "value_score": 0.47,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"sync workflows fail" manual workaround',
                    "competitor_query": '"sync workflows fail" integration tool',
                    "recurrence_queries": ['"sync workflows fail" manual workaround'],
                    "recurrence_state": "supported",
                    "recurrence_query_coverage": 1.0,
                    "recurrence_doc_count": 3,
                    "recurrence_domain_count": 2,
                    "recurrence_results_by_source": {"reddit": 1, "github": 2, "web": 0},
                    "recurrence_results_by_query": {'"sync workflows fail" manual workaround': 3},
                    "recurrence_docs": [
                        {
                            "title": "Sync workflows fail after changes",
                            "url": "https://www.reddit.com/r/sysadmin/comments/sync",
                            "snippet": "Ops falls back to manual exports after sync breaks.",
                            "source": "reddit.com",
                        },
                        {
                            "title": "Sync fails after upgrade",
                            "url": "https://github.com/org/repo/issues/123",
                            "snippet": "Manual workaround after sync fails and data handoff breaks.",
                            "source": "github.com",
                        },
                    ],
                    "competitor_docs": [],
                    "competitor_domains": [],
                    "pain_hits": 2,
                    "value_hits": 2,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="github-issue/org/repo",
                source_url="https://github.com/org/repo/issues/123",
                product_built="Sync fails after upgrade",
                outcome_summary="Teams fall back to manual exports after sync fails.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps(
                    {
                        "source_classification": {
                            "github_issue_type": "reusable_workflow_pain",
                            "github_specificity_score": 0.74,
                            "github_specificity_reasons": ["reusable_workflow_pattern"],
                        }
                    }
                ),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="github-issue/org/repo",
                source_type="github_issue",
                source_url="https://github.com/org/repo/issues/123",
                title="Sync fails after upgrade",
                body_excerpt="Teams fall back to manual exports after sync fails.",
                quote_text="Teams fall back to manual exports after sync fails.",
                role_hint="support lead",
                content_hash="cross-source-github-signal",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="cross-source-sync",
                segment="support teams",
                user_role="support lead",
                job_to_be_done="keep sync and data handoff workflows reliable",
                trigger_event="after deployment changes",
                pain_statement="Teams fall back to manual exports after sync fails.",
                failure_mode="sync workflows fail after changes",
                current_workaround="manual exports",
                current_tools="GitHub",
                urgency_clues="",
                frequency_clues="recurring",
                emotional_intensity=0.55,
                cost_consequence_clues="time, consequence",
                why_now_clues="",
                confidence=0.8,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        corroboration = self.db.get_corroboration(finding_id)
        self.assertEqual(corroboration.evidence["origin_source_family"], "github")
        self.assertEqual(corroboration.evidence["core_source_family_diversity"], 2)
        self.assertEqual(corroboration.evidence["source_group_diversity"], 2)
        self.assertIn("reddit", corroboration.evidence["source_families"])
        self.assertGreater(corroboration.evidence["cross_source_bonus"], 0.0)

    async def test_product_specific_review_gets_generalizability_penalty(self):
        async def fake_validate_problem(**_kwargs):
            return {
                "problem_score": 0.35,
                "value_score": 0.32,
                "feasibility_score": 0.61,
                "solution_gap_score": 0.57,
                "saturation_score": 0.43,
                "evidence": {
                    "query": '"plugin settings page broken"',
                    "competitor_query": "",
                    "recurrence_queries": ['"plugin settings page broken"'],
                    "recurrence_state": "thin",
                    "recurrence_query_coverage": 0.5,
                    "recurrence_doc_count": 1,
                    "recurrence_domain_count": 1,
                    "recurrence_results_by_source": {"web": 1},
                    "recurrence_results_by_query": {'"plugin settings page broken"': 1},
                    "recurrence_docs": [
                        {
                            "title": "Plugin settings page broken",
                            "url": "https://wordpress.org/support/topic/plugin-settings-page-broken/",
                            "snippet": "The settings page is unusable after the last update.",
                            "source": "wordpress.org",
                        }
                    ],
                    "competitor_docs": [],
                    "competitor_domains": [],
                    "pain_hits": 1,
                    "value_hits": 0,
                },
            }

        self.agent.toolkit.validate_problem = fake_validate_problem
        finding_id = self.db.insert_finding(
            Finding(
                source="wordpress-review/some-plugin",
                source_url="https://wordpress.org/support/topic/plugin-settings-page-broken/",
                product_built="Plugin settings page broken",
                outcome_summary="The settings page is unusable after the last update.",
                finding_kind="pain_point",
                source_class="pain_signal",
                evidence_json=json.dumps(
                    {
                        "source_classification": {
                            "review_issue_type": "product_specific_issue",
                            "review_generalizability_score": 0.12,
                            "review_generalizability_reasons": ["vendor_specific_complaint"],
                        }
                    }
                ),
            )
        )
        signal_id = self.db.insert_raw_signal(
            RawSignal(
                source_name="wordpress-review/some-plugin",
                source_type="review",
                source_url="https://wordpress.org/support/topic/plugin-settings-page-broken/",
                title="Plugin settings page broken",
                body_excerpt="The settings page is unusable after the last update.",
                quote_text="The settings page is unusable after the last update.",
                role_hint="",
                content_hash="product-specific-review-signal",
                finding_id=finding_id,
            )
        )
        self.db.insert_problem_atom(
            ProblemAtom(
                signal_id=signal_id,
                finding_id=finding_id,
                cluster_key="plugin-settings-page",
                segment="wordpress operators",
                user_role="operator",
                job_to_be_done="fix plugin settings page",
                trigger_event="after plugin update",
                pain_statement="The settings page is unusable after the last update.",
                failure_mode="plugin settings page breaks after update",
                current_workaround="",
                current_tools="WordPress",
                urgency_clues="",
                frequency_clues="",
                emotional_intensity=0.3,
                cost_consequence_clues="",
                why_now_clues="",
                confidence=0.45,
                atom_json="{}",
            )
        )

        await self.agent._enrich_finding({"finding_id": finding_id})

        corroboration = self.db.get_corroboration(finding_id)
        self.assertEqual(corroboration.evidence["generalizability_class"], "product_specific_issue")
        self.assertGreater(corroboration.evidence["generalizability_penalty"], 0.0)
