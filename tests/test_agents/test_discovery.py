"""Tests for the discovery agent."""

import asyncio
import os
import sys
import tempfile
import time
from datetime import UTC, datetime

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.agents.base import AgentStatus
from src.agents.discovery import DiscoveryAgent, is_wedge_ready_signal
from src.database import Database, Finding, Validation
from src.messaging import MessageType
from src.research_tools import DiscoveryQueryPlan


@pytest.fixture
def temp_db():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        yield db
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_default_initialization(temp_db):
    agent = DiscoveryAgent(temp_db)

    assert agent.name == "discovery"
    assert agent.status == AgentStatus.IDLE
    assert agent.sources == ["youtube", "reddit", "github"]
    assert agent.check_interval == 300
    assert agent.db is temp_db
    assert agent._message_queue is not None
    assert agent._seen_hashes == set()


def test_discover_once_refreshes_config_after_expansion(temp_db, tmp_path, monkeypatch):
    state_path = tmp_path / "discovery_expansion.json"
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "discovery": {
                "auto_expand": True,
                "expansion": {"state_path": str(state_path), "cooldown_hours": 0},
                "reddit": {"problem_keywords": ["base keyword"], "problem_subreddits": ["basesub"]},
            }
        },
    )

    monkeypatch.setattr(
        "agents.discovery.run_expansion",
        lambda db, config: {
            "expanded": True,
            "added_keywords": ["fresh keyword"],
            "added_subreddits": ["freshsub"],
        },
    )

    def fake_get_expanded_config(config):
        merged = dict(config)
        discovery = dict(merged.get("discovery", {}))
        reddit = dict(discovery.get("reddit", {}))
        reddit["problem_keywords"] = [*reddit.get("problem_keywords", []), "fresh keyword"]
        reddit["problem_subreddits"] = [*reddit.get("problem_subreddits", []), "freshsub"]
        discovery["reddit"] = reddit
        merged["discovery"] = discovery
        return merged

    monkeypatch.setattr("src.agents.discovery.get_expanded_config", fake_get_expanded_config)

    captured = {}

    class DummyToolkit:
        def __init__(self, config):
            captured["keywords"] = config["discovery"]["reddit"]["problem_keywords"]
            captured["subreddits"] = config["discovery"]["reddit"]["problem_subreddits"]

        def set_discovery_feedback(self, feedback):
            return None

    monkeypatch.setattr("src.agents.discovery.ResearchToolkit", DummyToolkit)

    async def no_prime():
        return None

    async def no_results(source):
        return []

    agent._prime_reddit_relay = no_prime
    agent._check_source = no_results

    asyncio.run(agent._discover_once())

    assert captured["keywords"] == ["base keyword", "fresh keyword"]
    assert captured["subreddits"] == ["basesub", "freshsub"]


def test_merge_problem_space_queries_normalizes_string_subreddit_payloads(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "discovery": {
                "reddit": {"problem_keywords": ["base keyword"], "problem_subreddits": ["accounting"]},
                "web": {"keywords": []},
                "github": {"problem_keywords": []},
            }
        },
    )

    class DummySpace:
        keywords = ["fresh keyword"]
        subreddits = "r/automation, notion, n"
        web_queries = []
        github_queries = []

    asyncio.run(agent._merge_problem_space_queries([DummySpace()]))

    reddit_cfg = agent.config["discovery"]["reddit"]
    assert "fresh keyword" in reddit_cfg["problem_keywords"]
    assert "automation" in reddit_cfg["problem_subreddits"]
    assert "notion" in reddit_cfg["problem_subreddits"]
    assert "n" not in reddit_cfg["problem_subreddits"]


def test_discover_once_closes_old_toolkit_when_refreshing_config(temp_db, tmp_path, monkeypatch):
    state_path = tmp_path / "discovery_expansion.json"
    toolkits = []

    class DummyToolkit:
        def __init__(self, config):
            self.config = config
            self.closed = False
            toolkits.append(self)

        async def close(self):
            self.closed = True

        def set_discovery_feedback(self, feedback):
            return None

    monkeypatch.setattr("src.agents.discovery.ResearchToolkit", DummyToolkit)

    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "discovery": {
                "auto_expand": True,
                "expansion": {"state_path": str(state_path), "cooldown_hours": 0},
                "reddit": {"problem_keywords": ["base keyword"], "problem_subreddits": ["basesub"]},
            }
        },
    )

    monkeypatch.setattr(
        "agents.discovery.run_expansion",
        lambda db, config: {
            "expanded": True,
            "added_keywords": ["fresh keyword"],
            "added_subreddits": ["freshsub"],
        },
    )

    def fake_get_expanded_config(config):
        merged = dict(config)
        discovery = dict(merged.get("discovery", {}))
        reddit = dict(discovery.get("reddit", {}))
        reddit["problem_keywords"] = [*reddit.get("problem_keywords", []), "fresh keyword"]
        reddit["problem_subreddits"] = [*reddit.get("problem_subreddits", []), "freshsub"]
        discovery["reddit"] = reddit
        merged["discovery"] = discovery
        return merged

    monkeypatch.setattr("src.agents.discovery.get_expanded_config", fake_get_expanded_config)

    async def no_prime():
        return None

    async def no_results(source):
        return []

    agent._prime_reddit_relay = no_prime
    agent._check_source = no_results

    asyncio.run(agent._discover_once())

    assert len(toolkits) >= 2
    assert toolkits[0].closed is True
    assert toolkits[-1].closed is False


def test_custom_sources_initialization(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["reddit", "github"])
    assert agent.sources == ["reddit", "github"]


def test_check_source_web_respects_focused_problem_only_and_configured_problem_queries(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["web"],
        config={
            "discovery": {
                "focused_problem_only": True,
                "web": {
                    "problem_keywords": ["stripe quickbooks reconciliation manual"],
                    "success_keywords": ["should not run"],
                    "market_keywords": ["should not run"],
                },
            }
        },
    )
    calls = []

    class DummyToolkit:
        def build_discovery_query_plan(self, source_name, queries, limit, cycle_index):
            return DiscoveryQueryPlan(source_name=source_name, queries=list(queries)[:limit], slice_size=min(limit, len(list(queries))))

        async def _discover_success_stories_on_web(self, queries, observer=None):
            calls.append(("web-success", list(queries)))
            return [{"source": "web-success"}]

        async def _discover_marketplace_problem_threads(self, queries, observer=None):
            calls.append(("market-problem", list(queries)))
            return [{"source": "market-problem"}]

        async def _discover_web_problem_threads(self, queries, observer=None):
            calls.append(("web-problem", list(queries)))
            return [{"source": "web-problem"}]

    agent.toolkit = DummyToolkit()

    results = asyncio.run(agent._check_source("web"))

    assert calls == [("web-problem", ["stripe quickbooks reconciliation manual"])]
    assert results == [{"source": "web-problem"}]


def test_check_source_github_respects_configured_problem_queries(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["github"],
        config={"discovery": {"github": {"problem_keywords": ['"stripe quickbooks reconciliation" issue']}}},
    )
    calls = []

    class DummyToolkit:
        _discovery_feedback = {}

        def build_discovery_query_plan(self, source_name, queries, limit, cycle_index):
            return DiscoveryQueryPlan(source_name=source_name, queries=list(queries)[:limit], slice_size=min(limit, len(list(queries))))

        async def _discover_github_problem_threads(self, queries, observer=None):
            calls.append(list(queries))
            return [{"source": "github-problem"}]

    agent.toolkit = DummyToolkit()

    results = asyncio.run(agent._check_source("github"))

    assert calls == [['"stripe quickbooks reconciliation" issue']]
    assert results == [{"source": "github-problem"}]


def test_process_finding_persists_qualified_problem_and_emits_message(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/smallbusiness/comments/1",
        "entrepreneur": "ops lead",
        "product_built": "Spreadsheet-heavy reconciliation workflow",
        "outcome_summary": (
            "Every day our ops team manually copy and paste payout data into spreadsheets. "
            "It breaks after pricing changes and we fall back to manual cleanup."
        ),
        "finding_kind": "pain_point",
    }

    finding_id = asyncio.run(agent._process_finding(finding_data))

    assert finding_id is not None
    finding = temp_db.get_finding(finding_id)
    assert finding is not None
    assert finding.status == "qualified"
    signals = temp_db.get_raw_signals_by_finding(finding_id)
    assert signals
    assert temp_db.get_problem_atoms_by_finding(finding_id)
    assert finding.evidence["high_leverage"]["status"] in {"candidate", "ordinary"}
    assert "high_leverage" in signals[0].metadata

    queued = asyncio.run(agent._message_queue.get_for_agent("orchestrator"))
    assert queued is not None
    assert queued.msg_type == MessageType.FINDING
    assert queued.payload["finding_id"] == finding_id


def test_process_finding_tolerates_missing_atom_json(temp_db, monkeypatch):
    agent = DiscoveryAgent(temp_db)

    def _build_atom(_signal_payload, _finding_data):
        return {
            "cluster_key": "stripe_qbo_mismatch",
            "segment": "finance operations",
            "user_role": "ops lead",
            "job_to_be_done": "reconcile Stripe payouts to QuickBooks",
            "trigger_event": "weekly close",
            "pain_statement": "exports do not match",
            "failure_mode": "stripe payouts do not match quickbooks exports",
            "current_workaround": "manually rebuild in spreadsheets",
            "current_tools": "Stripe, QuickBooks, spreadsheets",
            "urgency_clues": "close deadline",
            "frequency_clues": "weekly",
            "emotional_intensity": 0.4,
            "cost_consequence_clues": "hours of cleanup",
            "why_now_clues": "refund volume increased",
            "confidence": 0.75,
            "platform": "QuickBooks",
            "specificity_score": 0.84,
            "consequence_score": 0.7,
            "atom_extraction_method": "heuristic",
        }

    monkeypatch.setattr("src.agents.discovery.build_problem_atom", _build_atom)
    monkeypatch.setattr("src.agents.discovery.is_wedge_ready_signal", lambda *_args, **_kwargs: (True, "passed"))
    monkeypatch.setattr(
        "src.agents.discovery.classify_source_signal",
        lambda *_args, **_kwargs: {"source_class": "pain_signal", "reasons": []},
    )
    monkeypatch.setattr(
        "src.agents.discovery.qualify_problem_signal",
        lambda *_args, **_kwargs: {"accepted": True, "positive_signals": ["specific_failure"], "negative_signals": []},
    )

    finding_id = asyncio.run(
        agent._process_finding(
            {
                "source": "reddit-problem",
                "source_url": "https://reddit.com/r/smallbusiness/comments/2",
                "entrepreneur": "ops lead",
                "product_built": "Spreadsheet-heavy reconciliation workflow",
                "outcome_summary": (
                    "Stripe payouts do not match QuickBooks exports and the team rebuilds the ledger "
                    "manually in spreadsheets every week."
                ),
                "finding_kind": "pain_point",
            }
        )
    )

    assert finding_id is not None
    atom = temp_db.get_problem_atoms_by_finding(finding_id)[0]
    assert atom.atom_json


def test_process_finding_screens_out_generic_review_without_atom(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "wordpress-review/woocommerce",
        "source_url": "https://wordpress.org/plugins/woocommerce/",
        "product_built": "Great app",
        "outcome_summary": "Great app. We recommend it for all our clients.",
        "finding_kind": "problem_signal",
        "evidence": {"record_origin": "review"},
    }

    finding_id = asyncio.run(agent._process_finding(finding_data))

    assert finding_id is not None
    finding = temp_db.get_finding(finding_id)
    assert finding is not None
    assert finding.status == "screened_out"
    assert temp_db.get_raw_signals_by_finding(finding_id) == []
    assert temp_db.get_problem_atoms_by_finding(finding_id) == []


def test_is_wedge_ready_signal_rejects_meta_post():
    finding_data = {
        "product_built": "What should I automate?",
        "outcome_summary": "Looking to build a workflow automation tool and asking people what annoys them most at work.",
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is False
    assert reason == "meta_post"


def test_is_wedge_ready_signal_rejects_generic_task_bundle():
    finding_data = {
        "product_built": "If you're still doing this manually in your business, you're wasting hours every week",
        "outcome_summary": (
            "If you or your team are still doing repetitive tasks manually "
            "(data entry, moving info between tools, basic follow-ups, reporting, etc), "
            "you're probably losing hours every week."
        ),
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is False
    assert reason == "generic_prompt"


def test_is_wedge_ready_signal_accepts_specific_workflow_failure():
    finding_data = {
        "product_built": "Stripe imports into QuickBooks keep breaking for client books",
        "outcome_summary": (
            "We manually reformat Stripe CSV exports because duplicate rows and bad date columns "
            "cause QuickBooks imports to fail every month."
        ),
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is True
    assert reason == "passed"


def test_is_wedge_ready_signal_accepts_specific_manual_workflow_without_explicit_failure_verb():
    finding_data = {
        "product_built": "Shopify and Amazon revenue tracking without manual spreadsheets",
        "outcome_summary": (
            "Every week we spend hours reconciling Shopify and Amazon revenue in spreadsheets before month-end close "
            "because the team still rebuilds the shared ledger manually."
        ),
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is True
    assert reason == "passed"


def test_is_wedge_ready_signal_rejects_broad_finance_prompt():
    finding_data = {
        "product_built": "Growing team looking for the best virtual credit card for expense automation",
        "outcome_summary": (
            "We're drowning in receipts and evaluating Ramp and Brex because everyone uses personal cards "
            "for team spend and we want a corporate card solution."
        ),
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is False
    assert reason == "broad_finance_prompt"


def test_is_wedge_ready_signal_rejects_business_risk_story_without_operational_wedge():
    finding_data = {
        "product_built": "Perdi el 23% de mi revenue en un mes cuando se fue un cliente",
        "outcome_summary": (
            "Nuestro cliente mas grande se fue y ahora solo estamos crunching numbers in a spreadsheet "
            "to understand the revenue concentration risk."
        ),
    }

    is_ready, reason = is_wedge_ready_signal(finding_data)

    assert is_ready is False
    assert reason == "business_risk_or_career_post"


def test_process_finding_persists_pre_atom_filtered_signal_as_screened_out(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/entrepreneur/comments/meta",
        "product_built": "What workflow should I automate next?",
        "outcome_summary": "Looking to build an automation product and asking founders which workflow annoys them most.",
        "finding_kind": "problem_signal",
    }

    finding_id = asyncio.run(agent._process_finding(finding_data))

    assert finding_id is not None
    finding = temp_db.get_finding(finding_id)
    assert finding is not None
    assert finding.status == "screened_out"
    assert (finding.evidence or {}).get("pre_atom_filter", {}).get("accepted") is False
    assert temp_db.get_raw_signals_by_finding(finding_id) == []
    assert temp_db.get_problem_atoms_by_finding(finding_id) == []


def test_process_finding_screens_out_broad_manual_tasks_prompt(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/smallbusiness/comments/manual-tasks",
        "product_built": "If you're still doing this manually in your business, you're wasting hours every week",
        "outcome_summary": (
            "If you or your team are still doing repetitive tasks manually "
            "(data entry, moving info between tools, basic follow-ups, reporting, etc), "
            "you're probably losing hours every week."
        ),
        "finding_kind": "problem_signal",
    }

    finding_id = asyncio.run(agent._process_finding(finding_data))

    assert finding_id is not None
    finding = temp_db.get_finding(finding_id)
    assert finding is not None
    assert finding.status == "screened_out"
    assert (finding.evidence or {}).get("pre_atom_filter", {}).get("reason") == "generic_prompt"
    assert temp_db.get_raw_signals_by_finding(finding_id) == []
    assert temp_db.get_problem_atoms_by_finding(finding_id) == []


def test_process_finding_screens_out_broad_finance_vendor_prompt(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/smallbusiness/comments/virtual-cards",
        "product_built": "Growing team looking for the best virtual credit card for expense automation",
        "outcome_summary": (
            "We're drowning in receipts, use personal cards for software and ad spend, and have looked at "
            "standard options like Ramp and Brex for a corporate card solution."
        ),
        "finding_kind": "problem_signal",
    }

    finding_id = asyncio.run(agent._process_finding(finding_data))

    assert finding_id is not None
    finding = temp_db.get_finding(finding_id)
    assert finding is not None
    assert finding.status == "screened_out"
    assert (finding.evidence or {}).get("pre_atom_filter", {}).get("reason") == "broad_finance_prompt"
    assert temp_db.get_raw_signals_by_finding(finding_id) == []
    assert temp_db.get_problem_atoms_by_finding(finding_id) == []


def test_duplicate_pre_atom_filtered_signal_by_hash_not_stored(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/entrepreneur/comments/meta-dup",
        "product_built": "What workflow should I automate next?",
        "outcome_summary": "Looking to build an automation product and asking founders which workflow annoys them most.",
        "finding_kind": "problem_signal",
    }

    finding_id_1 = asyncio.run(agent._process_finding(finding_data))
    finding_id_2 = asyncio.run(agent._process_finding(finding_data))

    assert finding_id_1 is not None
    assert finding_id_2 is None
    assert temp_db.get_finding(finding_id_1).status == "screened_out"


def test_screened_out_retention_trims_oldest_findings(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"screened_out_retention": {"max_findings": 1}}},
    )
    first = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/entrepreneur/comments/weak-1",
        "product_built": "What workflow should I automate first?",
        "outcome_summary": "Looking to build an automation product and asking which workflow annoys people the most.",
        "finding_kind": "problem_signal",
    }
    second = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/entrepreneur/comments/weak-2",
        "product_built": "What task should I automate next?",
        "outcome_summary": "Looking to build an automation product and asking which task people want automated the most.",
        "finding_kind": "problem_signal",
    }

    first_id = asyncio.run(agent._process_finding(first))
    second_id = asyncio.run(agent._process_finding(second))

    assert first_id is not None
    assert second_id is not None
    assert temp_db.get_finding(first_id) is None
    assert temp_db.get_finding(second_id) is not None


def test_duplicate_finding_by_hash_not_stored(temp_db):
    agent = DiscoveryAgent(temp_db)
    finding_data = {
        "source": "reddit-problem",
        "source_url": "https://reddit.com/r/entrepreneur/comments/xyz",
        "entrepreneur": "Jane Smith",
        "product_built": "Content Generator",
        "outcome_summary": "Manual posting is expensive and we keep falling back to spreadsheets every day.",
        "finding_kind": "pain_point",
    }

    finding_id_1 = asyncio.run(agent._process_finding(finding_data))
    finding_id_2 = asyncio.run(agent._process_finding(finding_data))

    assert finding_id_1 is not None
    assert finding_id_2 is None


def test_discover_once_does_not_block_on_reddit_priming(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["reddit"])

    async def slow_prime():
        await asyncio.sleep(0.3)

    async def fast_check(source):
        await asyncio.sleep(0.01)
        return []

    agent._prime_reddit_relay = slow_prime
    agent._check_source = fast_check

    start = time.perf_counter()
    result = asyncio.run(agent._discover_once())
    elapsed = time.perf_counter() - start

    assert result == []
    assert elapsed < 0.2


def test_discover_once_stays_stable_when_bridge_only_reddit_returns_zero(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}},
    )

    async def no_prime():
        return None

    async def no_results(source):
        assert source == "reddit"
        return []

    agent._prime_reddit_relay = no_prime
    agent._check_source = no_results

    result = asyncio.run(agent._discover_once())

    assert result == []
    runtime = agent.reddit_runtime_summary()
    assert runtime["reddit_mode"] == "bridge_only"


def test_reddit_runtime_summary_preserves_seed_coverage_when_prime_is_cancelled(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["reddit"])

    async def slow_prime():
        agent._last_reddit_seed_summary = {
            "seeded_total_pairs": 24,
            "seeded_pairs_existing_cache": 14,
            "seeded_pairs_uncovered": 10,
        }
        await asyncio.sleep(0.3)

    async def fast_check(source):
        return []

    agent._prime_reddit_relay = slow_prime
    agent._check_source = fast_check

    asyncio.run(agent._discover_once())

    runtime = agent.reddit_runtime_summary()
    assert runtime["seeded_total_pairs"] == 24
    assert runtime["seeded_pairs_uncovered"] == 10


def test_discover_once_reaps_reddit_priming_exception(temp_db, caplog):
    agent = DiscoveryAgent(temp_db, sources=["reddit"])

    async def failing_prime():
        raise RuntimeError("prime failed")

    async def fast_check(source):
        return []

    agent._prime_reddit_relay = failing_prime
    agent._check_source = fast_check

    with caplog.at_level("WARNING"):
        result = asyncio.run(agent._discover_once())

    assert result == []
    assert "reddit relay priming task failed: prime failed" in caplog.text


def test_shopify_reviews_source_calls_toolkit(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["shopify_reviews"], config={})

    async def fake_shopify(*, app_handles=None, observer=None):
        return [{"source": "shopify-review/backup-and-sync", "source_url": "https://apps.shopify.com/backup-and-sync/reviews/1"}]

    agent.toolkit._discover_shopify_review_threads = fake_shopify

    result = asyncio.run(agent._check_source("shopify_reviews"))

    assert len(result) == 1
    assert result[0]["source"] == "shopify-review/backup-and-sync"


def test_wordpress_reviews_source_calls_toolkit(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["wordpress_reviews"], config={})

    async def fake_wp(*, plugin_slugs=None, observer=None):
        return [{"source": "wordpress-review/woocommerce", "source_url": "https://wordpress.org/support/topic/x"}]

    agent.toolkit._discover_wordpress_review_threads = fake_wp

    result = asyncio.run(agent._check_source("wordpress_reviews"))

    assert len(result) == 1
    assert "woocommerce" in result[0]["source"]


def test_github_source_timeout_does_not_stall_discovery(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["github"],
        config={"discovery": {"github": {"timeout_seconds": 0.01}}},
    )

    async def slow_github(*, queries, observer=None):
        await asyncio.sleep(0.1)
        return [{"source": "github-issue/example/repo"}]

    agent.toolkit._discover_github_problem_threads = slow_github

    result = asyncio.run(agent._check_source("github"))

    assert result == []


def test_web_success_timeout_does_not_stall_discovery(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["web"],
        config={"discovery": {"web": {"success_timeout_seconds": 0.01}}},
    )

    async def slow_success(*, queries, observer=None):
        await asyncio.sleep(0.1)
        return [{"source": "web-success/example"}]

    async def fast_market(*, queries, observer=None):
        return [{"source": "market-problem/example"}]

    async def fast_problem(*, queries, observer=None):
        return [{"source": "web-problem/example"}]

    agent.toolkit._discover_success_stories_on_web = slow_success
    agent.toolkit._discover_marketplace_problem_threads = fast_market
    agent.toolkit._discover_web_problem_threads = fast_problem

    result = asyncio.run(agent._check_source("web"))

    assert [item["source"] for item in result] == ["market-problem/example", "web-problem/example"]


def test_web_problem_timeout_does_not_stall_discovery(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["web"],
        config={"discovery": {"web": {"problem_timeout_seconds": 0.01}}},
    )

    async def fast_success(*, queries, observer=None):
        return [{"source": "web-success/example"}]

    async def fast_market(*, queries, observer=None):
        return [{"source": "market-problem/example"}]

    async def slow_problem(*, queries, observer=None):
        await asyncio.sleep(0.1)
        return [{"source": "web-problem/example"}]

    agent.toolkit._discover_success_stories_on_web = fast_success
    agent.toolkit._discover_marketplace_problem_threads = fast_market
    agent.toolkit._discover_web_problem_threads = slow_problem

    result = asyncio.run(agent._check_source("web"))

    assert [item["source"] for item in result] == ["web-success/example", "market-problem/example"]


def test_plan_queries_uses_configured_slice_limit_and_records_rotation_metadata(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"reddit-problem": 2}}},
    )
    queries = [
        "manual reconciliation",
        "spreadsheet workaround",
        "copy paste approvals",
        "duplicate entry workflow",
    ]

    first = agent._plan_queries("reddit-problem", queries, default_limit=4)
    first_meta = agent._cycle_strategy["reddit-problem"]
    agent._cycle_strategy = {}
    second = agent._plan_queries("reddit-problem", queries, default_limit=4)
    second_meta = agent._cycle_strategy["reddit-problem"]

    assert len(first) == 2
    assert len(second) == 2
    assert first != second
    assert first_meta["discovery_slice_size"] == 2
    assert second_meta["discovery_rotation_applied"] is True
    assert second_meta["discovery_cycle_query_offset"] > 0
    assert second_meta["rotated_queries_used"]


def test_planned_sources_prioritize_high_yield_and_rotate_low_yield(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit", "web", "github", "wordpress_reviews", "shopify_reviews", "youtube", "youtube-comments"],
        config={
            "discovery": {
                "source_selection": {
                    "always_run": ["reddit", "web"],
                    "exploratory_low_yield_sources_per_cycle": 1,
                    "low_yield_min_runs": 10,
                }
            }
        },
    )
    agent.toolkit.set_discovery_feedback(
        [
            {
                "source_name": "reddit-problem",
                "query_text": "manual reconciliation",
                "runs": 20,
                "validations": 5,
                "prototype_candidates": 1,
                "build_briefs": 1,
            },
            {
                "source_name": "web-problem",
                "query_text": "manual reconciliation forum",
                "runs": 20,
                "validations": 4,
                "prototype_candidates": 1,
                "build_briefs": 1,
            },
            {
                "source_name": "github-problem",
                "query_text": '"csv import" issue workflow',
                "runs": 20,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "wordpress-reviews",
                "query_text": "wordpress_reviews",
                "runs": 20,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "shopify-reviews",
                "query_text": "shopify_reviews",
                "runs": 20,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "youtube-success",
                "query_text": "AI startup success story revenue",
                "runs": 20,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "youtube-comments",
                "query_text": "shopify app review",
                "runs": 20,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
        ]
    )

    planned = agent._planned_sources_for_cycle()

    assert "reddit" in planned
    assert "web" in planned
    low_yield_selected = [source for source in planned if source in {"github", "wordpress_reviews", "shopify_reviews", "youtube", "youtube-comments"}]
    assert len(low_yield_selected) == 1


def test_web_source_uses_high_yield_problem_queries(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["web"])
    captured: dict[str, list[str]] = {}

    def fake_plan(source_name, candidates, *, default_limit):
        if source_name == "web-problem":
            captured["candidates"] = list(candidates)
        return list(candidates)[:default_limit]

    async def fast_success(*, queries, observer=None):
        return []

    async def fast_market(*, queries, observer=None):
        return []

    async def fast_problem(*, queries, observer=None):
        captured["planned"] = list(queries)
        return []

    agent._plan_queries = fake_plan
    agent.toolkit._discover_success_stories_on_web = fast_success
    agent.toolkit._discover_marketplace_problem_threads = fast_market
    agent.toolkit._discover_web_problem_threads = fast_problem

    asyncio.run(agent._check_source("web"))

    assert "manual reconciliation forum" in captured["candidates"]
    assert "manual handoff workflow forum" in captured["candidates"]
    assert '"manual process" every day' not in captured["candidates"]


def test_learned_theme_queries_are_injected_and_recorded(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"reddit-problem": 3}, "theme_query_limit_per_cycle": 1}},
    )
    temp_db.insert_finding(
        Finding(
            source="reddit-problem",
            source_url="https://reddit.com/r/smallbusiness/comments/a",
            content_hash="theme-a",
            product_built="Duct tape and spreadsheets ops pain",
            outcome_summary="Teams keep asking which spreadsheet is latest and miss manual handoffs.",
            status="qualified",
        )
    )
    temp_db.insert_finding(
        Finding(
            source="reddit-problem",
            source_url="https://reddit.com/r/smallbusiness/comments/b",
            content_hash="theme-b",
            product_built="Spreadsheet handoff confusion",
            outcome_summary="Operations run on spreadsheets, copy paste, and duct tape workflows.",
            status="qualified",
        )
    )

    agent._load_learning_feedback()
    planned = agent._plan_queries(
        "reddit-problem",
        ["manual reconciliation", "spreadsheet workaround", "copy paste approvals"],
        default_limit=3,
    )

    assert len(planned) == 3
    assert any("manual handoff workflow" in query for query in planned)
    strategy = agent._cycle_strategy["reddit-problem"]
    assert strategy["learned_theme_keys"] == ["workflow_fragility"]
    assert strategy["learned_theme_queries"] == ["manual handoff workflow"]


def test_learned_themes_can_be_refreshed_from_validation_backed_examples(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"web-problem": 3}, "theme_query_limit_per_cycle": 1, "theme_min_hits": 2}},
    )
    finding_id = temp_db.insert_finding(
        Finding(
            source="reddit-problem",
            source_url="https://reddit.com/r/smallbusiness/comments/c",
            content_hash="validation-theme-source",
            product_built="Operations workflow keeps breaking",
            outcome_summary="The team is frustrated but the text itself is generic.",
            status="qualified",
        )
    )
    temp_db.insert_validation(
        Validation(
            finding_id=finding_id,
            market_score=6.1,
            technical_score=6.0,
            distribution_score=5.8,
            overall_score=6.0,
            passed=False,
            evidence={
                "decision": "park",
                "decision_reason": "plausible_but_unproven",
                "selection_status": "prototype_candidate",
                "selection_reason": "prototype_candidate_gate",
                "recurrence_state": "supported",
                "matched_docs_by_source": {
                    "web": [
                        {
                            "source_family": "web",
                            "source": "web",
                            "query_text": "latest spreadsheet version confusion",
                            "normalized_url": "https://ops.example.com/latest-spreadsheet",
                            "title": "Teams keep asking which spreadsheet is latest",
                            "snippet": "Manual handoffs and spreadsheet glue keep the workflow brittle.",
                            "match_class": "strong",
                        }
                    ]
                },
                "partial_docs_by_source": {"web": []},
            },
            run_id="test-run",
        )
    )

    agent._load_learning_feedback()
    planned = agent._plan_queries(
        "web-problem",
        ["manual reporting burden", "frustrating workflow", "need a better way automate"],
        default_limit=3,
    )

    assert any(
        "latest spreadsheet version confusion" in query
        or "duct tape spreadsheets" in query
        or "spreadsheet version confusion forum" in query
        or "manual handoff workflow forum" in query
        for query in planned
    )
    strategy = agent._cycle_strategy["web-problem"]
    assert strategy["learned_theme_keys"] == ["workflow_fragility"]
    assert strategy["learned_theme_queries"]
    themes = temp_db.list_active_discovery_themes(limit=10)
    workflow_theme = next(row for row in themes if row["theme_key"] == "workflow_fragility")
    assert "excel shared workbook conflict" in workflow_theme["query_seeds"] or "latest spreadsheet version confusion" in workflow_theme["query_seeds"]


def test_market_problem_queries_include_market_shaped_learned_theme_variants(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"market-problem": 3}, "theme_query_limit_per_cycle": 1}},
    )
    temp_db.upsert_discovery_theme(
        "workflow_fragility",
        label="Workflow fragility and spreadsheet glue",
        query_seeds=["duct tape spreadsheets", "manual handoff workflow"],
        source_signals=["Ops are held together by duct tape and spreadsheets"],
        times_seen=3,
        yield_score=1.0,
        run_id="test-run",
    )

    planned = agent._plan_queries(
        "market-problem",
        ['"etsy seller" "wish there was" automation', '"google reviews" "too expensive" tool'],
        default_limit=3,
    )

    assert any(
        query in planned
        for query in [
            "spreadsheet workflow software",
            "manual handoff workflow software",
            "spreadsheet version control for operations",
        ]
    )
    strategy = agent._cycle_strategy["market-problem"]
    assert strategy["learned_theme_keys"] == ["workflow_fragility"]
    assert strategy["learned_theme_queries"]


def test_web_problem_queries_include_web_shaped_learned_theme_variants(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"web-problem": 3}, "theme_query_limit_per_cycle": 1}},
    )
    temp_db.upsert_discovery_theme(
        "workflow_fragility",
        label="Workflow fragility and spreadsheet glue",
        query_seeds=["duct tape spreadsheets", "manual handoff workflow"],
        source_signals=["Ops are held together by duct tape and spreadsheets"],
        times_seen=3,
        yield_score=1.0,
        run_id="test-run",
    )

    planned = agent._plan_queries(
        "web-problem",
        ['"manual process" every day', '"frustrating" workflow'],
        default_limit=3,
    )

    assert any(
        query in planned
        for query in [
            "spreadsheet version confusion forum",
            "manual handoff workflow forum",
            "outgrown spreadsheets operations",
        ]
    )
    strategy = agent._cycle_strategy["web-problem"]
    assert strategy["learned_theme_keys"] == ["workflow_fragility"]
    assert strategy["learned_theme_queries"]


def test_cycle_two_can_expand_learned_theme_slice(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_limits": {"web-problem": 4}, "theme_query_limit_per_cycle": 1}},
    )
    temp_db.upsert_discovery_theme(
        "workflow_fragility",
        label="Workflow fragility and spreadsheet glue",
        query_seeds=[
            "duct tape spreadsheets",
            "manual handoff workflow",
            "latest spreadsheet version confusion",
            "excel shared workbook conflict",
        ],
        source_signals=["Ops are held together by duct tape and spreadsheets"],
        times_seen=4,
        yield_score=1.0,
        run_id="test-run",
    )

    first = agent._plan_queries("web-problem", ['"manual process" every day', '"frustrating" workflow'], default_limit=4)
    second = agent._plan_queries("web-problem", ['"manual process" every day', '"frustrating" workflow'], default_limit=4)

    assert len(first) >= 3
    assert len(second) >= 3
    strategy = agent._cycle_strategy["web-problem"]
    assert len(strategy["learned_theme_queries"]) >= 2



def test_adaptive_github_timeout_shortens_after_repeated_zero_yield_feedback(temp_db):
    agent = DiscoveryAgent(temp_db, config={"discovery": {"github": {"timeout_seconds": 20}}})
    agent.toolkit.set_discovery_feedback(
        [
            {
                "source_name": "github-problem",
                "query_text": "manual process tool",
                "runs": 2,
                "findings_emitted": 0,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "github-problem",
                "query_text": "too expensive software",
                "runs": 2,
                "findings_emitted": 0,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
            {
                "source_name": "github-problem",
                "query_text": "feature request automation",
                "runs": 2,
                "findings_emitted": 0,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            },
        ]
    )

    assert agent._adaptive_github_timeout_seconds() == 6.0


def test_github_discovery_can_be_skipped_after_repeated_zero_yield_feedback(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        sources=["github"],
        config={"discovery": {"github": {"hard_skip_after_zero_yield": True}}},
    )
    agent.toolkit.set_discovery_feedback(
        [
            {
                "source_name": "github-problem",
                "query_text": f"query-{index}",
                "runs": 2,
                "findings_emitted": 0,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            }
            for index in range(4)
        ]
    )

    called = {"github": 0}

    async def fake_github(*, queries, observer=None):
        called["github"] += 1
        return [{"source": "github-issue/example/repo"}]

    agent.toolkit._discover_github_problem_threads = fake_github
    result = asyncio.run(agent._check_source("github"))

    assert result == []
    assert called["github"] == 0


def test_github_discovery_remains_enabled_by_default_after_zero_yield_feedback(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["github"])
    agent.toolkit.set_discovery_feedback(
        [
            {
                "source_name": "github-problem",
                "query_text": f"query-{index}",
                "runs": 2,
                "findings_emitted": 0,
                "validations": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
            }
            for index in range(4)
        ]
    )

    called = {"github": 0}

    async def fake_github(*, queries, observer=None):
        called["github"] += 1
        return [{"source": "github-issue/example/repo"}]

    agent.toolkit._discover_github_problem_threads = fake_github
    result = asyncio.run(agent._check_source("github"))

    assert result == [{"source": "github-issue/example/repo"}]
    assert called["github"] == 1


def test_prime_reddit_relay_includes_learned_theme_queries(temp_db, monkeypatch):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "reddit_bridge": {"enabled": True, "mode": "bridge_only", "base_url": "https://bridge.example"},
            "reddit_relay": {"auto_seed_on_discovery": True},
            "discovery": {"theme_query_limit_per_cycle": 2},
        },
    )
    temp_db.upsert_discovery_theme(
        "workflow_fragility",
        label="Workflow fragility and spreadsheet glue",
        query_seeds=["duct tape spreadsheets", "manual handoff workflow", "which spreadsheet is latest"],
        source_signals=["Ops are held together by duct tape and spreadsheets"],
        times_seen=3,
        yield_score=1.0,
        run_id="test-run",
    )

    captured: dict[str, list[str]] = {}

    class FakeSummary:
        total_pairs = 6
        searched_pairs = 0
        skipped_fresh_pairs = 0
        existing_cached_pairs = 0
        uncovered_pairs = 0
        cached_searches = 0
        cached_threads = 0
        thread_cache_hits = 0
        unique_urls = 0

    class FakeSeeder:
        def __init__(self, _config, **_kwargs):
            pass

        async def seed(self, *, subreddits, queries):
            captured["queries"] = list(queries)
            captured["subreddits"] = list(subreddits)
            captured["seed_queries"] = list(queries)
            return FakeSummary()

    monkeypatch.setattr("src.agents.discovery.RedditSeeder", FakeSeeder)

    asyncio.run(agent._prime_reddit_relay())

    assert "duct tape spreadsheets" in captured["seed_queries"]
    assert "manual handoff workflow" in captured["seed_queries"]


def test_load_learning_feedback_sets_query_cooldown_for_repeated_low_yield(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={"discovery": {"query_cooldown_hours": 6, "query_quarantine_min_runs": 3}},
    )
    for _ in range(3):
        temp_db.record_discovery_probe("reddit-problem", "manual reporting", docs_seen=0, latency_ms=10.0, status="ok")
        temp_db.record_discovery_screening(
            "reddit-problem",
            "manual reporting",
            accepted=False,
            source_class="low_signal_summary",
            screening_score=0.1,
        )

    agent._load_learning_feedback()

    rows = temp_db.get_discovery_feedback("reddit-problem")
    cooldown_until = rows[0]["cooldown_until"]
    assert cooldown_until
    assert datetime.fromisoformat(cooldown_until) > datetime.now(UTC)


def test_prime_reddit_relay_caps_seed_query_count(temp_db, monkeypatch):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "discovery": {"reddit_seed_query_limit": 3},
            "reddit_bridge": {"enabled": True, "base_url": "https://bridge.example"},
        },
    )

    monkeypatch.setattr(
        agent,
        "_plan_queries",
        lambda source_name, candidates, default_limit: [
            "latest spreadsheet version confusion",
            "manual reconciliation workflow",
            "manual audit evidence collection",
            "copy paste workflow handoff",
        ],
    )
    monkeypatch.setattr(
        agent,
        "_learned_theme_queries",
        lambda source_name: (["which spreadsheet is latest", "csv import cleanup workflow"], ["workflow_fragility"]),
    )

    captured = {}

    class FakeSummary:
        total_pairs = 3
        searched_pairs = 3
        skipped_fresh_pairs = 0
        existing_cached_pairs = 0
        uncovered_pairs = 0
        cached_searches = 3
        cached_threads = 0
        thread_cache_hits = 0
        unique_urls = 0

    class FakeSeeder:
        def __init__(self, config, bypass_cache=False):
            captured["bypass_cache"] = bypass_cache

        async def seed(self, *, subreddits, queries):
            captured["queries"] = list(queries)
            captured["subreddits"] = list(subreddits)
            captured["seed_queries"] = list(queries)
            return FakeSummary()

    monkeypatch.setattr("src.agents.discovery.RedditSeeder", FakeSeeder)

    asyncio.run(agent._prime_reddit_relay())

    assert "accounting" in captured["subreddits"]
    assert "smallbusiness" in captured["subreddits"]
    assert len(captured["seed_queries"]) == 3


def test_prime_reddit_relay_caps_seed_subreddit_count(temp_db, monkeypatch):
    agent = DiscoveryAgent(
        temp_db,
        sources=["reddit"],
        config={
            "discovery": {
                "reddit": {
                    "max_subreddits_per_wave": 2,
                    "problem_subreddits": ["accounting", "smallbusiness", "ecommerce", "shopify"],
                },
                "reddit_seed_query_limit": 2,
            },
            "reddit_bridge": {"enabled": True, "base_url": "https://bridge.example"},
        },
    )

    monkeypatch.setattr(
        agent,
        "_plan_queries",
        lambda source_name, candidates, default_limit: ["manual reconciliation workflow", "csv import cleanup workflow"],
    )
    monkeypatch.setattr(agent, "_learned_theme_queries", lambda source_name: ([], []))

    captured = {}

    class FakeSummary:
        total_pairs = 4
        searched_pairs = 4
        skipped_fresh_pairs = 0
        existing_cached_pairs = 0
        uncovered_pairs = 0
        cached_searches = 4
        cached_threads = 0
        thread_cache_hits = 0
        unique_urls = 0

    class FakeSeeder:
        def __init__(self, config, bypass_cache=False):
            pass

        async def seed(self, *, subreddits, queries):
            captured["subreddits"] = list(subreddits)
            captured["queries"] = list(queries)
            captured["seed_subreddits"] = list(subreddits)
            captured["seed_queries"] = list(queries)
            return FakeSummary()

    monkeypatch.setattr("src.agents.discovery.RedditSeeder", FakeSeeder)

    asyncio.run(agent._prime_reddit_relay())

    assert len(captured["seed_subreddits"]) == 2
    assert captured["seed_subreddits"] == ["accounting", "smallbusiness"]


def test_load_learning_feedback_decays_overused_weak_query_family(temp_db):
    agent = DiscoveryAgent(
        temp_db,
        config={
            "discovery": {
                "query_cooldown_hours": 4,
                "query_quarantine_min_runs": 3,
                "query_family_decay_hours": 10,
                "query_family_decay_min_queries": 2,
            }
        },
    )

    for query in ["manual reconciliation workflow", "spreadsheet reconciliation process"]:
        for _ in range(3):
            temp_db.record_discovery_probe("reddit-problem", query, docs_seen=3, latency_ms=10.0, status="ok")
            temp_db.record_discovery_screening(
                "reddit-problem",
                query,
                accepted=True,
                source_class="pain_signal",
                screening_score=0.65,
            )
            temp_db.record_validation_feedback(
                "reddit-problem",
                query,
                overall_score=0.28,
                passed=False,
                selection_status="research_more",
                decision="park",
                recurrence_state="thin",
            )

    agent._load_learning_feedback()

    rows = {row["query_text"]: row for row in temp_db.get_discovery_feedback("reddit-problem")}
    assert rows["manual reconciliation workflow"]["cooldown_until"]
    assert rows["spreadsheet reconciliation process"]["cooldown_until"]
