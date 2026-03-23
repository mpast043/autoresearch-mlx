"""Tests for the discovery agent."""

import asyncio
import os
import sys
import tempfile
import time

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agents.base import AgentStatus
from agents.discovery import DiscoveryAgent
from database import Database, Finding, Validation
from messaging import MessageType


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


def test_custom_sources_initialization(temp_db):
    agent = DiscoveryAgent(temp_db, sources=["reddit", "github"])
    assert agent.sources == ["reddit", "github"]


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
    assert temp_db.get_raw_signals_by_finding(finding_id)
    assert temp_db.get_problem_atoms_by_finding(finding_id)

    queued = asyncio.run(agent._message_queue.get_for_agent("orchestrator"))
    assert queued is not None
    assert queued.msg_type == MessageType.FINDING
    assert queued.payload["finding_id"] == finding_id


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
    assert any("duct tape spreadsheets" in query for query in planned)
    strategy = agent._cycle_strategy["reddit-problem"]
    assert strategy["learned_theme_keys"] == ["workflow_fragility"]
    assert strategy["learned_theme_queries"] == ["duct tape spreadsheets"]


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

    assert result == []
    assert called["github"] == 0


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

    class FakeCoverage:
        total_pairs = 6
        skipped_fresh_pairs = 0
        existing_cached_pairs = 0
        uncovered_pairs = 0

    class FakeSummary:
        total_pairs = 6
        searched_pairs = 0
        skipped_fresh_pairs = 0
        existing_cached_pairs = 0
        cached_searches = 0
        cached_threads = 0
        thread_cache_hits = 0
        unique_urls = 0

    class FakeSeeder:
        def __init__(self, _config):
            pass

        def coverage_report(self, *, subreddits, queries):
            captured["queries"] = list(queries)
            captured["subreddits"] = list(subreddits)
            return FakeCoverage()

        async def seed(self, *, subreddits, queries):
            captured["seed_queries"] = list(queries)
            return FakeSummary()

    monkeypatch.setattr("agents.discovery.RedditSeeder", FakeSeeder)

    asyncio.run(agent._prime_reddit_relay())

    assert "duct tape spreadsheets" in captured["queries"]
    assert "manual handoff workflow" in captured["queries"]
