"""Tests for research toolkit search and validation behavior."""

import asyncio
import os
import sys
import time
import types
from types import SimpleNamespace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reddit_bridge import BridgeError
from research_tools import ResearchToolkit, SearchDocument


class FakeDDGS:
    """Minimal DDGS stub used to validate search filtering."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, backend=None, max_results=None):
        assert backend == "api"
        return [
            {
                "title": "Wikipedia result",
                "href": "https://en.wikipedia.org/wiki/Workflow_automation",
                "body": "encyclopedia",
            },
            {
                "title": "Grokipedia result",
                "href": "https://grokipedia.com/page/workflow-automation",
                "body": "mirror",
            },
            {
                "title": "Useful 1",
                "href": "https://example.com/post-a",
                "body": "manual workflow pain",
            },
            {
                "title": "Useful 2",
                "href": "https://example.com/post-b",
                "body": "repetitive process pain",
            },
            {
                "title": "Useful 3 should be capped",
                "href": "https://example.com/post-c",
                "body": "same domain overflow",
            },
            {
                "title": "Wrapped result",
                "href": "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fanother.com%2Fitem",
                "body": "target result",
            },
        ]


class FakeValidationDDGS:
    """Search stub with validation junk and one real software result."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, backend=None, max_results=None):
        assert backend == "api"
        return [
            {
                "title": "Workflow Definition",
                "href": "https://www.merriam-webster.com/dictionary/workflow",
                "body": "dictionary definition",
            },
            {
                "title": "Business Hours",
                "href": "https://www.hours.com/workflow-automation",
                "body": "open to close hours",
            },
            {
                "title": "Workflow automation software",
                "href": "https://exampleapp.com/workflow-automation",
                "body": "workflow automation software for repetitive approvals",
            },
        ]


class FakeEmptyDDGS:
    """Search stub that yields no provider results."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, backend=None, max_results=None):
        assert backend == "api"
        return []


def test_search_web_filters_noise_and_caps_domains(monkeypatch):
    """Search results should drop wiki noise, unwrap redirects, and cap same-domain spam."""
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    toolkit = ResearchToolkit({"validation": {"search": {"max_results_per_domain": 2}}})

    docs = asyncio.run(toolkit.search_web("workflow automation", max_results=4))

    urls = [doc.url for doc in docs]
    assert "https://en.wikipedia.org/wiki/Workflow_automation" not in urls
    assert "https://grokipedia.com/page/workflow-automation" not in urls
    assert urls == [
        "https://example.com/post-a",
        "https://example.com/post-b",
        "https://another.com/item",
    ]


def test_validate_problem_builds_compact_query_and_caps_evidence():
    """Validation should use concise queries and keep bounded evidence samples."""
    toolkit = ResearchToolkit(
        {
            "validation": {
                "search": {
                    "query_terms": 10,
                    "recurrence_results": 6,
                    "competitor_results": 8,
                    "evidence_sample": 5,
                }
            }
        }
    )
    calls = []

    async def fake_search(query, max_results=8, site=None, intent="general"):
        calls.append((query, max_results, site, intent))
        docs = []
        for idx in range(max_results):
            docs.append(
                SearchDocument(
                    title=f"Doc {idx}",
                    url=f"https://domain{idx}.com/result-{idx}",
                    snippet="manual workflow every day cost issue",
                    source=site or "web",
                )
            )
        return docs

    async def fake_gather(queries, finding_kind, atom=None):
        docs = await fake_search(queries[0], max_results=6, intent="validation_recurrence")
        return docs, {
            "recurrence_score": 0.7,
            "recurrence_state": "supported",
            "query_coverage": 1.0,
            "doc_count": len(docs),
            "domain_count": len(docs),
            "results_by_query": {queries[0]: len(docs)},
            "results_by_source": {"web": len(docs)},
        }

    toolkit.search_web = fake_search
    toolkit.gather_recurrence_evidence = fake_gather

    result = asyncio.run(
        toolkit.validate_problem(
            title="[Feature Request] User-based Scoping for Workflows and Automation #9321",
            summary=(
                "Need a better way to scope workflow automation because teams use it every day "
                "and the manual process takes too long. Wish there was software that saved hours."
            ),
            finding_kind="pain_point",
        )
    )

    recurrence_query, recurrence_limit, _, recurrence_intent = calls[0]
    competitor_query, competitor_limit, _, competitor_intent = calls[1]

    assert recurrence_limit == 6
    assert competitor_limit == 8
    assert recurrence_intent == "validation_recurrence"
    assert competitor_intent == "validation_competitor"
    assert len(recurrence_query.split()) <= 10
    assert "feature" not in recurrence_query
    assert "request" not in recurrence_query
    assert competitor_query.endswith("software tool alternative")
    assert len(result["evidence"]["recurrence_docs"]) == 5
    assert len(result["evidence"]["competitor_docs"]) == 5
    assert result["evidence"]["query"]


def test_validate_problem_degrades_gracefully_when_inner_budgets_timeout():
    toolkit = ResearchToolkit(
        {
            "validation": {
                "search": {
                    "recurrence_budget_seconds": 0.01,
                    "competitor_budget_seconds": 0.01,
                }
            }
        }
    )

    async def slow_gather(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [], {}

    async def slow_search(*args, **kwargs):
        await asyncio.sleep(0.05)
        return []

    toolkit.gather_recurrence_evidence = slow_gather
    toolkit.search_web = slow_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Manual audit exports break every week",
            summary="We still merge exports manually and keep running custom scripts during audits.",
            finding_kind="pain_point",
        )
    )

    assert result["evidence"]["recurrence_state"] == "timeout"
    assert result["evidence"]["recurrence_timeout"] is True
    assert result["evidence"]["competitor_timeout"] is True
    assert result["evidence"]["recurrence_failure_class"] == "budget_exhausted"
    assert result["evidence"]["recurrence_docs"] == []
    assert result["evidence"]["competitor_docs"] == []


def test_generic_manual_prompt_uses_narrower_recurrence_budget():
    toolkit = ResearchToolkit()
    atom = type(
        "Atom",
        (),
        {
            "job_to_be_done": "keep a recurring workflow on track",
            "failure_mode": "manual work keeps piling up",
            "trigger_event": "",
            "current_workaround": "",
            "cost_consequence_clues": "",
            "segment": "small business operators",
            "user_role": "",
        },
    )()

    queries = toolkit.build_recurrence_queries(
        title="What manual process would you like to automate?",
        summary="People want to automate manual work but without a concrete failure mode.",
        atom=atom,
    )
    profile = toolkit._recurrence_budget_profile(atom)
    subreddits = toolkit._recurrence_subreddits(atom, limit=profile["subreddit_limit"])
    sites = toolkit._recurrence_site_plan(atom, subreddit_plan=subreddits, limit=profile["site_limit"])

    assert len(queries) == 2
    assert len(subreddits) == 2
    assert sites == [(None, "web")]


def test_nontechnical_spreadsheet_atom_deprioritizes_github_site_plan():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations manager",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    sites = toolkit._recurrence_site_plan(atom, subreddit_plan=["smallbusiness"], limit=4)

    assert ("github.com", "github") not in sites
    assert (None, "web") in sites


def test_technical_atom_keeps_github_in_site_plan():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="platform engineering",
        user_role="developer",
        job_to_be_done="keep deployment automation reliable",
        failure_mode="api integration rollback scripts fail during deploys",
        trigger_event="during releases",
        current_workaround="manual rollback script",
        cost_consequence_clues="incident risk",
        current_tools="github actions api webhook deployment config",
    )

    sites = toolkit._recurrence_site_plan(atom, subreddit_plan=["devops"], limit=4)

    assert ("github.com", "github") in sites


def test_choose_corroboration_action_deprioritizes_low_fit_github():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations manager",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=['"manual data entry" "spreadsheet cleanup"'],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={},
        matched_results_by_source={},
        partial_results_by_source={},
        family_confirmation_count=0,
        source_attempts_by_family={},
        budget_profile={"target_sources": 2, "specificity_score": 0.62},
        available_families=["github", "web"],
    )

    assert action.action == "GATHER_CORROBORATION"
    assert action.target_family == "web"
    assert action.skipped_families["github"] == "low_public_issue_fit"


def test_choose_corroboration_action_retries_zero_retrieval_with_strategy_switch():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations manager",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=['"manual data entry" "spreadsheet cleanup"'],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={"web": {"docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0}},
        matched_results_by_source={"web": 0},
        partial_results_by_source={"web": 0},
        family_confirmation_count=0,
        source_attempts_by_family={"web": 1},
        budget_profile={"target_sources": 2, "specificity_score": 0.62},
        available_families=["web"],
        current_family="web",
    )

    assert action.action == "RETRY_WITH_RESHAPED_QUERY"
    assert action.target_family == "web"
    assert action.reason == "zero_retrieval_strategy_switch"
    assert action.fallback_strategy == "decomposed_query_switch"


def test_choose_corroboration_action_stops_low_yield_web_retry_for_promising_near_miss():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="sales operations",
        user_role="crm administrator",
        job_to_be_done="keep sla follow-up workflows on track",
        failure_mode="manual follow-up reminders break after assignment",
        trigger_event="after lead assignment",
        current_workaround="manual reminders and status checks",
        cost_consequence_clues="missed sla and delayed follow-up",
        current_tools="d365 sales crm workflow automation",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["sla tracking follow-up workflow"],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={"web": {"docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0}},
        matched_results_by_source={"reddit": 1, "web": 0},
        partial_results_by_source={"reddit": 1, "web": 0},
        family_confirmation_count=1,
        source_attempts_by_family={"web": 1, "reddit": 1},
        budget_profile={"target_sources": 2, "specificity_score": 0.78},
        available_families=["web"],
        current_family="web",
        promotion_gap_class="corroboration_gap",
    )

    assert action.action == "STOP_FOR_BUDGET"
    assert action.reason == "zero_retrieval_confirmation_family_low_yield"


def test_choose_corroboration_action_stops_partial_only_web_retry_for_promising_near_miss():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="revenue operations",
        user_role="ops lead",
        job_to_be_done="keep brittle follow-up workflows reliable",
        failure_mode="spreadsheet handoffs still miss steps",
        trigger_event="after lead routing",
        current_workaround="manual spreadsheet checks",
        cost_consequence_clues="follow-ups slip",
        current_tools="spreadsheets",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=['"spreadsheet handoff"'],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={"web": {"docs_retrieved": 3, "docs_strong_match": 0, "docs_partial_match": 2}},
        matched_results_by_source={"reddit": 1, "web": 0},
        partial_results_by_source={"reddit": 1, "web": 2},
        family_confirmation_count=1,
        source_attempts_by_family={"web": 1, "reddit": 1},
        budget_profile={"target_sources": 2, "specificity_score": 0.78},
        available_families=["web"],
        current_family="web",
        promotion_gap_class="corroboration_gap",
    )

    assert action.action == "STOP_FOR_BUDGET"
    assert action.reason == "partial_only_confirmation_family_low_yield"


def test_choose_corroboration_action_stops_for_budget_when_attempts_exhausted():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="platform engineering",
        user_role="developer",
        job_to_be_done="keep deployment automation reliable",
        failure_mode="api integration rollback scripts fail during deploys",
        trigger_event="during releases",
        current_workaround="manual rollback script",
        cost_consequence_clues="incident risk",
        current_tools="github actions api webhook deployment config",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=['"rollback script" "deployment automation"'],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={"github": {"docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0}},
        matched_results_by_source={"github": 0},
        partial_results_by_source={"github": 0},
        family_confirmation_count=0,
        source_attempts_by_family={"github": plan.max_attempts_per_family},
        budget_profile={"target_sources": 2, "specificity_score": 0.8},
        available_families=["github"],
        current_family="github",
    )

    assert action.action == "STOP_FOR_BUDGET"


def test_classify_promotion_gap_distinguishes_corroboration_and_value_cases():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="sales ops",
        user_role="revenue operations manager",
        job_to_be_done="keep follow-up workflows on time",
        failure_mode="sla tracking and handoffs fail in crm",
        trigger_event="after lead assignment",
        current_workaround="manual follow-up reminders",
        cost_consequence_clues="missed deals",
        current_tools="crm spreadsheet",
    )

    corroboration_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="thin",
        recurrence_score=0.42,
        family_confirmation_count=2,
        strong_match_count=1,
        partial_match_count=1,
        query_coverage=0.6,
        value_signal=0.62,
    )
    value_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="supported",
        recurrence_score=0.58,
        family_confirmation_count=2,
        strong_match_count=2,
        partial_match_count=1,
        query_coverage=0.8,
        value_signal=0.34,
    )
    supported_multi_family_borderline_value_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="strong",
        recurrence_score=0.72,
        family_confirmation_count=2,
        strong_match_count=3,
        partial_match_count=1,
        query_coverage=0.8,
        value_signal=0.52,
    )
    supported_single_family_value_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="supported",
        recurrence_score=0.58,
        family_confirmation_count=1,
        strong_match_count=3,
        partial_match_count=0,
        query_coverage=0.75,
        value_signal=0.34,
    )
    supported_single_family_corroboration_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="supported",
        recurrence_score=0.58,
        family_confirmation_count=1,
        strong_match_count=2,
        partial_match_count=1,
        query_coverage=0.75,
        value_signal=0.52,
    )
    supported_single_family_borderline_value_stays_corroboration_gap = toolkit._classify_promotion_gap(
        atom=atom,
        recurrence_state="supported",
        recurrence_score=0.58,
        family_confirmation_count=1,
        strong_match_count=2,
        partial_match_count=1,
        query_coverage=0.75,
        value_signal=0.41,
    )

    assert corroboration_gap == "evidence_sufficiency_gap"
    assert value_gap == "value_gap"
    assert supported_multi_family_borderline_value_gap == "value_gap"
    assert supported_single_family_value_gap == "value_gap"
    assert supported_single_family_corroboration_gap == "corroboration_gap"
    assert supported_single_family_borderline_value_stays_corroboration_gap == "corroboration_gap"


def test_choose_corroboration_action_prioritizes_confirmation_completion_for_near_miss():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="sales operations",
        user_role="crm administrator",
        job_to_be_done="keep sla follow-up workflows on track",
        failure_mode="manual follow-up reminders break after assignment",
        trigger_event="after lead assignment",
        current_workaround="manual reminders and status checks",
        cost_consequence_clues="missed sla and delayed follow-up",
        current_tools="d365 sales crm workflow automation",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["sla tracking follow-up workflow"],
        finding_kind="problem_signal",
    )

    action = toolkit._choose_corroboration_action(
        atom=atom,
        corroboration_plan=plan,
        source_yield={
            "reddit": {"docs_retrieved": 3, "docs_strong_match": 1, "docs_partial_match": 1, "confirmed": True},
            "github": {"docs_retrieved": 1, "docs_strong_match": 0, "docs_partial_match": 1, "confirmed": False},
        },
        matched_results_by_source={"reddit": 1, "github": 0},
        partial_results_by_source={"reddit": 1, "github": 1},
        family_confirmation_count=1,
        source_attempts_by_family={"reddit": 1, "github": 1},
        budget_profile={"target_sources": 2, "specificity_score": 0.78},
        available_families=["github", "web"],
        promotion_gap_class="corroboration_gap",
    )

    assert action.action == "GATHER_CORROBORATION"
    assert action.target_family in {"github", "web"}
    assert action.sufficiency_priority_reason
    assert action.promotion_gap_class == "corroboration_gap"


def test_web_zero_retrieval_fallback_queries_are_materially_different_and_compact():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )
    prior_queries = ["manual data entry spreadsheet cleanup"]
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=prior_queries,
        finding_kind="problem_signal",
    )

    queries = toolkit._web_zero_retrieval_fallback_queries(
        atom=atom,
        plan=plan,
        prior_queries=prior_queries,
    )

    assert queries
    assert queries[0] != prior_queries[0]
    assert all('"' not in query for query in queries)
    assert any("manual csv cleanup" in query for query in queries)
    assert any("downtime risk" in query or "time risk" in query for query in queries)
    assert all(len(query.split()) <= 10 for query in queries)


def test_spreadsheet_operator_admin_cohort_pack_generates_operator_language_queries():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="office manager",
        job_to_be_done="keep status updates and reporting in sync",
        failure_mode="copy paste updates and duplicate spreadsheet entry slow the team down",
        trigger_event="during vendor reporting",
        current_workaround="using spreadsheets for status updates",
        cost_consequence_clues="time loss",
        current_tools="excel google sheets reporting",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["spreadsheet workflow pain"],
        finding_kind="problem_signal",
    )

    queries = toolkit._spreadsheet_operator_admin_web_queries(atom=atom, plan=plan)

    assert queries
    assert any("spreadsheet" in query or "excel" in query for query in queries)
    assert any("manual re-entry" in query or "copy paste" in query for query in queries)
    assert any("tracking" in query or "reporting" in query or "status updates" in query for query in queries)
    assert all('"' not in query for query in queries)
    assert all(len(query.split()) <= 10 for query in queries)


def test_workflow_fragility_web_queries_capture_brittle_handoff_language():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep operations data sync across teams",
        failure_mode="spreadsheet handoffs go out of sync and missed steps break follow-up",
        trigger_event="during status updates",
        current_workaround="copy paste updates between spreadsheets and email",
        cost_consequence_clues="time loss and missed follow-up",
        current_tools="excel spreadsheets email",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["duct tape spreadsheets workflow"],
        finding_kind="problem_signal",
    )

    queries = toolkit._workflow_fragility_web_queries(atom=atom, plan=plan)

    assert queries
    assert any("excel shared workbook conflict" in query for query in queries)
    assert any("google sheets collaborator changes not showing" in query for query in queries)
    assert any(
        "excel shared workbook conflict" in query
        or "shared spreadsheet saving conflicts" in query
        or "latest spreadsheet version confusion" in query
        for query in queries
    )
    assert any("handoff" in query or "copy paste" in query for query in queries)
    assert any("which spreadsheet is latest" in query or "latest spreadsheet version confusion" in query for query in queries)
    assert any("manual handoff workflow" in query or "copy paste workflow" in query for query in queries)
    assert any(
        "multiple people editing same spreadsheet latest version" in query
        or "workflow software" in query
        or "too expensive" in query
        for query in queries
    )
    assert all(len(query.split()) <= 10 for query in queries)
    assert queries[0].startswith("excel shared workbook conflict") or queries[0].startswith("shared spreadsheet saving conflicts")
    assert any(
        query.startswith("excel shared workbook conflict")
        or query.startswith("shared spreadsheet saving conflicts")
        or query.startswith("google sheets collaborator changes not showing")
        for query in queries[:3]
    )


def test_generic_operational_pain_does_not_trigger_workflow_fragility_pack():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep approvals and reporting moving",
        failure_mode="manual reporting takes too long during weekly updates",
        trigger_event="during status updates",
        current_workaround="checklists and reminders",
        cost_consequence_clues="time loss",
        current_tools="email docs",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["manual reporting workflow"],
        finding_kind="problem_signal",
    )

    assert toolkit._is_workflow_fragility_cohort(atom=atom, plan=plan) is False
    assert toolkit._workflow_fragility_web_queries(atom=atom, plan=plan) == []


def test_web_recurrence_queries_include_workflow_fragility_pack_for_seed_shape():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep operations data sync across teams",
        failure_mode="spreadsheet handoffs go out of sync and missed steps break follow-up",
        trigger_event="during status updates",
        current_workaround="copy paste updates between spreadsheets and email",
        cost_consequence_clues="time loss and missed follow-up",
        current_tools="excel spreadsheets email",
    )

    queries = toolkit._web_recurrence_queries_from_atom(
        atom=atom,
        signature_terms=["spreadsheet", "handoff", "manual", "copy paste"],
        role_terms=["operations lead"],
        segment_terms=["small business"],
        job_phrase="keep operations data sync across teams",
        failure_phrase="spreadsheet handoffs go out of sync",
        workaround_phrase="copy paste updates",
        cost_terms=["time loss", "missed follow-up"],
        ecosystem_hints=["excel", "spreadsheets"],
        reshape_reason="failure_missing",
    )

    assert any(
        "spreadsheet latest" in query
        or "latest spreadsheet version confusion" in query
        or "excel shared workbook conflict" in query
        or "shared spreadsheet saving conflicts" in query
        for query in queries
    )
    assert any("excel shared workbook conflict" in query or "shared spreadsheet saving conflicts" in query for query in queries)
    assert any("handoff" in query or "copy paste" in query for query in queries)
    assert any("workflow software" in query or "forum" in query or "too expensive" in query for query in queries)
    assert any(
        query.startswith("spreadsheet latest")
        or query.startswith("latest spreadsheet version confusion")
        or query.startswith("excel shared workbook conflict")
        or query.startswith("shared spreadsheet saving conflicts")
        for query in queries[:3]
    )


def test_workflow_fragility_atom_does_not_default_to_github_recurrence():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep order updates and team handoffs in sync",
        failure_mode="nobody knows which spreadsheet is latest and steps get missed",
        trigger_event="handoff between teams",
        current_workaround="copy paste between spreadsheets and status updates in Slack",
        cost_consequence_clues="hours lost each week fixing brittle workflow",
        current_tools="Google Sheets Slack Zapier",
    )

    assert toolkit._atom_supports_github_recurrence(atom=atom) is False


def test_workflow_fragility_web_routing_prefers_external_reviewable_surfaces():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep order updates and team handoffs in sync",
        failure_mode="nobody knows which spreadsheet is latest and steps get missed",
        trigger_event="handoff between teams",
        current_workaround="copy paste between spreadsheets and status updates in Slack",
        cost_consequence_clues="hours lost each week fixing brittle workflow",
        current_tools="Google Sheets Slack Zapier",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["duct tape spreadsheets workflow"],
        finding_kind="problem_signal",
    )

    sites, reason = toolkit._specialized_web_routing_sites(atom=atom, plan=plan, attempt_index=0)

    assert reason == "workflow_fragility_surface_first"
    assert ("superuser.com", "web") in sites
    assert ("webapps.stackexchange.com", "web") in sites
    assert ("community.atlassian.com", "web") in sites
    assert ("capterra.com", "web") in sites
    assert (None, "web") in sites


def test_workflow_fragility_shared_workbook_conflict_counts_as_strong_web_match():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep operations data sync across teams",
        failure_mode="nobody knows which spreadsheet is latest and shared edits get out of sync",
        trigger_event="handoff between teams",
        current_workaround="manual cleanup after copy paste conflicts",
        cost_consequence_clues="time loss fixing brittle workflow",
        current_tools="excel spreadsheets email",
    )
    doc = SearchDocument(
        title="Microsoft Excel 2010 Shared workbook - edit conflicts",
        url="https://superuser.com/questions/1107814/microsoft-excel-2010-shared-workbook-edit-conflicts",
        snippet="When multiple people update the same workbook, edit conflicts create manual cleanup and workflow confusion.",
        source="superuser.com",
        source_family="web",
        retrieval_query="excel shared workbook conflict",
    )

    match = toolkit._classify_recurrence_match(doc, atom, ["spreadsheet", "manual", "handoff", "latest version"])

    assert match == "strong"


def test_low_information_atom_decomposition_is_bounded_and_compact():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="",
        job_to_be_done="process lead tracking invoicing data",
        failure_mode="manual work keeps piling up",
        trigger_event="",
        current_workaround="",
        cost_consequence_clues="",
        current_tools="",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["manual process"],
        finding_kind="problem_signal",
    )

    queries = toolkit._decompose_low_information_atom(atom, plan)

    assert 1 <= len(queries) <= 3
    assert any("small business process lead tracking invoicing" in query for query in queries)
    assert any("manual work keeps piling" in query for query in queries)
    assert all(len(query.split()) <= 8 for query in queries)


def test_ecosystem_atom_prefers_specialized_web_sites_before_generic_web():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="shopify merchants",
        user_role="store operator",
        job_to_be_done="keep order status in sync without manual copy paste",
        failure_mode="shopify app updates break workflow handoffs",
        trigger_event="after app updates",
        current_workaround="manual order status updates",
        cost_consequence_clues="time loss",
        current_tools="shopify storefront app plugin",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["shopify app workflow"],
        finding_kind="problem_signal",
    )

    sites, reason = toolkit._specialized_web_routing_sites(atom=atom, plan=plan, attempt_index=0)

    assert reason == "shopify_community_first"
    assert sites == [("community.shopify.com", "web"), ("apps.shopify.com", "web")]


def test_low_information_atom_does_not_force_cohort_pack():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="",
        job_to_be_done="process leads",
        failure_mode="manual work piles up",
        trigger_event="",
        current_workaround="",
        cost_consequence_clues="",
        current_tools="",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["manual process"],
        finding_kind="problem_signal",
    )

    queries = toolkit._spreadsheet_operator_admin_web_queries(atom=atom, plan=plan)

    assert queries == []


def test_search_web_filters_validation_junk_domains(monkeypatch):
    """Validation search should drop dictionaries, hours sites, and keep relevant software results."""
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeValidationDDGS))
    toolkit = ResearchToolkit({"validation": {"search": {"max_results_per_domain": 2}}})

    docs = asyncio.run(
        toolkit.search_web(
            "workflow automation repetitive approvals software tool alternative",
            max_results=5,
            intent="validation_competitor",
        )
    )

    assert [doc.url for doc in docs] == ["https://exampleapp.com/workflow-automation"]


def test_validation_recurrence_site_search_can_fall_back_to_bing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeEmptyDDGS))
    monkeypatch.setitem(sys.modules, "duckduckgo_search", types.SimpleNamespace(DDGS=FakeEmptyDDGS))
    toolkit = ResearchToolkit({"validation": {"search": {"max_results_per_domain": 2}}})

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, timeout=None, headers=None):
        if "html.duckduckgo.com" in url:
            return FakeResponse("<html><body></body></html>")
        if "bing.com/search" in url:
            return FakeResponse(
                """
                <html><body>
                  <li class='b_algo'>
                    <h2><a href='https://community.atlassian.com/forums/App-Central-discussions/Which-spreadsheet-is-the-latest-version/td-p/123'>Which spreadsheet is the latest version?</a></h2>
                    <div class='b_caption'><p>Teams waste time because they keep editing the wrong file and handoffs drift out of sync.</p></div>
                  </li>
                </body></html>
                """
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("research_tools.requests.get", fake_get)

    docs = asyncio.run(
        toolkit.search_web(
            "spreadsheet latest operator",
            max_results=5,
            site="community.atlassian.com",
            intent="validation_recurrence",
        )
    )

    assert [doc.url for doc in docs] == [
        "https://community.atlassian.com/forums/App-Central-discussions/Which-spreadsheet-is-the-latest-version/td-p/123"
    ]


def test_validation_recurrence_stackexchange_site_search_returns_docs(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"max_results_per_domain": 2}}})

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        assert "api.stackexchange.com/2.3/search/advanced" in url
        assert params["site"] == "superuser"
        return FakeResponse(
            {
                "items": [
                    {
                        "title": "Microsoft Excel 2010 Shared workbook - edit conflicts",
                        "link": "https://superuser.com/questions/1107814/microsoft-excel-2010-shared-workbook-edit-conflicts",
                        "body": (
                            "<p>The shared spreadsheet creates conflicts and the team cannot trust the data "
                            "because edits land on the wrong line.</p>"
                        ),
                    }
                ]
            }
        )

    monkeypatch.setattr("research_tools.requests.get", fake_get)

    docs = asyncio.run(
        toolkit.search_web(
            "excel shared workbook conflict",
            max_results=5,
            site="superuser.com",
            intent="validation_recurrence",
        )
    )

    assert [doc.url for doc in docs] == [
        "https://superuser.com/questions/1107814/microsoft-excel-2010-shared-workbook-edit-conflicts"
    ]
    assert docs[0].source == "superuser.com"


def test_choose_query_plan_prefers_stronger_queries_but_keeps_exploration():
    """Adaptive query planning should prefer validated queries while leaving exploration room."""
    toolkit = ResearchToolkit()
    toolkit.set_discovery_feedback(
        [
            {
                "source_name": "github-problem",
                "query_text": '"feature request" automation',
                "runs": 4,
                "docs_seen": 12,
                "findings_emitted": 4,
                "validations": 3,
                "passes": 2,
                "avg_validation_score": 0.78,
            },
            {
                "source_name": "github-problem",
                "query_text": '"manual process" tool',
                "runs": 5,
                "docs_seen": 10,
                "findings_emitted": 2,
                "validations": 2,
                "passes": 0,
                "avg_validation_score": 0.52,
            },
        ]
    )

    plan = toolkit.choose_query_plan(
        "github-problem",
        [
            '"feature request" automation',
            '"wish there was" workflow',
            '"manual process" tool',
            '"too expensive" software',
        ],
        limit=3,
    )

    assert '"feature request" automation' in plan
    assert len(plan) == 3
    assert any(query in plan for query in ['"wish there was" workflow', '"too expensive" software'])


def test_build_discovery_query_plan_rotates_across_cycles():
    toolkit = ResearchToolkit()
    queries = [
        '"feature request" automation',
        '"wish there was" workflow',
        '"manual process" tool',
        '"too expensive" software',
        '"time consuming" issue',
    ]

    first = toolkit.build_discovery_query_plan("github-problem", queries, limit=3, cycle_index=0)
    second = toolkit.build_discovery_query_plan("github-problem", queries, limit=3, cycle_index=1)

    assert len(first.queries) == 3
    assert len(second.queries) == 3
    assert first.queries[0] == second.queries[0]
    assert first.rotated_queries_used != second.rotated_queries_used
    assert second.rotation_applied is True
    assert second.query_offset > 0


def test_build_discovery_query_plan_deprioritizes_repeated_zero_yield_queries():
    toolkit = ResearchToolkit()
    toolkit.set_discovery_feedback(
        [
            {
                "source_name": "web-problem",
                "query_text": '"frustrating" workflow',
                "runs": 3,
                "docs_seen": 0,
                "findings_emitted": 0,
                "validations": 0,
                "passes": 0,
                "avg_validation_score": 0.0,
            }
        ]
    )

    plan = toolkit.build_discovery_query_plan(
        "web-problem",
        [
            '"wish there was" software for',
            '"too expensive" current tool',
            '"manual process" every day',
            '"frustrating" workflow',
        ],
        limit=3,
        cycle_index=0,
    )

    assert '"frustrating" workflow' not in plan.queries


def test_build_discovery_query_plan_prefers_queries_that_produced_prototype_candidates():
    toolkit = ResearchToolkit()
    toolkit.set_discovery_feedback(
        [
            {
                "source_name": "reddit-problem",
                "query_text": "duct tape spreadsheets",
                "runs": 2,
                "docs_seen": 6,
                "findings_emitted": 2,
                "validations": 2,
                "passes": 0,
                "prototype_candidates": 1,
                "build_briefs": 1,
                "avg_validation_score": 0.44,
            },
            {
                "source_name": "reddit-problem",
                "query_text": "manual reporting",
                "runs": 2,
                "docs_seen": 6,
                "findings_emitted": 2,
                "validations": 2,
                "passes": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
                "avg_validation_score": 0.44,
            },
        ]
    )

    plan = toolkit.build_discovery_query_plan(
        "reddit-problem",
        ["manual reporting", "duct tape spreadsheets", "latest spreadsheet version confusion"],
        limit=2,
        cycle_index=0,
    )

    assert plan.queries[0] == "duct tape spreadsheets"


def test_discover_reddit_problem_threads_uses_bounded_concurrency():
    toolkit = ResearchToolkit({"discovery": {"reddit": {"pair_concurrency": 4, "context_concurrency": 4}}})

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        await asyncio.sleep(0.05)
        return [
            SearchDocument(
                title=f"{subreddit} {query} workflow breaks",
                url=f"https://reddit.com/{subreddit}/{query.replace(' ', '-')}",
                snippet="manual reconciliation keeps breaking every week",
                source=f"reddit/{subreddit}",
            )
        ]

    async def fake_thread_context(url):
        await asyncio.sleep(0.05)
        return {
            "title": "workflow breaks",
            "text": "manual reconciliation keeps breaking every week and teams fall back to spreadsheets",
            "description": "manual reconciliation keeps breaking every week",
            "comments": [],
        }

    toolkit.reddit_search = fake_reddit_search
    toolkit.reddit_thread_context = fake_thread_context

    start = time.perf_counter()
    findings = asyncio.run(
        toolkit._discover_reddit_problem_threads(
            subreddits=["ops", "finance"],
            queries=["manual reconciliation", "spreadsheet workaround"],
        )
    )
    elapsed = time.perf_counter() - start

    assert len(findings) == 4
    assert elapsed < 0.28


def test_discover_reddit_problem_threads_queries_multiple_sort_modes():
    toolkit = ResearchToolkit(
        {
            "discovery": {
                "reddit": {
                    "search_sorts": ["relevance", "new", "top", "comments"],
                    "per_sort_limit": 1,
                    "max_docs_per_pair": 4,
                }
            }
        }
    )
    seen_sorts: list[str] = []

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        seen_sorts.append(sort)
        return [
            SearchDocument(
                title=f"{subreddit} {query} {sort}",
                url=f"https://reddit.com/{subreddit}/{query.replace(' ', '-')}/{sort}",
                snippet="manual reconciliation keeps breaking every week",
                source=f"reddit/{subreddit}",
            )
        ]

    async def fake_thread_context(url):
        return {
            "title": "workflow breaks",
            "text": "manual reconciliation keeps breaking every week and teams fall back to spreadsheets",
            "description": "manual reconciliation keeps breaking every week",
            "comments": [],
        }

    toolkit.reddit_search = fake_reddit_search
    toolkit.reddit_thread_context = fake_thread_context

    findings = asyncio.run(
        toolkit._discover_reddit_problem_threads(
            subreddits=["ops"],
            queries=["manual reconciliation"],
        )
    )

    assert len(findings) == 4
    assert set(seen_sorts) == {"relevance", "new", "top", "comments"}
    assert {item["evidence"]["discovery_sort"] for item in findings} == {"relevance", "new", "top", "comments"}


class DummyRedditResponse:
    def __init__(self, children):
        self._children = children

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": {"children": self._children}}


def _reddit_child(title="Thread", permalink="/r/test/comments/1/test", selftext="Body"):
    return {"data": {"title": title, "permalink": permalink, "selftext": selftext}}


def test_reddit_bridge_only_hit_returns_results_without_public_fallback(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}})
    toolkit.node_bin = None

    async def fake_search_posts(**kwargs):
        return ([{"title": "Bridge post", "permalink": "https://reddit.com/r/test/comments/1", "body": "Bridge body", "subreddit": "test"}], "")

    monkeypatch.setattr(toolkit.reddit_bridge, "search_posts", fake_search_posts)
    monkeypatch.setattr("research_tools.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public reddit should not be hit")))

    docs = asyncio.run(toolkit.reddit_search("test", "workflow pain", limit=2))

    assert len(docs) == 1
    metrics = toolkit.get_reddit_runtime_metrics()
    assert metrics["reddit_mode"] == "bridge_only"
    assert metrics["reddit_bridge_hits"] == 1
    assert metrics["reddit_bridge_misses"] == 0
    assert metrics["reddit_fallback_queries"] == 0


def test_reddit_bridge_only_miss_returns_empty_without_public_fallback(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}})
    toolkit.node_bin = None

    async def fake_search_posts(**kwargs):
        raise BridgeError("no_cached_result", "no cached search result", 404)

    monkeypatch.setattr(toolkit.reddit_bridge, "search_posts", fake_search_posts)
    monkeypatch.setattr("research_tools.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public reddit should not be hit")))

    docs = asyncio.run(toolkit.reddit_search("test", "workflow pain", limit=2))

    assert docs == []
    metrics = toolkit.get_reddit_runtime_metrics()
    assert metrics["reddit_bridge_hits"] == 0
    assert metrics["reddit_bridge_misses"] == 1
    assert metrics["reddit_fallback_queries"] == 0


def test_reddit_bridge_with_fallback_uses_public_reddit_on_miss(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_with_fallback"}})
    toolkit.node_bin = None

    async def fake_search_posts(**kwargs):
        raise BridgeError("no_cached_result", "no cached search result", 404)

    monkeypatch.setattr(toolkit.reddit_bridge, "search_posts", fake_search_posts)
    monkeypatch.setattr("research_tools.requests.get", lambda *args, **kwargs: DummyRedditResponse([_reddit_child(title="Fallback post")]))

    docs = asyncio.run(toolkit.reddit_search("test", "workflow pain", limit=2))

    assert [doc.title for doc in docs] == ["Fallback post"]
    metrics = toolkit.get_reddit_runtime_metrics()
    assert metrics["reddit_mode"] == "bridge_with_fallback"
    assert metrics["reddit_bridge_misses"] == 1
    assert metrics["reddit_fallback_queries"] == 1


def test_reddit_public_direct_bypasses_bridge(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "public_direct"}})
    toolkit.node_bin = None

    async def should_not_call_bridge(**kwargs):
        raise AssertionError("bridge should not be called in public_direct mode")

    monkeypatch.setattr(toolkit.reddit_bridge, "search_posts", should_not_call_bridge)
    monkeypatch.setattr("research_tools.requests.get", lambda *args, **kwargs: DummyRedditResponse([_reddit_child(title="Direct post")]))

    docs = asyncio.run(toolkit.reddit_search("test", "workflow pain", limit=2))

    assert [doc.title for doc in docs] == ["Direct post"]
    metrics = toolkit.get_reddit_runtime_metrics()
    assert metrics["reddit_mode"] == "public_direct"
    assert metrics["reddit_bridge_hits"] == 0
    assert metrics["reddit_bridge_misses"] == 0
    assert metrics["reddit_fallback_queries"] == 0
    assert metrics["reddit_public_direct_queries"] == 1


def test_warm_reddit_validation_queries_seeds_missing_pairs_in_bridge_only(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}})

    class FakeSeeder:
        def __init__(self, _config):
            self.calls = []

        def coverage_report(self, *, subreddits=None, queries=None):
            self.calls.append(("coverage", tuple(subreddits or []), tuple(queries or [])))
            if len(self.calls) == 1:
                return SimpleNamespace(uncovered_pairs=4)
            return SimpleNamespace(uncovered_pairs=0)

        async def seed(self, *, subreddits=None, queries=None):
            self.calls.append(("seed", tuple(subreddits or []), tuple(queries or [])))
            return SimpleNamespace(cached_searches=4)

    monkeypatch.setattr("reddit_seed.RedditSeeder", FakeSeeder)

    warmed = asyncio.run(
        toolkit.warm_reddit_validation_queries(
            subreddits=["smallbusiness", "sysadmin"],
            queries=['"manual reconciliation" ops', '"manual reconciliation" ops'],
        )
    )

    assert warmed["seed_runs"] == 1
    assert warmed["seeded_pairs"] == 2
    assert warmed["seeded_searches"] == 4
    assert warmed["uncovered_before"] == 4
    assert warmed["uncovered_after"] == 0
    metrics = toolkit.get_reddit_runtime_metrics()
    assert metrics["reddit_validation_seed_runs"] == 1
    assert metrics["reddit_validation_seeded_pairs"] == 2
    assert metrics["reddit_validation_seed_searches"] == 4
    assert metrics["reddit_validation_seed_uncovered_before"] == 4
    assert metrics["reddit_validation_seed_uncovered_after"] == 0


def test_gather_recurrence_evidence_warms_bridge_only_queries(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}})

    async def fake_warm(*, subreddits, queries):
        assert subreddits == ["smallbusiness", "sysadmin"]
        assert queries == ['"manual reconciliation" "small business"']
        return {
            "seed_runs": 1,
            "seeded_pairs": 2,
            "seeded_searches": 2,
            "uncovered_before": 2,
            "uncovered_after": 0,
        }

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        return [SearchDocument(title=f"{subreddit} hit", url=f"https://reddit.com/r/{subreddit}/comments/1", snippet="body", source=f"reddit/{subreddit}")]

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        return []

    monkeypatch.setattr(toolkit, "warm_reddit_validation_queries", fake_warm)
    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations manager",
        job_to_be_done="keep operations data in sync",
        failure_mode="manual reconciliation delays reporting",
        trigger_event="after payout reconciliation",
        current_tools="Excel",
    )

    docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"manual reconciliation" "small business"'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert len(docs) == 2
    assert meta["warmed_validation_queries"]["seed_runs"] == 1
    assert meta["warmed_validation_queries"]["seeded_pairs"] == 2
    assert meta["recurrence_failure_class"] == "single_source_only"


def test_gather_recurrence_evidence_stops_early_on_generic_probe_miss(monkeypatch):
    toolkit = ResearchToolkit()
    reddit_calls = []
    web_calls = []

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        reddit_calls.append((subreddit, query, limit))
        return []

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        web_calls.append((query, site, max_results, intent))
        return []

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="",
        job_to_be_done="keep a recurring workflow on track",
        failure_mode="manual work keeps piling up",
        trigger_event="",
        current_workaround="",
        cost_consequence_clues="",
        current_tools="",
    )

    docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"manual work" ops', '"spreadsheet workaround" ops'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert docs == []
    assert meta["recurrence_gap_reason"] in {"search_breadth_likely_insufficient", "no_independent_confirmations"}
    assert meta["recurrence_failure_class"] in {"breadth_limited", "no_corroboration_found"}
    assert meta["recurrence_probe_summary"]["probe_hit_count"] == 0
    assert meta["recurrence_probe_summary"]["branched_after_probe"] is False
    assert len(reddit_calls) == 1
    assert len(web_calls) == 1


def test_gather_recurrence_evidence_branches_after_specific_probe_miss(monkeypatch):
    toolkit = ResearchToolkit()
    reddit_calls = []
    web_calls = []

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        reddit_calls.append((subreddit, query, limit))
        return []

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        web_calls.append((query, site, max_results, intent))
        return []

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="developer",
        job_to_be_done="keep sync and data handoff workflows reliable",
        failure_mode="sync workflows fail after changes",
        trigger_event="after deployment changes",
        current_workaround="manual rollback scripts",
        cost_consequence_clues="downtime risk",
        current_tools="shell scripts",
    )

    docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"backup restore" operator', '"restore jobs fail" downtime'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert docs == []
    assert meta["recurrence_probe_summary"]["probe_hit_count"] == 0
    assert meta["recurrence_probe_summary"]["branched_after_probe"] is True
    assert meta["recurrence_failure_class"] in {"no_corroboration_found", "breadth_limited"}
    assert len(reddit_calls) > 1
    assert len(web_calls) > 1


def test_gather_recurrence_evidence_branches_to_multi_source_after_single_source_hit(monkeypatch):
    toolkit = ResearchToolkit()
    search_calls = {"web": 0}

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        return [
            SearchDocument(
                title=f"{subreddit} corroboration",
                url=f"https://reddit.com/r/{subreddit}/comments/abc",
                snippet="operators keep manual rollback checklists after restore failures",
                source=f"reddit/{subreddit}",
            )
        ]

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        if site is None:
            search_calls["web"] += 1
            if search_calls["web"] >= 3:
                return [
                    SearchDocument(
                        title="Ops blog corroboration",
                        url="https://ops.example.com/restore-runbook",
                        snippet="teams still keep manual rollback runbooks when restores fail",
                        source="web",
                    )
                ]
        return []

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep backup restore and recovery reliable",
        failure_mode="restored environments stay unreachable",
        trigger_event="after restore jobs fail",
        current_workaround="manual rollback scripts",
        cost_consequence_clues="downtime risk",
        current_tools="backup console",
    )

    docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"backup restore" operator'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert len(docs) >= 2
    assert meta["recurrence_source_branch"]["triggered"] is True
    assert meta["recurrence_failure_class"] == "confirmed"
    assert meta["recurrence_state"] in {"supported", "strong"}
    assert meta["results_by_source"]["reddit"] > 0
    assert meta["results_by_source"]["web"] > 0


def test_recurrence_source_specific_queries_shape_by_family():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    github_queries = toolkit._recurrence_source_specific_queries(
        selected_queries=['"manual data entry" "spreadsheets manual work email"'],
        atom=atom,
        source_label="github",
        limit=3,
    )
    web_queries = toolkit._recurrence_source_specific_queries(
        selected_queries=['"manual data entry" "spreadsheets manual work email"'],
        atom=atom,
        source_label="web",
        limit=3,
    )

    assert github_queries == []
    assert web_queries
    assert any(
        "excel shared workbook conflict" in query
        or "shared spreadsheet saving conflicts" in query
        or "google sheets collaborator changes not showing" in query
        or "spreadsheet latest" in query
        or "latest spreadsheet version confusion" in query
        or "workflow software" in query
        or "forum" in query
        or "duct tape spreadsheets" in query
        for query in web_queries
    )


def test_classify_recurrence_match_strong_partial_none():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel csv import",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=['"manual data entry" "spreadsheets manual work email"'],
        finding_kind="problem_signal",
    )

    strong = toolkit._classify_recurrence_match(
        SearchDocument(
            title="Spreadsheet cleanup workflow keeps ops teams stuck",
            url="https://ops.example.com/spreadsheet-cleanup",
            snippet="Operations leads still rely on manual csv cleanup when imports arrive with duplicates.",
            source="web",
        ),
        atom,
        plan.signature_terms,
    )
    partial = toolkit._classify_recurrence_match(
        SearchDocument(
            title="CSV import cleanup checklist",
            url="https://ops.example.com/checklist",
            snippet="Teams use csv import cleanup checklists before automation.",
            source="web",
        ),
        atom,
        plan.signature_terms,
    )
    none = toolkit._classify_recurrence_match(
        SearchDocument(
            title="Plugin settings page broken after update",
            url="https://wordpress.org/support/topic/plugin-settings-page-broken/",
            snippet="The settings page is unusable after the update.",
            source="web",
        ),
        atom,
        plan.signature_terms,
    )
    fragility = toolkit._classify_recurrence_match(
        SearchDocument(
            title="Duct tape spreadsheets causing cross-team handoff failures",
            url="https://example.com/ops-handoff",
            snippet="Copy paste updates leave teams out of sync and missed steps pile up.",
            source="web",
        ),
        atom,
        plan.signature_terms,
    )

    assert strong == "strong"
    assert partial == "partial"
    assert none == "none"
    assert fragility in {"partial", "strong"}


def test_gather_recurrence_evidence_uses_family_specific_queries_for_branch(monkeypatch):
    toolkit = ResearchToolkit()
    captured_calls: list[tuple[str | None, str]] = []

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        return [
            SearchDocument(
                title="Reddit corroboration",
                url=f"https://reddit.com/r/{subreddit}/comments/abc",
                snippet="operators keep manual rollback checklists after restore failures",
                source=f"reddit/{subreddit}",
            )
        ]

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        captured_calls.append((site, query))
        if site is None and ("spreadsheet import" in query or "csv import cleanup" in query or "manual spreadsheet" in query):
            return [
                SearchDocument(
                    title="Operator workflow corroboration",
                    url="https://ops.example.com/spreadsheet-cleanup",
                    snippet="operations leads still rely on manual csv cleanup when spreadsheet imports arrive with duplicates and cause downtime risk",
                    source="web",
                )
            ]
        return []

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"manual data entry" "spreadsheets manual work email"'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert len(docs) >= 1
    assert meta["matched_results_by_source"]["web"] >= 0
    source_attempts = meta["recurrence_source_branch"]["source_attempts"]
    assert any(attempt["source"] == "web" for attempt in source_attempts)
    github_queries = [query for site, query in captured_calls if site == "github.com"]
    assert github_queries == []
    reviewable_web_calls = [
        query
        for site, query in captured_calls
        if site in {
            None,
            "superuser.com",
            "webapps.stackexchange.com",
            "community.atlassian.com",
            "community.monday.com",
            "capterra.com",
            "g2.com",
        }
    ]
    assert reviewable_web_calls
    assert any(
        "excel shared workbook conflict" in query
        or "shared spreadsheet saving conflicts" in query
        or "google sheets collaborator changes not showing" in query
        or "spreadsheet latest version" in query
        or "latest spreadsheet version confusion" in query
        or "multiple people editing same spreadsheet latest version" in query
        for query in reviewable_web_calls
    )


def test_expand_recurrence_source_families_warms_reddit_branch_queries(monkeypatch):
    toolkit = ResearchToolkit({"reddit_bridge": {"enabled": True, "base_url": "https://bridge.example", "mode": "bridge_only"}})
    warm_calls = []

    async def fake_warm(*, subreddits, queries):
        warm_calls.append((tuple(subreddits), tuple(queries)))
        return {
            "seed_runs": 1,
            "seeded_pairs": len(subreddits) * len(queries),
            "seeded_searches": len(subreddits) * len(queries),
            "uncovered_before": len(subreddits) * len(queries),
            "uncovered_after": 0,
        }

    async def fake_run_recurrence_collection(*, queries, subreddit_plan, site_plan, atom, per_source_limit, stop_after_docs, allow_fallback):
        return [], {query: 0 for query in queries}, {"reddit": 0}, {
            "retrieved_by_source": {"reddit": 0},
            "deduped_by_source": {"reddit": 0},
            "docs_by_source": {"reddit": []},
            "queries_by_source": {"reddit": list(queries)},
        }

    monkeypatch.setattr(toolkit, "warm_reddit_validation_queries", fake_warm)
    monkeypatch.setattr(toolkit, "_run_recurrence_collection", fake_run_recurrence_collection)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    docs, results_by_query, results_by_source, _collection_meta, branch = asyncio.run(
        toolkit._expand_recurrence_source_families(
            selected_queries=['"manual data entry" "spreadsheets manual work email"'],
            atom=atom,
            all_subreddits=["smallbusiness", "sysadmin"],
            all_sites=[],
            current_docs=[],
            current_results_by_query={},
            current_results_by_source={},
            current_collection_meta={},
            budget_profile={"target_sources": 2, "target_docs": 4, "early_stop_docs": 5},
            corroboration_plan=toolkit._build_corroboration_plan(
                atom=atom,
                queries=['"manual data entry" "spreadsheets manual work email"'],
                finding_kind="problem_signal",
            ),
        )
    )

    assert docs == []
    assert results_by_query
    assert '"manual workflow" "small business"' in results_by_query
    assert '"spreadsheet workaround" "small business"' in results_by_query
    assert results_by_source == {"reddit": 0}
    assert branch["triggered"] is True
    assert warm_calls
    subreddits, queries = warm_calls[0]
    assert subreddits == ("smallbusiness", "sysadmin")
    assert any("manual workflow" in query or "spreadsheet workaround" in query for query in queries)


def test_expand_recurrence_source_families_records_reshape_retry(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"recurrence_results": 6}}})
    atom = SimpleNamespace(
        segment="platform engineering",
        user_role="developer",
        job_to_be_done="keep deployment automation reliable",
        failure_mode="api integration rollback scripts fail during deploys",
        trigger_event="during releases",
        current_workaround="manual rollback script",
        cost_consequence_clues="downtime risk",
        current_tools="github actions api webhook deployment config",
    )

    call_queries: list[list[str]] = []

    async def fake_run_recurrence_collection(*, queries, subreddit_plan, site_plan, atom, per_source_limit, stop_after_docs, allow_fallback):
        call_queries.append(list(queries))
        return [
            SearchDocument(
                title="Theme color bug in spreadsheet add-on",
                url="https://github.com/example/repo/issues/99",
                snippet="Cosmetic issue in spreadsheet add-on theme colors.",
                source="github",
            )
        ], {query: 1 for query in queries}, {"github": len(queries)}, {
            "retrieved_by_source": {"github": len(queries)},
            "deduped_by_source": {"github": 1},
            "docs_by_source": {"github": [
                SearchDocument(
                    title="Theme color bug in spreadsheet add-on",
                    url="https://github.com/example/repo/issues/99",
                    snippet="Cosmetic issue in spreadsheet add-on theme colors.",
                    source="github",
                )
            ]},
            "queries_by_source": {"github": list(queries)},
        }

    monkeypatch.setattr(toolkit, "_run_recurrence_collection", fake_run_recurrence_collection)

    _docs, _results_by_query, _results_by_source, _collection_meta, branch = asyncio.run(
        toolkit._expand_recurrence_source_families(
            selected_queries=['"rollback script" "deployment automation"'],
            atom=atom,
            all_subreddits=[],
            all_sites=[("github.com", "github")],
            current_docs=[],
            current_results_by_query={},
            current_results_by_source={"reddit": 1},
            current_collection_meta={},
            budget_profile={"target_sources": 2, "target_docs": 4, "early_stop_docs": 5},
            corroboration_plan=toolkit._build_corroboration_plan(
                atom=atom,
                queries=['"rollback script" "deployment automation"'],
                finding_kind="problem_signal",
            ),
        )
    )

    assert branch["triggered"] is True
    assert any(attempt["source"] == "github" and attempt["attempts"] == 2 for attempt in branch["source_attempts"])
    assert branch["reshaped_query_history"]
    assert len(call_queries) == 2
    assert call_queries[0] != call_queries[1]


def test_expand_recurrence_source_families_records_reshape_retry_on_empty_retrieval(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"recurrence_results": 6}}})
    atom = SimpleNamespace(
        segment="compliance teams",
        user_role="compliance lead",
        job_to_be_done="keep compliance evidence collection reliable",
        failure_mode="manual m365 audit exports block audits",
        trigger_event="before audits",
        current_workaround="manual export checklists",
        cost_consequence_clues="audit delay risk",
        current_tools="m365 compliance export",
    )

    call_queries: list[list[str]] = []

    async def fake_run_recurrence_collection(*, queries, subreddit_plan, site_plan, atom, per_source_limit, stop_after_docs, allow_fallback):
        call_queries.append(list(queries))
        return [], {query: 0 for query in queries}, {"github": 0}, {
            "retrieved_by_source": {"github": 0},
            "deduped_by_source": {"github": 0},
            "docs_by_source": {"github": []},
            "queries_by_source": {"github": list(queries)},
        }

    monkeypatch.setattr(toolkit, "_run_recurrence_collection", fake_run_recurrence_collection)

    _docs, _results_by_query, _results_by_source, _collection_meta, branch = asyncio.run(
        toolkit._expand_recurrence_source_families(
            selected_queries=["manual m365 export workflow"],
            atom=atom,
            all_subreddits=[],
            all_sites=[("github.com", "github")],
            current_docs=[],
            current_results_by_query={},
            current_results_by_source={"reddit": 1},
            current_collection_meta={},
            budget_profile={"target_sources": 2, "target_docs": 4, "early_stop_docs": 5},
            corroboration_plan=toolkit._build_corroboration_plan(
                atom=atom,
                queries=["manual m365 export workflow"],
                finding_kind="problem_signal",
            ),
        )
    )

    assert len(call_queries) == 2
    assert call_queries[0] != call_queries[1]
    assert any(attempt["source"] == "github" and attempt["attempts"] == 2 for attempt in branch["source_attempts"])
    assert branch["reshaped_query_history"]


def test_web_zero_retrieval_retry_switches_to_decomposed_queries(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"recurrence_results": 6}}})
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    call_queries: list[list[str]] = []

    async def fake_run_recurrence_collection(*, queries, subreddit_plan, site_plan, atom, per_source_limit, stop_after_docs, allow_fallback):
        call_queries.append(list(queries))
        return [], {query: 0 for query in queries}, {"web": 0}, {
            "retrieved_by_source": {"web": 0},
            "deduped_by_source": {"web": 0},
            "docs_by_source": {"web": []},
            "queries_by_source": {"web": list(queries)},
        }

    monkeypatch.setattr(toolkit, "_run_recurrence_collection", fake_run_recurrence_collection)

    _docs, _results_by_query, _results_by_source, _collection_meta, branch = asyncio.run(
        toolkit._expand_recurrence_source_families(
            selected_queries=['"manual data entry" "spreadsheet cleanup"'],
            atom=atom,
            all_subreddits=[],
            all_sites=[(None, "web")],
            current_docs=[],
            current_results_by_query={},
            current_results_by_source={"reddit": 1},
            current_collection_meta={},
            budget_profile={"target_sources": 2, "target_docs": 4, "early_stop_docs": 5},
            corroboration_plan=toolkit._build_corroboration_plan(
                atom=atom,
                queries=['"manual data entry" "spreadsheet cleanup"'],
                finding_kind="problem_signal",
            ),
        )
    )

    assert len(call_queries) == 2
    assert call_queries[0] != call_queries[1]
    assert any("manual re-entry" in q or "tracking manual" in q for q in call_queries[1])
    assert any("spreadsheet keep operations data sync" in q for q in call_queries[1])
    assert branch["reshaped_query_history"]
    assert branch["decomposed_atom_queries"]
    assert branch["fallback_strategy_used"] == "decomposed_query_switch"
    assert branch["cohort_query_pack_used"] is True
    assert branch["cohort_query_pack_name"] == "spreadsheet_operator_admin"
    assert branch["web_query_strategy_path"] == [
        "atom_shaped",
        "cohort_pack",
        "specialized_surface_targeting",
        "fallback_workaround_friction",
        "decomposition",
    ]
    assert branch["routing_override_reason"] == "workflow_fragility_surface_first"
    assert branch["specialized_surface_targeting_used"] is True


def test_low_information_atom_retry_uses_decomposed_web_fallback():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="",
        job_to_be_done="process like lead tracking invoicing data",
        failure_mode="manual work keeps piling up",
        trigger_event="",
        current_workaround="",
        cost_consequence_clues="",
        current_tools="",
    )

    queries = toolkit._web_recurrence_queries_from_atom(
        atom=atom,
        signature_terms=["lead", "tracking", "invoicing"],
        role_terms=[],
        segment_terms=["small", "business"],
        job_phrase="process like lead tracking invoicing data",
        failure_phrase="manual work keeps piling up",
        workaround_phrase="",
        cost_terms=[],
        ecosystem_hints=[],
        reshape_reason="cost_missing",
    )

    assert any("process like lead tracking small business" in q for q in queries)
    assert any("manual work keeps piling" in q for q in queries)


def test_specialized_web_routing_metadata_populates(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"recurrence_results": 6}}})
    atom = SimpleNamespace(
        segment="shopify merchants",
        user_role="store operator",
        job_to_be_done="keep order status in sync without manual copy paste",
        failure_mode="shopify app updates break workflow handoffs",
        trigger_event="after app updates",
        current_workaround="manual order status updates",
        cost_consequence_clues="time loss",
        current_tools="shopify storefront app plugin",
    )

    captured_sites: list[list[tuple[object, str]]] = []

    async def fake_run_recurrence_collection(*, queries, subreddit_plan, site_plan, atom, per_source_limit, stop_after_docs, allow_fallback):
        captured_sites.append(list(site_plan))
        return [], {query: 0 for query in queries}, {"web": 0}, {
            "retrieved_by_source": {"web": 0},
            "deduped_by_source": {"web": 0},
            "docs_by_source": {"web": []},
            "queries_by_source": {"web": list(queries)},
        }

    monkeypatch.setattr(toolkit, "_run_recurrence_collection", fake_run_recurrence_collection)

    _docs, _results_by_query, _results_by_source, _collection_meta, branch = asyncio.run(
        toolkit._expand_recurrence_source_families(
            selected_queries=["shopify app workflow"],
            atom=atom,
            all_subreddits=[],
            all_sites=toolkit._recurrence_site_plan(atom, subreddit_plan=["smallbusiness"], limit=4),
            current_docs=[],
            current_results_by_query={},
            current_results_by_source={"reddit": 1},
            current_collection_meta={},
            budget_profile={"target_sources": 2, "target_docs": 4, "early_stop_docs": 5},
            corroboration_plan=toolkit._build_corroboration_plan(
                atom=atom,
                queries=["shopify app workflow"],
                finding_kind="problem_signal",
            ),
        )
    )

    assert captured_sites
    assert captured_sites[0] == [("community.shopify.com", "web"), ("apps.shopify.com", "web")]
    assert branch["routing_override_reason"] == "shopify_community_first"


def test_cohort_specific_pack_used_before_generic_web_fallback():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations coordinator",
        job_to_be_done="keep approvals and reporting in sync",
        failure_mode="copy paste spreadsheet updates create duplicate entry",
        trigger_event="during weekly reporting",
        current_workaround="using spreadsheets for approvals tracking",
        cost_consequence_clues="time loss",
        current_tools="excel approvals tracker",
    )

    queries = toolkit._web_recurrence_queries_from_atom(
        atom=atom,
        signature_terms=["spreadsheet", "duplicate", "approvals", "reporting"],
        role_terms=["operations", "coordinator"],
        segment_terms=["small", "business"],
        job_phrase="keep approvals reporting sync",
        failure_phrase="copy paste spreadsheet updates duplicate entry",
        workaround_phrase="using spreadsheets approvals tracking",
        cost_terms=["time", "loss"],
        ecosystem_hints=["excel"],
        reshape_reason="failure_missing",
    )

    assert any("manual re-entry" in q or "replace spreadsheet for" in q or "using spreadsheets for" in q for q in queries)
    assert len(queries) <= 5


def test_operator_admin_atom_triggers_specialized_surface_queries():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="office manager",
        job_to_be_done="keep reporting and approvals in sync",
        failure_mode="spreadsheet handoffs create duplicate entry and slow reporting",
        trigger_event="during weekly vendor updates",
        current_workaround="using spreadsheets for approvals tracking",
        cost_consequence_clues="time loss",
        current_tools="excel reporting tracker",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["spreadsheet reporting duplicate entry"],
        finding_kind="problem_signal",
    )

    queries = toolkit._specialized_operator_surface_queries(atom=atom, plan=plan)

    assert queries
    assert any("replace spreadsheet" in q for q in queries)
    assert any("software" in q or "community" in q or "forum" in q for q in queries)
    assert len(queries) <= 4


def test_low_information_atom_does_not_trigger_specialized_surface_queries():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="",
        job_to_be_done="keep work moving",
        failure_mode="manual work piles up",
        trigger_event="",
        current_workaround="",
        cost_consequence_clues="",
        current_tools="",
    )
    plan = toolkit._build_corroboration_plan(
        atom=atom,
        queries=["manual work piles up"],
        finding_kind="problem_signal",
    )

    assert toolkit._specialized_operator_surface_queries(atom=atom, plan=plan) == []


def test_validate_problem_propagates_web_strategy_metadata():
    toolkit = ResearchToolkit()

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.31,
            "recurrence_state": "thin",
            "query_coverage": 0.5,
            "doc_count": 0,
            "domain_count": 0,
            "results_by_query": {},
            "results_by_source": {"reddit": 2, "web": 0},
            "matched_results_by_source": {"reddit": 1, "web": 0},
            "partial_results_by_source": {"reddit": 1, "web": 0},
            "matched_docs_by_source": {
                "web": [
                    {
                        "source_family": "web",
                        "source": "web",
                        "query_text": "spreadsheet cleanup workflow",
                        "normalized_url": "https://ops.example.com/manual-spreadsheet-cleanup",
                        "title": "Spreadsheet cleanup workflow still manual",
                        "snippet": "Teams still rely on manual csv cleanup after spreadsheet imports break.",
                        "match_class": "strong",
                    }
                ]
            },
            "partial_docs_by_source": {"web": []},
            "family_confirmation_count": 1,
            "source_yield": {"web": {"attempts": 2, "docs_retrieved": 0}},
            "reshaped_query_history": [{"source": "web", "attempt": 2, "reason": "failure_missing"}],
            "queries_considered": ['"spreadsheet cleanup"'],
            "queries_executed": ["spreadsheet cleanup workflow"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {},
            "last_action": "RETRY_WITH_RESHAPED_QUERY",
            "last_transition_reason": "zero_retrieval_strategy_switch",
            "chosen_family": "web",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {},
            "skipped_families": {"github": "low_public_issue_fit"},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "decomposed_query_switch",
            "decomposed_atom_queries": ["small business reporting"],
            "routing_override_reason": "operator_surface_queries_first",
            "cohort_query_pack_used": True,
            "cohort_query_pack_name": "spreadsheet_operator_admin",
            "web_query_strategy_path": ["atom_shaped", "cohort_pack", "specialized_surface_targeting"],
            "specialized_surface_targeting_used": True,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "single_source_confirmation_only",
            "recurrence_failure_class": "single_source_only",
        }

    async def fake_search(*_args, **_kwargs):
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Spreadsheet cleanup still breaks approvals",
            summary="Office managers keep using spreadsheets for approvals and reporting, which causes duplicate entry.",
            finding_kind="pain_point",
        )
    )

    assert result["evidence"]["cohort_query_pack_used"] is True
    assert result["evidence"]["cohort_query_pack_name"] == "spreadsheet_operator_admin"
    assert result["evidence"]["web_query_strategy_path"] == [
        "atom_shaped",
        "cohort_pack",
        "specialized_surface_targeting",
    ]
    assert result["evidence"]["matched_docs_by_source"]["web"][0]["match_class"] == "strong"
    assert result["evidence"]["matched_docs_by_source"]["web"][0]["normalized_url"] == "https://ops.example.com/manual-spreadsheet-cleanup"
    assert result["evidence"]["specialized_surface_targeting_used"] is True
    assert result["evidence"]["routing_override_reason"] == "operator_surface_queries_first"


def test_validate_problem_runs_targeted_value_enrichment_for_supported_value_gap():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="finance ops",
        user_role="controller",
        job_to_be_done="close the books without export failures",
        failure_mode="month-end export timeouts block close",
        trigger_event="during month-end close",
        current_workaround="manual export retries and spreadsheet cleanup",
        cost_consequence_clues="hours lost and reporting delays",
        current_tools="qbo exports spreadsheets",
    )
    search_calls = []

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.59,
            "recurrence_state": "supported",
            "query_coverage": 0.75,
                "doc_count": 3,
                "domain_count": 2,
                "results_by_query": {"month end export timeout": 3},
                "results_by_source": {"reddit": 2, "web": 1},
                "matched_results_by_source": {"reddit": 1, "web": 1},
                "partial_results_by_source": {"reddit": 1, "web": 0},
                "family_confirmation_count": 2,
                "source_yield": {
                    "reddit": {"attempts": 1, "docs_retrieved": 2, "docs_strong_match": 1, "docs_partial_match": 1, "confirmed": True},
                    "web": {"attempts": 1, "docs_retrieved": 1, "docs_strong_match": 1, "docs_partial_match": 0, "confirmed": True},
                },
            "reshaped_query_history": [],
            "queries_considered": ["month end export timeout"],
            "queries_executed": ["month end export timeout"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {},
            "last_action": "GATHER_MARKET_ENRICHMENT",
            "last_transition_reason": "supported_recurrence_value_gap",
            "chosen_family": "reddit",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {"reddit": {"attempts": 1}},
            "skipped_families": {},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "",
            "decomposed_atom_queries": [],
            "routing_override_reason": "",
            "cohort_query_pack_used": False,
            "cohort_query_pack_name": "",
            "web_query_strategy_path": [],
            "specialized_surface_targeting_used": False,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "single_source_confirmation_only",
            "recurrence_failure_class": "single_source_only",
        }

    async def fake_search(query, max_results=8, site=None, intent="general"):
        search_calls.append((query, intent))
        if intent == "validation_competitor":
            return []
        if intent == "validation_value_enrichment":
            return [
                SearchDocument(
                    title="Manual close process costs finance teams hours",
                    url="https://example.com/finance-close",
                    snippet="Teams spend hours on manual exports and spreadsheet retries every month.",
                    source="web",
                )
            ]
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Month-end close bottlenecks with export timeouts",
            summary="Finance teams keep retrying exports and cleaning spreadsheets manually during close.",
            finding_kind="pain_point",
            atom=atom,
        )
    )

    assert result["evidence"]["promotion_gap_class"] in {"value_gap", "mixed_gap", "confirmed"}
    assert result["evidence"]["near_miss_enrichment_action"] == "GATHER_MARKET_ENRICHMENT"
    assert result["evidence"]["value_enrichment_used"] is True
    assert result["evidence"]["value_enrichment_queries"]
    assert result["evidence"]["value_enrichment_docs"]
    assert any(intent == "validation_value_enrichment" for _, intent in search_calls)


def test_validate_problem_runs_value_enrichment_for_strong_multi_family_borderline_value_gap():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep shared spreadsheets aligned across teams",
        failure_mode="teams lose track of which spreadsheet version is latest and handoffs break",
        trigger_event="during weekly status updates and customer follow-ups",
        current_workaround="copy paste updates and manual reconciliation",
        cost_consequence_clues="hours lost chasing status and fixing missed steps",
        current_tools="excel email slack",
    )
    search_calls = []

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.76,
            "recurrence_state": "strong",
            "query_coverage": 0.8,
            "doc_count": 4,
            "domain_count": 2,
            "results_by_query": {"spreadsheet handoff confusion": 4},
            "results_by_source": {"reddit": 2, "web": 2},
            "matched_results_by_source": {"reddit": 2, "web": 2},
            "partial_results_by_source": {"reddit": 0, "web": 0},
            "family_confirmation_count": 2,
            "source_yield": {
                "reddit": {"attempts": 1, "docs_retrieved": 2, "docs_strong_match": 2, "docs_partial_match": 0, "confirmed": True},
                "web": {"attempts": 1, "docs_retrieved": 2, "docs_strong_match": 2, "docs_partial_match": 0, "confirmed": True},
            },
            "reshaped_query_history": [],
            "queries_considered": ["spreadsheet handoff confusion"],
            "queries_executed": ["spreadsheet handoff confusion"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {
                "near_miss_enrichment_action": "GATHER_CORROBORATION",
                "promotion_gap_class": "corroboration_gap",
                "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
            },
            "last_action": "GATHER_CORROBORATION",
            "last_transition_reason": "highest_information_gain:community_operator_fit",
            "chosen_family": "web",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {"reddit": {"attempts": 1}, "web": {"attempts": 1}},
            "skipped_families": {},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "",
            "decomposed_atom_queries": [],
            "routing_override_reason": "workflow_fragility_surface_first",
            "cohort_query_pack_used": True,
            "cohort_query_pack_name": "workflow_fragility",
            "web_query_strategy_path": ["atom_shaped", "specialized_surface_targeting"],
            "specialized_surface_targeting_used": True,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "",
            "recurrence_failure_class": "confirmed",
            "promotion_gap_class": "corroboration_gap",
            "near_miss_enrichment_action": "GATHER_CORROBORATION",
            "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
        }

    async def fake_search(query, max_results=8, site=None, intent="general"):
        search_calls.append((query, intent))
        if intent == "validation_competitor":
            return []
        if intent == "validation_value_enrichment":
            return [
                SearchDocument(
                    title="Teams outgrow Excel when status chasing becomes a full-time job",
                    url="https://example.com/outgrow-excel",
                    snippet="Operations teams describe hours lost to manual reconciliation, version confusion, and paying for workflow software to escape spreadsheet drag.",
                    source="web",
                )
            ]
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Duct tape spreadsheets keep ops aligned until they break",
            summary="Teams lose hours to spreadsheet version confusion, status chasing, and manual handoffs.",
            finding_kind="pain_point",
            atom=atom,
        )
    )

    assert result["evidence"]["promotion_gap_class"] in {"value_gap", "confirmed"}
    assert result["evidence"]["near_miss_enrichment_action"] == "GATHER_MARKET_ENRICHMENT"
    assert result["evidence"]["value_enrichment_used"] is True
    assert result["evidence"]["value_enrichment_queries"]
    assert result["evidence"]["value_enrichment_docs"]
    assert any(intent == "validation_value_enrichment" for _, intent in search_calls)


def test_validate_problem_runs_value_enrichment_for_supported_single_family_value_gap():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="finance ops",
        user_role="controller",
        job_to_be_done="close the books without export failures",
        failure_mode="month-end export timeouts block close",
        trigger_event="during month-end close",
        current_workaround="manual export retries and spreadsheet cleanup",
        cost_consequence_clues="hours lost and reporting delays",
        current_tools="qbo exports spreadsheets",
    )
    search_calls = []

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.59,
            "recurrence_state": "supported",
            "query_coverage": 0.75,
            "doc_count": 3,
            "domain_count": 1,
            "results_by_query": {"month end export timeout": 3},
            "results_by_source": {"reddit": 3},
            "matched_results_by_source": {"reddit": 3},
            "partial_results_by_source": {"reddit": 0},
            "family_confirmation_count": 1,
            "source_yield": {
                "reddit": {"attempts": 1, "docs_retrieved": 3, "docs_strong_match": 3, "docs_partial_match": 0, "confirmed": True},
                "web": {"attempts": 2, "docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0, "confirmed": False},
            },
            "reshaped_query_history": [],
            "queries_considered": ["month end export timeout"],
            "queries_executed": ["month end export timeout"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {
                "near_miss_enrichment_action": "GATHER_CORROBORATION",
                "promotion_gap_class": "corroboration_gap",
                "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
            },
            "last_action": "GATHER_CORROBORATION",
            "last_transition_reason": "highest_information_gain:community_operator_fit",
            "chosen_family": "reddit",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {"reddit": {"attempts": 1}},
            "skipped_families": {},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "",
            "decomposed_atom_queries": [],
            "routing_override_reason": "",
            "cohort_query_pack_used": False,
            "cohort_query_pack_name": "",
            "web_query_strategy_path": [],
            "specialized_surface_targeting_used": False,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "single_source_confirmation_only",
            "recurrence_failure_class": "single_source_only",
            "promotion_gap_class": "corroboration_gap",
            "near_miss_enrichment_action": "GATHER_CORROBORATION",
            "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
        }

    async def fake_search(query, max_results=8, site=None, intent="general"):
        search_calls.append((query, intent))
        if intent == "validation_competitor":
            return []
        if intent == "validation_value_enrichment":
            return [
                SearchDocument(
                    title="Finance teams still lose hours to manual export retries",
                    url="https://example.com/qbo-close",
                    snippet="Controllers describe month-end export retries and spreadsheet cleanup as costly recurring work.",
                    source="web",
                )
            ]
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Month-end close bottlenecks with export timeouts",
            summary="Finance teams keep retrying exports and cleaning spreadsheets manually during close.",
            finding_kind="pain_point",
            atom=atom,
        )
    )

    assert result["evidence"]["promotion_gap_class"] in {"value_gap", "confirmed"}
    assert result["evidence"]["near_miss_enrichment_action"] == "GATHER_MARKET_ENRICHMENT"
    assert result["evidence"]["value_enrichment_used"] is True
    assert result["evidence"]["value_enrichment_queries"]
    assert result["evidence"]["value_enrichment_docs"]
    assert any(intent == "validation_value_enrichment" for _, intent in search_calls)


def test_validate_problem_runs_value_enrichment_even_when_competitor_times_out():
    toolkit = ResearchToolkit(
        {
            "validation": {
                "search": {
                    "competitor_budget_seconds": 0.01,
                }
            }
        }
    )
    atom = SimpleNamespace(
        segment="finance ops",
        user_role="controller",
        job_to_be_done="close the books without export failures",
        failure_mode="month-end export timeouts block close",
        trigger_event="during month-end close",
        current_workaround="manual export retries and spreadsheet cleanup",
        cost_consequence_clues="hours lost and reporting delays",
        current_tools="qbo exports spreadsheets",
    )
    search_calls = []

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.59,
            "recurrence_state": "supported",
            "query_coverage": 0.75,
            "doc_count": 3,
            "domain_count": 1,
            "results_by_query": {"month end export timeout": 3},
            "results_by_source": {"reddit": 3},
            "matched_results_by_source": {"reddit": 3},
            "partial_results_by_source": {"reddit": 0},
            "family_confirmation_count": 1,
            "source_yield": {
                "reddit": {"attempts": 1, "docs_retrieved": 3, "docs_strong_match": 3, "docs_partial_match": 0, "confirmed": True},
                "web": {"attempts": 2, "docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0, "confirmed": False},
            },
            "reshaped_query_history": [],
            "queries_considered": ["month end export timeout"],
            "queries_executed": ["month end export timeout"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {
                "near_miss_enrichment_action": "GATHER_CORROBORATION",
                "promotion_gap_class": "corroboration_gap",
                "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
            },
            "last_action": "GATHER_CORROBORATION",
            "last_transition_reason": "highest_information_gain:community_operator_fit",
            "chosen_family": "reddit",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {"reddit": {"attempts": 1}},
            "skipped_families": {},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "",
            "decomposed_atom_queries": [],
            "routing_override_reason": "",
            "cohort_query_pack_used": False,
            "cohort_query_pack_name": "",
            "web_query_strategy_path": [],
            "specialized_surface_targeting_used": False,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "single_source_confirmation_only",
            "recurrence_failure_class": "single_source_only",
            "promotion_gap_class": "corroboration_gap",
            "near_miss_enrichment_action": "GATHER_CORROBORATION",
            "sufficiency_priority_reason": "single_or_thin_family_support_blocks_selection",
        }

    async def fake_search(query, max_results=8, site=None, intent="general"):
        search_calls.append((query, intent))
        if intent == "validation_competitor":
            await asyncio.sleep(0.05)
            return []
        if intent == "validation_value_enrichment":
            return [
                SearchDocument(
                    title="Finance teams still lose hours to manual export retries",
                    url="https://example.com/qbo-close",
                    snippet="Controllers describe month-end export retries and spreadsheet cleanup as costly recurring work.",
                    source="web",
                )
            ]
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Month-end close bottlenecks with export timeouts",
            summary="Finance teams keep retrying exports and cleaning spreadsheets manually during close.",
            finding_kind="pain_point",
            atom=atom,
        )
    )

    assert result["evidence"]["competitor_timeout"] is True
    assert result["evidence"]["promotion_gap_class"] in {"value_gap", "confirmed"}
    assert result["evidence"]["near_miss_enrichment_action"] == "GATHER_MARKET_ENRICHMENT"
    assert result["evidence"]["value_enrichment_used"] is True
    assert result["evidence"]["value_enrichment_queries"]
    assert result["evidence"]["value_enrichment_docs"]
    assert any(intent == "validation_value_enrichment" for _, intent in search_calls)


def test_workflow_fragility_value_enrichment_queries_include_cost_and_replacement_signals():
    toolkit = ResearchToolkit()
    atom = SimpleNamespace(
        segment="small business operations",
        user_role="operations lead",
        job_to_be_done="keep operations data sync across teams",
        failure_mode="shared spreadsheets get out of sync and no one knows which version is latest",
        trigger_event="during status updates and handoffs",
        current_workaround="copy paste updates between spreadsheets and email",
        cost_consequence_clues="hours lost fixing manual cleanup and missed steps",
        current_tools="excel spreadsheets email whatsapp",
    )

    queries = toolkit._build_value_enrichment_queries(
        title="Duct tape and Excel ops pain",
        summary="Teams lose time to spreadsheet version confusion and manual handoffs.",
        atom=atom,
    )

    assert any("outgrew excel operations hours lost" in query for query in queries)
    assert any("spreadsheet version confusion error cost" in query for query in queries)
    assert any("status chasing manual handoff headcount drag" in query for query in queries)
    assert any("replace spreadsheets workflow too expensive" in query for query in queries)


def test_workflow_fragility_value_enrichment_bonus_counts_cost_and_replacement_signals():
    toolkit = ResearchToolkit()
    docs = [
        SearchDocument(
            title="We outgrew Excel and lost hours every week",
            url="https://example.com/outgrew-excel",
            snippet="Manual reconciliation, status chasing, and headcount drag pushed us to pay for software to replace spreadsheets.",
            source="web",
        )
    ]

    bonus = toolkit._value_enrichment_signal_bonus(docs)

    assert bonus >= 0.09


def test_validate_problem_preserves_recurrence_action_when_top_level_action_empty():
    toolkit = ResearchToolkit()

    async def fake_gather(*_args, **_kwargs):
        return [], {
            "recurrence_score": 0.48,
            "recurrence_state": "thin",
            "query_coverage": 0.6,
            "doc_count": 2,
            "domain_count": 1,
            "results_by_query": {"sla tracking workflow": 2},
            "results_by_source": {"reddit": 2, "web": 0},
            "matched_results_by_source": {"reddit": 1, "web": 0},
            "partial_results_by_source": {"reddit": 1, "web": 0},
            "family_confirmation_count": 1,
            "source_yield": {
                "reddit": {"attempts": 1, "docs_retrieved": 2, "docs_strong_match": 1, "docs_partial_match": 1, "confirmed": True},
                "web": {"attempts": 2, "docs_retrieved": 0, "docs_strong_match": 0, "docs_partial_match": 0, "confirmed": False},
            },
            "reshaped_query_history": [{"source": "web", "attempt": 2, "reason": "cost_missing"}],
            "queries_considered": ["sla tracking workflow"],
            "queries_executed": ["sla tracking workflow"],
            "recurrence_budget_profile": {"remaining_beta": 1},
            "candidate_meaningful": {"meaningful_candidate": True},
            "recurrence_probe_summary": {},
            "recurrence_source_branch": {},
            "last_action": "RETRY_WITH_RESHAPED_QUERY",
            "last_transition_reason": "zero_retrieval_strategy_switch",
            "chosen_family": "web",
            "expected_gain_class": "medium",
            "source_attempts_snapshot": {"web": {"attempts": 2}},
            "skipped_families": {},
            "controller_actions": [],
            "budget_snapshot": {"remaining_beta": 1},
            "fallback_strategy_used": "decomposed_query_switch",
            "decomposed_atom_queries": ["sla tracking workflow"],
            "routing_override_reason": "",
            "cohort_query_pack_used": False,
            "cohort_query_pack_name": "",
            "web_query_strategy_path": ["atom_shaped", "decomposition"],
            "specialized_surface_targeting_used": False,
            "warmed_validation_queries": {},
            "recurrence_gap_reason": "single_source_confirmation_only",
            "recurrence_failure_class": "single_source_only",
            "promotion_gap_class": "corroboration_gap",
            "near_miss_enrichment_action": "GATHER_CORROBORATION",
            "sufficiency_priority_reason": "partial_match_family_can_raise_sufficiency",
        }

    async def fake_search(*_args, **_kwargs):
        return []

    toolkit.gather_recurrence_evidence = fake_gather
    toolkit.search_web = fake_search

    result = asyncio.run(
        toolkit.validate_problem(
            title="Manual SLA tracking still breaks follow-up workflows",
            summary="Revenue ops teams still rely on reminders and spreadsheet checks.",
            finding_kind="pain_point",
        )
    )

    assert result["evidence"]["promotion_gap_class"] == "corroboration_gap"
    assert result["evidence"]["near_miss_enrichment_action"] == "GATHER_CORROBORATION"
    assert result["evidence"]["sufficiency_priority_reason"] == "partial_match_family_can_raise_sufficiency"


def test_gather_recurrence_evidence_records_yield_fields(monkeypatch):
    toolkit = ResearchToolkit({"validation": {"search": {"recurrence_results": 6}}})
    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        return [
            SearchDocument(
                title="Operators still do manual spreadsheet cleanup",
                url=f"https://reddit.com/r/{subreddit}/comments/abc",
                snippet="Manual csv cleanup is still common after spreadsheet imports break.",
                source=f"reddit/{subreddit}",
            )
        ]

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        if site is None:
            return [
                SearchDocument(
                    title="Spreadsheet cleanup workflow still manual",
                    url="https://ops.example.com/manual-spreadsheet-cleanup",
                    snippet="Teams still rely on manual csv cleanup after spreadsheet imports break.",
                    source="web",
                )
            ]
        return []

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    _docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"manual data entry" "spreadsheets manual work email"'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert "source_yield" in meta
    assert "matched_results_by_source" in meta
    assert "partial_results_by_source" in meta
    assert "matched_docs_by_source" in meta
    assert "partial_docs_by_source" in meta
    assert "family_confirmation_count" in meta
    assert "reshaped_query_history" in meta
    assert meta["source_yield"]["web"]["docs_retrieved"] >= 1
    assert meta["source_yield"]["web"]["docs_strong_match"] >= 0
    assert meta["matched_docs_by_source"]["web"]
    assert len(meta["matched_docs_by_source"]["web"]) >= min(
        meta["matched_results_by_source"]["web"],
        5,
    )
    first_match = meta["matched_docs_by_source"]["web"][0]
    assert first_match["source_family"] == "web"
    assert first_match["match_class"] == "strong"
    assert first_match["normalized_url"] == "https://ops.example.com/manual-spreadsheet-cleanup"
    assert first_match["query_text"]
    assert first_match["title"] == "Spreadsheet cleanup workflow still manual"
    assert meta["recurrence_failure_class"] in {"single_source_only", "partial_confirmation_only", "confirmed"}


def test_retrieved_docs_without_strong_matches_do_not_look_confirmed(monkeypatch):
    toolkit = ResearchToolkit()

    async def fake_reddit_search(subreddit, query, limit=2, sort="relevance"):
        return []

    async def fake_search_web(query, max_results=6, site=None, intent=""):
        return [
            SearchDocument(
                title="Theme settings bug report",
                url="https://github.com/example/repo/issues/1" if site == "github.com" else "https://forum.example.com/theme-bug",
                snippet="Theme colors and plugin settings are broken after update.",
                source="github" if site == "github.com" else "web",
            )
        ]

    monkeypatch.setattr(toolkit, "reddit_search", fake_reddit_search)
    monkeypatch.setattr(toolkit, "search_web", fake_search_web)

    atom = SimpleNamespace(
        segment="small business operators",
        user_role="operations lead",
        job_to_be_done="keep operations data in sync without manual cleanup",
        failure_mode="spreadsheet imports arrive with duplicates and broken formats",
        trigger_event="after vendor spreadsheets arrive",
        current_workaround="manual csv cleanup in excel",
        cost_consequence_clues="downtime risk",
        current_tools="excel google sheets csv import",
    )

    _docs, meta = asyncio.run(
        toolkit.gather_recurrence_evidence(
            ['"manual data entry" "spreadsheets manual work email"'],
            finding_kind="problem_signal",
            atom=atom,
        )
    )

    assert meta["source_yield"]["web"]["docs_retrieved"] >= 1
    assert meta["source_yield"]["web"]["docs_strong_match"] == 0
    assert meta["family_confirmation_count"] == 0
    assert meta["recurrence_state"] in {"weak", "thin"}
    assert meta["recurrence_failure_class"] in {"no_corroboration_found", "breadth_limited"}
