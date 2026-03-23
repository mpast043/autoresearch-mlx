"""Tests for explicit GitHub issue/discussion source adapter."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from github_sources import GitHubIssueAdapter
from research_tools import SearchDocument


def test_github_adapter_extracts_structured_issue_fields():
    async def fake_search_web(query, max_results=4, site=None):
        return [
            SearchDocument(
                title="Issue: backup restore fails after migration",
                url="https://github.com/org/repo/issues/123",
                snippet="Restore fails every time after migration and we keep rerunning it manually.",
                source="github.com",
            )
        ]

    async def fake_fetch_content(url):
        return {
            "description": "Steps to reproduce: migrate site, trigger restore, restore fails every time.",
            "text": (
                "After migration, restore fails every time. "
                "We manually rerun recovery and keep backup copies. "
                "This blocks site cutovers for hours."
            ),
        }

    adapter = GitHubIssueAdapter(search_web=fake_search_web, fetch_content=fake_fetch_content)
    async def fake_search_issue_api(session, query, per_page):
        return [
            {
                "html_url": "https://github.com/org/repo/issues/123",
                "title": "Issue: backup restore fails after migration",
                "body": (
                    "Steps to reproduce: migrate site, trigger restore. "
                    "After migration, restore fails every time. "
                    "We manually rerun recovery and keep backup copies. "
                    "This blocks site cutovers for hours."
                ),
            }
        ]
    adapter._search_issue_api = fake_search_issue_api
    findings = asyncio.run(adapter.discover_items(queries=['"manual workaround" issue']))

    assert len(findings) == 1
    finding = findings[0]
    meta = finding["evidence"]["github_metadata"]
    assert finding["source"] == "github-issue/org/repo"
    assert finding["tool_used"] == ""
    assert meta["repository"] == "org/repo"
    assert meta["item_type"] == "issue"
    assert "restore fails every time" in meta["failure_mode"].lower()
    assert "manually rerun recovery" in meta["workaround"].lower()
    assert "migration" in meta["reproduction_context"].lower()
    assert "restore fails" in meta["reproduction_context"].lower()


def test_github_adapter_resolves_placeholder_token_to_empty(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    async def fake_search_web(query, max_results=4, site=None):
        return []

    async def fake_fetch_content(url):
        return {}

    adapter = GitHubIssueAdapter(
        search_web=fake_search_web,
        fetch_content=fake_fetch_content,
        token="${GITHUB_TOKEN}",
    )

    assert adapter.token == ""
