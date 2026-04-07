"""Explicit source adapters for GitHub issue and discussion discovery lanes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import aiohttp

GITHUB_BOILERPLATE_PHRASES = [
    "write this issue in english",
    "searched existing issues",
    "searched the existing issues",
    "doctor output",
    "no response",
    "translated by claude",
    "issue checklist",
    "is there an existing issue",
    "my issue is not listed",
    "looked at pinned issues",
    "filled in short, clear headings",
    "confirmed that i am using the latest",
    "this is a question",
    "not a bug report or feature request",
    "context optional",
    "couldn't find the answer",
    "question this is my configuration",
    "and not a suggestion",
    "i understand that issues are for feedback and problem solving",
    "not for complaining in the comment section",
    "provide as much information as possible",
]


def _compact(text: str, limit: int = 2000) -> str:
    return " ".join((text or "").split())[:limit]


def _clean_text(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\|[-: ]+\|", " ", cleaned)
    cleaned = re.sub(r"\|", " ", cleaned)
    cleaned = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[\w./-]*support/logs[\w./-]*\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bworkflow run\b[:\s#-]*\d+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcommit\b[:\s#-]*[a-f0-9]{7,40}", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcontact\s*:\s*\S+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(issue checklist|logs stored in repo|logs stored|expected behavior|actual behavior)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", _compact(cleaned, 4000))
    return cleaned.strip()


def _strip_issue_boilerplate(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r">\s*\[!note\]\s*this issue was translated by claude\.?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"prerequisites\s*(?:\[[ xX]\]\s*[^.]{0,180}){1,6}",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bissue checklist\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bis there an existing issue(?: for this)?\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy issue is not listed(?: in the)?\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi've looked at \*?\*?pinned issues\*?\*?[^.]{0,200}\.", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi've filled in short, clear headings[^.]{0,220}\.", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi've confirmed that i am using the latest[^.]{0,120}\.", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdoctor output(?:\s*\(optional\))?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\badditional information\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bquestion category\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bthis is a question\s*\(not a bug report or feature request\)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcontext\s*\(optional\)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcould(?: not|n't) find the answer\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bquestion\s+this is my configuration\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band not \"a suggestion\", \"stuck\", etc\.?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcurrent behavior\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbug description\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsteps to reproduce\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\banything else\??\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bplatform\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bversion\b[:\s-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b_no response_\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"i understand that issues are for feedback and problem solving, not for complaining in the comment section,? and will provide as much information as possible to help solve the problem\.?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[[ xX]\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [part.strip() for part in parts if part.strip()]


def _pick_sentence(
    text: str,
    terms: list[str],
    *,
    fallback: str = "",
    limit: int = 220,
    allow_first_sentence_fallback: bool = False,
) -> str:
    lowered_terms = [term.lower() for term in terms]
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if "prerequisites" in lowered or any(noise in lowered for noise in GITHUB_BOILERPLATE_PHRASES):
            continue
        if any(term in lowered for term in lowered_terms):
            return _compact(sentence, limit)
    if allow_first_sentence_fallback:
        sentences = _sentences(text)
        if sentences:
            return _compact(sentences[0], limit)
    return _compact(fallback or "", limit)


def _extract_repo_and_kind(url: str) -> tuple[str, str, str]:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) < 4:
        return "", "", ""
    repo = "/".join(path_parts[:2])
    item_type = path_parts[2]
    item_number = path_parts[3]
    return repo, item_type, item_number


def _resolve_env(value: str | None) -> str:
    raw = (value or "").strip()
    match = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", raw)
    if match:
        return os.getenv(match.group(1), "")
    return raw


@dataclass
class GitHubIssueRecord:
    repository: str
    item_type: str
    item_number: str
    title: str
    url: str
    issue_text: str
    snippet: str
    trigger: str
    failure_mode: str
    workaround: str
    reproduction_context: str
    cost_friction: str
    discovery_query: str

    def as_finding(self) -> dict[str, Any]:
        evidence = {
            "source_plan": "github-issues-discussions",
            "discovery_query": self.discovery_query,
            "snippet": self.snippet,
            "page_excerpt": self.issue_text,
            "github_metadata": {
                "repository": self.repository,
                "item_type": self.item_type,
                "item_number": self.item_number,
                "trigger": self.trigger,
                "failure_mode": self.failure_mode,
                "workaround": self.workaround,
                "reproduction_context": self.reproduction_context,
                "cost_friction": self.cost_friction,
            },
        }
        summary_parts = [
            self.failure_mode,
            self.trigger,
            self.reproduction_context,
            self.workaround,
            self.cost_friction,
        ]
        outcome_summary = ". ".join(part for part in summary_parts if part) or _compact(self.issue_text, 420)
        return {
            "source": f"github-{self.item_type}/{self.repository}",
            "source_url": self.url,
            "entrepreneur": "GitHub issue or discussion",
            "tool_used": "",
            "product_built": self.title,
            "monetization_method": "",
            "outcome_summary": _compact(outcome_summary, 420),
            "finding_kind": "problem_signal",
            "evidence": evidence,
        }


class GitHubIssueAdapter:
    """Collects explicit GitHub issue/discussion findings from search + page fetch hooks."""

    def __init__(
        self,
        *,
        search_web: Callable[..., Awaitable[list[Any]]],
        fetch_content: Callable[[str], Awaitable[dict[str, Any]]],
        token: str = "",
        user_agent: str = "AutoResearcher/1.0",
    ):
        self.search_web = search_web
        self.fetch_content = fetch_content
        self.token = _resolve_env(token)
        self.user_agent = user_agent
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=20)
            headers = {"User-Agent": self.user_agent, "Accept": "application/vnd.github+json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def discover_items(
        self,
        *,
        queries: list[str],
        max_results_per_query: int = 4,
        observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        session = self._get_session()
        for query in queries:
            api_items = await self._search_issue_api(session, query, per_page=max_results_per_query)
            if observer:
                observer(
                    {
                        "source_name": "github-problem",
                        "query_text": query,
                        "docs_seen": len(api_items),
                        "status": "ok",
                        "error": "",
                    }
                )
            for item in api_items:
                html_url = str(item.get("html_url", "") or "")
                if not html_url or html_url in seen_urls:
                    continue
                seen_urls.add(html_url)
                record = self._build_issue_record_from_api(item, query)
                if record:
                    findings.append(record.as_finding())

            docs = await self.search_web(query, max_results=max_results_per_query, site="github.com")
            if observer:
                observer(
                    {
                        "source_name": "github-problem",
                        "query_text": f"{query} [discussion-fallback]",
                        "docs_seen": len(docs),
                        "status": "ok",
                        "error": "",
                    }
                )
            for doc in docs:
                if doc.url in seen_urls or not any(token in doc.url for token in ["/issues/", "/discussions/"]):
                    continue
                repo, item_type, item_number = _extract_repo_and_kind(doc.url)
                if item_type not in {"issues", "discussions"} or not repo:
                    continue
                seen_urls.add(doc.url)
                content = await self.fetch_content(doc.url)
                record = self._build_record_from_doc(doc, content, query, repo, item_type, item_number)
                if record:
                    findings.append(record.as_finding())
        return findings

    async def search_issue_records(
        self,
        query: str,
        *,
        max_results: int = 4,
    ) -> list[dict[str, Any]]:
        session = self._get_session()
        api_items = await self._search_issue_api(session, query, per_page=max_results)
        findings: list[dict[str, Any]] = []
        for item in api_items:
            record = self._build_issue_record_from_api(item, query)
            if record:
                findings.append(record.as_finding())
        return findings

    async def _search_issue_api(
        self,
        session: aiohttp.ClientSession,
        query: str,
        *,
        per_page: int,
    ) -> list[dict[str, Any]]:
        api_query = f'{query} is:issue archived:false'
        async with session.get(
            "https://api.github.com/search/issues",
            params={"q": api_query, "per_page": min(max(per_page, 1), 10)},
        ) as response:
            if response.status >= 400:
                return []
            payload = await response.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        return [item for item in items if isinstance(item, dict) and not item.get("pull_request")]

    def _build_issue_record_from_api(self, item: dict[str, Any], query: str) -> Optional[GitHubIssueRecord]:
        html_url = str(item.get("html_url", "") or "")
        repo, item_type, item_number = _extract_repo_and_kind(html_url)
        if item_type != "issues" or not repo:
            return None
        body_text = _strip_issue_boilerplate(_clean_text(str(item.get("body", "") or "")))
        title = _compact(str(item.get("title", "") or ""), 240)
        text = _clean_text(f"{title} {body_text}")
        if not text:
            return None
        return GitHubIssueRecord(
            repository=repo,
            item_type="issue",
            item_number=item_number,
            title=title,
            url=html_url,
            issue_text=_compact(text, 1800),
            snippet=_compact(str(item.get("body", "") or ""), 240),
            trigger=_pick_sentence(body_text, ["when", "after", "during", "on ", "while ", "once ", "whenever"]),
            failure_mode=_pick_sentence(body_text, ["fail", "fails", "failing", "broken", "breaks", "error", "does not", "doesn't", "cannot", "can't", "stuck", "stops", "stopped", "disappeared"]),
            workaround=_pick_sentence(body_text, ["workaround", "manually", "manual", "temporary", "manual fallback", "copy", "custom script", "manual retry"]),
            reproduction_context=_pick_sentence(body_text, ["steps to reproduce", "reproduce", "every time", "always", "happens when", "triggered by"]),
            cost_friction=_pick_sentence(body_text, ["time", "slow", "hours", "blocked", "friction", "pain", "frustrating", "overhead", "risk"]),
            discovery_query=query,
        )

    def _build_record_from_doc(
        self,
        doc: Any,
        content: dict[str, Any],
        query: str,
        repo: str,
        item_type: str,
        item_number: str,
    ) -> Optional[GitHubIssueRecord]:
        body_text = _strip_issue_boilerplate(
            _clean_text(
            " ".join(
                part
                for part in [
                    content.get("description", ""),
                    content.get("text", ""),
                ]
                if part
            )
            )
        )
        text = _clean_text(
            " ".join(
                part
                for part in [
                    doc.title,
                    doc.snippet,
                    body_text,
                ]
                if part
            )
        )
        if not text:
            return None
        return GitHubIssueRecord(
            repository=repo,
            item_type="issue" if item_type == "issues" else "discussion",
            item_number=item_number,
            title=_compact(doc.title, 240),
            url=doc.url,
            issue_text=_compact(text, 1800),
            snippet=_compact(doc.snippet, 240),
            trigger=_pick_sentence(body_text, ["when", "after", "during", "on ", "while ", "once ", "whenever"]),
            failure_mode=_pick_sentence(body_text, ["fail", "fails", "failing", "broken", "breaks", "error", "does not", "doesn't", "cannot", "can't", "stuck", "stops", "stopped", "disappeared"]),
            workaround=_pick_sentence(body_text, ["workaround", "manually", "manual", "temporary", "manual fallback", "copy", "custom script", "manual retry"]),
            reproduction_context=_pick_sentence(body_text, ["steps to reproduce", "reproduce", "every time", "always", "happens when", "triggered by"]),
            cost_friction=_pick_sentence(body_text, ["time", "slow", "hours", "blocked", "friction", "pain", "frustrating", "overhead", "risk"]),
            discovery_query=query,
        )
