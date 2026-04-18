"""
DeepResearchAgent — multi-source weak-signal synthesis agent.

Fetches raw pain signals from Reddit, GitHub, and web search simultaneously,
extracts structured problem atoms, and creates finding records ready for
the ValidationAgent gate.

This is the "heavy research" counterpart to DiscoveryAgent's lightweight
seeded crawling. It is invoked:
  1. Manually via:  python cli.py run-deep-research --vertical <name>
  2. Programmatically when the pipeline needs to dig deeper into a niche
     that seeded discovery could not cover.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.database import Database, Finding, ProblemAtom, RawSignal
from src.messaging import MessageQueue, MessageType

logger = logging.getLogger(__name__)


# ── Vertical pain-signal keywords ────────────────────────────────────────────

VERTICAL_KEYWORDS: dict[str, dict[str, Any]] = {
    "devtools": {
        "reddit_subreddits": ["devops", "DevOps", "programming", "SoftwareEngineering",
                              "AskProgramming", "webdev", "typescript"],
        "reddit_queries": [
            "devtools migration pain points",
            "DX frustration automation tooling",
            "build system migration problems",
            "CLI tool problems developers hate",
            "infrastructure as code frustration",
        ],
        "github_queries": [
            "migration toil OR upgrade pain OR broken DX OR poor documentation",
            "deprecated OR forced migration OR breaking changes",
        ],
        "web_queries": [
            "developer tools frustrations pain points 2024",
            "devtools migration problems",
            "IDE extension frustration survey",
        ],
    },
    "ecommerce": {
        "reddit_subreddits": ["ecommerce", "shopify", "smallbusiness", "Entrepreneur"],
        "reddit_queries": [
            "ecommerce analytics pain points",
            "shopify app frustration",
            "inventory management problems small business",
            "payment processor problems",
        ],
        "github_queries": [
            "ecommerce platform problems OR migration OR API limitations",
        ],
        "web_queries": [
            "ecommerce seller pain points 2024",
            "shopify app problems reviews",
        ],
    },
}


# ── Per-source result types ───────────────────────────────────────────────────

@dataclass
class PainSignal:
    title: str
    url: str
    source: str          # "reddit" | "github" | "web"
    subreddit: Optional[str] = None
    author: Optional[str] = None
    score: int = 0
    num_comments: int = 0
    timestamp: Optional[str] = None
    body_excerpt: Optional[str] = None


# ── Agent ────────────────────────────────────────────────────────────────────

class DeepResearchAgent(BaseAgent):
    """
    Conducts deep, multi-source research into a named vertical.

    Produces :class:`Finding` records (source='deep_research') that are
    dispatched directly to the ValidationAgent via
    ``MessageType.FINDING_UNSEEDED``.
    """

    def __init__(
        self,
        name: str,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        vertical: str = "devtools",
    ):
        super().__init__(name, message_queue)
        self.db = db
        self.vertical = vertical
        self.keywords = VERTICAL_KEYWORDS.get(vertical, VERTICAL_KEYWORDS["devtools"])

    def _get_tavily_key(self) -> Optional[str]:
        import os
        return os.getenv("TAVILY_API_KEY")

    # ── Lightweight text extraction ────────────────────────────────────────────

    _PAIN_KEYWORDS = (
        "hate", "frustrat", "annoying", "broken", "can't", "cannot", "unable",
        "struggle", "painful", "waste", "slow", "manual", "tedious", "error",
        "bug", "fail", "crash", "missing", "wrong", "doesn't work", "doesn't support",
        "no way to", "impossible", "ridiculous", "horrible", "terrible",
        "have to", "forced to", "workaround", "hack",
    )

    def _extract_failure_mode(self, text: str) -> str:
        """Extract the most failure-like sentence from the body text."""
        sentences = [s.strip() for s in text.replace("\n", ".").split(".") if len(s.strip()) > 20]
        if not sentences:
            return ""
        best = ""
        best_count = 0
        for s in sentences[:20]:
            count = sum(1 for kw in self._PAIN_KEYWORDS if kw in s.lower())
            if count > best_count:
                best_count = count
                best = s
        return best[:300]

    def _extract_pain_statement(self, text: str) -> str:
        """Extract the most pain-relevant text from the body."""
        # Prefer failure mode if found, otherwise first meaningful sentence
        failure = self._extract_failure_mode(text)
        if failure:
            return failure
        sentences = [s.strip() for s in text.replace("\n", ".").split(".") if len(s.strip()) > 15]
        return sentences[0][:500] if sentences else ""

    # ── BaseAgent abstract method ─────────────────────────────────────────────

    async def process(self, message) -> dict[str, Any]:
        """Process incoming messages — dispatch to deep research if requested."""
        payload = message.payload if hasattr(message, 'payload') else {}
        vertical = payload.get("vertical", self.vertical)
        if vertical != self.vertical:
            self.vertical = vertical
            self.keywords = VERTICAL_KEYWORDS.get(vertical, VERTICAL_KEYWORDS["devtools"])
        result = await self.run_deep_research()
        return {"status": "completed", "vertical": vertical, "result": result}

    # ── Public entry point ──────────────────────────────────────────────────

    async def run_deep_research(
        self,
        max_signals_per_source: int = 15,
        output_dir: str = "outputs/deep_research",
    ) -> dict[str, Any]:
        """
        Run full deep-research loop for the configured vertical.

        Returns a summary dict with counts of signals found and findings created.
        """
        from pathlib import Path
        import json

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"dr_{self.vertical}_{ts}"

        logger.info("[DeepResearch] starting run_id=%s vertical=%s", run_id, self.vertical)
        logger.info(f"deep_research_start vertical={self.vertical}")

        # 1) Gather from all sources in parallel
        t0 = time.time()
        reddit_signals, github_signals, web_signals = await asyncio.gather(
            self._search_reddit(max_signals_per_source),
            self._search_github(max_signals_per_source),
            self._search_web(max_signals_per_source),
        )
        logger.info(
            "[DeepResearch] signals gathered — reddit=%d github=%d web=%d (%.1fs)",
            len(reddit_signals), len(github_signals), len(web_signals),
            time.time() - t0,
        )

        # 2) Deduplicate across sources
        all_signals = reddit_signals + github_signals + web_signals
        deduped = self._deduplicate(all_signals)
        logger.info("[DeepResearch] after dedup=%d", len(deduped))

        # 3) Score and filter
        scored = self._score_signals(deduped)
        filtered = [s for s in scored if s.score >= 0]
        logger.info("[DeepResearch] after scoring/filtering=%d", len(filtered))

        # 4) Persist as RawSignals + create Findings
        finding_ids = await self._persist_signals_and_findings(filtered, run_id)

        # 5) Dispatch findings directly to ValidationAgent
        dispatched = await self._dispatch_to_validation(finding_ids)

        elapsed = time.time() - t0
        summary = {
            "run_id": run_id,
            "vertical": self.vertical,
            "reddit_signals": len(reddit_signals),
            "github_signals": len(github_signals),
            "web_signals": len(web_signals),
            "after_dedup": len(deduped),
            "after_filter": len(filtered),
            "findings_created": len(finding_ids),
            "dispatched_to_validation": dispatched,
            "elapsed_s": round(elapsed, 1),
        }

        # Save run report
        report_path = out_path / f"{run_id}_report.json"
        report_path.write_text(json.dumps(summary, indent=2))
        logger.info("[DeepResearch] report saved to %s", report_path)
        logger.info(
            f"deep_research_complete findings={len(finding_ids)} "
            f"dispatched={dispatched} elapsed={elapsed:.1f}s"
        )

        return summary

    # ── Source fetchers ─────────────────────────────────────────────────────

    async def _search_reddit(self, limit: int) -> list[PainSignal]:
        """Search Reddit via ddgs (site:reddit.com format), then fetch thread content."""
        import ddgs

        signals: list[PainSignal] = []
        for query in self.keywords.get("reddit_queries", []):
            try:
                with ddgs.DDGS() as ddg:
                    results = list(ddg.text(f"site:reddit.com {query}", max_results=limit))
                for item in results:
                    signals.append(PainSignal(
                        title=item.get("title", "Untitled") or "Untitled",
                        url=item.get("href", "") or item.get("url", "") or "",
                        source="reddit",
                        subreddit=None,
                        score=0,
                        num_comments=0,
                        body_excerpt=item.get("description"),
                    ))
            except Exception as exc:
                logger.warning("[DeepResearch] reddit query '%s' failed: %s", query, exc)

        # Enrich top Reddit signals with actual thread content so atoms have
        # real pain language instead of empty body_excerpts.
        signals = await self._enrich_reddit_threads(signals)
        return signals

    async def _enrich_reddit_threads(
        self, signals: list[PainSignal], max_enrich: int = 20
    ) -> list[PainSignal]:
        """Fetch actual thread content for Reddit signals that lack body_excerpt.

        Uses the public JSON API (no auth required) to get post selftext
        and top comments. Caps at max_enrich to avoid rate-limiting.
        """
        import aiohttp

        to_enrich = [
            s for s in signals
            if s.source == "reddit" and s.url and not s.body_excerpt
        ][:max_enrich]
        if not to_enrich:
            return signals

        enriched_count = 0
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            for sig in to_enrich:
                # Convert reddit.com URL to .json endpoint
                url = sig.url.rstrip("/")
                if not url.startswith("http"):
                    continue
                json_url = url + ".json"
                try:
                    async with session.get(
                        json_url,
                        headers={"User-Agent": "autoresearch-deep-research/1.0"},
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json(content_type=None)
                        # Extract selftext + top comments
                        parts: list[str] = []
                        if isinstance(data, list) and len(data) >= 1:
                            post = data[0].get("data", {}).get("children", [])
                            if post:
                                selftext = post[0].get("data", {}).get("selftext", "")
                                if selftext:
                                    parts.append(selftext[:1000])
                            # Top comments
                            if len(data) >= 2:
                                comments = data[1].get("data", {}).get("children", [])
                                for c in comments[:5]:
                                    body = c.get("data", {}).get("body", "")
                                    if body and len(body) > 20:
                                        parts.append(body[:500])
                        if parts:
                            sig.body_excerpt = "\n".join(parts)[:2000]
                            enriched_count += 1
                except Exception:
                    continue

        if enriched_count:
            logger.info("[DeepResearch] enriched %d Reddit threads with body content", enriched_count)
        return signals

    async def _search_github(self, limit: int) -> list[PainSignal]:
        """Search GitHub issues via gh CLI (same pattern as unseeded_loop.py)."""
        import json, subprocess

        signals: list[PainSignal] = []
        for query in self.keywords.get("github_queries", []):
            try:
                result = subprocess.run(
                    ["gh", "search", "issues", query, "--limit", str(limit),
                     "--json", "title,body,url,createdAt,state,comments"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    for issue in json.loads(result.stdout):
                        signals.append(PainSignal(
                            title=issue.get("title", "Untitled") or "Untitled",
                            url=issue.get("url", "") or "",
                            source="github",
                            score=0,
                            num_comments=len(issue.get("comments", []) or []),
                            timestamp=issue.get("createdAt"),
                            body_excerpt=(issue.get("body") or "")[:500],
                        ))
            except FileNotFoundError:
                logger.warning("[DeepResearch] gh CLI not installed — skipping GitHub search")
            except Exception as exc:
                logger.warning("[DeepResearch] github query '%s' failed: %s", query, exc)
        return signals

    async def _search_web(self, limit: int) -> list[PainSignal]:
        """Broad web search — tries Tavily API first, falls back to ddgs."""
        import ddgs

        signals: list[PainSignal] = []
        for query in self.keywords.get("web_queries", []):
            # Try Tavily first
            api_key = self._get_tavily_key()
            if api_key:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            "https://api.tavily.com/search",
                            json={"query": query, "max_results": limit},
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=15.0,
                        )
                        r.raise_for_status()
                        for item in r.json().get("results", []):
                            signals.append(PainSignal(
                                title=item.get("title", "Untitled") or "Untitled",
                                url=item.get("href", "") or item.get("url", "") or "",
                                source="web",
                                timestamp=item.get("published"),
                                body_excerpt=item.get("content", None),
                            ))
                    continue  # skip ddgs if Tavily succeeded
                except Exception:
                    pass  # fall through to ddgs
            # ddgs fallback
            try:
                with ddgs.DDGS() as ddg:
                    results = list(ddg.text(query, max_results=limit))
                for item in results:
                    signals.append(PainSignal(
                        title=item.get("title", "Untitled") or "Untitled",
                        url=item.get("href", "") or item.get("url", "") or "",
                        source="web",
                        timestamp=item.get("published"),
                        body_excerpt=item.get("description"),
                    ))
            except Exception as exc:
                logger.warning("[DeepResearch] web query '%s' failed: %s", query, exc)
        return signals

    # ── Deduplication ───────────────────────────────────────────────────────

    def _deduplicate(self, signals: list[PainSignal]) -> list[PainSignal]:
        """Drop signals with duplicate URLs within each source, keeping highest-scoring."""
        # Dedupe per-source to avoid Reddit/Web having the same URL collapsing everything
        by_source: dict[str, dict[str, PainSignal]] = {}
        for sig in signals:
            key = sig.url.split("?")[0]
            if sig.source not in by_source:
                by_source[sig.source] = {}
            if key not in by_source[sig.source] or sig.score > by_source[sig.source][key].score:
                by_source[sig.source][key] = sig
        result = []
        for source_dict in by_source.values():
            result.extend(source_dict.values())
        return result

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_signals(self, signals: list[PainSignal]) -> list[PainSignal]:
        """
        Lightweight heuristic scoring:
        - reddit score > 50   → +2
        - has body/excerpt     → +1
        - github issues (often actionable) → +1
        - web (noisy)          → -1
        """
        for sig in signals:
            score = 0
            if sig.source == "reddit":
                if sig.score and sig.score > 50:
                    score += 2
                if sig.num_comments and sig.num_comments > 10:
                    score += 1
            if sig.source == "github":
                score += 1
            if sig.body_excerpt:
                score += 1
            if sig.source == "web":
                score -= 1  # web is noisier
            sig.score = max(0, score)
        return signals

    # ── Persistence ────────────────────────────────────────────────────────

    async def _persist_signals_and_findings(
        self, signals: list[PainSignal], run_id: str
    ) -> int:
        """Write signals to DB, cluster into findings, return count created."""
        from src.database import RawSignal

        finding_ids: list[int] = []
        for sig in signals:
            if not sig.url:
                continue

            # Create or update finding first (need the ID for signal + atom)
            content_hash = hashlib.sha256(sig.url.encode()).hexdigest()[:16]
            finding_id = self._upsert_finding(sig, run_id, content_hash)
            if not finding_id:
                continue
            finding_ids.append(finding_id)

            # Persist raw signal linked to finding
            signal = RawSignal(
                finding_id=finding_id,
                source_name=sig.source,
                source_type=sig.source,
                source_url=sig.url,
                title=sig.title,
                body_excerpt=sig.body_excerpt or "",
                content_hash=content_hash,
                source_class="pain_signal",
                metadata={
                    "run_id": run_id,
                    "vertical": self.vertical,
                    "subreddit": sig.subreddit,
                    "reddit_score": sig.score,
                    "num_comments": sig.num_comments,
                },
            )
            try:
                signal_id = self.db.insert_raw_signal(signal)
            except Exception:
                signal_id = None

            # Create a problem atom so the finding is actionable by the pipeline.
            # Extract structured fields from the signal content rather than dumping
            # the raw body — the screening gate requires real pain language, failure
            # modes, and workarounds to classify a finding as pain_signal.
            if signal_id and not self.db.get_problem_atoms_by_finding(finding_id):
                body_text = (sig.body_excerpt or sig.title or "")
                title_text = (sig.title or "")
                atom = ProblemAtom(
                    finding_id=finding_id,
                    raw_signal_id=int(signal_id) if signal_id else 0,
                    signal_id=int(signal_id) if signal_id else 0,
                    segment=sig.source or "",
                    user_role="",
                    job_to_be_done=title_text[:200],
                    trigger_event="",
                    failure_mode=self._extract_failure_mode(body_text) or title_text[:200],
                    current_workaround="",
                    current_tools="",
                    pain_statement=self._extract_pain_statement(body_text) or body_text[:500],
                    source_quote=body_text[:300],
                    confidence=0.5,
                    confidence_score=0.5,
                    metadata={
                        "run_id": run_id,
                        "vertical": self.vertical,
                        "auto_created": True,
                    },
                )
                try:
                    self.db.insert_problem_atom(atom)
                except Exception:
                    pass

        return finding_ids

    def _upsert_finding(self, sig: PainSignal, run_id: str, content_hash: str) -> Optional[int]:
        """Create a finding, raw signal, and problem atom from a pain signal.

        Without the signal and atom records, the finding cannot enter the
        backlog workbench or produce useful recurrence queries.  The deep-
        research agent extracts enough structure from the signal to build a
        heuristic atom.
        """
        # Check if URL already has a finding
        existing = self.db.get_finding_by_url(sig.url)
        if existing is not None:
            return int(existing.id) if existing.id else None

        source_label = sig.source or "web"
        finding = Finding(
            source="deep_research",
            source_url=sig.url,
            source_class="pain_signal",
            finding_kind="problem_signal",
            product_built=sig.title[:200],
            outcome_summary=(sig.body_excerpt or sig.title)[:2000],
            entrepreneur="",
            monetization_method="",
            content_hash=content_hash,
            status="qualified",  # routed to evidence/validation
            evidence={
                "run_id": run_id,
                "vertical": self.vertical,
                "reddit_score": sig.score,
                "num_comments": sig.num_comments,
            },
        )
        finding_id = self.db.insert_finding(finding)
        if not finding_id:
            return None
        finding_id = int(finding_id)

        # Create a raw_signal so the backlog workbench and evidence agent can
        # reference it.
        raw_signal = RawSignal(
            finding_id=finding_id,
            source_name=f"deep_research/{source_label}",
            source_type=source_label,
            source_url=sig.url,
            title=sig.title[:200],
            body_excerpt=(sig.body_excerpt or "")[:2000],
            content_hash=content_hash,
            source_class="pain_signal",
            quote_text=(sig.body_excerpt or "")[:500],
            role_hint="",
            published_at=sig.timestamp or "",
            timestamp_hint=sig.timestamp or "",
            metadata={
                "vertical": self.vertical,
                "subreddit": sig.subreddit,
                "author": sig.author,
                "score": sig.score,
                "num_comments": sig.num_comments,
            },
        )
        signal_id = self.db.insert_raw_signal(raw_signal)

        # Create a heuristic problem atom from the signal title and excerpt.
        # This gives the evidence agent something to work with for recurrence
        # queries, even though a full LLM extraction would be richer.
        body_text = sig.body_excerpt or sig.title or ""
        atom = ProblemAtom(
            finding_id=finding_id,
            raw_signal_id=signal_id,
            signal_id=signal_id,
            job_to_be_done=sig.title[:300],
            pain_statement=body_text[:500],
            failure_mode=body_text[:300],
            source_quote=body_text[:500],
            atom_extraction_method="deep_research_heuristic",
            confidence=0.4,
            confidence_score=0.4,
            atom_json='{}',
        )
        self.db.insert_problem_atom(atom)

        return finding_id

    # ── Dispatch ───────────────────────────────────────────────────────────

    async def _dispatch_to_validation(self, finding_ids: list[int]) -> int:
        """Send findings to ValidationAgent via orchestrator queue."""
        dispatched = 0
        for finding_id in finding_ids:
            try:
                await self.send_message(
                    to_agent="orchestrator",
                    msg_type=MessageType.FINDING_UNSEEDED,
                    payload={"finding_id": finding_id},
                    priority=1,
                )
                dispatched += 1
            except Exception as exc:
                logger.warning("[DeepResearch] dispatch for finding %d failed: %s", finding_id, exc)
        return dispatched
