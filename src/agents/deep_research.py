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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from agents.base import BaseAgent
from database import Database, Finding, RawSignal
from messaging import MessageQueue, MessageType

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
        self.status_tracker.log(f"deep_research_start vertical={self.vertical}")

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
        filtered = [s for s in scored if s.score >= 2]
        logger.info("[DeepResearch] after scoring/filtering=%d", len(filtered))

        # 4) Persist as RawSignals + create Findings
        findings_created = await self._persist_signals_and_findings(filtered, run_id)

        # 5) Dispatch findings directly to ValidationAgent
        dispatched = await self._dispatch_to_validation(findings_created)

        elapsed = time.time() - t0
        summary = {
            "run_id": run_id,
            "vertical": self.vertical,
            "reddit_signals": len(reddit_signals),
            "github_signals": len(github_signals),
            "web_signals": len(web_signals),
            "after_dedup": len(deduped),
            "after_filter": len(filtered),
            "findings_created": findings_created,
            "dispatched_to_validation": dispatched,
            "elapsed_s": round(elapsed, 1),
        }

        # Save run report
        report_path = out_path / f"{run_id}_report.json"
        report_path.write_text(json.dumps(summary, indent=2))
        logger.info("[DeepResearch] report saved to %s", report_path)
        self.status_tracker.log(
            f"deep_research_complete findings={findings_created} "
            f"dispatched={dispatched} elapsed={elapsed:.1f}s"
        )

        return summary

    # ── Source fetchers ─────────────────────────────────────────────────────

    async def _search_reddit(self, limit: int) -> list[PainSignal]:
        """Search Reddit via research_tools (ddgs)."""
        from research_tools import RedditSearchResult, search_reddit

        signals: list[PainSignal] = []
        for query in self.keywords.get("reddit_queries", []):
            try:
                raw = await asyncio.to_thread(search_reddit, query, limit=limit)
                for item in raw:
                    if isinstance(item, dict):
                        item = RedditSearchResult(**item)
                    signals.append(PainSignal(
                        title=item.title or "Untitled",
                        url=item.url or "",
                        source="reddit",
                        subreddit=item.subreddit or None,
                        author=item.author or None,
                        score=getattr(item, "score", 0) or 0,
                        num_comments=getattr(item, "num_comments", 0) or 0,
                        timestamp=getattr(item, "created_utc", None),
                        body_excerpt=getattr(item, "body", None),
                    ))
            except Exception as exc:
                logger.warning("[DeepResearch] reddit query '%s' failed: %s", query, exc)
        return signals

    async def _search_github(self, limit: int) -> list[PainSignal]:
        """Search GitHub issues via research_tools."""
        from research_tools import search_github_issues

        signals: list[PainSignal] = []
        for query in self.keywords.get("github_queries", []):
            try:
                raw = await asyncio.to_thread(search_github_issues, query, limit=limit)
                for item in raw:
                    url = getattr(item, "url", "") or getattr(item, "html_url", "") or ""
                    signals.append(PainSignal(
                        title=getattr(item, "title", "Untitled") or "Untitled",
                        url=url,
                        source="github",
                        author=getattr(item, "author", None) or None,
                        score=getattr(item, "score", 0) or 0,
                        num_comments=getattr(item, "num_comments", 0) or 0,
                        timestamp=getattr(item, "created_at", None),
                    ))
            except Exception as exc:
                logger.warning("[DeepResearch] github query '%s' failed: %s", query, exc)
        return signals

    async def _search_web(self, limit: int) -> list[PainSignal]:
        """Broad web search via ddgs."""
        import ddgs
        signals: list[PainSignal] = []
        for query in self.keywords.get("web_queries", []):
            try:
                with ddgs.DDGS() as ddg:
                    results = list(ddg.text(query, max_results=limit))
                for item in results:
                    signals.append(PainSignal(
                        title=item.get("title", "Untitled") or "Untitled",
                        url=item.get("url", "") or "",
                        source="web",
                        timestamp=item.get("published", None),
                        body_excerpt=item.get("description", None),
                    ))
            except Exception as exc:
                logger.warning("[DeepResearch] web query '%s' failed: %s", query, exc)
        return signals

    # ── Deduplication ───────────────────────────────────────────────────────

    def _deduplicate(self, signals: list[PainSignal]) -> list[PainSignal]:
        """Drop signals with duplicate URLs, keeping highest-scoring duplicate."""
        seen: dict[str, PainSignal] = {}
        for sig in signals:
            key = sig.url.split("?")[0]  # strip query params
            if key not in seen or sig.score > seen[key].score:
                seen[key] = sig
        return list(seen.values())

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
        from database import RawSignal

        findings_created = 0
        for sig in signals:
            if not sig.url:
                continue

            # Persist raw signal
            content_hash = hashlib.sha256(f"{sig.url}".encode()).hexdigest()[:16]
            existing = self.db.get_raw_signal_by_hash(content_hash)
            if existing:
                signal_id = existing.id
            else:
                signal = RawSignal(
                    finding_id=None,
                    source=sig.source,
                    source_url=sig.url,
                    content_hash=content_hash,
                    content=sig.body_excerpt or sig.title,
                    metadata_={
                        "run_id": run_id,
                        "vertical": self.vertical,
                        "subreddit": sig.subreddit,
                        "reddit_score": sig.score,
                        "num_comments": sig.num_comments,
                    },
                )
                signal_id = self.db.insert_raw_signal(signal)
                self.db.update_raw_signal_signal_id(signal_id)

            # Create or update finding
            finding_id = self._upsert_finding(sig, run_id, content_hash)
            if finding_id:
                findings_created += 1

        return findings_created

    def _upsert_finding(self, sig: PainSignal, run_id: str, content_hash: str) -> Optional[int]:
        """Create a finding from a pain signal. Returns finding_id."""
        # Check if URL already has a finding
        existing_findings = self.db.get_findings(limit=500)
        for f in existing_findings:
            if f.source_url == sig.url:
                return int(f.id) if f.id else None

        finding = Finding(
            source="deep_research",
            source_url=sig.url,
            source_class="pain_signal",
            finding_kind="reddit_post" if sig.source == "reddit" else "github_issue" if sig.source == "github" else "web_content",
            product_built=sig.title[:200],
            outcome_summary=(sig.body_excerpt or sig.title)[:2000],
            entrepreneur="",
            monetization_method="",
            content_hash=content_hash,
            status="discovery_filter",  # will be routed to validation
            metadata_={
                "run_id": run_id,
                "vertical": self.vertical,
                "reddit_score": sig.score,
                "num_comments": sig.num_comments,
            },
        )
        finding_id = self.db.insert_finding(finding)
        return int(finding_id) if finding_id else None

    # ── Dispatch ───────────────────────────────────────────────────────────

    async def _dispatch_to_validation(self, finding_ids: list[int]) -> int:
        """Send findings to ValidationAgent via orchestrator queue."""
        from messaging import Message

        dispatched = 0
        for finding_id in finding_ids:
            try:
                msg = Message(
                    msg_type=MessageType.FINDING_UNSEEDED,
                    payload={"finding_id": finding_id},
                    from_agent="deep_research",
                    to_agent="validation",
                )
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
