"""
Unseeded Discovery Loop — cold-start pain signal discovery.

Runs without DB seeding: takes a vertical/topic, searches multiple sources,
scores signals, routes through validation → build_prep, and returns
prototype_candidate briefs.

Usage:
    python -m src.unseeded_loop --vertical devtools
    python -m src.unseeded_loop --vertical ecommerce --max-findings 10
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.database import Database
from src.agents.validation import ValidationAgent

logger = logging.getLogger(__name__)


# ─── Vertical contexts ─────────────────────────────────────────────────────────

VERTICAL_NICHES = {
    "devtools": {
        "keywords": [
            "migration pain", "slow build", "poor docs", "breakage",
            "vendor lock-in", "undocumented", "deprecation", "tooling",
            "CI/CD", "DX", "cognitive load", "undocumented API",
        ],
        "source_priority": {"github_issues": 0.9, "reddit": 0.6, "wordpress_shopify": 0.1},
        "reddit_subreddits": ["r/devops", "r/programming", "r/webdev"],
    },
    "ecommerce": {
        "keywords": [
            "inventory sync", "checkout pain", "shopify app", "payment failed",
            "order management", "analytics wrong", "VAT", "tax calculation",
            "abandoned cart", "fraud detection",
        ],
        "source_priority": {"wordpress_shopify": 0.9, "reddit": 0.5, "github_issues": 0.2},
        "reddit_subreddits": ["r/ecommerce", "r/shopify", "r/smallbusiness"],
    },
    "healthtech": {
        "keywords": [
            "HIPAA compliance", "patient data", "telehealth", "EHR integration",
            "medical billing", "appointment scheduling", "claims processing",
        ],
        "source_priority": {"github_issues": 0.5, "reddit": 0.7, "wordpress_shopify": 0.3},
        "reddit_subreddits": ["r/healthIT", "r/healthcare", "r/medicalschool"],
    },
}


@dataclass
class UnseededRawSignal:
    """Lightweight signal container for unseeded discovery (distinct from src.database.RawSignal)."""
    title: str
    body: str
    url: str
    source_type: str
    source_name: str
    timestamp: Optional[str] = None
    vertical: str = "general"
    pain_keywords: list[str] = field(default_factory=list)


@dataclass
class UnseededResult:
    raw_signals_found: int
    promoted_to_validation: int
    promoted_to_build_prep: int
    prototype_candidates: list[dict]
    errors: list[str]


async def _search_reddit(config: dict, vertical: str, niche: dict) -> list[UnseededRawSignal]:
    """Search Reddit for pain signals in vertical subreddits."""
    from src.research_tools import RedditClient
    signals = []
    client = RedditClient(config)

    for subreddit in niche.get("reddit_subreddits", []):
        try:
            for kw in niche["keywords"][:3]:  # top 3 keywords per subreddit
                query = f"{subreddit.replace('r/', '')} {kw}"
                results = await client.search_subreddit(query, limit=5)
                for r in results:
                    signals.append(UnseededRawSignal(
                        title=r.get("title", ""),
                        body=r.get("body", ""),
                        url=r.get("url", ""),
                        source_type="reddit",
                        source_name=subreddit,
                        vertical=vertical,
                        pain_keywords=[kw],
                    ))
        except Exception as e:
            logger.warning(f"Reddit search failed for {subreddit}: {e}")

    return signals


async def _search_github(config: dict, vertical: str, niche: dict) -> list[UnseededRawSignal]:
    """Search GitHub issues for pain signals."""
    signals = []
    for kw in niche["keywords"][:5]:
        query = f"{kw} is:issue is:open"
        # gh CLI search (sync)
        import subprocess
        try:
            result = subprocess.run(
                ["gh", "search", "issues", query, "--limit", "10",
                 "--json", "title,body,url,createdAt,state"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                for issue in json.loads(result.stdout):
                    signals.append(UnseededRawSignal(
                        title=issue["title"],
                        body=issue.get("body", "")[:500],
                        url=issue["url"],
                        source_type="github_issues",
                        source_name="gh",
                        timestamp=issue.get("createdAt"),
                        vertical=vertical,
                        pain_keywords=[kw],
                    ))
        except Exception as e:
            logger.warning(f"GitHub search failed for '{query}': {e}")
    return signals


async def _search_web(config: dict, vertical: str, niche: dict) -> list[UnseededRawSignal]:
    """Search web for pain signals via Tavily or ddgs."""
    signals = []
    for kw in niche["keywords"][:5]:
        query = f"{kw} pain point frustration problem"
        # Use Tavily if key available
        api_key = config.get("tavily", {}).get("api_key") or config.get("TAVILY_API_KEY")
        if api_key:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "https://api.tavily.com/search",
                        json={"query": query, "max_results": 5},
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=15.0,
                    )
                    r.raise_for_status()
                    for item in r.json().get("results", []):
                        signals.append(UnseededRawSignal(
                            title=item.get("title", ""),
                            body=item.get("content", ""),
                            url=item.get("url", ""),
                            source_type="web",
                            source_name="tavily",
                            vertical=vertical,
                            pain_keywords=[kw],
                        ))
            except Exception as e:
                logger.warning(f"Tavily search failed for '{query}': {e}")
        else:
            # Fallback to ddgs
            from src.research_tools import search_web
            try:
                results = await search_web(query, config, limit=5)
                for item in results:
                    signals.append(UnseededRawSignal(
                        title=item.get("title", ""),
                        body=item.get("snippet", ""),
                        url=item.get("url", ""),
                        source_type="web",
                        source_name="ddgs",
                        vertical=vertical,
                        pain_keywords=[kw],
                    ))
            except Exception as e:
                logger.warning(f"ddgs search failed for '{query}': {e}")
    return signals


def _score_raw_signal(signal: UnseededRawSignal, niche: dict) -> float:
    """
    Quick score for a raw signal to decide if it merits full validation.
    Returns 0.0-1.0.
    """
    score = 0.0

    # Source priority bonus
    priority = niche.get("source_priority", {}).get(signal.source_type, 0.5)
    score += priority * 0.3

    # Keyword density (how many niche keywords appear in title/body)
    text = (signal.title + " " + signal.body).lower()
    keyword_hits = sum(1 for kw in niche["keywords"] if kw.lower() in text)
    score += min(1.0, keyword_hits / 3) * 0.4

    # Has body content (not just title)
    if len(signal.body) > 100:
        score += 0.15

    # Has URL (verifiable)
    if signal.url:
        score += 0.15

    return min(1.0, score)


async def run_unseeded(
    vertical: str,
    config: dict,
    db: Database,
    max_findings: int = 20,
    min_signal_score: float = 0.35,
) -> UnseededResult:
    """
    Run the unseeded discovery loop for a given vertical.

    Steps:
    1. Load vertical context
    2. Parallel search across Reddit, GitHub, Web
    3. Score and filter raw signals
    4. Write qualifying signals to DB as findings (state=discovery_filter)
    5. Run ValidationAgent on each
    6. Run build_prep on validated
    7. Return prototype_candidate results
    """
    t0 = time.time()
    niche = VERTICAL_NICHES.get(vertical, {
        "keywords": [vertical],
        "source_priority": {},
        "reddit_subreddits": [],
    })

    print(f"\n[Unseeded Loop] Starting for vertical='{vertical}'")
    print(f"  Keywords: {niche['keywords'][:5]}")

    # ── Step 1: Parallel source search ─────────────────────────────
    print(f"\n[1/5] Searching sources...")
    results = await asyncio.gather(
        _search_reddit(config, vertical, niche),
        _search_github(config, vertical, niche),
        _search_web(config, vertical, niche),
        return_exceptions=True,
    )
    reddit_signals, github_signals, web_signals = results
    all_signals = [
        *(reddit_signals if isinstance(reddit_signals, list) else []),
        *(github_signals if isinstance(github_signals, list) else []),
        *(web_signals if isinstance(web_signals, list) else []),
    ]
    print(f"  Raw signals: {len(all_signals)} total "
          f"(reddit={len(reddit_signals) if isinstance(reddit_signals, list) else 'err'}, "
          f"github={len(github_signals) if isinstance(github_signals, list) else 'err'}, "
          f"web={len(web_signals) if isinstance(web_signals, list) else 'err'})")

    # ── Step 2: Score and filter ──────────────────────────────────
    print(f"\n[2/5] Scoring signals...")
    scored = [(s, _score_raw_signal(s, niche)) for s in all_signals]
    filtered = [(s, score) for s, score in scored if score >= min_signal_score]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:max_findings]
    print(f"  Passed threshold ({min_signal_score}): {len(filtered)}")

    # ── Step 3: Write to DB as findings ────────────────────────────
    print(f"\n[3/5] Writing {len(filtered)} findings to DB...")
    finding_ids = []
    for signal, score in filtered:
        import uuid
        finding_id = str(uuid.uuid4())[:8]
        db.insert_finding({
            "id": finding_id,
            "source": f"unseeded_{signal.source_type}",
            "source_class": signal.source_type,
            "status": "discovery_filter",
            "finding_kind": "problem_signal",
            "product_built": signal.title[:200] if signal.title else "N/A",
            "outcome_summary": signal.body[:1000] if signal.body else "N/A",
            "signal_strength": float(score),
            "evidence_json": json.dumps({
                "url": signal.url,
                "source_name": signal.source_name,
                "vertical": vertical,
                "pain_keywords": signal.pain_keywords,
            }),
            "vertical": vertical,
        })
        finding_ids.append((finding_id, score))

    # ── Step 4: Run ValidationAgent on each ────────────────────────
    print(f"\n[4/5] Running validation...")
    validation_agent = ValidationAgent(db=db, config=config)
    validated_count = 0
    build_prep_count = 0
    prototype_candidates = []
    errors = []

    for finding_id, raw_score in finding_ids:
        try:
            # Run validation synchronously in-process
            result = await validation_agent.process(finding_id)
            if result and result.get("passed"):
                validated_count += 1

                # Check if it also qualifies for build_prep
                conn = db._get_connection()
                row = conn.execute(
                    "SELECT id, passed, selection_status FROM validations WHERE finding_id=? ORDER BY id DESC LIMIT 1",
                    (finding_id,)
                ).fetchone()
                if row and row["selection_status"] == "prototype_candidate":
                    build_prep_count += 1
                    prototype_candidates.append({
                        "finding_id": finding_id,
                        "raw_score": raw_score,
                        "selection_status": row["selection_status"],
                        "url": json.loads(db.get_finding(finding_id).evidence_json or "{}").get("url", ""),
                    })
        except Exception as e:
            errors.append(f"Validation {finding_id}: {e}")
            logger.warning(f"Unseeded validation error for {finding_id}: {e}")

    print(f"\n[5/5] Done.")
    print(f"  → Validated:       {validated_count}/{len(filtered)}")
    print(f"  → Build_prep:      {build_prep_count}")
    print(f"  → Prototype cands: {len(prototype_candidates)}")
    print(f"  → Errors:          {len(errors)}")
    print(f"  → Duration:        {time.time()-t0:.1f}s")

    if prototype_candidates:
        print(f"\n  Prototype Candidates:")
        for pc in prototype_candidates:
            print(f"    [{pc['finding_id']}] score={pc['raw_score']:.2f} {pc['url']}")

    return UnseededResult(
        raw_signals_found=len(all_signals),
        promoted_to_validation=validated_count,
        promoted_to_build_prep=build_prep_count,
        prototype_candidates=prototype_candidates,
        errors=errors,
    )


async def main():
    parser = argparse.ArgumentParser(description="Unseeded discovery loop")
    parser.add_argument("--vertical", default="devtools", help="Vertical to search (devtools, ecommerce, healthtech)")
    parser.add_argument("--max-findings", type=int, default=20, help="Max findings to process")
    parser.add_argument("--min-score", type=float, default=0.35, help="Min raw signal score")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_path = Path(__file__).parent.parent / "data" / "autoresearch.db"
    db = Database(str(db_path))

    result = await run_unseeded(
        vertical=args.vertical,
        config=config,
        db=db,
        max_findings=args.max_findings,
        min_signal_score=args.min_score,
    )

    # Save result
    output_dir = Path(__file__).parent.parent / "data" / "unseeded_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{args.vertical}_{int(time.time())}.json"
    out_file.write_text(json.dumps({
        "vertical": args.vertical,
        "raw_signals": result.raw_signals_found,
        "validated": result.promoted_to_validation,
        "build_prep": result.promoted_to_build_prep,
        "prototype_candidates": result.prototype_candidates,
        "errors": result.errors,
    }, indent=2))
    print(f"\nResult saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
