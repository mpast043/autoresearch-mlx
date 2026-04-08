"""
CompetitorIntel — extract structured competitor intelligence from review sites.

Used by BuildPrepAgent after a build brief is validated, to understand
existing solutions before writing a spec.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Competitor:
    name: str
    product_url: str
    ratings: dict[str, float]          # e.g. {"g2": 4.2, "capterra": 3.8}
    pros: list[str]
    cons: list[str]
    pricing_model: str                 # e.g. "per-seat", "usage-based", "free"
    description: str
    source: str = "manual"


def _parse_apify_reviews(raw: list[dict]) -> list[Competitor]:
    competitors = {}
    for item in raw:
        source_url = item.get("sourceUrl", "")
        name = item.get("name", source_url.split("/")[2] if "/" in source_url else "unknown")
        if name not in competitors:
            competitors[name] = Competitor(
                name=name,
                product_url=source_url,
                ratings={},
                pros=[],
                cons=[],
                pricing_model=item.get("pricingModel", "unknown"),
                description=item.get("description", ""),
            )
        rating = item.get("rating")
        if isinstance(rating, (int, float)):
            competitors[name].ratings[item.get("sourceName", "unknown")] = float(rating)
        for review in item.get("reviews", []):
            if review.get("type") == "PRO":
                competitors[name].pros.append(review.get("text", "")[:200])
            elif review.get("type") == "CON":
                competitors[name].cons.append(review.get("text", "")[:200])
    return list(competitors.values())


async def fetch_g2_reviews(product_name: str, limit: int = 20) -> list[Competitor]:
    """
    Scrape G2 for competitor reviews using Apify.
    Requires APIFY_API_KEY in environment.
    """
    import os

    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        logger.warning("[CompetitorIntel] no APAFY_API_KEY — skipping G2 fetch")
        return []

    try:
        from apify_client import ApifyClient

        client = ApifyClient(api_key)
        run_input = {
            "queries": [product_name],
            "maxResults": limit,
        }
        logger.info("[CompetitorIntel] calling Apify for '%s'", product_name)
        run = client.actor("jupri/g2-reviews-scraper").start(run_input=run_input)
        # Poll for completion (up to 60s)
        for _ in range(12):
            await asyncio.sleep(5)
            status = client.run(run["id"]).get("status")
            if status in ("SUCCEEDED", "FAILED"):
                break
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        return _parse_apify_reviews(dataset_items)
    except Exception as exc:
        logger.warning("[CompetitorIntel] Apify fetch failed: %s", exc)
        return []


async def fetch_capterra_reviews(product_name: str, limit: int = 20) -> list[Competitor]:
    """Scrape Capterra reviews via Apify."""
    import os

    api_key = os.getenv("APIFY_API_KEY")
    if not api_key:
        return []

    try:
        from apify_client import ApifyClient

        client = ApifyClient(api_key)
        run_input = {"productName": product_name, "maxResults": limit}
        run = client.actor("apify/capterra-reviews").start(run_input=run_input)
        for _ in range(12):
            await asyncio.sleep(5)
            status = client.run(run["id"]).get("status")
            if status in ("SUCCEEDED", "FAILED"):
                break
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
        return _parse_apify_reviews(dataset_items)
    except Exception as exc:
        logger.warning("[CompetitorIntel] Capterra fetch failed: %s", exc)
        return []


async def fetch_all_reviews(product_name: str, limit: int = 20) -> dict[str, list[Competitor]]:
    """
    Fetch reviews from all competitor-intel sources in parallel.
    Returns {source_name: [Competitor]}.
    """
    g2_task = asyncio.create_task(fetch_g2_reviews(product_name, limit))
    capterra_task = asyncio.create_task(fetch_capterra_reviews(product_name, limit))

    g2_results, capterra_results = await asyncio.gather(g2_task, capterra_task)

    return {
        "g2": g2_results,
        "capterra": capterra_results,
    }


def build_competitor_summary(competitors: list[Competitor]) -> dict:
    """
    Convert a list of Competitor records into a structured summary
    suitable for passing to BuildPrepAgent as context.
    """
    if not competitors:
        return {"competitors": [], "summary": "No competitor data available."}

    avg_rating = 0.0
    all_pros: dict[str, int] = {}
    all_cons: dict[str, int] = {}

    for c in competitors:
        if c.ratings:
            avg_rating = sum(c.ratings.values()) / len(c.ratings)
        for pro in c.pros:
            all_pros[pro.lower().strip()] = all_pros.get(pro.lower().strip(), 0) + 1
        for con in c.cons:
            all_cons[con.lower().strip()] = all_cons.get(con.lower().strip(), 0) + 1

    top_pros = sorted(all_pros.items(), key=lambda x: -x[1])[:5]
    top_cons = sorted(all_cons.items(), key=lambda x: -x[1])[:5]

    return {
        "competitors": [
            {
                "name": c.name,
                "product_url": c.product_url,
                "ratings": c.ratings,
                "pricing_model": c.pricing_model,
                "top_pros": c.pros[:3],
                "top_cons": c.cons[:3],
            }
            for c in competitors
        ],
        "average_rating": round(avg_rating, 2),
        "top_mentioned_pros": [p for p, _ in top_pros],
        "top_mentioned_cons": [c for c, _ in top_cons],
        "summary": (
            f"{len(competitors)} competitors found. "
            f"Avg rating: {avg_rating:.1f}/5. "
            f"Top pain points: {', '.join(c for c, _ in top_cons[:3])}."
        ),
    }
