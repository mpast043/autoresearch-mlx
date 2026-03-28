"""Explicit source adapters for review-origin discovery lanes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _compact(text: str, limit: int = 2000) -> str:
    return " ".join((text or "").split())[:limit]


def _parse_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_active_installs(text: str) -> int:
    cleaned = (text or "").lower().replace(",", " ").strip()
    if not cleaned:
        return 0
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(million|thousand)?", cleaned)
    if match:
        base = float(match.group(1))
        scale = match.group(2) or ""
        if scale == "million":
            return int(base * 1_000_000)
        if scale == "thousand":
            return int(base * 1_000)
        return int(base)
    digits = re.findall(r"\d+", cleaned)
    if not digits:
        return 0
    return int("".join(digits))


def _parse_star_rating(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s+out of 5", text or "", re.IGNORECASE)
    return _parse_float(match.group(1) if match else 0)


def _derive_review_title(text: str, fallback: str = "Shopify review") -> str:
    cleaned = _compact(text, 240)
    if not cleaned:
        return fallback
    parts = re.split(r"[.!?]\s+", cleaned, maxsplit=1)
    title = _compact(parts[0], 120)
    words = title.split()
    if len(words) > 12:
        title = " ".join(words[:12]).rstrip(",:;.-")
    return title or fallback


def _extract_labeled_value(text: str, label: str, next_labels: list[str]) -> str:
    if not text:
        return ""
    next_pattern = "|".join(re.escape(item) for item in next_labels)
    pattern = rf"{re.escape(label)}\s+(.*?)(?=\s+(?:{next_pattern})\s+|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    return _compact(match.group(1), 260) if match else ""


def _find_json_ld_object(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            type_value = item.get("@type", [])
            types = type_value if isinstance(type_value, list) else [type_value]
            if "SoftwareApplication" in types or "Product" in types:
                return item
    return {}


@dataclass
class WordPressPluginReview:
    plugin_slug: str
    plugin_name: str
    review_title: str
    review_text: str
    review_url: str
    review_date: str
    review_rating: float
    reviewer_name: str
    reviewer_type: str
    aggregate_rating: float
    review_count: int
    active_installs: int
    pricing: str
    version: str
    last_updated: str
    tested_up_to: str
    wordpress_requires: str

    def as_finding(self) -> dict[str, Any]:
        evidence = {
            "source_plan": "wordpress-reviews",
            "discovery_query": f"{self.plugin_slug}::reviews",
            "page_excerpt": self.review_text,
            "published_at": self.review_date,
            "review_metadata": {
                "review_source": "wordpress_plugin_directory",
                "record_origin": "review_text",
                "plugin_slug": self.plugin_slug,
                "product_name": self.plugin_name,
                "review_rating": self.review_rating,
                "aggregate_rating": self.aggregate_rating,
                "review_count": self.review_count,
                "review_date": self.review_date,
                "reviewer_name": self.reviewer_name,
                "reviewer_type": self.reviewer_type,
                "active_installs": self.active_installs,
                "pricing": self.pricing,
                "version": self.version,
                "last_updated": self.last_updated,
                "tested_up_to": self.tested_up_to,
                "wordpress_requires": self.wordpress_requires,
            },
        }
        return {
            "source": f"wordpress-review/{self.plugin_slug}",
            "source_url": self.review_url,
            "entrepreneur": self.reviewer_name or "WordPress reviewer",
            "tool_used": self.plugin_name,
            "product_built": self.review_title,
            "monetization_method": self.pricing,
            "outcome_summary": self.review_text,
            "finding_kind": "problem_signal",
            "evidence": evidence,
        }


class WordPressPluginReviewAdapter:
    """Fetches explicit review records from the WordPress Plugin Directory."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; AutoResearcher/1.0)"

    async def fetch_reviews(
        self,
        *,
        plugin_slugs: list[str],
        reviews_per_plugin: int = 3,
        star_filters: Optional[list[int]] = None,
    ) -> list[WordPressPluginReview]:
        if not plugin_slugs:
            return []
        star_filters = star_filters or [1]
        timeout = aiohttp.ClientTimeout(total=25)
        headers = {"User-Agent": self.user_agent}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [
                self._fetch_plugin_reviews(
                    session,
                    slug=slug,
                    reviews_per_plugin=reviews_per_plugin,
                    star_filters=star_filters,
                )
                for slug in plugin_slugs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        reviews: list[WordPressPluginReview] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            reviews.extend(result)
        return reviews

    async def _fetch_plugin_reviews(
        self,
        session: aiohttp.ClientSession,
        *,
        slug: str,
        reviews_per_plugin: int,
        star_filters: list[int],
    ) -> list[WordPressPluginReview]:
        metadata = await self._fetch_plugin_metadata(session, slug)
        if not metadata:
            return []

        listing_rows: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for star in star_filters:
            url = f"https://wordpress.org/support/plugin/{slug}/reviews/?filter={star}"
            html = await self._fetch_html(session, url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.select("li.bbp-topic-title"):
                link = item.select_one("a.bbp-topic-permalink")
                if link is None:
                    continue
                review_url = urljoin(url, link.get("href", "").strip())
                if not review_url or review_url in seen_urls:
                    continue
                seen_urls.add(review_url)
                rating = _parse_star_rating((item.select_one(".wporg-ratings") or {}).get("title", ""))
                reviewer = item.select_one(".bbp-author-name")
                listing_rows.append(
                    {
                        "review_url": review_url,
                        "review_title": _compact(link.get_text(" ", strip=True), 240),
                        "review_rating": rating or float(star),
                        "reviewer_name": _compact(reviewer.get_text(" ", strip=True) if reviewer else "", 80),
                    }
                )
                if len(listing_rows) >= reviews_per_plugin:
                    break
            if len(listing_rows) >= reviews_per_plugin:
                break

        detail_tasks = [
            self._fetch_review_detail(session, row["review_url"], row["review_title"], row["reviewer_name"])
            for row in listing_rows[:reviews_per_plugin]
        ]
        details = await asyncio.gather(*detail_tasks, return_exceptions=True)

        reviews: list[WordPressPluginReview] = []
        for row, detail in zip(listing_rows[:reviews_per_plugin], details):
            if isinstance(detail, Exception) or not detail:
                continue
            review_text = _compact(detail.get("review_text", ""), 1600)
            if not review_text:
                continue
            reviews.append(
                WordPressPluginReview(
                    plugin_slug=slug,
                    plugin_name=metadata.get("plugin_name", slug),
                    review_title=row["review_title"],
                    review_text=review_text,
                    review_url=row["review_url"],
                    review_date=detail.get("review_date", ""),
                    review_rating=float(row["review_rating"] or metadata.get("aggregate_rating") or 0.0),
                    reviewer_name=row["reviewer_name"],
                    reviewer_type=detail.get("reviewer_type", ""),
                    aggregate_rating=float(metadata.get("aggregate_rating", 0.0) or 0.0),
                    review_count=int(metadata.get("review_count", 0) or 0),
                    active_installs=int(metadata.get("active_installs", 0) or 0),
                    pricing=metadata.get("pricing", "free"),
                    version=metadata.get("version", ""),
                    last_updated=metadata.get("last_updated", ""),
                    tested_up_to=metadata.get("tested_up_to", ""),
                    wordpress_requires=metadata.get("wordpress_requires", ""),
                )
            )
        return reviews

    async def _fetch_plugin_metadata(self, session: aiohttp.ClientSession, slug: str) -> dict[str, Any]:
        url = f"https://wordpress.org/plugins/{slug}/"
        html = await self._fetch_html(session, url)
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        title = _compact((soup.select_one(".plugin-title") or soup.title).get_text(" ", strip=True), 120) if (soup.select_one(".plugin-title") or soup.title) else slug
        json_ld = _find_json_ld_object(soup)
        aggregate = json_ld.get("aggregateRating", {}) if isinstance(json_ld.get("aggregateRating"), dict) else {}
        metadata = {
            "plugin_name": title,
            "aggregate_rating": _parse_float(aggregate.get("ratingValue")),
            "review_count": _parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount")),
            "active_installs": 0,
            "pricing": "free",
            "version": "",
            "last_updated": "",
            "tested_up_to": "",
            "wordpress_requires": "",
        }
        for item in soup.select(".plugin-meta li"):
            line = _compact(item.get_text(" ", strip=True), 160)
            lowered = line.lower()
            if lowered.startswith("version "):
                metadata["version"] = line.replace("Version", "", 1).strip()
            elif lowered.startswith("last updated "):
                metadata["last_updated"] = line.replace("Last updated", "", 1).strip()
            elif lowered.startswith("active installations "):
                installs_text = line.replace("Active installations", "", 1).strip()
                metadata["active_installs"] = _parse_active_installs(installs_text)
            elif lowered.startswith("tested up to "):
                metadata["tested_up_to"] = line.replace("Tested up to", "", 1).strip()
            elif lowered.startswith("wordpress version "):
                metadata["wordpress_requires"] = line.replace("WordPress version", "", 1).strip()
        return metadata

    async def _fetch_review_detail(
        self,
        session: aiohttp.ClientSession,
        review_url: str,
        review_title: str,
        reviewer_name: str,
    ) -> dict[str, Any]:
        html = await self._fetch_html(session, review_url)
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one(".bbp-topic-content") or soup.select_one(".entry-content")
        review_text = _compact(body.get_text(" ", strip=True) if body else "", 2400)
        if review_text:
            review_text = re.sub(r"\bViewing\s+\d+\s+replies?.*$", "", review_text, flags=re.IGNORECASE).strip()
            review_text = re.sub(r"\bThis topic was modified.*$", "", review_text, flags=re.IGNORECASE).strip()
            for prefix in [review_title, reviewer_name]:
                prefix = _compact(prefix, 120)
                if prefix and review_text.lower().startswith(prefix.lower()):
                    review_text = review_text[len(prefix) :].strip(" :-")
        og_description = soup.select_one('meta[property="og:description"]')
        if len(review_text.split()) < 12 and og_description is not None:
            review_text = _compact(og_description.get("content", ""), 1600)
        published = soup.select_one('meta[property="article:published_time"]')
        modified = soup.select_one('meta[property="article:modified_time"]')
        return {
            "review_text": review_text,
            "review_date": (published.get("content") if published else "") or (modified.get("content") if modified else ""),
            "reviewer_type": "",
        }

    async def _fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url) as response:
            if response.status != 200:
                return ""
            return await response.text()


@dataclass
class ShopifyAppReview:
    app_handle: str
    app_name: str
    listing_url: str
    review_url: str
    category: str
    review_title: str
    review_text: str
    review_date: str
    review_rating: float
    reviewer_name: str
    reviewer_type: str
    reviewer_country: str
    aggregate_rating: float
    review_count: int
    pricing: str
    popularity_proxy: int
    version: str
    last_updated: str
    launched_at: str
    developer_name: str
    built_for_shopify: bool

    def as_finding(self) -> dict[str, Any]:
        evidence = {
            "source_plan": "shopify-reviews",
            "discovery_query": f"{self.app_handle}::reviews",
            "page_excerpt": self.review_text,
            "published_at": self.review_date,
            "review_metadata": {
                "review_source": "shopify_app_store",
                "record_origin": "review_text",
                "app_handle": self.app_handle,
                "product_name": self.app_name,
                "listing_url": self.listing_url,
                "category": self.category,
                "review_rating": self.review_rating,
                "aggregate_rating": self.aggregate_rating,
                "review_count": self.review_count,
                "review_title": self.review_title,
                "review_date": self.review_date,
                "reviewer_name": self.reviewer_name,
                "reviewer_type": self.reviewer_type,
                "reviewer_country": self.reviewer_country,
                "pricing": self.pricing,
                "popularity_proxy": self.popularity_proxy,
                "version": self.version,
                "last_updated": self.last_updated,
                "launched_at": self.launched_at,
                "developer_name": self.developer_name,
                "built_for_shopify": self.built_for_shopify,
            },
        }
        return {
            "source": f"shopify-review/{self.app_handle}",
            "source_url": self.review_url,
            "entrepreneur": self.reviewer_name or self.reviewer_type or "Shopify merchant",
            "tool_used": self.app_name,
            "product_built": self.review_title or _derive_review_title(self.review_text, self.app_name),
            "monetization_method": self.pricing,
            "outcome_summary": self.review_text,
            "finding_kind": "problem_signal",
            "evidence": evidence,
        }


class ShopifyAppReviewAdapter:
    """HTTP scrape of public Shopify App Store listing + reviews HTML — not the Shopify Admin API."""

    SITEMAP_URL = "https://apps.shopify.com/sitemap.xml"
    LISTING_BASE_URL = "https://apps.shopify.com/"

    def __init__(self, user_agent: str):
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; AutoResearcher/1.0)"

    async def fetch_reviews(
        self,
        *,
        app_handles: Optional[list[str]] = None,
        max_apps: int = 2,
        reviews_per_app: int = 2,
        rating_filters: Optional[list[int]] = None,
        sort_by: str = "newest",
        use_sitemap_discovery: bool = True,
    ) -> list[ShopifyAppReview]:
        rating_filters = rating_filters or [1]
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {"User-Agent": self.user_agent}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            handles = [handle.strip().strip("/") for handle in (app_handles or []) if handle.strip()]
            if not handles and use_sitemap_discovery:
                handles = await self._discover_app_handles_from_sitemap(session, max_apps=max_apps)
            handles = handles[:max_apps]
            if not handles:
                return []
            tasks = [
                self._fetch_app_reviews(
                    session,
                    app_handle=handle,
                    reviews_per_app=reviews_per_app,
                    rating_filters=rating_filters,
                    sort_by=sort_by,
                )
                for handle in handles
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        reviews: list[ShopifyAppReview] = []
        for handle, result in zip(handles, results):
            if isinstance(result, Exception):
                logger.warning("shopify review fetch failed handle=%s error=%s", handle, result)
                continue
            reviews.extend(result)
        return reviews

    async def _discover_app_handles_from_sitemap(
        self,
        session: aiohttp.ClientSession,
        *,
        max_apps: int,
    ) -> list[str]:
        xml = await self._fetch_html(session, self.SITEMAP_URL)
        if not xml:
            return []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            logger.warning("shopify sitemap parse failed: %s", exc)
            return []
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        handles: list[str] = []
        seen: set[str] = set()
        for node in root.findall(".//sm:loc", namespace) + root.findall(".//loc"):
            loc = (node.text or "").strip()
            if not loc.startswith(self.LISTING_BASE_URL):
                continue
            handle = loc.removeprefix(self.LISTING_BASE_URL).strip("/")
            if not handle or "/" in handle or handle in seen:
                continue
            seen.add(handle)
            handles.append(handle)
            if len(handles) >= max_apps:
                break
        return handles

    async def _fetch_app_reviews(
        self,
        session: aiohttp.ClientSession,
        *,
        app_handle: str,
        reviews_per_app: int,
        rating_filters: list[int],
        sort_by: str,
    ) -> list[ShopifyAppReview]:
        listing_url = urljoin(self.LISTING_BASE_URL, app_handle)
        listing_html = await self._fetch_html(session, listing_url)
        if not listing_html:
            logger.info("shopify listing skipped handle=%s reason=missing_listing", app_handle)
            return []
        metadata = self._parse_listing_metadata(listing_html, app_handle=app_handle, listing_url=listing_url)
        if not metadata.get("product_name"):
            logger.info("shopify listing skipped handle=%s reason=missing_product_name", app_handle)
            return []

        reviews: list[ShopifyAppReview] = []
        seen_urls: set[str] = set()
        for rating in rating_filters:
            review_url = f"{listing_url}/reviews?ratings%5B%5D={int(rating)}&sort_by={sort_by}"
            reviews_html = await self._fetch_html(session, review_url)
            if not reviews_html:
                logger.info("shopify reviews skipped handle=%s reason=missing_reviews_html", app_handle)
                continue
            parsed = self._parse_reviews_html(
                reviews_html,
                app_handle=app_handle,
                listing_metadata=metadata,
                max_reviews=reviews_per_app,
            )
            for item in parsed:
                if item.review_url in seen_urls:
                    continue
                seen_urls.add(item.review_url)
                reviews.append(item)
                if len(reviews) >= reviews_per_app:
                    break
            if len(reviews) >= reviews_per_app:
                break
        return reviews

    def _parse_listing_metadata(self, html: str, *, app_handle: str, listing_url: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        text = _compact(" ".join(soup.stripped_strings), 16000)
        json_ld = _find_json_ld_object(soup)
        aggregate = json_ld.get("aggregateRating", {}) if isinstance(json_ld.get("aggregateRating"), dict) else {}
        brand = json_ld.get("brand")
        if isinstance(brand, dict):
            developer_name = str(brand.get("name", "") or "").strip()
        else:
            developer_name = str(brand or "").strip()
        labels = [
            "Pricing",
            "Rating",
            "Developer",
            "Install",
            "Categories",
            "Works with",
            "Support",
            "Featured images gallery",
            "Highlights",
            "About this app",
            "Launched",
            "Built for Shopify",
            "More apps like this",
        ]
        pricing = _extract_labeled_value(text, "Pricing", labels)
        category_text = _extract_labeled_value(text, "Categories", labels)
        works_with = _extract_labeled_value(text, "Works with", labels)
        launched_at = _extract_labeled_value(text, "Launched", labels)
        launched_at = launched_at.replace("Built for Shopify", "").strip()
        if not developer_name:
            developer_name = _extract_labeled_value(text, "Developer", labels)
        if not pricing and "$" not in text:
            pricing = ""
        product_name = (
            str(json_ld.get("name", "") or "").strip()
            or _compact((soup.title.get_text(" ", strip=True) if soup.title else "").split(" - ", 1)[0], 120)
            or app_handle.replace("-", " ").title()
        )
        return {
            "product_name": product_name,
            "listing_url": listing_url,
            "aggregate_rating": _parse_float(aggregate.get("ratingValue")),
            "review_count": _parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount")),
            "category": category_text.split(" Show features", 1)[0].strip(),
            "pricing": pricing,
            "popularity_proxy": _parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount")),
            "version": "",
            "last_updated": "",
            "launched_at": launched_at,
            "developer_name": developer_name,
            "built_for_shopify": "Built for Shopify" in text,
            "works_with": works_with,
        }

    def _parse_reviews_html(
        self,
        html: str,
        *,
        app_handle: str,
        listing_metadata: dict[str, Any],
        max_reviews: int,
    ) -> list[ShopifyAppReview]:
        soup = BeautifulSoup(html, "html.parser")
        reviews: list[ShopifyAppReview] = []
        for node in soup.select('div[id^="review-"]'):
            parsed = self._parse_review_node(node, app_handle=app_handle, listing_metadata=listing_metadata)
            if parsed is None:
                continue
            reviews.append(parsed)
            if len(reviews) >= max_reviews:
                break
        return reviews

    def _parse_review_node(
        self,
        node: Any,
        *,
        app_handle: str,
        listing_metadata: dict[str, Any],
    ) -> Optional[ShopifyAppReview]:
        review_id = (node.get("id", "") or "").replace("review-", "").strip()
        content_root = node.select_one("[data-review-content-id]")
        if not review_id.isdigit() or content_root is None:
            return None
        reply_node = node.select_one("[data-merchant-review-reply]")
        if reply_node is not None:
            reply_node.extract()
        body_node = content_root.select_one("[data-truncate-content-copy]")
        review_text = _compact(body_node.get_text(" ", strip=True) if body_node else "", 1800)
        review_text = review_text.replace("Show more", "").strip()
        if len(review_text.split()) < 3:
            logger.info("shopify review skipped app=%s review_id=%s reason=empty_body", app_handle, review_id)
            return None
        rating_node = content_root.select_one('[aria-label*="out of 5 stars"]') or node.select_one('[aria-label*="out of 5 stars"]')
        review_rating = _parse_star_rating(rating_node.get("aria-label", "") if rating_node else "")
        strings = [item for item in content_root.stripped_strings if item and item != "Show more"]
        review_date = ""
        reviewer_name = ""
        reviewer_country = ""
        reviewer_type = ""
        for item in strings:
            cleaned = _compact(item, 120)
            if not review_date and re.search(r"(edited\s+)?[a-z]+\s+\d{1,2},\s+\d{4}", cleaned, re.IGNORECASE):
                review_date = re.sub(r"^Edited\s+", "", cleaned, flags=re.IGNORECASE)
            elif not reviewer_type and re.search(r"using the app", cleaned, re.IGNORECASE):
                reviewer_type = cleaned
        tail_candidates = []
        if review_text:
            tail_candidates = [item for item in strings if item not in {review_date, reviewer_type} and item not in review_text]
        if tail_candidates:
            reviewer_name = _compact(tail_candidates[-2] if len(tail_candidates) >= 2 else tail_candidates[0], 120)
            if len(tail_candidates) >= 2:
                reviewer_country = _compact(tail_candidates[-1], 80)
        review_url = f"{listing_metadata.get('listing_url', '').rstrip('/')}/reviews/{review_id}"
        return ShopifyAppReview(
            app_handle=app_handle,
            app_name=str(listing_metadata.get("product_name", "") or app_handle),
            listing_url=str(listing_metadata.get("listing_url", "") or ""),
            review_url=review_url,
            category=str(listing_metadata.get("category", "") or ""),
            review_title=_derive_review_title(review_text, str(listing_metadata.get("product_name", "") or "Shopify review")),
            review_text=review_text,
            review_date=review_date,
            review_rating=review_rating,
            reviewer_name=reviewer_name,
            reviewer_type=reviewer_type,
            reviewer_country=reviewer_country,
            aggregate_rating=_parse_float(listing_metadata.get("aggregate_rating")),
            review_count=_parse_int(listing_metadata.get("review_count")),
            pricing=str(listing_metadata.get("pricing", "") or ""),
            popularity_proxy=_parse_int(listing_metadata.get("popularity_proxy")),
            version=str(listing_metadata.get("version", "") or ""),
            last_updated=str(listing_metadata.get("last_updated", "") or ""),
            launched_at=str(listing_metadata.get("launched_at", "") or ""),
            developer_name=str(listing_metadata.get("developer_name", "") or ""),
            built_for_shopify=bool(listing_metadata.get("built_for_shopify", False)),
        )

    async def _fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.info("shopify fetch skipped url=%s status=%s", url, response.status)
                    return ""
                return await response.text()
        except Exception as exc:  # pragma: no cover - network variance
            logger.warning("shopify fetch failed url=%s error=%s", url, exc)
            return ""
