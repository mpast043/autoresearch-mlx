"""Tests for explicit review-source adapters."""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from review_sources import ShopifyAppReviewAdapter, WordPressPluginReviewAdapter


PLUGIN_HTML = """
<html>
  <head>
    <title>WooCommerce</title>
    <script type="application/ld+json">
      [
        {
          "@context": "http://schema.org",
          "@type": ["SoftwareApplication", "Product"],
          "name": "WooCommerce",
          "aggregateRating": {
            "@type": "AggregateRating",
            "ratingValue": "4.4",
            "reviewCount": "4582"
          }
        }
      ]
    </script>
  </head>
  <body>
    <h1 class="plugin-title">WooCommerce</h1>
    <ul class="plugin-meta">
      <li>Version 10.6.1</li>
      <li>Last updated 6 days ago</li>
      <li>Active installations 7+ million</li>
      <li>WordPress version 6.8 or higher</li>
      <li>Tested up to 6.9.4</li>
    </ul>
  </body>
</html>
"""


REVIEW_HTML = """
<html>
  <head>
    <meta property="article:published_time" content="2026-02-23T02:56:55+00:00" />
    <meta property="og:description" content="Love the flexibility, but analytics still require paid plugins and manual GA4 work." />
  </head>
  <body>
    <div class="bbp-topic-content">
      Slow development, dated features, far behind shopify…
      regedy1
      Love the flexibility, but analytics still require paid plugins and manual GA4 work.
      This topic was modified 3 weeks ago.
    </div>
  </body>
</html>
"""


SHOPIFY_LISTING_HTML = """
<html>
  <head>
    <title>AppsByB: Backup & Sync - Backup, view historical product versions and restore with ease | Shopify App Store</title>
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "AppsByB: Backup & Sync",
        "brand": "AppsByB",
        "aggregateRating": {
          "@type": "AggregateRating",
          "ratingValue": 3.7,
          "ratingCount": 118
        }
      }
    </script>
  </head>
  <body>
    Pricing $19/month. Free trial available. Rating 3.7 (118) Developer AppsByB Install
    Works with Shopify Flow Google Drive Categories Store management Backups
    Launched January 14, 2021 Built for Shopify
  </body>
</html>
"""


SHOPIFY_REVIEWS_HTML = """
<html>
  <body>
    <div id="review-101" class="review">
      <div aria-label="1 out of 5 stars"></div>
      <div data-merchant-review data-review-content-id="101">
        <div>Edited June 1, 2024</div>
        <div data-truncate-content-copy>
          Restore jobs fail every time, so we keep manual recovery checklists and rebuild stores by hand.
        </div>
        <div>Agency Ops Ltd</div>
        <div>Australia</div>
        <div>Over 1 year using the app</div>
      </div>
      <button data-review-share-link="/reviews/101" data-review-id="101"></button>
    </div>
    <div id="review-102" class="review">
      <div aria-label="5 out of 5 stars"></div>
      <div data-merchant-review data-review-content-id="102">
        <div>March 3, 2024</div>
        <div data-truncate-content-copy>
          Great app. Highly recommend it to everyone.
        </div>
        <div>Helpful Shop</div>
        <div>United States</div>
        <div>6 months using the app</div>
      </div>
      <button data-review-share-link="/reviews/102" data-review-id="102"></button>
    </div>
  </body>
</html>
"""


def test_wordpress_adapter_parses_plugin_metadata():
    adapter = WordPressPluginReviewAdapter("Mozilla/5.0")

    async def fake_fetch_html(_session, _url):
        return PLUGIN_HTML

    adapter._fetch_html = fake_fetch_html

    metadata = asyncio.run(adapter._fetch_plugin_metadata(None, "woocommerce"))

    assert metadata["plugin_name"] == "WooCommerce"
    assert metadata["aggregate_rating"] == 4.4
    assert metadata["review_count"] == 4582
    assert metadata["active_installs"] == 7000000
    assert metadata["version"] == "10.6.1"


def test_wordpress_adapter_parses_review_detail():
    adapter = WordPressPluginReviewAdapter("Mozilla/5.0")

    async def fake_fetch_html(_session, _url):
        return REVIEW_HTML

    adapter._fetch_html = fake_fetch_html

    detail = asyncio.run(
        adapter._fetch_review_detail(
            None,
            "https://wordpress.org/support/topic/slow-development/",
            "Slow development, dated features, far behind shopify…",
            "regedy1",
        )
    )

    assert detail["review_date"] == "2026-02-23T02:56:55+00:00"
    assert "manual GA4 work" in detail["review_text"]


def test_shopify_adapter_parses_listing_metadata():
    adapter = ShopifyAppReviewAdapter("Mozilla/5.0")

    metadata = adapter._parse_listing_metadata(
        SHOPIFY_LISTING_HTML,
        app_handle="backup-and-sync",
        listing_url="https://apps.shopify.com/backup-and-sync",
    )

    assert metadata["product_name"] == "AppsByB: Backup & Sync"
    assert metadata["aggregate_rating"] == 3.7
    assert metadata["review_count"] == 118
    assert metadata["category"] == "Store management Backups"
    assert metadata["pricing"] == "$19/month. Free trial available."
    assert metadata["developer_name"] == "AppsByB"
    assert metadata["launched_at"] == "January 14, 2021"
    assert metadata["built_for_shopify"] is True


def test_shopify_adapter_parses_review_nodes_and_handles_missing_fields():
    adapter = ShopifyAppReviewAdapter("Mozilla/5.0")
    listing_metadata = adapter._parse_listing_metadata(
        SHOPIFY_LISTING_HTML,
        app_handle="backup-and-sync",
        listing_url="https://apps.shopify.com/backup-and-sync",
    )

    reviews = adapter._parse_reviews_html(
        SHOPIFY_REVIEWS_HTML,
        app_handle="backup-and-sync",
        listing_metadata=listing_metadata,
        max_reviews=2,
    )

    assert len(reviews) == 2
    assert reviews[0].review_rating == 1.0
    assert "manual recovery checklists" in reviews[0].review_text
    assert reviews[0].reviewer_name == "Agency Ops Ltd"
    assert reviews[0].reviewer_country == "Australia"
    assert reviews[0].reviewer_type == "Over 1 year using the app"
    assert reviews[0].review_url.endswith("/reviews/101")
    assert reviews[1].review_title == "Great app"


def test_shopify_adapter_skips_fetches_during_rate_limit_cooldown():
    adapter = ShopifyAppReviewAdapter("Mozilla/5.0", rate_limit_cooldown_seconds=300)
    adapter._rate_limited_until = datetime.now(UTC) + timedelta(seconds=300)

    result = asyncio.run(
        adapter.fetch_reviews(
            app_handles=["backup-and-sync"],
            max_apps=1,
            reviews_per_app=1,
            use_sitemap_discovery=False,
        )
    )

    assert result == []


def test_shopify_adapter_marks_rate_limit_on_429():
    adapter = ShopifyAppReviewAdapter("Mozilla/5.0", rate_limit_cooldown_seconds=300)

    class FakeResponse:
        status = 429

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return ""

    class FakeSession:
        def get(self, url):
            return FakeResponse()

    html = asyncio.run(adapter._fetch_html(FakeSession(), "https://apps.shopify.com/backup-and-sync"))

    assert html == ""
    assert adapter._rate_limited_until is not None


def test_shopify_adapter_stops_batch_after_first_rate_limit():
    adapter = ShopifyAppReviewAdapter("Mozilla/5.0", rate_limit_cooldown_seconds=300)
    seen_handles: list[str] = []

    async def fake_fetch_app_reviews(session, *, app_handle, reviews_per_app, rating_filters, sort_by):
        seen_handles.append(app_handle)
        adapter._mark_shopify_rate_limited()
        return []

    adapter._fetch_app_reviews = fake_fetch_app_reviews

    result = asyncio.run(
        adapter.fetch_reviews(
            app_handles=["backup-and-sync", "parcel-intelligence", "matrixify"],
            max_apps=3,
            reviews_per_app=1,
            use_sitemap_discovery=False,
        )
    )

    assert result == []
    assert seen_handles == ["backup-and-sync"]
