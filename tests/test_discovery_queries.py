"""Tests for discovery query defaults."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discovery_queries import reddit_discovery_subreddits, reddit_problem_keywords, reddit_problem_subreddits


def test_reddit_discovery_subreddits_respects_use_r_all():
    cfg = {"discovery": {"reddit": {"use_r_all": True, "problem_subreddits": ["smallbusiness"]}}}
    assert reddit_discovery_subreddits(cfg) == ["all"]


def test_reddit_discovery_subreddits_defaults_to_problem_subreddits():
    cfg = {"discovery": {"reddit": {"use_r_all": False, "problem_subreddits": ["foo", "bar"]}}}
    assert reddit_discovery_subreddits(cfg) == reddit_problem_subreddits(cfg)


def test_reddit_problem_keywords_merge_curated_operator_pack():
    cfg = {"discovery": {"reddit": {"problem_keywords": ["manual reconciliation", "custom niche pain"]}}}
    keywords = reddit_problem_keywords(cfg)

    assert keywords[0] == "manual reconciliation"
    assert "custom niche pain" in keywords
    assert "month end close spreadsheet" in keywords
    assert "sales channel reconciliation spreadsheet" in keywords
    assert "invoice reminder spreadsheet workflow" in keywords
    assert "pdf collaboration version control" in keywords


def test_reddit_problem_subreddits_prioritize_practitioner_lanes_ahead_of_meta():
    cfg = {
        "discovery": {
            "reddit": {
                "problem_subreddits": [
                    "projectmanagement",
                    "automation",
                    "accounting",
                    "smallbusiness",
                    "shopify",
                    "indiehackers",
                ]
            }
        }
    }

    subreddits = reddit_problem_subreddits(cfg)

    assert subreddits[:7] == ["accounting", "Bookkeeping", "Excel", "smallbusiness", "ecommerce", "shopify", "EtsySellers"]
    assert subreddits.index("projectmanagement") > subreddits.index("shopify")
    assert subreddits.index("automation") > subreddits.index("EtsySellers")
