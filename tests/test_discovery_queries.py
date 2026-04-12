"""Tests for discovery query defaults."""

import os
import sys

import yaml

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
    assert "invoice reconciliation" in keywords
    assert "payment reconciliation" in keywords
    assert "shopify payout reconciliation" in keywords
    assert "bank deposits not matching invoices" in keywords
    assert "credit memo tracking spreadsheet" in keywords


def test_reddit_problem_subreddits_prioritize_practitioner_lanes_ahead_of_meta():
    cfg = {
        "discovery": {
            "reddit": {
                "problem_subreddits": [
                    "accounting",
                    "quickbooksonline",
                    "smallbusiness",
                    "Bookkeeping",
                    "Netsuite",
                ]
            }
        }
    }

    subreddits = reddit_problem_subreddits(cfg)

    assert subreddits[:5] == ["accounting", "Bookkeeping", "quickbooksonline", "Netsuite", "smallbusiness"]


def test_repo_config_defaults_bias_toward_reddit_only_discovery():
    root = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(root, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    discovery = cfg["discovery"]

    assert discovery["sources"] == ["reddit"]
    assert discovery["candidate_filter"]["min_score"] == 2
    assert discovery["candidate_filter"]["behavioral_min_signals"] == 2
    assert discovery["auto_expand"] is False
    assert discovery["llm_expansion"]["enabled"] is False
    assert discovery["reddit"]["use_r_all"] is False
    assert discovery["reddit"]["search_time_filter"] == "month"
    assert discovery["reddit"]["search_sorts"] == ["new"]
    assert discovery["reddit"]["problem_subreddits"] == [
        "accounting",
        "Bookkeeping",
        "quickbooksonline",
        "Netsuite",
        "smallbusiness",
    ]
