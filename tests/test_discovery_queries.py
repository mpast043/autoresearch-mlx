"""Tests for discovery query defaults."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discovery_queries import reddit_discovery_subreddits, reddit_problem_subreddits


def test_reddit_discovery_subreddits_respects_use_r_all():
    cfg = {"discovery": {"reddit": {"use_r_all": True, "problem_subreddits": ["smallbusiness"]}}}
    assert reddit_discovery_subreddits(cfg) == ["all"]


def test_reddit_discovery_subreddits_defaults_to_problem_subreddits():
    cfg = {"discovery": {"reddit": {"use_r_all": False, "problem_subreddits": ["foo", "bar"]}}}
    assert reddit_discovery_subreddits(cfg) == reddit_problem_subreddits(cfg)
