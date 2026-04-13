"""Tests for promotion/park threshold resolution (config drift guard)."""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validation_thresholds import resolve_promotion_park_thresholds


def test_defaults_when_empty():
    p, k = resolve_promotion_park_thresholds({})
    assert p == 0.62
    assert k == 0.48


def test_top_level_validation_keys_used():
    """config.yaml uses validation.promotion_threshold — must not be ignored."""
    p, k = resolve_promotion_park_thresholds(
        {"validation": {"promotion_threshold": 0.66, "park_threshold": 0.42}},
    )
    assert p == 0.66
    assert k == 0.42


def test_decisions_override_top_level():
    p, k = resolve_promotion_park_thresholds(
        {
            "validation": {
                "promotion_threshold": 0.66,
                "park_threshold": 0.42,
                "decisions": {"promote_score": 0.71, "park_score": 0.4},
            },
        },
    )
    assert p == 0.71
    assert k == 0.4


def test_repo_config_yaml_matches_expectation():
    root = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(root, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p, k = resolve_promotion_park_thresholds(cfg)
    assert p == pytest.approx(0.45)
    assert k == pytest.approx(0.35)
