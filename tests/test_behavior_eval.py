"""Tests for the gold-set behavioral eval harness."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from behavior_eval import run_behavior_eval


def test_behavior_eval_gold_set_passes():
    fixtures = Path(__file__).resolve().parents[1] / "evals" / "behavior_gold.json"
    report = run_behavior_eval(fixtures)

    assert report["failed_cases"] == 0
    assert report["total_cases"] >= 4
    assert report["pass_rate"] == 1.0
