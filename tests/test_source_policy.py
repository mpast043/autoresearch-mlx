"""Tests for the first-class source policy layer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from source_policy import atom_generation_allowed, normalize_source_class, policy_for


def test_success_signal_is_not_atom_eligible():
    policy = policy_for("success_signal")
    assert policy.atom_eligible is False
    assert policy.use_for_search_seed is True
    assert policy.discovery_status == "screened_out"


def test_pain_signal_is_atom_eligible():
    policy = policy_for("pain_signal")
    assert policy.atom_eligible is True
    assert policy.exclude_from_active_path is False


def test_legacy_success_kind_normalizes_to_success_signal():
    assert normalize_source_class("", "success_signal") == "success_signal"
    assert atom_generation_allowed("", "success_signal") is False
