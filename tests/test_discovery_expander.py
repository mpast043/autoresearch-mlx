import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database
from src.discovery_expander import (
    get_expanded_config,
    load_expansion_state,
    run_expansion,
    save_expansion_state,
)


@pytest.fixture
def temp_db():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        yield db
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_get_expanded_config_handles_missing_discovery_branch(tmp_path):
    state_path = tmp_path / "discovery_expansion.json"
    save_expansion_state(
        {"keywords": ["manual handoff"], "subreddits": ["operations"], "last_expansion_ts": 0},
        path=state_path,
    )

    cfg = {"discovery": {"expansion": {"state_path": str(state_path)}}}
    expanded = get_expanded_config(cfg)

    assert expanded["discovery"]["reddit"]["problem_keywords"] == ["manual handoff"]
    assert expanded["discovery"]["reddit"]["problem_subreddits"] == ["operations"]


def test_get_expanded_config_merges_and_dedupes_stably(tmp_path):
    state_path = tmp_path / "discovery_expansion.json"
    save_expansion_state(
        {
            "keywords": ["Spreadsheet glue", "manual handoff"],
            "subreddits": ["Operations", "excel"],
            "last_expansion_ts": 0,
        },
        path=state_path,
    )

    cfg = {
        "discovery": {
            "expansion": {"state_path": str(state_path)},
            "reddit": {
                "problem_keywords": ["spreadsheet glue", "copy paste"],
                "problem_subreddits": ["excel", "sysadmin"],
            },
        }
    }

    expanded = get_expanded_config(cfg)

    assert expanded["discovery"]["reddit"]["problem_keywords"] == [
        "spreadsheet glue",
        "copy paste",
        "manual handoff",
    ]
    assert expanded["discovery"]["reddit"]["problem_subreddits"] == [
        "excel",
        "sysadmin",
        "Operations",
    ]


def test_run_expansion_honors_separate_keyword_and_subreddit_limits(temp_db, tmp_path, monkeypatch):
    state_path = tmp_path / "discovery_expansion.json"
    cfg = {
        "discovery": {
            "auto_expand": True,
            "expansion": {
                "state_path": str(state_path),
                "cooldown_hours": 0,
                "max_keywords_per_wave": 2,
                "max_subreddits_per_wave": 1,
            },
            "reddit": {
                "problem_keywords": ["existing keyword"],
                "problem_subreddits": ["existingsub"],
            },
        }
    }

    monkeypatch.setattr(
        "discovery_expander.get_winning_patterns",
        lambda db, min_score=0.5: {
            "keywords": ["winner one", "winner two", "winner three"],
            "subreddits": ["ops", "excel"],
        },
    )
    monkeypatch.setattr(
        "discovery_expander.build_discovery_suggestions",
        lambda db, max_keywords=0: {
            "suggested_keywords": ["winner two", "suggested three"],
            "suggested_subreddits_from_findings": ["ops", "finance"],
        },
    )

    result = run_expansion(temp_db, cfg)
    state = load_expansion_state(path=state_path)

    assert result["expanded"] is True
    assert result["added_keywords"] == ["winner one", "winner two"]
    assert result["added_subreddits"] == ["ops"]
    assert state["keywords"] == ["winner one", "winner two"]
    assert state["subreddits"] == ["ops"]
