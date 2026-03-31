"""Self-expanding discovery: automatically add new keywords/subreddits based on what works."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.discovery_suggestions import build_discovery_suggestions
from src.runtime.paths import resolve_project_path

logger = logging.getLogger(__name__)

DEFAULT_EXPANSION_PATH = Path("data/discovery_expansion.json")


def _expansion_file(config: dict[str, Any] | None = None, *, path: str | Path | None = None) -> Path:
    if path is not None:
        return resolve_project_path(path)
    configured = (config or {}).get("discovery", {}).get("expansion", {}).get("state_path")
    return resolve_project_path(configured, default=DEFAULT_EXPANSION_PATH)


def load_expansion_state(
    config: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load current expansion state from file."""
    expansion_file = _expansion_file(config, path=path)
    if expansion_file.exists():
        try:
            return json.loads(expansion_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "keywords": [],
        "subreddits": [],
        "last_expansion_ts": 0,
    }


def save_expansion_state(
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> None:
    """Save expansion state to file."""
    expansion_file = _expansion_file(config, path=path)
    expansion_file.parent.mkdir(parents=True, exist_ok=True)
    expansion_file.write_text(json.dumps(state, indent=2))


def get_winning_patterns(db, min_score: float = 0.5) -> dict[str, list[str]]:
    """Find keywords and subreddits that have high validation scores.

    Returns dict with 'keywords' and 'subreddits' lists.
    """
    conn = db._get_connection()

    # Get keywords with good validation scores
    keyword_rows = conn.execute("""
        SELECT query_text, avg_validation_score, prototype_candidates
        FROM discovery_feedback
        WHERE source_name LIKE 'reddit%'
          AND avg_validation_score >= ?
          AND prototype_candidates > 0
        ORDER BY avg_validation_score DESC, prototype_candidates DESC
        LIMIT 20
    """, (min_score,)).fetchall()

    winning_keywords = [row["query_text"] for row in keyword_rows if row["query_text"]]

    # Get subreddits that yielded findings with good validation scores
    subreddit_rows = conn.execute("""
        SELECT DISTINCT
            substr(source_url, 9, instr(substr(source_url, 9), '/') - 1) as sub
        FROM findings f
        JOIN discovery_feedback df ON df.query_text LIKE '%' || substr(f.source_url, 9, instr(substr(f.source_url, 9), '/') - 1) || '%'
        WHERE f.source LIKE 'reddit%'
          AND f.status IN ('promoted', 'qualified')
          AND df.avg_validation_score >= ?
        LIMIT 20
    """, (min_score,)).fetchall()

    winning_subreddits = [row["sub"] for row in subreddit_rows if row["sub"]]

    return {
        "keywords": winning_keywords,
        "subreddits": winning_subreddits,
    }


def _merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*(existing or []), *(new_items or [])]:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def generate_candidates(
    db,
    base_keywords: list[str],
    base_subreddits: list[str],
    *,
    max_new_keywords: int = 3,
    max_new_subreddits: int = 2,
    winning: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Generate new keyword candidates based on winning patterns.

    Uses existing problem atoms to extract similar language patterns.
    """
    suggestions = build_discovery_suggestions(
        db,
        max_keywords=max(max_new_keywords, max_new_subreddits) * 3,
    )

    winning = winning or {"keywords": [], "subreddits": []}
    suggested_keywords = _merge_unique(
        winning.get("keywords", []),
        suggestions.get("suggested_keywords", []),
    )
    suggested_subs = _merge_unique(
        winning.get("subreddits", []),
        suggestions.get("suggested_subreddits_from_findings", []),
    )

    base_keyword_keys = {bk.lower() for bk in base_keywords}
    base_subreddit_keys = {bs.lower() for bs in base_subreddits}

    new_keywords = [k for k in suggested_keywords if k.lower() not in base_keyword_keys]
    new_subreddits = [s for s in suggested_subs if s.lower() not in base_subreddit_keys]

    return {
        "keywords": new_keywords[:max_new_keywords],
        "subreddits": new_subreddits[:max_new_subreddits],
    }


def run_expansion(db, config: dict[str, Any]) -> dict[str, Any]:
    """Run expansion logic if enabled and cooldown has passed.

    Returns dict with 'expanded' bool and 'added_keywords', 'added_subreddits' lists.
    """
    expansion_config = config.get("discovery", {}).get("expansion", {})
    auto_expand = config.get("discovery", {}).get("auto_expand", False)

    if not auto_expand:
        return {"expanded": False, "added_keywords": [], "added_subreddits": []}

    # Check cooldown
    cooldown_hours = expansion_config.get("cooldown_hours", 24)
    state = load_expansion_state(config)
    last_ts = state.get("last_expansion_ts", 0)
    hours_since = (time.time() - last_ts) / 3600

    if hours_since < cooldown_hours:
        logger.debug(f"Expansion cooldown: {hours_since:.1f}h < {cooldown_hours}h")
        return {"expanded": False, "added_keywords": [], "added_subreddits": [], "reason": "cooldown"}

    # Get limits
    max_keywords = expansion_config.get("max_keywords_per_wave", 3)
    max_subreddits = expansion_config.get("max_subreddits_per_wave", 2)
    min_score = expansion_config.get("min_validation_score", 0.5)

    # Get base config
    base_config = config.get("discovery", {}).get("reddit", {})
    base_keywords = base_config.get("problem_keywords", [])
    base_subreddits = base_config.get("problem_subreddits", [])

    # Get winning patterns
    winning = get_winning_patterns(db, min_score)
    logger.info(f"Found {len(winning['keywords'])} winning keywords, {len(winning['subreddits'])} winning subreddits")

    # Generate candidates
    candidates = generate_candidates(
        db,
        base_keywords,
        base_subreddits,
        max_new_keywords=max_keywords,
        max_new_subreddits=max_subreddits,
        winning=winning,
    )
    logger.info(f"Generated {len(candidates['keywords'])} new keyword candidates, {len(candidates['subreddits'])} subreddit candidates")

    # Add to state
    state["keywords"] = _merge_unique(state.get("keywords", []), candidates["keywords"])
    state["subreddits"] = _merge_unique(state.get("subreddits", []), candidates["subreddits"])
    state["last_expansion_ts"] = time.time()

    save_expansion_state(state, config)

    logger.info(f"Expansion complete: added {len(candidates['keywords'])} keywords, {len(candidates['subreddits'])} subreddits")

    return {
        "expanded": True,
        "added_keywords": candidates["keywords"],
        "added_subreddits": candidates["subreddits"],
    }


def get_expanded_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge base config with expanded keywords/subreddits.

    Returns a modified config dict with expanded discovery scope.
    """
    state = load_expansion_state(config)
    expanded_keywords = state.get("keywords", [])
    expanded_subreddits = state.get("subreddits", [])

    if not expanded_keywords and not expanded_subreddits:
        return config

    # Deep copy to avoid modifying original
    import copy
    merged = copy.deepcopy(config)

    reddit_config = merged.get("discovery", {}).get("reddit", {})
    base_keywords = reddit_config.get("problem_keywords", [])
    base_subreddits = reddit_config.get("problem_subreddits", [])

    # Merge
    all_keywords = _merge_unique(base_keywords, expanded_keywords)
    all_subreddits = _merge_unique(base_subreddits, expanded_subreddits)

    reddit_section = merged.setdefault("discovery", {}).setdefault("reddit", {})
    reddit_section["problem_keywords"] = all_keywords
    reddit_section["problem_subreddits"] = all_subreddits

    logger.info(f"Merged config: {len(all_keywords)} keywords ({len(expanded_keywords)} expanded), "
                f"{len(all_subreddits)} subreddits ({len(expanded_subreddits)} expanded)")

    return merged