"""Self-expanding discovery: automatically add new keywords/subreddits based on what works."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from src.discovery_suggestions import build_discovery_suggestions
from src.discovery_term_lifecycle import TermLifecycleManager
from src.discovery_next_wave import generate_next_wave
from src.runtime.paths import resolve_project_path

logger = logging.getLogger(__name__)

DEFAULT_EXPANSION_PATH = Path("data/discovery_expansion.json")
_SUBREDDIT_NAME_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")


def _normalize_term_list(value: Any, *, subreddit: bool = False) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        items = [piece for piece in re.split(r"[\n,]+", value) if piece]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if subreddit:
            text = re.sub(r"^/?r/", "", text, flags=re.IGNORECASE).strip()
            if not _SUBREDDIT_NAME_RE.fullmatch(text):
                continue
        if text not in normalized:
            normalized.append(text)
    return normalized


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
            state = json.loads(expansion_file.read_text())
            if isinstance(state, dict):
                return {
                    "keywords": _normalize_term_list(state.get("keywords", [])),
                    "subreddits": _normalize_term_list(state.get("subreddits", []), subreddit=True),
                    "last_expansion_ts": state.get("last_expansion_ts", 0),
                }
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

    Now filters by term state AND prioritizes by wedge quality.
    Prefers terms with high plugin/add-on/microSaaS fit over vague abstractions.
    Returns dict with 'keywords' and 'subreddits' lists.
    """
    # First get high-performing terms from lifecycle manager, prioritized by wedge quality
    lifecycle = TermLifecycleManager(db)

    # Get terms sorted by wedge quality (best plugin/add-on fit first)
    wedge_quality_keywords = lifecycle.get_terms_for_expansion_by_wedge_quality("keyword", limit=30)
    wedge_quality_subreddits = lifecycle.get_terms_for_expansion_by_wedge_quality("subreddit", limit=20)

    # Also get high-performing terms
    high_perf_keywords = lifecycle.get_terms_for_expansion("keyword", limit=30)
    high_perf_subreddits = lifecycle.get_terms_for_expansion("subreddit", limit=20)

    # Combine: prioritize wedge quality, then high performing
    winning_keywords = []
    winning_subreddits = []

    # First add wedge quality terms
    for t in wedge_quality_keywords:
        if t.get("state") in ("high_performing", "active", "used"):
            winning_keywords.append(t["term_value"])

    # Fill in with high performing terms not already included
    seen_kw = set(winning_keywords)
    for t in high_perf_keywords:
        if t["term_value"] not in seen_kw and t.get("state") in ("high_performing", "active", "used"):
            winning_keywords.append(t["term_value"])
            seen_kw.add(t["term_value"])

    # Same for subreddits
    for t in wedge_quality_subreddits:
        if t.get("state") in ("high_performing", "active", "used"):
            winning_subreddits.append(t["term_value"])

    seen_sub = set(winning_subreddits)
    for t in high_perf_subreddits:
        if t["term_value"] not in seen_sub and t.get("state") in ("high_performing", "active", "used"):
            winning_subreddits.append(t["term_value"])
            seen_sub.add(t["term_value"])

    # Fall back to query-level feedback for terms not yet in lifecycle table
    conn = db._get_connection()

    # Get keywords with good validation scores (from discovery_feedback)
    keyword_rows = conn.execute("""
        SELECT query_text, avg_validation_score, prototype_candidates
        FROM discovery_feedback
        WHERE source_name LIKE 'reddit%'
          AND avg_validation_score >= ?
          AND prototype_candidates > 0
        ORDER BY avg_validation_score DESC, prototype_candidates DESC
        LIMIT 20
    """, (min_score,)).fetchall()

    # Filter out keywords that are exhausted/banned/paused
    existing_keywords = {t["term_value"] for t in lifecycle.db.list_search_terms(term_type="keyword", limit=500)}
    for row in keyword_rows:
        q = row["query_text"]
        if q and q not in existing_keywords:
            winning_keywords.append(q)
        elif q:
            # Check if term exists and its state
            term = lifecycle.db.get_search_term("keyword", q)
            if term and term.get("state") in ("new", "active", "used", "high_performing"):
                winning_keywords.append(q)

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

    for row in subreddit_rows:
        sub = row["sub"]
        if sub and sub not in winning_subreddits:
            # Check subreddit state
            term = lifecycle.db.get_search_term("subreddit", sub)
            if term is None or term.get("state") in ("new", "active", "used", "high_performing"):
                winning_subreddits.append(sub)

    # Dedupe
    winning_keywords = list(dict.fromkeys(winning_keywords))[:30]
    winning_subreddits = list(dict.fromkeys(winning_subreddits))[:20]

    logger.info(f"Winning patterns: {len(winning_keywords)} keywords, {len(winning_subreddits)} subreddits (filtered by state)")

    return {
        "keywords": winning_keywords,
        "subreddits": winning_subreddits,
    }


def get_winning_patterns_hybrid(db, max_keywords: int = 5, max_subreddits: int = 5) -> dict[str, list[str]]:
    """Get next-wave terms using hybrid quality+output selector.

    Uses the new hybrid selector that combines:
    - Quality component (60%): wedge_quality, specificity, consequence, platform_native, plugin_fit
    - Output component (40%): findings, validations, prototype_candidates, build_briefs

    Includes:
    - Locked term seeding (until selector proves itself)
    - Challenger replacement rules
    - Regression guardrails

    Returns dict with 'keywords' and 'subreddits' lists.
    """
    result = generate_next_wave(
        db,
        max_keywords=max_keywords,
        max_subreddits=max_subreddits,
        use_locked_as_seed=True,
        allow_replacement=True,
    )

    keywords = [kw['term_value'] for kw in result['keywords']]
    subreddits = [sub['term_value'] for sub in result['subreddits']]

    # Log observability
    obs = result['observability']
    logger.info(f"Hybrid next wave: {len(keywords)} keywords, {len(subreddits)} subreddits")
    logger.info(f"  Keywords retained: {obs['keyword_retained']}")
    if obs['keyword_replaced']:
        logger.info(f"  Keywords replaced: {obs['keyword_replaced']}")
    if obs['keyword_rejected']:
        logger.info(f"  Keywords rejected: {len(obs['keyword_rejected'])}")
    logger.info(f"  Subreddits retained: {obs['sub_retained']}")

    return {
        "keywords": keywords,
        "subreddits": subreddits,
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

    # Filter out exhausted/banned terms using lifecycle manager
    lifecycle = TermLifecycleManager(db)
    existing_terms = {t["term_value"].lower(): t["state"] for t in lifecycle.db.list_search_terms(limit=500)}

    def is_allowed_term(term_value: str) -> bool:
        """Check if a term is allowed for discovery."""
        lower = term_value.lower()
        # Check if already exists in our term tracking
        if lower in existing_terms:
            state = existing_terms[lower]
            # Only allow if not exhausted/banned/paused
            return state in ("new", "active", "used", "high_performing")
        # New terms are allowed
        return True

    new_keywords = [k for k in suggested_keywords if k.lower() not in base_keyword_keys and is_allowed_term(k)]
    new_subreddits = [s for s in suggested_subs if s.lower() not in base_subreddit_keys and is_allowed_term(s)]

    logger.info(f"Generated {len(new_keywords)} keyword candidates (filtered), {len(new_subreddits)} subreddit candidates (filtered)")

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

    # Register new terms in lifecycle system
    lifecycle = TermLifecycleManager(db)
    for kw in candidates.get("keywords", []):
        if kw:
            lifecycle.ensure_term_exists("keyword", kw)
    for sub in candidates.get("subreddits", []):
        if sub:
            lifecycle.ensure_term_exists("subreddit", sub)

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
    expanded_keywords = _normalize_term_list(state.get("keywords", []))
    expanded_subreddits = _normalize_term_list(state.get("subreddits", []), subreddit=True)

    if not expanded_keywords and not expanded_subreddits:
        return config

    # Deep copy to avoid modifying original
    import copy
    merged = copy.deepcopy(config)

    reddit_config = merged.get("discovery", {}).get("reddit", {})
    base_keywords = _normalize_term_list(reddit_config.get("problem_keywords", []))
    base_subreddits = _normalize_term_list(reddit_config.get("problem_subreddits", []), subreddit=True)
    if not bool(reddit_config.get("use_r_all")):
        expanded_subreddits = [sub for sub in expanded_subreddits if sub.lower() != "all"]

    # Merge
    all_keywords = _merge_unique(base_keywords, expanded_keywords)
    all_subreddits = _merge_unique(base_subreddits, expanded_subreddits)

    reddit_section = merged.setdefault("discovery", {}).setdefault("reddit", {})
    reddit_section["problem_keywords"] = all_keywords
    reddit_section["problem_subreddits"] = all_subreddits

    logger.info(f"Merged config: {len(all_keywords)} keywords ({len(expanded_keywords)} expanded), "
                f"{len(all_subreddits)} subreddits ({len(expanded_subreddits)} expanded)")

    return merged


def get_winning_patterns_from_problem_spaces(db: Any) -> list[dict[str, Any]]:
    """Read problem space metrics as an alternative source for expansion patterns.

    Returns a list of dicts with 'keywords' and 'subreddits' from validated
    and exploring problem spaces, weighted by yield_score.
    """
    from src.problem_space import VALIDATED
    from src.problem_space_lifecycle import ProblemSpaceLifecycleManager

    try:
        spaces = db.list_problem_spaces(limit=50)
    except Exception:
        return []

    patterns: list[dict[str, Any]] = []
    for space in spaces:
        if space.status not in (VALIDATED, "exploring"):
            continue
        if space.yield_score < 0.1:
            continue
        patterns.append({
            "space_key": space.space_key,
            "label": space.label,
            "keywords": space.keywords or [],
            "subreddits": space.subreddits or [],
            "yield_score": space.yield_score,
        })

    patterns.sort(key=lambda p: p["yield_score"], reverse=True)
    return patterns
