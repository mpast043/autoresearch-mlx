"""Hybrid next-wave term selector.

Combines quality (60%) and historical output (40%) to select next-wave terms.
Includes locked-term seeding, challenger replacement rules, and regression guardrails.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.database import Database

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_QUALITY_WEIGHT = 0.6
DEFAULT_OUTPUT_WEIGHT = 0.4
DEFAULT_MAX_KEYWORDS = 5
DEFAULT_MAX_SUBREDDITS = 5
DEFAULT_CHALLENGER_MARGIN = 0.2
DEFAULT_MIN_HYBRID_SCORE = 0.05
DEFAULT_LOCKED_SEED_MIN_QUALITY = 0.1

# Regression blocklist - terms always penalized
REGRESSION_BLOCKLIST = [
    'keep sync and data handoff',
    'operator - keep sync and data handoff',
    'copy paste workflow',
    'complex logic managed entirely through',
    'spreadsheet hell',
    'held together by spreadsheets',
    'duct tape spreadsheets',
    'i am trying to make our month-end',
]

# Locked default terms (seed until selector proves itself)
LOCKED_KEYWORDS = [
    'manual handoff workflow',
    'bank deposit reconciliation spreadsheet',
    'manual reconciliation',
    'sales channel reconciliation spreadsheet',
    'returns workflow spreadsheet',
]

LOCKED_SUBREDDITS = [
    'airtable',
    'excel',
    'notion',
    'shopify',
    'automation',
]


def load_locked_terms(config_path: str | Path | None = None) -> dict[str, list[str]]:
    """Load locked term set from config file or return defaults."""
    if config_path is None:
        config_path = Path('data/next_wave_locked.json')

    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass

    return {'keywords': LOCKED_KEYWORDS, 'subreddits': LOCKED_SUBREDDITS}


def normalize_term(term: str) -> str:
    """Canonicalize a term for alias deduplication."""
    term_lower = term.lower().strip()

    # Remove common prefixes
    for prefix in ('operator - ', 'developer - ', 'finance - '):
        if term_lower.startswith(prefix):
            term_lower = term_lower[len(prefix):]

    return ' '.join(term_lower.split())


def calculate_hybrid_score(
    term_value: str,
    term_type: str,
    db_conn,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    output_weight: float = DEFAULT_OUTPUT_WEIGHT,
) -> dict[str, Any] | None:
    """Calculate hybrid score combining quality (60%) and output (40%).

    Quality component (60%):
    - wedge_quality_score (base)
    - abstraction penalty
    - vague bucket penalty
    - regression penalty

    Output component (40%):
    - findings_emitted
    - validations
    - prototype_candidates (highest weight)
    - build_briefs (highest weight)
    - screened_out penalty
    """

    row = db_conn.execute('''
        SELECT
            wedge_quality_score, specificity_score, consequence_score,
            platform_native_score, plugin_fit_score,
            vague_bucket_count, abstraction_collapse_count,
            findings_emitted, validations, passes,
            prototype_candidates, build_briefs, screened_out, state
        FROM discovery_search_terms
        WHERE term_value = ? AND term_type = ?
    ''', (term_value, term_type)).fetchone()

    if not row:
        return None

    # === QUALITY COMPONENT ===
    base_quality = row['wedge_quality_score'] or 0.0

    # Abstraction penalty
    abstr_penalty = min(0.2, (row['abstraction_collapse_count'] or 0) * 0.05)

    # Vague bucket penalty
    vague_penalty = 0.3 if (row['vague_bucket_count'] or 0) > 0 else 0

    # Regression penalty (hard block for known bad terms)
    regress_penalty = 0.0
    term_lower = term_value.lower()
    for pattern in REGRESSION_BLOCKLIST:
        if pattern in term_lower:
            regress_penalty = 0.4
            break

    quality_score = max(0.0, base_quality - abstr_penalty - vague_penalty - regress_penalty)

    # === OUTPUT COMPONENT ===
    output_score = 0.0

    findings = row['findings_emitted'] or 0
    validations = row['validations'] or 0
    pc = row['prototype_candidates'] or 0
    bf = row['build_briefs'] or 0
    screened = row['screened_out'] or 0

    # Weighted output scoring
    output_score += min(1.0, findings / 50.0) * 0.15
    output_score += min(1.0, validations / 20.0) * 0.15
    output_score += min(1.0, pc / 10.0) * 0.25
    output_score += min(1.0, bf / 5.0) * 0.25

    # Penalize high screened-out rate
    if screened > 20:
        output_score *= 0.5

    # === HYBRID COMBINATION ===
    final_score = (quality_score * quality_weight) + (output_score * output_weight)

    return {
        'term_value': term_value,
        'term_type': term_type,
        'state': row['state'],
        'hybrid_score': round(final_score, 3),
        'quality_component': round(quality_score, 3),
        'output_component': round(output_score, 3),
        'wedge_quality': row['wedge_quality_score'] or 0,
        'specificity': row['specificity_score'] or 0,
        'consequence': row['consequence_score'] or 0,
        'platform_native': row['platform_native_score'] or 0,
        'plugin_fit': row['plugin_fit_score'] or 0,
        'vague_bucket': row['vague_bucket_count'] or 0,
        'abstraction_collapse': row['abstraction_collapse_count'] or 0,
        'findings_emitted': findings,
        'validations': validations,
        'prototype_candidates': pc,
        'build_briefs': bf,
        'screened_out': screened,
    }


def is_excluded_by_lifecycle(state: str) -> bool:
    """Check if term should be excluded by lifecycle state."""
    return state in ('exhausted', 'paused', 'banned', 'completed')


def generate_next_wave(
    db: Database,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    max_subreddits: int = DEFAULT_MAX_SUBREDDITS,
    use_locked_as_seed: bool = True,
    allow_replacement: bool = True,
    challenger_margin: float = DEFAULT_CHALLENGER_MARGIN,
    min_hybrid_score: float = DEFAULT_MIN_HYBRID_SCORE,
    locked_seed_min_quality_score: float = DEFAULT_LOCKED_SEED_MIN_QUALITY,
    locked_config: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Generate next-wave terms using hybrid selector.

    Args:
        db: Database instance
        max_keywords: Maximum keywords to select
        max_subreddits: Maximum subreddits to select
        use_locked_as_seed: Use locked set as seed (default True)
        allow_replacement: Allow challengers to replace locked terms
        challenger_margin: How much better a challenger must be to replace
        min_hybrid_score: Minimum hybrid score to be included
        locked_seed_min_quality_score: Minimum quality score for historical locked seeds
        locked_config: Locked term config (defaults to LOCKED_*)

    Returns:
        dict with selected keywords, subreddits, and observability data
    """

    if locked_config is None:
        locked_config = load_locked_terms()

    locked_keywords_list = list(locked_config.get('keywords', LOCKED_KEYWORDS))
    locked_subreddits_list = list(locked_config.get('subreddits', LOCKED_SUBREDDITS))
    locked_keywords = set(locked_keywords_list)
    locked_subreddits = set(locked_subreddits_list)

    conn = db._get_connection()
    conn.row_factory = sqlite3.Row

    # Get all non-excluded keywords
    keyword_scores = []
    for row in conn.execute('''
        SELECT term_value, state
        FROM discovery_search_terms
        WHERE term_type = 'keyword'
          AND state NOT IN ('exhausted', 'paused', 'banned')
    ''').fetchall():
        score = calculate_hybrid_score(row['term_value'], 'keyword', conn)
        if score and score['hybrid_score'] >= min_hybrid_score:
            keyword_scores.append(score)

    # Get all non-excluded subreddits
    sub_scores = []
    for row in conn.execute('''
        SELECT term_value, state
        FROM discovery_search_terms
        WHERE term_type = 'subreddit'
          AND state NOT IN ('exhausted', 'paused', 'banned')
    ''').fetchall():
        score = calculate_hybrid_score(row['term_value'], 'subreddit', conn)
        if score and score['hybrid_score'] >= min_hybrid_score:
            sub_scores.append(score)

    # Sort by hybrid score descending
    keyword_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
    sub_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)

    placeholder_baseline = max(min_hybrid_score, locked_seed_min_quality_score)

    def _placeholder_score(term_value: str, term_type: str) -> dict[str, Any]:
        return {
            'term_value': term_value,
            'term_type': term_type,
            'state': 'locked_default',
            'hybrid_score': round(placeholder_baseline, 3),
            'quality_component': round(placeholder_baseline, 3),
            'output_component': 0.0,
            'wedge_quality': round(placeholder_baseline, 3),
            'specificity': round(placeholder_baseline, 3),
            'consequence': round(placeholder_baseline, 3),
            'platform_native': 0.0,
            'plugin_fit': 0.0,
            'vague_bucket': 0,
            'abstraction_collapse': 0,
            'findings_emitted': 0,
            'validations': 0,
            'prototype_candidates': 0,
            'build_briefs': 0,
            'screened_out': 0,
            'score_source': 'placeholder_no_history',
            'quality_known': False,
        }

    def _locked_seed_score(term_value: str, term_type: str) -> dict[str, Any] | None:
        score = calculate_hybrid_score(term_value, term_type, conn)
        if score is None:
            return _placeholder_score(term_value, term_type)
        seeded = dict(score)
        seeded['score_source'] = 'historical'
        seeded['quality_known'] = True
        if is_excluded_by_lifecycle(str(seeded.get('state', ''))):
            return None
        quality_history = any(
            float(seeded.get(key, 0.0) or 0.0) > 0.0
            for key in ('wedge_quality', 'specificity', 'consequence', 'platform_native', 'plugin_fit')
        ) or any(
            int(seeded.get(key, 0) or 0) > 0
            for key in ('vague_bucket', 'abstraction_collapse')
        )
        if not quality_history:
            placeholder = _placeholder_score(term_value, term_type)
            placeholder['state'] = seeded.get('state', 'locked_default')
            placeholder['score_source'] = 'placeholder_missing_quality_history'
            return placeholder
        if float(seeded.get('quality_component', 0.0) or 0.0) < locked_seed_min_quality_score:
            return None
        return seeded

    # === SELECT KEYWORDS ===
    selected_keywords = []
    keyword_retained = []
    keyword_replaced = []
    keyword_rejected = []

    if use_locked_as_seed:
        for term in locked_keywords_list[:max_keywords]:
            seeded = dict(_locked_seed_score(term, 'keyword') or {})
            if not seeded:
                keyword_rejected.append({
                    'term': term,
                    'reason': 'locked_seed_quality_below_threshold',
                    'score': 0.0,
                })
                continue
            seeded['selection_reason'] = 'locked_default'
            seeded['retained'] = True
            selected_keywords.append(seeded)
            keyword_retained.append(term)

    for score in keyword_scores:
        term = score['term_value']
        if term in [s['term_value'] for s in selected_keywords]:
            continue

        if len(selected_keywords) < max_keywords:
            # Challenger term
            selected_keywords.append({
                **score,
                'selection_reason': 'selected' if not use_locked_as_seed else 'fallback',
                'retained': False
            })
        elif allow_replacement:
            retained_locked = [
                (index, selected)
                for index, selected in enumerate(selected_keywords)
                if selected.get('retained')
            ]
            if not retained_locked:
                keyword_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})
                continue

            weakest_index, weakest_locked = min(
                retained_locked,
                key=lambda item: item[1].get('hybrid_score', 0.0),
            )
            if score['hybrid_score'] >= weakest_locked.get('hybrid_score', 0.0) + challenger_margin:
                selected_keywords[weakest_index] = {
                    **score,
                    'selection_reason': 'challenger_outperformed',
                    'retained': False,
                }
                keyword_replaced.append(term)
            else:
                keyword_rejected.append({'term': term, 'reason': 'insufficient_margin', 'score': score['hybrid_score']})
        else:
            keyword_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})

    # Fill remaining slots if needed
    for score in keyword_scores:
        if len(selected_keywords) >= max_keywords:
            break
        term = score['term_value']
        if term not in [s['term_value'] for s in selected_keywords]:
            selected_keywords.append({
                **score,
                'selection_reason': 'fallback',
                'retained': False
            })
            keyword_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})

    # === SELECT SUBREDDITS ===
    selected_subs = []
    sub_retained = []
    sub_replaced = []
    sub_rejected = []

    if use_locked_as_seed:
        for term in locked_subreddits_list[:max_subreddits]:
            seeded = dict(_locked_seed_score(term, 'subreddit') or {})
            if not seeded:
                sub_rejected.append({
                    'term': term,
                    'reason': 'locked_seed_quality_below_threshold',
                    'score': 0.0,
                })
                continue
            seeded['selection_reason'] = 'locked_default'
            seeded['retained'] = True
            selected_subs.append(seeded)
            sub_retained.append(term)

    for score in sub_scores:
        term = score['term_value']
        if term in [s['term_value'] for s in selected_subs]:
            continue

        if len(selected_subs) < max_subreddits:
            selected_subs.append({
                **score,
                'selection_reason': 'selected' if not use_locked_as_seed else 'fallback',
                'retained': False
            })
        elif allow_replacement:
            retained_locked = [
                (index, selected)
                for index, selected in enumerate(selected_subs)
                if selected.get('retained')
            ]
            if not retained_locked:
                sub_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})
                continue

            weakest_index, weakest_locked = min(
                retained_locked,
                key=lambda item: item[1].get('hybrid_score', 0.0),
            )
            if score['hybrid_score'] >= weakest_locked.get('hybrid_score', 0.0) + challenger_margin:
                selected_subs[weakest_index] = {
                    **score,
                    'selection_reason': 'challenger_outperformed',
                    'retained': False
                }
                sub_replaced.append(term)
            else:
                sub_rejected.append({'term': term, 'reason': 'insufficient_margin', 'score': score['hybrid_score']})
        else:
            sub_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})

    # Fill remaining slots
    for score in sub_scores:
        if len(selected_subs) >= max_subreddits:
            break
        term = score['term_value']
        if term not in [s['term_value'] for s in selected_subs]:
            selected_subs.append({
                **score,
                'selection_reason': 'fallback',
                'retained': False
            })
            sub_rejected.append({'term': term, 'reason': 'budget_filled', 'score': score['hybrid_score']})

    return {
        'keywords': selected_keywords[:max_keywords],
        'subreddits': selected_subs[:max_subreddits],
        'observability': {
            'locked_keywords': list(locked_keywords),
            'locked_subreddits': list(locked_subreddits),
            'keyword_retained': keyword_retained,
            'keyword_replaced': keyword_replaced,
            'keyword_rejected': keyword_rejected,
            'sub_retained': sub_retained,
            'sub_replaced': sub_replaced,
            'sub_rejected': sub_rejected,
        }
    }


def save_next_wave_config(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Save generated next-wave config to file."""
    if output_path is None:
        output_path = Path('data/next_wave_generated.json')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create simplified config
    config = {
        'keywords': [kw['term_value'] for kw in result['keywords']],
        'subreddits': [sub['term_value'] for sub in result['subreddits']],
    }

    # Also save full observability
    full_path = output_path.parent / 'next_wave_full.json'
    full_path.write_text(json.dumps(result, indent=2))

    output_path.write_text(json.dumps(config, indent=2))

    return output_path


def get_next_wave(
    db: Database,
    config: dict[str, Any] | None = None,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    max_subreddits: int = DEFAULT_MAX_SUBREDDITS,
) -> dict[str, Any]:
    """Convenience function to get next-wave terms.

    Combines loading config, generating wave, and saving output.
    """
    result = generate_next_wave(
        db,
        max_keywords=max_keywords,
        max_subreddits=max_subreddits,
        use_locked_as_seed=True,
        allow_replacement=True,
    )

    # Log observability
    obs = result['observability']
    logger.info(f"Next wave: {len(result['keywords'])} keywords, {len(result['subreddits'])} subreddits")
    logger.info(f"  Retained: {len(obs['keyword_retained'])} keywords, {len(obs['sub_retained'])} subs")
    if obs['keyword_replaced']:
        logger.info(f"  Replaced: {obs['keyword_replaced']}")
    if obs['keyword_rejected']:
        logger.info(f"  Rejected: {len(obs['keyword_rejected'])} candidates")

    return result
