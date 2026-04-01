"""Research package - modular research tools.

Submodules:
- scraping: Web search, fetching, Reddit, YouTube
- classification: Signal classification, keyword matching
- enrichment: Recurrence queries, corroboration planning
- scoring: Validation scoring logic

Can import from this package or directly from src.research_tools.
"""

from __future__ import annotations

# Re-export classification constants
from src.research.classification import (
    AI_TOOL_KEYWORDS,
    PAIN_KEYWORDS,
    VALUE_KEYWORDS,
    RECURRENCE_KEYWORDS,
    contains_ai_keyword,
    contains_pain_keyword,
    contains_value_keyword,
    contains_recurrence_keyword,
    classify_signal,
)

# Re-export enrichment functions
from src.research.enrichment import (
    RecurrenceQueryBuilder,
    CorroborationPlanner,
    build_recurrence_query,
    build_competitor_query,
    build_value_enrichment_query,
    decompose_recurrence_queries,
    decompose_low_info_atom,
    prioritize_queries,
)

# Re-export scoring functions
from src.research.scoring import (
    score_market_fit,
    score_technical_fit,
    score_distribution_fit,
    compute_composite_score,
    make_decision,
    score_recurrence,
    score_corroboration,
)

# Re-export research_tools for backward compatibility
from src.research_tools import (
    ResearchToolkit,
    REDDIT_MODES,
    SOFTWARE_DOMAINS,
    NOISY_SEARCH_DOMAINS,
    SEARCH_ENGINE_RESULT_DOMAINS,
    STACKEXCHANGE_SITE_ALIASES,
    QUERY_STOPWORDS,
    WEAK_VALIDATION_TERMS,
    RECURRENCE_NOISE_TERMS,
    COMPETITOR_INTENT_HINTS,
    NON_PROBLEM_DISCOVERY_PATTERNS,
    WORKAROUND_SIGNAL_TERMS,
    FREQUENCY_SIGNAL_TERMS,
    COST_SIGNAL_TERMS,
    HELP_URL_PATTERNS,
    BLOG_LIKE_PATTERNS,
    VALIDATION_QUERY_PHRASES,
)

# Re-export utility functions from utils.text for convenience
from src.utils.text import (
    compact_text,
    slugify,
    domain_for,
    unwrap_search_result_url,
    normalize_search_url,
    url_path,
    contains_keyword,
    first_match,
    infer_recurrence_key,
)