"""Suggest discovery.reddit keywords/subs from DB state (lightweight adaptation)."""

from __future__ import annotations

import re
from typing import Any

REDDIT_SOURCE = re.compile(r"^reddit-problem/([^/]+)\s*$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_COMMENT_HINT = re.compile(
    r"\b(i|we)\b.*\b(can't|cannot|stuck|hate|pain|problem|manual|slow|annoying|frustrating|wish)\b",
    re.I,
)
_MONEY_CLAIM = re.compile(
    r"(\$[\d,.]+(?:\s?[kKmM])?(?:\s*(?:mrr|arr|revenue|sales|profit|monthly))?|\b\d+\s*(?:customers|users|clients)\b)",
    re.I,
)
_BUILD_HINT = re.compile(r"\b(built|building|launched|shipping|prototype|mvp|discovered|found)\b", re.I)
_MONEY_STRONG = re.compile(r"\b(mrr|arr|revenue|profit|stripe|dashboard|screenshot)\b", re.I)
_MONEY_MEDIUM = re.compile(r"\b(sales|customers|users|clients|launched|built|found|discovered)\b", re.I)


def _clean_phrase(text: str, *, max_len: int = 90) -> str:
    t = " ".join((text or "").split())
    if len(t) < 12:
        return ""
    return t[:max_len].strip(" .,;:!\"'-")


def _subs_from_findings(rows: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        src = getattr(row, "source", None) or ""
        m = REDDIT_SOURCE.match(str(src).strip())
        if not m:
            continue
        sub = m.group(1).strip()
        if sub and sub.lower() not in seen:
            seen.add(sub.lower())
            out.append(sub)
    return out


def _as_theme_map(theme_keywords: Any) -> dict[str, list[str]]:
    if not isinstance(theme_keywords, dict):
        return {}
    out: dict[str, list[str]] = {}
    for theme, items in theme_keywords.items():
        if isinstance(items, list):
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            if cleaned:
                out[str(theme).strip()] = cleaned
    return out


def _extract_comment_phrases(signals: list[Any], *, max_items: int = 20) -> list[str]:
    """Pull short complaint-style phrases from Reddit signal excerpts."""
    out: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        source_type = (getattr(signal, "source_type", "") or "").lower()
        source_name = (getattr(signal, "source_name", "") or "").lower()
        if "reddit" not in source_type and "reddit" not in source_name:
            continue
        text = " ".join(
            [
                str(getattr(signal, "body_excerpt", "") or ""),
                str(getattr(signal, "quote_text", "") or ""),
            ]
        )
        for sentence in _SENTENCE_SPLIT.split(text):
            phrase = _clean_phrase(sentence, max_len=110)
            if len(phrase) < 24:
                continue
            if not _COMMENT_HINT.search(phrase):
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(phrase)
            if len(out) >= max_items:
                return out
    return out


def _extract_money_claims(findings: list[Any], *, max_items: int = 20) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for finding in findings:
        text = " ".join(
            [
                str(getattr(finding, "monetization_method", "") or ""),
                str(getattr(finding, "outcome_summary", "") or ""),
                str(getattr(finding, "product_built", "") or ""),
            ]
        )
        for match in _MONEY_CLAIM.findall(text):
            claim = " ".join((match or "").split())[:60].strip(" .,;:!\"'-")
            if not claim:
                continue
            key = claim.lower()
            if key in seen:
                continue
            seen.add(key)
            source = str(getattr(finding, "source", "") or "")
            context = str(getattr(finding, "outcome_summary", "") or "")
            monetization = str(getattr(finding, "monetization_method", "") or "")
            product = str(getattr(finding, "product_built", "") or "")
            confidence = _money_claim_confidence(
                claim=claim,
                source=source,
                context=context,
                monetization=monetization,
                product=product,
            )
            out.append(
                {
                    "claim": claim,
                    "confidence": confidence,
                    "source": source,
                    "url": str(getattr(finding, "source_url", "") or ""),
                    "context": _clean_phrase(context, max_len=120),
                }
            )
            if len(out) >= max_items:
                return out
    return out


def _money_claim_confidence(*, claim: str, source: str, context: str, monetization: str, product: str) -> str:
    text = " ".join([claim, source, context, monetization, product])
    has_dollar = "$" in claim
    source_success = "success" in source.lower()
    if has_dollar and (_MONEY_STRONG.search(text) or source_success):
        return "high"
    if has_dollar or _MONEY_MEDIUM.search(text):
        return "medium"
    return "low"


def _ranked_money_claims(claims: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {"high": [], "medium": [], "low": []}
    for claim in claims:
        tier = str(claim.get("confidence", "low")).lower()
        grouped[tier if tier in grouped else "low"].append(claim)
    return grouped


def _filter_money_claims(claims: list[dict[str, str]], *, min_confidence: str = "low") -> list[dict[str, str]]:
    rank = {"low": 0, "medium": 1, "high": 2}
    floor = rank.get(str(min_confidence or "low").lower(), 0)
    out: list[dict[str, str]] = []
    for claim in claims:
        tier = str(claim.get("confidence", "low")).lower()
        if rank.get(tier, 0) >= floor:
            out.append(claim)
    return out


def _extract_build_discovery_signals(findings: list[Any], *, max_items: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        text = " ".join(
            [
                str(getattr(finding, "product_built", "") or ""),
                str(getattr(finding, "outcome_summary", "") or ""),
            ]
        )
        if not _BUILD_HINT.search(text):
            continue
        for sentence in _SENTENCE_SPLIT.split(text):
            phrase = _clean_phrase(sentence, max_len=100)
            if len(phrase) < 18:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(phrase)
            if len(out) >= max_items:
                return out
    return out


def build_discovery_suggestions(
    db: Any,
    *,
    min_atoms: int = 2,
    limit_clusters: int = 40,
    limit_atoms: int = 120,
    limit_findings: int = 400,
    max_keywords: int = 28,
    theme_keywords: Any = None,
    money_claim_min_confidence: str = "low",
) -> dict[str, Any]:
    """
    Pull recurring language from clusters/atoms and Reddit sources from findings.
    Operator pastes into ``discovery.reddit.problem_keywords`` / ``problem_subreddits``.
    """
    clusters = [c for c in db.get_clusters(limit=limit_clusters) if int(getattr(c, "atom_count", 0) or 0) >= min_atoms]
    clusters.sort(
        key=lambda c: (int(getattr(c, "atom_count", 0) or 0), float(getattr(c, "evidence_quality", 0) or 0.0)),
        reverse=True,
    )

    phrases: list[str] = []
    seen: set[str] = set()

    def add_phrase(raw: str) -> None:
        p = _clean_phrase(raw)
        if not p:
            return
        key = p.lower()
        if key in seen or len(key) < 12:
            return
        seen.add(key)
        phrases.append(p)

    for cl in clusters:
        add_phrase(str(getattr(cl, "label", "") or ""))
        add_phrase(str(getattr(cl, "job_to_be_done", "") or ""))
        add_phrase(str(getattr(cl, "trigger_summary", "") or ""))

    for atom in db.get_problem_atoms(limit=limit_atoms):
        add_phrase(str(getattr(atom, "pain_statement", "") or ""))
        add_phrase(str(getattr(atom, "failure_mode", "") or ""))

    keywords = phrases[:max_keywords]
    theme_map = _as_theme_map(theme_keywords)
    themed: dict[str, list[str]] = {}
    for theme, seeds in theme_map.items():
        theme_terms = [seed.lower() for seed in seeds]
        themed_hits: list[str] = []
        local_seen: set[str] = set()
        for phrase in phrases:
            lowered = phrase.lower()
            if not any(term in lowered for term in theme_terms):
                continue
            if lowered in local_seen:
                continue
            local_seen.add(lowered)
            themed_hits.append(phrase)
            if len(themed_hits) >= 10:
                break
        # Backfill with seed terms if DB does not yet have enough matches.
        if len(themed_hits) < 5:
            for seed in seeds:
                cleaned = _clean_phrase(seed, max_len=90)
                if not cleaned:
                    continue
                seed_key = cleaned.lower()
                if seed_key in local_seen:
                    continue
                local_seen.add(seed_key)
                themed_hits.append(cleaned)
                if len(themed_hits) >= 10:
                    break
        themed[theme] = themed_hits

    findings = db.get_findings(limit=limit_findings)
    subs = _subs_from_findings(findings)
    signals = db.get_raw_signals(limit=limit_findings)
    comment_phrases = _extract_comment_phrases(signals, max_items=20)
    money_claims_all = _extract_money_claims(findings, max_items=40)
    money_claims = _filter_money_claims(money_claims_all, min_confidence=money_claim_min_confidence)[:20]
    ranked_money_claims = _ranked_money_claims(money_claims)
    build_discovery_signals = _extract_build_discovery_signals(findings, max_items=24)

    return {
        "suggested_keywords": keywords,
        "suggested_keywords_by_theme": themed,
        "suggested_subreddits_from_findings": subs[:40],
        "suggested_comment_phrases": comment_phrases,
        "suggested_money_claims": money_claims,
        "suggested_money_claims_by_confidence": ranked_money_claims,
        "suggested_build_discovery_signals": build_discovery_signals,
        "cluster_rows_used": len(clusters),
        "hint": "Merge into config.yaml under discovery.reddit — trim noisy lines; dedupe with existing lists.",
    }
