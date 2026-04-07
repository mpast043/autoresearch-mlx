"""Opportunity engine helpers extracted from opportunity_engine."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlparse

# Re-export compact_text for internal use
from src.research_tools import compact_text, infer_recurrence_key

from src.database import ProblemAtom


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _value(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    # Handle sqlite3.Row which supports key access but not attribute access
    if hasattr(obj, "__getitem__"):
        try:
            return obj[name]
        except (KeyError, TypeError):
            pass
    return getattr(obj, name, default)


def json_dumps(value: Any) -> str:
    return json.dumps(value)


def _pick_first_sentence(text: str, hints: list[str]) -> str:
    text = compact_text(text or "", 1600)
    if not text:
        return ""
    parts = [part.strip(" .:-") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    for part in parts:
        lowered = _normalized(part)
        if any(hint in lowered for hint in hints):
            return compact_text(part, 220)
    return compact_text(parts[0], 220) if parts else ""


def _clean_fragment(text: str) -> str:
    cleaned = compact_text(re.sub(r"\s+", " ", (text or "").strip(" .:-")), 140)
    return cleaned.rstrip(".")


def _is_generic_phrase(text: str) -> bool:
    lowered = _normalized(text)
    return lowered in {
        "",
        "keep a recurring workflow reliable without manual cleanup",
        "keep a recurring workflow on track",
        "operators with recurring workflow pain",
        "remove repeated operational bottlenecks",
    }


def _has_phrase(text: str, phrases: list[str]) -> bool:
    lowered = _normalized(text)
    return any(phrase in lowered for phrase in phrases)


def _strip_title_prefix(title: str, body: str) -> str:
    title = compact_text(title or "", 240).strip()
    body = compact_text(body or "", 1600).strip()
    if not title or not body:
        return body
    pattern = r"^\s*" + re.escape(title) + r"[\s:|.\-]*"
    stripped = re.sub(pattern, "", body, count=1, flags=re.IGNORECASE).strip()
    return stripped or body


def _normalize_problem_fragment(text: str, *, fallback: str = "", limit: int = 96) -> str:
    cleaned = compact_text(re.sub(r"\s+", " ", (text or "").strip()), 240)
    if not cleaned:
        return fallback
    cleaned = re.sub(r"https?://\S+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(contact|logs stored|stack trace|version|windows v\d[\w.\-]*)\b[: ]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^\s*(feature request:?|request:?|issue:?|discussion:?|question:?|trouble with|looking for|is anyone|does anyone|need help with|the app doesn't work how it should)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-")
    return compact_text(cleaned, limit) or fallback


def _match_rule(text: str, rules: list[tuple[str, str]], fallback: str) -> str:
    lowered = _normalized(text)
    for pattern, label in rules:
        if re.search(pattern, lowered, re.IGNORECASE):
            return label
    return fallback


def _segment_inference_context(finding_data: dict[str, Any], signal_text: str) -> str:
    product = _value(finding_data, "product_built", "")
    outcome = _value(finding_data, "outcome_summary", "")
    return compact_text(f"{product} {outcome} {signal_text}", 500)


def _extract_workarounds(text: str) -> list[str]:
    text = compact_text(text or "", 800)
    if not text:
        return []
    patterns = [
        r"(?:we |they |i )?have? to (?:\w+ )?manual(?:ly)? ([\w\s]+?)(?: to | because |\.|$)",
        r"(?:we |they |i )?manual(?:ly)? ([\w\s]+?)(?: to | because |\.|$)",
        r"(?:use |uses |using )?(\w+ )?script(?:s)?(?: to | for |\.|$)",
        r"(?:export|import|convert) (?:to |from )?(\w+)",
    ]
    workarounds: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        workarounds.extend([m.strip() for m in matches if m.strip()])
    return list(dict.fromkeys(workarounds))[:5]


def _extract_clues(text: str, terms: list[str]) -> list[str]:
    lowered = _normalized(text)
    return [term for term in terms if term in lowered]


def _extract_cost_clues(text: str) -> list[str]:
    text = (text or "").lower()
    cost_terms = [
        "cost", "expensive", "cheap", "budget", "price", "pay", "free",
        "subscription", "monthly", "annual", "license", "per user", "seat",
    ]
    return [term for term in cost_terms if term in text]


def _normalize_tools(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").lower())
    replacements = {
        "google sheets": "spreadsheet",
        "excel": "spreadsheet",
        "airtable": "spreadsheet",
        "notion": "wiki",
        "confluence": "wiki",
        "slack": "chat",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def _looks_like_meta_prompt(text: str) -> bool:
    lowered = (text or "").lower()
    meta_indicators = [
        "you are a",
        "your task is to",
        "write a program",
        "generate code",
        "create a script",
    ]
    return any(indicator in lowered for indicator in meta_indicators)


def _is_clean_context(text: str) -> bool:
    if not text or len(text) < 20:
        return False
    if _looks_like_meta_prompt(text):
        return False
    banned = ["lorem ipsum", "placeholder", "TODO", "FIXME"]
    lowered = text.lower()
    return not any(banned in lowered for banned in banned)


def _normalize_workaround_phrase(text: str, *, fallback: str = "manual workarounds") -> str:
    text = (text or "").lower().strip()
    if not text:
        return fallback
    replacements = {
        "manual processes": "manual workarounds",
        "manual process": "manual workarounds",
        "manual steps": "manual workarounds",
        "manual work": "manual workarounds",
        "workaround": "manual workarounds",
    }
    for old, new in replacements.items():
        if old in text:
            return new
    return fallback


def _normalize_cost_phrase(text: str) -> str:
    text = (text or "").lower().strip()
    if "free" in text:
        return "free"
    if any(term in text for term in ["cost", "price", "expensive", "cheap", "budget", "pay"]):
        return "paid"
    return ""


def _infer_specific_job(job: str, fallback: str) -> str:
    job = _normalized(job)
    if not job:
        return fallback
    if "reconcil" in job or "match" in job:
        return "data reconciliation"
    if "report" in job or "dashboard" in job:
        return "reporting"
    if "import" in job or "export" in job:
        return "data import/export"
    if "sync" in job or "integrat" in job:
        return "data sync"
    return fallback


def _select_specific_context(*values: str, limit: int = 72) -> str:
    for value in values:
        if value and _is_clean_context(value):
            return compact_text(value, limit)
    return ""


def _summarize_context(text: str) -> str:
    return compact_text(text or "", 140)


def _fallback_context_from_atom(atom: ProblemAtom) -> str:
    if not atom:
        return ""
    return _summarize_context(f"{atom.job_to_be_done} {atom.failure_summary or ''}")


def _review_generalizability(text: str, source_name: str, source_url: str) -> dict[str, Any]:
    text = text or ""
    lowered = text.lower()
    score = 0.5
    flags: list[str] = []
    if len(text) < 50:
        score -= 0.3
        flags.append("too_short")
    if "every user" in lowered or "all customers" in lowered:
        score += 0.2
        flags.append("generalizable_claim")
    if source_name == "reddit":
        if "/r/sysadmin" in source_url or "/r/devops" in source_url:
            score += 0.1
            flags.append("technical_audience")
    return {"score": clamp(score), "flags": flags}


def _github_generalizability(text: str) -> dict[str, Any]:
    text = text or ""
    lowered = text.lower()
    score = 0.5
    flags: list[str] = []
    if len(text) < 100:
        score -= 0.2
        flags.append("too_short")
    if "feature request" in lowered:
        score += 0.15
        flags.append("feature_request")
    if "would be nice" in lowered or "would be great" in lowered:
        score += 0.1
        flags.append("wishlist")
    return {"score": clamp(score), "flags": flags}


def infer_source_type(source_name: str, source_url: str) -> str:
    if not source_name and not source_url:
        return "unknown"
    url = source_url or ""
    if source_name == "reddit" or "reddit.com" in url:
        return "reddit"
    if source_name == "github" or "github.com" in url:
        return "github"
    if source_name == "youtube" or "youtube.com" in url:
        return "youtube"
    if source_name in ("wordpress_reviews", "shopify_reviews") or "wordpress" in url or "shopify" in url:
        return "reviews"
    if source_name == "web" or source_name == "duckduckgo":
        return "web"
    if "stackexchange.com" in url or "stackoverflow.com" in url:
        return "stackoverflow"
    return source_name or "unknown"