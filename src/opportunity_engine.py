"""Deterministic extraction and scoring helpers for the evidence-first pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from src.database import OpportunityCluster, ProblemAtom, RawSignal, ValidationExperiment
from src.research_tools import compact_text, infer_recurrence_key

# =============================================================================
# SCORING VERSION CONTROL
# =============================================================================
# Bump this version whenever formula weights, thresholds, or logic change.
# The version is stored with each scorecard for audit trail.
#
# Version History:
#   v1 (legacy): Original formula with weak penalties (0.12/0.11/0.11)
#   v2: Stronger penalties (0.18/0.16/0.16), nonlinear evidence multiplier,
#       lower thresholds (composite 0.45, plausibility 0.55), frequency gate
#   v3: Removed duplicate counting (urgency_hits, cost_hits, recurrence_score, etc.)
#   v4: Split scoring into PTS/RRS with blended decision_score
CURRENT_SCORING_VERSION = "v4"
CURRENT_FORMULA_VERSION = "pts_rrs_v1"
CURRENT_THRESHOLD_VERSION = "2025_q2"

# Version-specific thresholds (can be referenced by other code)
SCORING_THRESHOLDS = {
    "v1": {"promotion": 0.55, "park": 0.35},
    "v2": {"promotion": 0.45, "park": 0.35, "plausibility": 0.55},
    "v3": {"promotion": 0.45, "park": 0.35},
    "v4": {"promotion": 0.18, "park": 0.15, "pts_floor": 0.11, "rrs_floor": 0.22},
}

# Re-export helpers for backward compatibility
from src.utils.opportunity_helpers import (
    clamp,
    infer_source_type,
    json_dumps,
)


PAIN_KEYWORDS = [
    "pain",
    "manual",
    "broken",
    "fails",
    "failed",
    "error",
    "friction",
    "workaround",
    "spreadsheet",
    "restore",
    "recovery",
    "sync",
    "compliance",
    "unreachable",
    "fallback",
]
PAIN_SIGNAL_HINTS = [
    "doesn't work",
    "doesnt work",
    "stopped working",
    "fails every time",
    "every time",
    "manual workaround",
    "fall back to",
    "falls back to",
    "restore fails",
    "sync fails",
    "support deleted",
    "stays manual",
    "csv export",
]
EMOTION_TERMS = [
    "frustrating",
    "annoying",
    "losing my mind",
    "urgent",
    "panic",
    "painful",
    "hate",
]
WHY_NOW_TERMS = [
    "after",
    "when",
    "during",
    "because",
    "suddenly",
    "recently",
    "now",
    "migration",
    "upgrade",
    "publish",
    "restore",
]
URGENCY_TERMS = [
    "urgent",
    "asap",
    "immediately",
    "must",
    "critical",
    "blocked",
    "outage",
    "down",
]
FREQUENCY_TERMS = [
    "every day",
    "daily",
    "every week",
    "weekly",
    "every time",
    "repeatedly",
    "again",
    "always",
    "constantly",
]
PROMOTIONAL_PATTERNS = [
    "great app",
    "excellent support",
    "highly recommend",
    "we recommend",
    "best tool",
    "love this app",
]
GENERIC_PROMPT_PATTERNS = [
    "any recommendations",
    "looking for",
    "does anyone know",
    "feature request",
    "roadmap",
    "idea",
]
GENERIC_REQUEST_PATTERNS = [
    "can anyone suggest",
    "help choosing",
    "need help choosing",
    "what app should i use",
    "need a recommendation",
    "what's the best",
]
ADVICE_SEEKING_PATTERNS = [
    "looking for advice",
    "how do you handle",
    "how are you managing",
    "how are u guys managing",
    "how do you plan on handling",
    "is this possible",
]
ROI_SHOPPING_PATTERNS = [
    "what was the first automation you paid for",
    "trying to gauge the roi",
    "worth the money",
    "outsourcing an automation project",
    "break the bank",
]
PRODUCT_COMPLAINT_PATTERNS = [
    "should focus on improving",
    "pricing is terrible",
    "pricing change",
    "pricing of",
    "custom agents",
    "approval timeline",
    "feature request",
    "roadmap",
    "useless ai features",
]
OPERATIONAL_CONTEXT_HINTS = [
    "invoice",
    "payment",
    "deposit",
    "reconciliation",
    "csv",
    "export",
    "import",
    "client",
    "booking",
    "handoff",
    "approval",
    "permissions",
    "versioning",
    "source of truth",
    "database",
    "workflow",
    "order",
    "intake",
    "form response",
]
ACTIONABLE_WORKFLOW_HINTS = [
    "->",
    "through whatsapp",
    "bank deposit",
    "bank deposits",
    "label generation",
    "keep up",
    "state between",
    "returns in january",
]
GENERIC_WHY_NOW_FILLERS = {
    "after",
    "because",
    "during",
    "now",
    "recently",
    "suddenly",
    "when",
}
VENTING_PATTERNS = [
    "i am done with",
    "just want vent",
    "not looking for product recommendations",
    "bane of my existence",
    "just vent my frustrations",
]
SOLICITATION_PROMPT_PATTERNS = [
    "tell me your most annoying manual",
    "what manual process would you like to automate",
    "what is the most annoying manual task",
    "i'm a developer looking to build",
    "instead of guessing what people need",
    "wanted to ask directly",
    "i might automate it for free",
    "working on some automation stuff",
    "looking for some real-world problems",
]
HELP_PAGE_PATTERNS = [
    "help center",
    "documentation",
    "docs",
    "knowledge base",
    "support article",
    "api reference",
]
CAREER_GUIDANCE_PATTERNS = [
    "roast my resume",
    "resume review",
    "review my resume",
    "feeling lost with career",
    "transitioning to a larger firm",
]
TUTORIAL_SHARE_PATTERNS = [
    "thought i would put this information all together",
    "return the favor",
    "here are the steps",
    "step by step",
    "walkthrough",
]
META_GUIDANCE_PATTERNS = [
    "roadmap skill iteration",
    "intent 001",
    "internal task",
    "meta guidance",
    "evaluation harness",
    # Developer solicitation / idea gathering - NOT actual pain
    "looking to build",
    "want to automate for free",
    "what problem should i solve",
    "what problems should i solve",
    "what problem would you like",
    "most annoying manual task",
    "weekend puzzle",
    "automate it for free",
    "looking for a weekend",
    "tell me your most annoying",
    "i can help automate",
    "developer looking to",
    "building a new tool",
    "looking to automate",
    "working on some automation",
    "building custom systems",
    "help me build",
    "what should i build",
    "what should i automate",
    # Meta requests
    "would like to automate",
    "think about automating",
    "consider automating",
]
DEMAND_SIGNAL_PATTERNS = [
    "search volume",
    "trend",
    "growth",
    "reviews",
    "rating",
    "rating count",
]
COMPETITION_SIGNAL_PATTERNS = [
    "competitor",
    "alternative",
    "pricing page",
    "app store listing",
]
SEGMENT_RULES = [
    ("smallbusiness", "small business operations"),
    ("small business", "small business operations"),
    ("etsy", "etsy sellers"),
    ("seller", "sellers"),
    ("shopify", "shopify merchants"),
    ("woocommerce", "woocommerce merchants"),
    ("wordpress", "wordpress operators"),
    ("compliance", "compliance teams"),
    ("operator", "operations teams"),
    ("developer", "developers"),
    ("support", "support teams"),
    ("backup", "operators with backup and recovery workflows"),
]
ROLE_RULES = [
    ("compliance", "compliance lead"),
    ("support", "support lead"),
    ("seller", "seller"),
    ("merchant", "merchant"),
    ("developer", "developer"),
    ("engineer", "developer"),
    ("finance", "finance lead"),
    ("operator", "operator"),
    ("ops", "operations lead"),
    ("small business", "operations lead"),
]
JTBD_RULES = [
    ("backup", "keep backup restore and recovery reliable"),
    ("restore", "keep backup restore and recovery reliable"),
    ("recovery", "keep backup restore and recovery reliable"),
    ("sync", "keep sync and data handoff workflows reliable"),
    ("integration", "keep sync and data handoff workflows reliable"),
    ("shipping", "keep listing setup and shipping settings reliable"),
    ("compliance", "keep multi-framework compliance evidence and monitoring reliable"),
    ("audit", "keep multi-framework compliance evidence and monitoring reliable"),
    ("spreadsheet", "keep operations data in sync without manual cleanup"),
    ("manual data entry", "keep operations data in sync without manual cleanup"),
    ("template", "keep template application and onboarding reliable"),
    ("review", "respond to reviews and reputation issues consistently"),
]

# Platform extraction patterns for atom enhancement
PLATFORM_PATTERNS = [
    # Explicit platform names
    (r"\bquickbooks\b", "quickbooks"),
    (r"\bqbo\b", "quickbooks"),
    (r"\bxero\b", "xero"),
    (r"\bshopify\b", "shopify"),
    (r"\betsy\b", "etsy"),
    (r"\bnotion\b", "notion"),
    (r"\bairtable\b", "airtable"),
    (r"\bgoogle sheets?\b", "google_sheets"),
    (r"\bgoogle docs?\b", "google_sheets"),
    (r"\bmicrosoft excel\b", "excel"),
    (r"\bmsexcel\b", "excel"),
    (r"\bexcel\b(?!\s+add)", "excel"),
    (r"\bwordpress\b", "wordpress"),
    (r"\bwp\b(?=\s+(plugin|theme|admin))", "wordpress"),
    (r"\bwooCommerce\b", "woocommerce"),
    (r"\bslack\b", "slack"),
    (r"\bgmail\b", "gmail"),
    (r"\bsalesforce\b", "salesforce"),
    (r"\bhubspot\b", "hubspot"),
    (r"\bstripe\b", "stripe"),
    (r"\bpaypal\b", "paypal"),
    (r"\bzapier\b", "zapier"),
    (r"\bmake\.com\b", "make"),
    (r"\bnocode\b", "nocode"),
]

# Expanded generic phrase detection for atom validation
GENERIC_PHRASES = {
    "",
    "keep a recurring workflow reliable without manual cleanup",
    "keep a recurring workflow on track",
    "operators with recurring workflow pain",
    "remove repeated operational bottlenecks",
    "keep operations data in sync without manual cleanup",
    "keep sync and data handoff workflows reliable",
    "keep template application and onboarding reliable",
    "keep backup restore and recovery reliable",
    "keep multi-framework compliance evidence and monitoring reliable",
    "respond to reviews and reputation issues consistently",
}
SOURCE_TYPE_HINTS = [
    ("reddit", "forum"),
    ("github", "github_issue"),
    ("wordpress-review", "review"),
    ("shopify-review", "review"),
    ("review", "review"),
]
WORKAROUND_TERMS = {
    "spreadsheet": "spreadsheets",
    "excel": "spreadsheets",
    "csv": "csv exports",
    "manual": "manual work",
    "email": "email routing",
    "copy": "copy/paste",
    "script": "custom scripts",
    "workaround": "manual workarounds",
}
REVIEW_NEGATIVE_TERMS = [
    "useless",
    "no support",
    "terrible support",
    "scam",
    "waste of money",
]
INTERNAL_IDENTIFIER_PATTERNS = [
    "intent 001",
    "skill iteration",
    "roadmap",
    "ticket only",
    "template id",
]


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
            return obj[name]  # sqlite3.Row supports key access
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
    """Check if text is a generic cluster label rather than specific extraction."""
    lowered = _normalized(text)
    return lowered in GENERIC_PHRASES


def _extract_platform_from_text(text: str) -> str:
    """Extract platform from signal text for atom enrichment."""
    lowered = _normalized(text)
    for pattern, platform in PLATFORM_PATTERNS:
        if re.search(pattern, lowered):
            return platform
    return ""


def _extract_specific_workflow(text: str) -> str:
    """Try to extract specific workflow description from text.

    Looks for specific task descriptions rather than generic phrases.
    """
    lowered = _normalized(text)

    # Pattern: "manually X" - extract the specific task
    match = re.search(r'manually\s+([a-z\s]{5,40})', lowered)
    if match:
        task = match.group(1).strip()
        if len(task) > 5:
            return task

    # Pattern: "spending X hours Y" - extract the time-consuming task
    match = re.search(r'spend(?:ing|s)?\s+\d+\s+hours?\s+(?:on|doing|for)\s+([a-z\s]{5,40})', lowered)
    if match:
        task = match.group(1).strip()
        if len(task) > 5:
            return task

    # Pattern: "have to X" - extract obligation
    match = re.search(r'have to\s+([a-z]{4,30})', lowered)
    if match:
        task = match.group(1).strip()
        if len(task) > 4:
            return task

    return ""


def _has_meaningful_consequence(text: str) -> bool:
    """Check if text contains meaningful consequence (not just 'consequence' placeholder)."""
    lowered = _normalized(text)

    # Must have actual consequence keywords
    consequence_terms = [
        "time", "hours", "days", "week",
        "money", "dollar", "cost", "expensive", "waste",
        "lost", "missed", "late", "risk", "error", "mistake",
        "revenue", "profit", "penalty", "fine",
    ]

    return any(term in lowered for term in consequence_terms)


def _validate_atom_quality(atom: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Validate and enhance atom quality, returning quality signals."""
    quality = {
        "is_valid": True,
        "quality_issues": [],
        "specificity_score": 0.0,
        "consequence_score": 0.0,
        "platform_score": 0.0,
        "should_reject": False,
    }

    # Check 1: Generic workflow
    if _is_generic_phrase(atom.get("job_to_be_done", "")):
        quality["quality_issues"].append("generic_workflow")
        quality["is_valid"] = False

    # Check 2: Empty or placeholder consequence
    consequence = atom.get("cost_consequence_clues", "")
    if not consequence or consequence == "consequence":
        quality["quality_issues"].append("missing_consequence")
    elif _has_meaningful_consequence(consequence):
        quality["consequence_score"] = 0.8

    # Check 3: Failure mode quality
    failure = atom.get("failure_mode", "")
    if not failure or len(failure) < 15:
        quality["quality_issues"].append("weak_failure_mode")
    else:
        quality["specificity_score"] += 0.3

    # Check 4: Platform detection
    platform = _extract_platform_from_text(source_text)
    if platform:
        quality["platform_score"] = 1.0
    else:
        quality["quality_issues"].append("no_platform")

    # Check 5: Specific workflow extraction
    specific_workflow = _extract_specific_workflow(source_text)
    if specific_workflow:
        quality["specificity_score"] += 0.4

    # Calculate overall specificity
    if quality["specificity_score"] > 0:
        quality["specificity_score"] = min(1.0, quality["specificity_score"])

    # Reject only if too many issues
    if len(quality["quality_issues"]) >= 3:
        quality["should_reject"] = True

    return quality


# Patterns that indicate low-quality signals (meta-posts, generic questions)
LOW_QUALITY_PATTERNS = [
    r"^what (is|are) .* (annoying|frustrating|painful)",
    r"looking to build",
    r"want to automate (for free|it)",
    r"what problem (should i|would you)",
    r"automate it for free",
    r"weekend puzzle",
    r"tell me your most annoying",
    r"i can help automate",
    r"developer looking to",
    r"building a new tool",
    r"working on some automation",
    r"what should i build",
    r"what should i automate",
]


def _signal_quality_score(text: str, atom_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Score signal quality. Returns dict with:
    - quality_score: 0.0-1.0 (higher is better)
    - is_low_quality: bool
    - quality_issues: list of issues found
    """
    lowered = _normalized(text)
    issues: list[str] = []
    score = 1.0

    # Check for meta-prompt patterns
    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, lowered):
            issues.append("meta_prompt_pattern")
            score -= 0.4
            break

    # Check for meta-prompt in classification
    if _looks_like_meta_prompt(text):
        issues.append("meta_guidance_detected")
        score -= 0.3

    # Check if pain statement is too short or empty
    if not atom_payload.get("pain_statement") or len(atom_payload.get("pain_statement", "")) < 30:
        issues.append("no_specific_pain")
        score -= 0.25

    # Check if too generic JTBD
    if _is_generic_phrase(atom_payload.get("job_to_be_done", "")):
        issues.append("generic_jtbd")
        score -= 0.2

    # Check for question marks (likely asking for help, not complaining)
    if lowered.endswith("?"):
        issues.append("question_not_complaint")
        score -= 0.15

    # Check for first-person building intent
    if re.search(r"\bi('m| am) (building|working on|creating|developing)", lowered):
        issues.append("builder_not_user")
        score -= 0.3

    # Bonus: has specific workaround
    if atom_payload.get("current_workaround"):
        score += 0.1

    # Bonus: has cost/time clues
    if atom_payload.get("cost_consequence_clues"):
        score += 0.1

    score = clamp(score)
    return {
        "quality_score": score,
        "is_low_quality": score < 0.5,
        "quality_issues": issues,
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
    if not cleaned:
        cleaned = fallback
    return compact_text(cleaned, limit).rstrip(".")


def _match_rule(text: str, rules: list[tuple[str, str]], fallback: str) -> str:
    lowered = _normalized(text)
    for needle, label in rules:
        if needle in lowered:
            return label
    return fallback


def _segment_inference_context(finding_data: dict[str, Any], signal_text: str) -> str:
    """Context string for SEGMENT_RULES.

    Reddit discovery encodes the subreddit in ``source`` (e.g. ``reddit-problem/sysadmin``). Including
    that label can skew ``segment`` so the same recurring pain in different communities gets
    different ``cluster_key`` values. For Reddit we match segments on post text only; Shopify /
    WordPress / GitHub findings still pass source/URL hints for vertical rules.
    """
    source = str(finding_data.get("source", "") or "")
    url = str(finding_data.get("source_url", "") or "")
    if "reddit.com" in url.lower() or source.lower().startswith("reddit-problem"):
        return signal_text
    return f"{source} {url} {signal_text}"


def _extract_workarounds(text: str) -> list[str]:
    lowered = _normalized(text)
    hits: list[str] = []
    for term, label in WORKAROUND_TERMS.items():
        if term in lowered and label not in hits:
            hits.append(label)
    return hits


def _extract_clues(text: str, terms: list[str]) -> list[str]:
    lowered = _normalized(text)
    return [term for term in terms if term in lowered]


def _extract_cost_clues(text: str) -> list[str]:
    lowered = _normalized(text)
    clues: list[str] = []
    if re.search(r"\$[\d,.]+", text):
        clues.append("money")
    if re.search(r"\b\d+\s+(hours?|days?|weeks?)\b", lowered):
        clues.append("time")
    if any(term in lowered for term in ["expensive", "waste", "lost", "late", "missed", "downtime", "risk", "consequence"]):
        clues.append("consequence")
    return clues


def _normalize_tools(text: str) -> str:
    return compact_text(re.sub(r"\s+", " ", (text or "").replace("|", ",").replace("/", ", ")).strip(" ,"), 120)


# Assumptions that can be extracted from signal text
ASSUMPTION_PATTERNS = {
    # Team size
    (r"\bsolo\b", r"\bfounder\b", r"\bi'm a one[- ]?person\b", r"\bone[- ]?person\b"): "team_size:solo",
    (r"\bsmall team\b", r"\bsmall business\b", r"\bsmb\b", r"\bstartup\b"): "team_size:small",
    (r"\bmid[- ]?sized\b", r"\b50\+ employees\b", r"\b70\+ employees\b"): "team_size:mid",
    (r"\benterprise\b", r"\blarge (company|corporation)\b", r"\b100\+ employees\b"): "team_size:enterprise",
    # Cloud vs local
    (r"\bcloud\b", r"\bsaas\b", r"\bhosted\b", r"\bonline\b"): "deployment:cloud",
    (r"\blocal\b", r"\bon[- ]?premise\b", r"\bself[- ]?hosted\b", r"\bdesktop\b"): "deployment:local",
    (r"\bmicrosoft 365\b", r"\bm365\b", r"\boffice 365\b"): "tool:microsoft_365",
    (r"\bgoogle (sheets|docs|workspace)\b", r"\bgsuite\b"): "tool:google_workspace",
    (r"\bquickbooks\b", r"\bstripe\b", r"\bxero\b"): "tool:accounting",
    # Budget signals
    (r"\bfree\b", r"\bfree trial\b"): "budget:free_only",
    (r"\bcheap\b", r"\baffordable\b", r"\bbudget\b"): "budget:low",
    (r"\bwilling to pay\b", r"\bpay for\b", r"\bpremium\b"): "budget:paid",
    (r"\bexpensive\b", r"\benterprise pricing\b"): "budget:high",
    # Integration needs
    (r"\bapi\b", r"\bintegrate\b", r"\bconnect\b", r"\bwebhook\b"): "needs:integration",
    (r"\bautomate\b", r"\bautomation\b"): "needs:automation",
}


def _extract_assumptions(text: str) -> dict[str, str]:
    """Extract implicit assumptions from signal text about team, tools, budget, deployment."""
    lowered = _normalized(text)
    assumptions: dict[str, str] = {}
    for patterns, label in ASSUMPTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                key, value = label.split(":")
                assumptions[key] = value
                break
    return assumptions


# ============================================================
# SPECIFIC PATTERN DETECTION - identifies exact problems
# ============================================================

# Patterns for specific integration problems (these become product opportunities)
SPECIFIC_INTEGRATION_PATTERNS = [
    # Payment processor <-> Accounting
    (r"stripe.*quickbooks|stripe.*xero|stripe.*accounting", "stripe_to_accounting"),
    (r"shopify.*quickbooks|shopify.*xero|shopify.*accounting", "shopify_to_accounting"),
    (r"amazon.*quickbooks|amazon.*xero|amazon.*accounting", "amazon_to_accounting"),
    # Multi-channel e-commerce
    (r"amazon.*shopify.*etsy|multi.*channel|channel.*revenue", "multi_channel_ecom"),
    (r"shopify.*etsy|etsy.*amazon", "channel_to_channel"),
    # Bank reconciliation
    (r"bank.*reconcil|reconcil.*bank|bank.*feed", "bank_reconciliation"),
    (r"csv.*import.*bank|bank.*csv", "csv_bank_import"),
    # Invoice/Payment
    (r"unpaid.*invoice|invoice.*reminder|payment.*follow.*up", "invoice_follow_up"),
    (r"invoice.*sync|invoice.*match", "invoice_matching"),
    # Spreadsheet specific
    (r"spreadsheet.*version|version.*control.*spreadsheet|latest.*version.*spreadsheet", "spreadsheet_versioning"),
    (r"multiple.*spreadsheet|spreadsheet.*sync|spreadsheet.*merge", "spreadsheet_sync"),
    # Revenue reporting
    (r"revenue.*report|channel.*profit|sales.*report", "revenue_reporting"),
    # Payroll/Staffing
    (r"payroll.*sync|payroll.*reconcil", "payroll_reconciliation"),
    (r"shift.*schedul.*excel|schedul.*payroll", "scheduling_payroll"),
]

# Tools mentioned in signals - helps identify what systems are involved
TOOL_MENTIONS = {
    r"\bstripe\b": "stripe",
    r"\bquickbooks\b": "quickbooks",
    r"\bxero\b": "xero",
    r"\bshopify\b": "shopify",
    r"\bamazon\b": "amazon",
    r"\betsy\b": "etsy",
    r"\bgoogle.?sheet\b": "google_sheets",
    r"\bexcel\b": "excel",
    r"\bmicrosoft.?365\b": "ms365",
    r"\bnotion\b": "notion",
    r"\bazure\b": "azure",
    r"\baws\b": "aws",
    r"\bwebflow\b": "webflow",
}


# Mapping from detected patterns to focused discovery queries
# Used when running: python3 cli.py run --pattern <pattern>
PATTERN_TO_DISCOVERY_QUERIES = {
    "spreadsheet_versioning": [
        "spreadsheet version control multiple users",
        "excel version conflict team",
        "spreadsheet sync problems",
        "latest spreadsheet version confusion",
        "excel file merge conflicts",
        "shared spreadsheet version tracking",
    ],
    "bank_reconciliation": [
        "bank reconciliation manual process",
        "bank feed vs csv import small business",
        "bank statement reconciliation automation",
        "manual bank reconciliation time",
    ],
    "stripe_to_quickbooks": [
        "stripe quickbooks reconciliation manual",
        "stripe xero sync automation",
        "stripe accounting software integration",
    ],
    "multi_channel_ecom": [
        "shopify amazon etsy sales reconciliation",
        "multi channel ecommerce reporting manual",
        "amazon shopify etsy revenue tracking",
    ],
    "invoice_follow_up": [
        "unpaid invoice reminder automation",
        "invoice follow up manual process",
        "payment reminder automation small business",
    ],
}


def _extract_specific_patterns(text: str) -> list[dict[str, str]]:
    """
    Extract specific integration patterns from signal text.
    Returns list of {pattern_name, matched_text, confidence}.
    """
    lowered = _normalized(text)
    patterns_found: list[dict[str, str]] = []

    for regex, pattern_name in SPECIFIC_INTEGRATION_PATTERNS:
        if re.search(regex, lowered):
            patterns_found.append({
                "pattern": pattern_name,
                "confidence": 0.9,  # Specific pattern match = high confidence
            })

    # Also extract tool mentions
    tools_found: set[str] = set()
    for tool_regex, tool_name in TOOL_MENTIONS.items():
        if re.search(tool_regex, lowered):
            tools_found.add(tool_name)

    if tools_found and patterns_found:
        for p in patterns_found:
            p["tools"] = list(tools_found)

    return patterns_found


def _is_specific_problem(text: str, atom_payload: dict[str, Any]) -> bool:
    """
    Returns True if the signal describes a SPECIFIC problem (not generic "manual work").
    Specific = mentions exact tools/systems and specific workflow.
    """
    patterns = _extract_specific_patterns(text)
    if patterns:
        return True

    # Check for tool combinations (at least 2 tools mentioned)
    lowered = _normalized(text)
    tool_count = sum(1 for regex in TOOL_MENTIONS if re.search(regex, lowered))

    # Check for specific workflow language
    workflow_indicators = [
        r"reconcile", r"sync", r"import", r"export",
        r"match.*transaction", r"connect.*api", r"integrate"
    ]
    has_workflow = any(re.search(w, lowered) for w in workflow_indicators)

    return tool_count >= 2 or (tool_count >= 1 and has_workflow)


def _calculate_pattern_emergence_score(db_path: str, pattern_name: str) -> dict[str, Any]:
    """
    Analyze how many signals mention a specific pattern.
    Returns {count, trend, urgency}.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Count signals with this pattern in body_excerpt
    cursor = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM raw_signals
        WHERE LOWER(body_excerpt) LIKE ?
    """, (f"%{pattern_name.replace('_', ' ')}%",))

    count = cursor.fetchone()["cnt"] if cursor else 0

    # Get cluster count too
    cursor = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM problem_atoms
        WHERE LOWER(pain_statement) LIKE ? OR LOWER(job_to_be_done) LIKE ?
    """, (f"%{pattern_name.replace('_', ' ')}%", f"%{pattern_name.replace('_', ' ')}%"))

    cluster_count = cursor.fetchone()["cnt"] if cursor else 0
    conn.close()

    return {
        "pattern": pattern_name,
        "signal_count": count,
        "atom_count": cluster_count,
        "urgency": "high" if cluster_count >= 5 else "medium" if cluster_count >= 2 else "low",
    }


def get_patterns_for_discovery(db_path: str, min_atoms: int = 2) -> list[dict[str, Any]]:
    """
    Return list of specific patterns that have emerged from signals.
    Used to guide focused discovery.

    Searches for tool combination patterns in raw signals.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    all_patterns = []

    # Define tool pair patterns to search for
    tool_pairs = [
        (["stripe", "quickbooks"], "stripe_to_quickbooks"),
        (["stripe", "xero"], "stripe_to_xero"),
        (["shopify", "quickbooks"], "shopify_to_quickbooks"),
        (["shopify", "xero"], "shopify_to_xero"),
        (["amazon", "quickbooks"], "amazon_to_quickbooks"),
        (["amazon", "shopify"], "amazon_shopify"),
        (["amazon", "etsy"], "amazon_etsy"),
        (["shopify", "etsy"], "shopify_etsy"),
        (["bank", "reconcil"], "bank_reconciliation"),
        (["invoice", "unpaid"], "invoice_follow_up"),
        (["spreadsheet", "version"], "spreadsheet_versioning"),
    ]

    for tools, pattern_name in tool_pairs:
        # Build query for multiple terms
        like_clause = " AND ".join([f"LOWER(body_excerpt) LIKE '%{t}%'" for t in tools])
        cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM raw_signals WHERE {like_clause}")
        count = cursor.fetchone()["cnt"] if cursor else 0

        if count >= min_atoms:
            all_patterns.append({
                "pattern": pattern_name,
                "signal_count": count,
                "tools": tools,
                "urgency": "high" if count >= 5 else "medium" if count >= 2 else "low",
            })

    conn.close()

    # Sort by signal count (most common first)
    all_patterns.sort(key=lambda x: x["signal_count"], reverse=True)
    return all_patterns


def _looks_like_meta_prompt(text: str) -> bool:
    lowered = _normalized(text)
    return any(pattern in lowered for pattern in META_GUIDANCE_PATTERNS + INTERNAL_IDENTIFIER_PATTERNS)


def _is_clean_context(text: str) -> bool:
    lowered = _normalized(text)
    if not lowered:
        return False
    if any(pattern in lowered for pattern in HELP_PAGE_PATTERNS + INTERNAL_IDENTIFIER_PATTERNS):
        return False
    if re.search(r"\b(v?\d+\.\d+(?:\.\d+)*)\b", lowered):
        return False
    return True


def _normalize_workaround_phrase(text: str, *, fallback: str = "manual workarounds") -> str:
    phrase = _normalize_problem_fragment(text, fallback=fallback, limit=72).lower()
    return phrase or fallback


def _normalize_cost_phrase(text: str) -> str:
    lowered = _normalized(text)
    if "money" in lowered:
        return "real money and time loss"
    if "consequence" in lowered:
        return "missed work and operational risk"
    if "time" in lowered:
        return "wasted time"
    return "workflow instability"


def _infer_specific_job(job: str, fallback: str) -> str:
    cleaned = _normalize_problem_fragment(job, fallback=fallback, limit=96)
    if _is_generic_phrase(cleaned):
        return fallback
    return cleaned


def _select_specific_context(*values: str, limit: int = 72) -> str:
    for value in values:
        cleaned = _normalize_problem_fragment(value, fallback="", limit=limit)
        if cleaned and _is_clean_context(cleaned):
            return cleaned
    return ""


def _summarize_context(text: str) -> str:
    return _normalize_problem_fragment(text, fallback="", limit=72)


def _fallback_context_from_atom(atom: Any) -> str:
    for value in [
        _value(atom, "failure_mode", ""),
        _value(atom, "pain_statement", ""),
        _value(atom, "trigger_event", ""),
    ]:
        cleaned = _normalize_problem_fragment(value, fallback="", limit=72)
        if cleaned:
            return cleaned
    return "the workflow breaks"


def _review_generalizability(text: str, source_name: str, source_url: str) -> dict[str, Any]:
    lowered = _normalized(f"{source_name} {source_url} {text}")
    reasons: list[str] = []
    score = 0.0
    if any(term in lowered for term in ["restore", "recovery", "sync", "manual workaround", "manual work", "fallback", "csv", "spreadsheet"]):
        score += 0.55
        reasons.append("workflow_relevance")
    if any(term in lowered for term in ["support deleted", "no support", "terrible support", "useless"]):
        score -= 0.42
        reasons.append("vendor_specific_complaint")
    if any(term in lowered for term in ["plugin", "app", "extension"]) and not reasons:
        score += 0.05
    issue_type = "reusable_workflow_pain" if score >= 0.42 else "product_specific_issue"
    return {
        "review_issue_type": issue_type,
        "review_generalizability_score": round(clamp(score), 4),
        "review_generalizability_reasons": reasons or (["workflow_relevance"] if issue_type == "reusable_workflow_pain" else ["vendor_specific_complaint"]),
    }


def _github_generalizability(text: str) -> dict[str, Any]:
    lowered = _normalized(text)
    reasons: list[str] = []
    score = 0.0
    if any(term in lowered for term in ["restore", "backup", "recovery", "sync", "manual workaround", "fall back", "unreachable"]):
        score += 0.58
        reasons.append("workflow_failure")
    if any(term in lowered for term in INTERNAL_IDENTIFIER_PATTERNS):
        score -= 0.55
        reasons.append("internal_identifier_github_issue")
    if any(term in lowered for term in ["feature request", "wishlist", "nice to have"]) and score < 0.4:
        score -= 0.25
        reasons.append("generic_feature_request")
    issue_type = "reusable_workflow_pain" if score >= 0.42 else "product_specific_issue"
    return {
        "github_issue_type": issue_type,
        "github_generalizability_score": round(clamp(score), 4),
        "github_generalizability_reasons": reasons or (["workflow_failure"] if issue_type == "reusable_workflow_pain" else ["product_specific_issue"]),
    }


def infer_source_type(source_name: str, source_url: str) -> str:
    haystack = f"{source_name} {source_url}".lower()
    for term, label in SOURCE_TYPE_HINTS:
        if term in haystack:
            return label
    domain = urlparse(source_url).netloc.lower()
    if "reddit.com" in domain:
        return "forum"
    if "github.com" in domain:
        return "github_issue"
    return "web"


def build_raw_signal_payload(finding_data: dict[str, Any]) -> dict[str, Any]:
    evidence = finding_data.get("evidence", {}) or {}
    title = finding_data.get("product_built") or "Untitled signal"
    excerpt = compact_text(
        " ".join(
            part
            for part in [
                finding_data.get("outcome_summary") or "",
                evidence.get("page_excerpt") or "",
                evidence.get("snippet") or "",
                evidence.get("review_text") or "",
                evidence.get("transcript_excerpt") or "",
            ]
            if part
        ),
        1400,
    )
    source_name = finding_data.get("source", "unknown")
    source_url = finding_data.get("source_url", "")
    signal_text = compact_text(f"{title}. {excerpt}", 1600)
    role_hint = _match_rule(signal_text, ROLE_RULES, "")
    metadata_json = {
        "finding_kind": finding_data.get("finding_kind", "problem_signal"),
        "source_class": finding_data.get("source_class", ""),
        "tool_used": finding_data.get("tool_used", ""),
        "discovery_query": evidence.get("discovery_query", ""),
        "source_plan": evidence.get("source_plan", ""),
        "record_origin": evidence.get("record_origin", ""),
        "evidence": evidence,
    }
    return {
        "source_name": source_name,
        "source_type": infer_source_type(source_name, source_url),
        "source_url": source_url,
        "title": compact_text(title, 240),
        "body_excerpt": excerpt or compact_text(title, 400),
        "quote_text": _pick_first_sentence(signal_text, PAIN_KEYWORDS + EMOTION_TERMS + WHY_NOW_TERMS),
        "role_hint": role_hint,
        "published_at": evidence.get("published_at") or evidence.get("date"),
        "timestamp_hint": evidence.get("timestamp_hint", ""),
        "metadata_json": metadata_json,
    }


# =============================================================================
# FAILURE EVENT EXTRACTION - Extract specific events, not generic summaries
# =============================================================================

def _extract_failure_event(text: str) -> str:
    """
    Extract specific failure event from text.
    Returns a description like:
    - "QuickBooks invoices do not match Stripe payouts during weekly reconciliation"
    - "Excel formulas break when copying between sheets"
    NOT generic like "data sync issue"
    """
    lowered = _normalized(text)

    # Pattern: "X does not match Y" / "X doesn't match Y" / "X not matching Y"
    match = re.search(r'(\w+(?:\s+\w+){0,5})\s+(?:does not|doesn\'t|don\'t|not)\s+match(?:es)?\s+(\w+(?:\s+\w+){0,5})', lowered)
    if match:
        return f"{match.group(1)} does not match {match.group(2)}"

    # Pattern: "X and Y don't reconcile"
    match = re.search(r'(\w+)\s+and\s+(\w+)\s+(?:don\'t|don\'t|doesn\'t)\s+reconcile', lowered)
    if match:
        return f"{match.group(1)} and {match.group(2)} don't reconcile"

    # Pattern: "X is out of sync with Y"
    match = re.search(r'(\w+)\s+(?:is|are)\s+out of sync (?:with|and)\s+(\w+)', lowered)
    if match:
        return f"{match.group(1)} is out of sync with {match.group(2)}"

    # Pattern: "X breaks when Y"
    match = re.search(r'(\w+(?:\s+\w+){0,3})\s+break(?:s|ing)?\s+(?:when|after|during)\s+([^\.]{5,40})', lowered)
    if match:
        return f"{match.group(1)} breaks when {match.group(2)}"

    # Pattern: "error when X" / "errors in X"
    match = re.search(r'(?:error|errors|erroring)\s+(?:in|when|while)\s+([^\.]{5,50})', lowered)
    if match:
        return f"Error when {match.group(1)}"

    # Pattern: "duplicate X" / "duplicates in X"
    match = re.search(r'(?:duplicate|duplicates)\s+(?:in|on|for)\s+([^\.]{5,40})', lowered)
    if match:
        return f"Duplicates in {match.group(1)}"

    # Pattern: "manually X" - extract specific task
    match = re.search(r'manually\s+([a-z\s]{5,40})', lowered)
    if match:
        task = match.group(1).strip()
        if len(task) > 5:
            return f"Manually {task}"

    # Pattern: "have to manually X"
    match = re.search(r'have to\s+manually\s+([a-z]{4,30})', lowered)
    if match:
        return f"Have to manually {match.group(1)}"

    # Pattern: "spending X hours Y" - specific time waste
    match = re.search(r'spend(?:ing|s)?\s+(\d+)\s+hours?\s+(?:on|doing|for)\s+([^\.]{5,40})', lowered)
    if match:
        return f"Spending {match.group(1)} hours {match.group(2)}"

    # Pattern: "X takes too long" / "X takes hours"
    match = re.search(r'([a-z\s]{5,30})\s+takes?\s+(?:too\s+)?long', lowered)
    if match:
        return f"{match.group(1)} takes too long"

    # Pattern: "missed X" / "missing X"
    match = re.search(r'(?:missed|missing)\s+([a-z\s]{5,30})', lowered)
    if match:
        return f"Missed {match.group(1)}"

    return ""


def _extract_trigger_moment(text: str) -> str:
    """
    Extract when the problem occurs.
    Returns like:
    - "at month-end reconciliation"
    - "when importing CSV"
    - "during weekly sync"
    """
    lowered = _normalized(text)

    # Pattern: "when X" / "when I'm X" / "when we X"
    match = re.search(r'when\s+([^\.]{5,40})', lowered)
    if match:
        trigger = match.group(1).strip()
        # Clean up trigger
        if trigger.startswith('i ') or trigger.startswith('i\'m '):
            trigger = trigger[2:] if trigger.startswith('i ') else trigger[4:]
        if len(trigger) > 3:
            return f"when {trigger}"

    # Pattern: "at X" (specific times)
    match = re.search(r'at\s+(month[\s-]?end|week[\s-]?end|quarter[\s-]?end|daily|weekly|monthly|year[\s-]?end)', lowered)
    if match:
        return f"at {match.group(1)}"

    # Pattern: "during X"
    match = re.search(r'during\s+([a-z\s]{5,30})', lowered)
    if match:
        return f"during {match.group(1)}"

    # Pattern: "after X"
    match = re.search(r'after\s+([a-z\s]{5,30})', lowered)
    if match:
        return f"after {match.group(1)}"

    # Pattern: "every time X"
    match = re.search(r'every\s+time\s+([^\.]{5,30})', lowered)
    if match:
        return f"every time {match.group(1)}"

    return ""


def _extract_consequence(text: str) -> str:
    """
    Extract specific consequence - time, money, risk.
    """
    lowered = _normalized(text)

    # Time cost patterns
    time_match = re.search(r'(\d+)\s+hours?\s+(?:per|a|each|every)\s+(day|week|month)', lowered)
    if time_match:
        return f"{time_match.group(1)} hours lost per {time_match.group(2)}"

    time_match2 = re.search(r'spend(?:ing|s)?\s+(\d+)\s+hours?', lowered)
    if time_match2:
        return f"{time_match2.group(1)} hours wasted"

    # Money cost patterns
    money_match = re.search(r'\$[\d,]+(?:\.\d{2})?\s+(?:lost|waste|cost|missed)', lowered)
    if money_match:
        return "money lost"

    money_match2 = re.search(r'(?:lost|missed|waste|due)\s+\$[\d,]+', lowered)
    if money_match2:
        return "money lost"

    # Error/risk patterns
    if 'error' in lowered and ('data' in lowered or 'record' in lowered or 'entry' in lowered):
        return "data errors"

    if 'late' in lowered and ('payment' in lowered or 'invoice' in lowered):
        return "late payments"

    if 'missed' in lowered and ('payment' in lowered or 'deadline' in lowered):
        return "missed payments"

    return ""


def _extract_specific_workflow(text: str, platform: str = "") -> str:
    """
    Extract specific workflow being performed.
    """
    lowered = _normalized(text)

    # If platform detected, look for platform-specific workflows
    if platform == 'quickbooks':
        if 'invoice' in lowered:
            return "processing invoices in QuickBooks"
        if 'reconcile' in lowered:
            return "reconciling transactions in QuickBooks"

    if platform == 'shopify':
        if 'order' in lowered:
            return "processing Shopify orders"
        if 'inventory' in lowered:
            return "managing Shopify inventory"

    if platform == 'excel' or platform == 'google_sheets':
        if 'formula' in lowered:
            return "working with spreadsheet formulas"
        if 'vlookup' in lowered or 'xlookup' in lowered:
            return "using spreadsheet lookups"
        if 'copy' in lowered:
            return "copying spreadsheet data"

    if platform == 'notion':
        if 'database' in lowered:
            return "managing Notion databases"
        if 'sync' in lowered:
            return "syncing Notion data"

    # Generic workflow extraction - look for specific task mentions
    workflows = [
        'reconcil', 'import', 'export', 'sync', 'update', 'copy', 'paste',
        'match', 'merge', 'dedupe', 'clean', 'format', 'validate',
    ]

    for wf in workflows:
        if wf in lowered:
            # Find context around the workflow
            match = re.search(r'([a-z\s]{0,20})' + wf + r'([a-z\s]{0,20})', lowered)
            if match:
                prefix = match.group(1).strip()
                suffix = match.group(2).strip()
                if prefix or suffix:
                    return f"{prefix} {wf} {suffix}".strip()

    return ""


def _is_failure_event_atom(text: str) -> bool:
    """
    Check if text contains a specific failure event, not generic summary.
    """
    lowered = _normalized(text)

    # Must have specific failure indicators
    failure_indicators = [
        'does not match', 'don\'t match', 'doesn\'t match',
        'out of sync', 'out of date',
        'break', 'broken', 'error', 'mistake',
        'duplicate', 'mismatch', 'inconsistent',
        'manually', 'have to manually',
        'spend hours', 'takes too long',
        'missed', 'missing',
    ]

    return any(ind in lowered for ind in failure_indicators)


def build_problem_atom(signal_payload: dict[str, Any], finding_data: dict[str, Any]) -> dict[str, Any]:
    title_text = compact_text(signal_payload.get("title", ""), 400)
    body_text = compact_text(signal_payload.get("body_excerpt", ""), 1600)
    cleaned_body_text = _strip_title_prefix(title_text, body_text)
    text = compact_text(f"{title_text}. {body_text}", 1800)
    segment = _match_rule(
        _segment_inference_context(finding_data, text),
        SEGMENT_RULES,
        "operators with recurring workflow pain",
    )
    user_role = signal_payload.get("role_hint") or _match_rule(text, ROLE_RULES, "operator")

    # Extract platform first (needed for context)
    platform = _extract_platform_from_text(text)

    # NEW: Extract specific failure event instead of generic summary
    # This is the key change - extract real events, not cluster labels
    failure_event = _extract_failure_event(text)
    trigger_moment = _extract_trigger_moment(text)
    specific_consequence = _extract_consequence(text)
    specific_workflow = _extract_specific_workflow(text, platform)

    # Use extracted failure event as job_to_be_done if available
    if failure_event:
        job_to_be_done = failure_event
    elif specific_workflow:
        job_to_be_done = specific_workflow
    else:
        # Fallback to generic only if no specific extraction
        job_to_be_done = _match_rule(text, JTBD_RULES, "keep a recurring workflow reliable without manual cleanup")

    descriptive_text = cleaned_body_text or body_text or text

    # Use extracted failure event for failure_mode
    if failure_event:
        failure_mode = failure_event
    else:
        failure_mode = _normalize_problem_fragment(
            _pick_first_sentence(
                descriptive_text,
                [
                    "manual",
                    "break",
                    "broken",
                    "fails",
                    "failed",
                    "error",
                    "can't",
                    "cant",
                    "unreachable",
                    "stuck",
                    "reset",
                    "deleted",
                    "fallback",
                ],
            ),
            fallback="",
            limit=120,
        )

    # Use extracted trigger moment
    trigger_event = trigger_moment if trigger_moment else _normalize_problem_fragment(
        _pick_first_sentence(descriptive_text, WHY_NOW_TERMS + ["when", "after", "during", "because"]),
        fallback="",
        limit=120,
    )
    workarounds = _extract_workarounds(text)
    current_tools = _normalize_tools(finding_data.get("tool_used") or "")
    assumptions = _extract_assumptions(text)
    urgency_clues = _extract_clues(text, URGENCY_TERMS)
    frequency_clues = _extract_clues(text, FREQUENCY_TERMS)
    why_now_clues = _extract_clues(text, WHY_NOW_TERMS)
    cost_clues = _extract_cost_clues(text)

    # Add specific consequence if extracted
    if specific_consequence and specific_consequence not in cost_clues:
        cost_clues.append(specific_consequence)

    emotional_hits = _extract_clues(text, EMOTION_TERMS)

    # Use extracted trigger if available, otherwise use old logic
    if not trigger_event:
        trigger_event = _normalize_problem_fragment(
            _pick_first_sentence(descriptive_text, WHY_NOW_TERMS + ["when", "after", "during", "because"]),
        fallback="",
        limit=120,
    )
    if trigger_event == pain_statement:
        trigger_event = ""
    if _looks_like_meta_prompt(pain_statement):
        pain_statement = ""
    if _looks_like_meta_prompt(failure_mode):
        failure_mode = ""
    if _looks_like_meta_prompt(trigger_event):
        trigger_event = ""
    if trigger_event and not _is_clean_context(trigger_event):
        trigger_event = ""
    if not failure_mode and workarounds:
        failure_mode = f"teams fall back to {', '.join(workarounds)}"

    if _is_generic_phrase(job_to_be_done) and failure_mode:
        job_to_be_done = _normalize_problem_fragment(failure_mode, fallback=job_to_be_done, limit=120)
    else:
        job_to_be_done = _normalize_problem_fragment(job_to_be_done, fallback=job_to_be_done, limit=120)

    filled_fields = sum(
        bool(value)
        for value in [
            segment,
            user_role,
            job_to_be_done,
            pain_statement,
            failure_mode,
            workarounds,
            current_tools,
            urgency_clues,
            frequency_clues,
            why_now_clues,
            cost_clues,
        ]
    )

    # Extract specific integration patterns (the "specific problem" detection)
    specific_patterns = _extract_specific_patterns(text)
    is_specific = _is_specific_problem(text, {})

    # Boost confidence for specific problems
    specific_boost = 0.15 if is_specific and specific_patterns else 0.0
    confidence = clamp(0.35 + filled_fields * 0.045 + specific_boost)
    emotional_intensity = clamp(0.2 + len(emotional_hits) * 0.16 + min(text.count("!"), 3) * 0.04)
    cluster_key = infer_recurrence_key(f"{segment} {user_role} {job_to_be_done} {failure_mode} {' '.join(workarounds)}")

    # Extract platform for atom enrichment
    platform = _extract_platform_from_text(text)

    # Validate atom quality
    quality_signals = _validate_atom_quality(
        {"job_to_be_done": job_to_be_done, "failure_mode": failure_mode, "cost_consequence_clues": ", ".join(cost_clues)},
        text
    )

    return {
        "cluster_key": cluster_key,
        "segment": segment,
        "user_role": user_role,
        "job_to_be_done": job_to_be_done,
        "trigger_event": trigger_event,
        "pain_statement": pain_statement,
        "failure_mode": failure_mode,
        "current_workaround": ", ".join(workarounds),
        "current_tools": current_tools,
        "urgency_clues": ", ".join(urgency_clues),
        "frequency_clues": ", ".join(frequency_clues),
        "emotional_intensity": emotional_intensity,
        "cost_consequence_clues": ", ".join(cost_clues),
        "why_now_clues": ", ".join(why_now_clues),
        "confidence": confidence,
        "assumptions": assumptions,
        "specific_patterns": specific_patterns,
        "is_specific_problem": is_specific,
        # New quality signals for wedge pipeline
        "platform": platform,
        "specificity_score": quality_signals.get("specificity_score", 0.0),
        "consequence_score": quality_signals.get("consequence_score", 0.0),
        "quality_issues": quality_signals.get("quality_issues", []),
        "atom_json": {
            "source_type": signal_payload.get("source_type", "web"),
            "workaround_terms": workarounds,
            "specific_patterns": specific_patterns,
            "is_specific_problem": is_specific,
            "urgency_terms": urgency_clues,
            "frequency_terms": frequency_clues,
            "cost_terms": cost_clues,
            "why_now_terms": why_now_clues,
            "emotional_terms": emotional_hits,
            "assumptions": assumptions,
            # Include quality signals
            "platform": platform,
            "quality_issues": quality_signals.get("quality_issues", []),
        },
    }


def classify_source_signal(
    finding_data: dict[str, Any],
    signal_payload: dict[str, Any],
    atom_payload: dict[str, Any],
) -> dict[str, Any]:
    evidence = signal_payload.get("metadata_json", {}).get("evidence", {})
    record_origin = str(signal_payload.get("metadata_json", {}).get("record_origin", "") or evidence.get("record_origin", "")).strip()
    text = _normalized(
        " ".join(
            [
                finding_data.get("source", ""),
                finding_data.get("source_url", ""),
                signal_payload.get("title", ""),
                signal_payload.get("body_excerpt", ""),
                json_dumps(signal_payload.get("metadata_json", {})),
            ]
        )
    )
    context_text = _normalized(
        " ".join(
            [
                finding_data.get("source", ""),
                finding_data.get("source_url", ""),
                evidence.get("source_plan", ""),
                evidence.get("discovery_query", ""),
                record_origin,
            ]
        )
    )
    reasons: list[str] = []

    if record_origin in {"listing", "app_description", "marketing_copy"}:
        reasons.append("listing_or_marketing_copy")
        return {"source_class": "low_signal_summary", "reasons": reasons}
    if _has_phrase(text, HELP_PAGE_PATTERNS):
        reasons.append("help_or_generic_summary_content")
        return {"source_class": "low_signal_summary", "reasons": reasons}
    if _has_phrase(context_text, DEMAND_SIGNAL_PATTERNS):
        reasons.append("search_or_trend_signal")
        return {"source_class": "demand_signal", "reasons": reasons}
    if _has_phrase(context_text, COMPETITION_SIGNAL_PATTERNS):
        reasons.append("competition_or_alternative_signal")
        return {"source_class": "competition_signal", "reasons": reasons}
    if _has_phrase(text, META_GUIDANCE_PATTERNS) or _has_phrase(text, INTERNAL_IDENTIFIER_PATTERNS):
        reasons.append("methodology_or_guidance_signal")
        return {"source_class": "meta_guidance", "reasons": reasons}
    if _has_phrase(text, PROMOTIONAL_PATTERNS):
        reasons.append("promo_or_generic_praise")
        return {"source_class": "low_signal_summary", "reasons": reasons}

    review_profile: dict[str, Any] = {}
    github_profile: dict[str, Any] = {}
    source_name = str(finding_data.get("source", "") or "")
    source_type = str(signal_payload.get("source_type", "") or "")
    finding_kind = str(finding_data.get("finding_kind", "") or "")

    if finding_kind == "success_signal":
        return {"source_class": "success_signal", "reasons": ["success_signal_finding_kind"]}
    if finding_kind == "meta_guidance":
        return {"source_class": "meta_guidance", "reasons": ["meta_guidance_finding_kind"]}

    if "review" in source_type or "wordpress-review" in source_name or "shopify-review" in source_name:
        review_profile = _review_generalizability(text, source_name, str(finding_data.get("source_url", "") or ""))
        if review_profile["review_issue_type"] == "product_specific_issue":
            reasons.append("review_product_specific_issue")
    if "github" in source_type or "github" in source_name.lower():
        github_profile = _github_generalizability(text)
        if github_profile["github_issue_type"] == "product_specific_issue":
            reasons.append("github_product_specific_issue")

    specificity = sum(
        bool(value)
        for value in [
            atom_payload.get("user_role"),
            atom_payload.get("segment"),
            atom_payload.get("job_to_be_done"),
            atom_payload.get("trigger_event"),
            atom_payload.get("failure_mode"),
            atom_payload.get("current_workaround"),
            atom_payload.get("cost_consequence_clues"),
            atom_payload.get("frequency_clues"),
        ]
    )
    has_consequence = bool(atom_payload.get("cost_consequence_clues") or atom_payload.get("urgency_clues"))
    has_structure = bool(atom_payload.get("failure_mode")) and bool(atom_payload.get("current_workaround") or has_consequence)

    if (
        finding_data.get("finding_kind") in {"pain_point", "problem_signal"}
        and specificity >= 5
        and has_structure
        and "review_product_specific_issue" not in reasons
        and "github_product_specific_issue" not in reasons
    ):
        return {"source_class": "pain_signal", "reasons": ["specific_structured_pain_evidence"], **review_profile, **github_profile}

    return {
        "source_class": "low_signal_summary",
        "reasons": reasons or ["insufficient_problem_specificity"],
        **review_profile,
        **github_profile,
    }


def qualify_problem_signal(
    finding_data: dict[str, Any],
    signal_payload: dict[str, Any],
    atom_payload: dict[str, Any],
) -> dict[str, Any]:
    finding_kind = finding_data.get("finding_kind", "")
    source_class = finding_data.get("source_class") or signal_payload.get("metadata_json", {}).get("source_class", "")
    text = _normalized(f"{signal_payload.get('title', '')}. {signal_payload.get('body_excerpt', '')}")
    positive_signals: list[str] = []
    negative_signals: list[str] = []
    score = 0
    source_name = str(finding_data.get("source", "") or "")
    why_now_text = _normalized(atom_payload.get("why_now_clues", ""))
    why_now_tokens = {token for token in re.split(r"[\s,;/]+", why_now_text) if token}
    has_descriptive_why_now = bool(why_now_tokens - GENERIC_WHY_NOW_FILLERS)
    behavior_signal_count = sum(
        bool(atom_payload.get(field))
        for field in [
            "current_workaround",
            "failure_mode",
            "urgency_clues",
            "frequency_clues",
            "cost_consequence_clues",
            "why_now_clues",
        ]
    )
    stakes_signal_count = sum(
        [
            bool(atom_payload.get("urgency_clues")),
            bool(atom_payload.get("frequency_clues")),
            bool(atom_payload.get("cost_consequence_clues")),
            has_descriptive_why_now,
        ]
    )
    workflow_context = _normalized(
        " ".join(
            [
                text,
                atom_payload.get("job_to_be_done", ""),
                atom_payload.get("trigger_event", ""),
                atom_payload.get("pain_statement", ""),
                atom_payload.get("failure_mode", ""),
                atom_payload.get("current_workaround", ""),
            ]
        )
    )
    actionable_workflow = _has_phrase(workflow_context, ACTIONABLE_WORKFLOW_HINTS)

    if finding_kind in {"pain_point", "problem_signal"}:
        positive_signals.append("problem_finding_kind")
        score += 2
    else:
        negative_signals.append("non_problem_finding_kind")
        score -= 4
    if source_class == "pain_signal":
        positive_signals.append("pain_signal_source_class")
        score += 2
    elif source_class:
        negative_signals.append(f"source_class_{source_class}")
        score -= 5

    if atom_payload.get("current_workaround"):
        positive_signals.append("workaround_detected")
        score += 2
    if atom_payload.get("failure_mode"):
        positive_signals.append("failure_mode_detected")
        score += 1
    if atom_payload.get("urgency_clues"):
        positive_signals.append("urgency_detected")
        score += 1
    if atom_payload.get("frequency_clues"):
        positive_signals.append("frequency_detected")
        score += 1
    if atom_payload.get("cost_consequence_clues"):
        positive_signals.append("cost_detected")
        score += 1
    if atom_payload.get("why_now_clues"):
        positive_signals.append("why_now_detected")
        score += 1

    pain_hits = sum(1 for keyword in PAIN_KEYWORDS if keyword in text)
    if pain_hits >= 2:
        positive_signals.append("pain_language")
        score += 1

    if _has_phrase(text, HELP_PAGE_PATTERNS):
        negative_signals.append("support_or_help_page")
        score -= 5
    if _has_phrase(text, CAREER_GUIDANCE_PATTERNS):
        negative_signals.append("career_guidance_thread")
        score -= 6
    title_lower = str(signal_payload.get("title", "") or "").lower().strip()
    if title_lower.startswith("how to ") and _has_phrase(text, TUTORIAL_SHARE_PATTERNS):
        negative_signals.append("tutorial_or_instructional_post")
        score -= 5
    if _has_phrase(text, REVIEW_NEGATIVE_TERMS) and not atom_payload.get("failure_mode"):
        negative_signals.append("vague_negative_review")
        score -= 3
    if _has_phrase(text, PROMOTIONAL_PATTERNS):
        negative_signals.append("promotional_or_celebratory")
        score -= 3
    if _has_phrase(text, ROI_SHOPPING_PATTERNS):
        negative_signals.append("roi_or_vendor_shopping_prompt")
        score -= 5
    if _has_phrase(text, GENERIC_REQUEST_PATTERNS):
        negative_signals.append("generic_request_or_vendor_shopping")
        score -= 3
    if _has_phrase(text, ADVICE_SEEKING_PATTERNS) and stakes_signal_count == 0 and not actionable_workflow:
        negative_signals.append("advice_seeking_without_actionable_stakes")
        score -= 4
    if _has_phrase(text, GENERIC_PROMPT_PATTERNS) and not atom_payload.get("current_workaround"):
        negative_signals.append("generic_prompt_without_behavioral_pain")
        score -= 3
    if _has_phrase(text, SOLICITATION_PROMPT_PATTERNS):
        negative_signals.append("solicitation_for_problem_examples")
        score -= 5
    if _has_phrase(text, VENTING_PATTERNS) and stakes_signal_count < 2 and behavior_signal_count < 4:
        negative_signals.append("venting_without_transferable_workflow_problem")
        score -= 5
    if _has_phrase(text, PRODUCT_COMPLAINT_PATTERNS):
        if not _has_phrase(workflow_context, OPERATIONAL_CONTEXT_HINTS):
            negative_signals.append("product_specific_complaint_without_workflow_context")
            score -= 5
    if "?" in signal_payload.get("title", "") and not atom_payload.get("current_workaround"):
        negative_signals.append("question_without_workaround")
        score -= 1
    if "github" in source_name.lower() and "feature request" in text:
        behavior_signals = sum(
            bool(atom_payload.get(field))
            for field in ["current_workaround", "frequency_clues", "urgency_clues", "cost_consequence_clues"]
        )
        if behavior_signals < 2:
            negative_signals.append("feature_request_without_behavioral_evidence")
            score -= 3

    accepted = (
        score >= 3
        and "non_problem_finding_kind" not in negative_signals
        and "support_or_help_page" not in negative_signals
        and "career_guidance_thread" not in negative_signals
        and "tutorial_or_instructional_post" not in negative_signals
        and "vague_negative_review" not in negative_signals
        and "roi_or_vendor_shopping_prompt" not in negative_signals
        and "advice_seeking_without_actionable_stakes" not in negative_signals
        and "solicitation_for_problem_examples" not in negative_signals
        and "venting_without_transferable_workflow_problem" not in negative_signals
        and "product_specific_complaint_without_workflow_context" not in negative_signals
        and (not source_class or source_class == "pain_signal")
    )
    if accepted and _has_phrase(text, GENERIC_REQUEST_PATTERNS) and (score < 7 or not actionable_workflow):
        accepted = False
        negative_signals.append("too_generic_after_review")

    return {
        "accepted": accepted,
        "score": score,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
    }


def build_cluster_summary(atoms: list[Any], signals: list[Any]) -> dict[str, Any]:
    if not atoms:
        return {
            "label": "Empty cluster",
            "segment": "unknown",
            "user_role": "unknown",
            "job_to_be_done": "unknown",
            "trigger_summary": "",
            "signal_count": 0,
            "atom_count": 0,
            "evidence_quality": 0.0,
            "summary_json": {},
        }

    segment = Counter(_value(atom, "segment", "") for atom in atoms if _value(atom, "segment", "")).most_common(1)
    user_role = Counter(_value(atom, "user_role", "") for atom in atoms if _value(atom, "user_role", "")).most_common(1)
    job = Counter(_value(atom, "job_to_be_done", "") for atom in atoms if _value(atom, "job_to_be_done", "")).most_common(1)
    trigger_summary = Counter(_value(atom, "trigger_event", "") for atom in atoms if _value(atom, "trigger_event", "")).most_common(1)
    dominant_failure = Counter(_value(atom, "failure_mode", "") for atom in atoms if _value(atom, "failure_mode", "")).most_common(1)
    dominant_workaround = Counter(_value(atom, "current_workaround", "") for atom in atoms if _value(atom, "current_workaround", "")).most_common(1)
    # NEW: Extract specific pain statements for more descriptive labels
    pain_statements = Counter(_value(atom, "pain_statement", "") for atom in atoms if _value(atom, "pain_statement", "") and len(_value(atom, "pain_statement", "")) > 20).most_common(1)

    segment_value = segment[0][0] if segment else "operators with recurring workflow pain"
    user_role_value = user_role[0][0] if user_role else "operator"
    job_value = job[0][0] if job else "keep a recurring workflow on track"
    # NEW: Get the most specific pain statement available
    specific_pain = pain_statements[0][0] if pain_statements else ""
    if not specific_pain and dominant_failure:
        specific_pain = _normalize_problem_fragment(dominant_failure[0][0], fallback="", limit=96)
    if _is_generic_phrase(job_value) and dominant_failure:
        job_value = _normalize_problem_fragment(dominant_failure[0][0], fallback=job_value, limit=96)
    trigger_value = trigger_summary[0][0] if trigger_summary else ""
    if dominant_failure and (
        not trigger_value
        or "audit" in _normalized(trigger_value)
        or _is_generic_phrase(trigger_value)
        or len(trigger_value.split()) <= 2
    ):
        trigger_value = dominant_failure[0][0]
    trigger_value = _select_specific_context(trigger_value, dominant_failure[0][0] if dominant_failure else "", limit=88)
    workaround_value = dominant_workaround[0][0] if dominant_workaround else ""

    source_types = Counter(_value(signal, "source_type", "web") for signal in signals)
    evidence_quality = clamp(
        0.26
        + min(len(atoms), 4) * 0.08
        + min(len([signal for signal in signals if _value(signal, "quote_text", "")]), 3) * 0.05
        + min(len(source_types), 3) * 0.07
        + min(len([atom for atom in atoms if _value(atom, "current_workaround", "")]), 3) * 0.05
    )

    # Use specific pain statement for the label if available, otherwise fall back to generic
    if specific_pain and len(specific_pain) > 15:
        label = f"{user_role_value} - {specific_pain}"
    else:
        label = f"{user_role_value} - {job_value}"
        if trigger_value:
            label += f" when {trigger_value.lower()}"
    label = compact_text(label, 130)

    return {
        "label": label,
        "segment": segment_value,
        "user_role": user_role_value,
        "job_to_be_done": job_value,
        "trigger_summary": trigger_value,
        "signal_count": len(signals),
        "atom_count": len(atoms),
        "evidence_quality": round(evidence_quality, 4),
        "summary_json": {
            "dominant_failure": dominant_failure[0][0] if dominant_failure else "",
            "dominant_workaround": workaround_value,
            "source_types": dict(source_types),
            "cluster_context": trigger_value,
            "sample_pains": [compact_text(_value(atom, "pain_statement", ""), 120) for atom in atoms[:3]],
            "sample_failures": [compact_text(_value(atom, "failure_mode", ""), 120) for atom in atoms[:3] if _value(atom, "failure_mode", "")],
        },
    }


def assess_market_gap(cluster_summary: dict[str, Any], validation_evidence: dict[str, Any]) -> dict[str, Any]:
    scores = validation_evidence.get("scores", {})
    solution_gap = float(scores.get("solution_gap_score", 0.0) or 0.0)
    saturation = float(scores.get("saturation_score", 0.0) or 0.0)
    recurrence = float(scores.get("problem_score", 0.0) or 0.0)
    summary = cluster_summary.get("summary_json", {})
    why_now_text = " ".join(summary.get("sample_failures", [])) + " " + " ".join(summary.get("sample_pains", []))
    why_now_strength = clamp(0.15 + 0.18 * len(_extract_clues(why_now_text, WHY_NOW_TERMS)))
    evidence_quality = float(cluster_summary.get("evidence_quality", 0.0) or 0.0)

    if recurrence < 0.18 and evidence_quality < 0.4:
        gap = "likely_false_signal"
        recurrence_state = "weak"
    elif recurrence < 0.45 or evidence_quality < 0.5:
        gap = "needs_more_recurrence_evidence"
        recurrence_state = "thin"
    elif why_now_strength >= 0.55:
        gap = "newly_emerging_due_to_environment_change"
        recurrence_state = "supported"
    elif solution_gap >= 0.6 and saturation <= 0.45:
        gap = "underserved_edge_case"
        recurrence_state = "supported"
    elif solution_gap >= 0.35:
        gap = "partially_solved"
        recurrence_state = "supported"
    else:
        gap = "already_solved_well"
        recurrence_state = "supported"

    return {
        "market_gap": gap,
        "recurrence_state": recurrence_state,
        "why_now_strength": round(why_now_strength, 4),
        "solution_gap_score": solution_gap,
        "saturation_score": saturation,
    }


# Service-first keywords for income optimization
SERVICE_FIRST_KEYWORDS = [
    "consulting", "agency", "implementation", "integration",
    "setup", "migration", "training", "managed", "support"
]


def detect_service_first_bonus(atom_text: str) -> float:
    """Detect service-first / productized-service opportunity indicators.

    Returns 0.04 bonus if any keyword found in atom text.
    """
    text_lower = atom_text.lower()
    if any(kw in text_lower for kw in SERVICE_FIRST_KEYWORDS):
        return 0.04
    return 0.0


def compute_problem_truth_score(scores: dict[str, Any]) -> float:
    """Compute Problem Truth Score (PTS).

    Answers: Is this a real, recurring, evidenced, costly problem worth solving?

    Formula:
    PTS = clamp(
        pain_severity * 0.16
        + frequency_score * 0.14
        + cost_of_inaction * 0.12
        + workaround_density * 0.10
        + evidence_quality * 0.18
        + corroboration_strength * 0.10
        + segment_concentration * 0.08
        - education_burden * 0.14
        - dependency_risk * 0.12
        - adoption_friction * 0.12
    )
    """
    pain_severity = scores.get("pain_severity", 0.3)
    frequency_score = scores.get("frequency_score", 0.25)
    cost_of_inaction = scores.get("cost_of_inaction", 0.2)
    workaround_density = scores.get("workaround_density", 0.2)
    evidence_quality = scores.get("evidence_quality", 0.3)
    corroboration_strength = scores.get("corroboration_strength", 0.25)
    segment_concentration = scores.get("segment_concentration", 0.3)
    education_burden = scores.get("education_burden", 0.2)
    dependency_risk = scores.get("dependency_risk", 0.15)
    adoption_friction = scores.get("adoption_friction", 0.15)

    pts = (
        pain_severity * 0.16
        + frequency_score * 0.14
        + cost_of_inaction * 0.12
        + workaround_density * 0.10
        + evidence_quality * 0.18
        + corroboration_strength * 0.10
        + segment_concentration * 0.08
        - education_burden * 0.14
        - dependency_risk * 0.12
        - adoption_friction * 0.12
    )

    return clamp(pts, 0.0, 1.0)


def compute_revenue_readiness_score(
    scores: dict[str, Any],
    market_enrichment: dict[str, Any],
    atom_text: str = ""
) -> float:
    """Compute Revenue Readiness Score (RRS).

    Answers: Is there a clear buyer, willingness to pay, reachability, path to revenue?

    Formula:
    RRS = clamp(
        value_support * 0.18
        + willingness_to_pay_proxy * 0.16
        + reachability * 0.14
        + buildability * 0.10
        + expansion_potential * 0.08
        + operational_buyer * 0.14
        + service_first_bonus * 0.04
        + cost_of_inaction * 0.06
    )
    """
    value_support = scores.get("value_support", 0.25)
    willingness_to_pay_proxy = scores.get("willingness_to_pay_proxy", 0.2)
    reachability = scores.get("reachability", 0.4)
    buildability = scores.get("buildability", 0.3)
    expansion_potential = scores.get("expansion_potential", 0.25)
    cost_of_inaction = scores.get("cost_of_inaction", 0.2)

    operational_buyer = float(market_enrichment.get("operational_buyer_score", 0.0) or 0.0)
    if not operational_buyer:
        operational_buyer = 0.15  # fallback if not enriched

    # Service-first bonus
    service_bonus = detect_service_first_bonus(atom_text)

    rrs = (
        value_support * 0.18
        + willingness_to_pay_proxy * 0.16
        + reachability * 0.14
        + buildability * 0.10
        + expansion_potential * 0.08
        + operational_buyer * 0.14
        + service_bonus
        + cost_of_inaction * 0.06
    )

    return clamp(rrs, 0.0, 1.0)


def compute_decision_score(pts: float, rrs: float) -> float:
    """Compute blended decision score from PTS and RRS.

    Uses 55% PTS / 45% RRS weighting.
    """
    return clamp(pts * 0.55 + rrs * 0.45, 0.0, 1.0)


def score_opportunity(
    atom: Any,
    signal: Any,
    cluster_summary: dict[str, Any],
    validation_evidence: dict[str, Any],
    market_gap: dict[str, Any],
    *,
    review_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = validation_evidence.get("scores", {}) or {}
    evidence = validation_evidence.get("evidence", {}) or {}
    corroboration = validation_evidence.get("corroboration", {}) or {}
    market_enrichment = validation_evidence.get("market_enrichment", {}) or {}

    recurrence_score = float(scores.get("problem_score", 0.0) or 0.0)
    feasibility = float(scores.get("feasibility_score", 0.0) or 0.0)
    value_score = float(scores.get("value_score", 0.0) or 0.0)
    recurrence_coverage = float(evidence.get("recurrence_query_coverage", 0.0) or 0.0)
    recurrence_doc_count = int(evidence.get("recurrence_doc_count", 0) or 0)
    recurrence_domain_count = int(evidence.get("recurrence_domain_count", 0) or 0)
    recurrence_results_by_source = evidence.get("recurrence_results_by_source", {}) or {}

    urgency_hits = len([item for item in str(_value(atom, "urgency_clues", "")).split(", ") if item])
    frequency_hits = len([item for item in str(_value(atom, "frequency_clues", "")).split(", ") if item])
    why_now_hits = len([item for item in str(_value(atom, "why_now_clues", "")).split(", ") if item])
    cost_hits = len([item for item in str(_value(atom, "cost_consequence_clues", "")).split(", ") if item])
    workaround_hits = len([item for item in str(_value(atom, "current_workaround", "")).split(", ") if item])
    clear_trigger = 1 if _value(atom, "trigger_event", "") else 0
    clear_failure = 1 if _value(atom, "failure_mode", "") else 0
    source_type = _value(signal, "source_type", "web")
    operational_haystack = _normalized(
        " ".join(
            [
                str(_value(atom, "segment", "")),
                str(_value(atom, "user_role", "")),
                str(_value(atom, "job_to_be_done", "")),
                str(_value(atom, "failure_mode", "")),
                str(_value(atom, "why_now_clues", "")),
            ]
        )
    )

    operational_buyer = float(market_enrichment.get("operational_buyer_score", 0.0) or 0.0)
    if not operational_buyer:
        operational_buyer = clamp(
            0.15
            + (0.18 if any(term in operational_haystack for term in ["operations", "operator", "seller", "support", "finance", "account", "compliance"]) else 0.0)
            + (0.14 if any(term in operational_haystack for term in ["billing", "payout", "shipping", "device setup", "evidence", "compliance", "order"]) else 0.0)
        )
    compliance_burden = float(market_enrichment.get("compliance_burden_score", 0.0) or 0.0)
    cost_pressure = float(market_enrichment.get("cost_pressure_score", 0.0) or 0.0)
    buyer_intent_score = float(market_enrichment.get("buyer_intent_score", 0.0) or 0.0)
    demand_score = float(market_enrichment.get("demand_score", 0.0) or 0.0)
    competition_score = float(market_enrichment.get("competition_score", 0.0) or 0.0)
    trend_score = float(market_enrichment.get("trend_score", 0.0) or 0.0)
    review_signal_score = float(market_enrichment.get("review_signal_score", 0.0) or 0.0)
    willingness_to_pay_signal = float(market_enrichment.get("willingness_to_pay_signal", 0.0) or 0.0)
    wedge_value_lift = float(market_enrichment.get("wedge_value_lift", 0.0) or 0.0)
    multi_source_value_lift = float(market_enrichment.get("multi_source_value_lift", 0.0) or 0.0)
    corroboration_score = float(corroboration.get("corroboration_score", 0.0) or 0.0)
    evidence_sufficiency_hint = float(corroboration.get("evidence_sufficiency", 0.0) or 0.0)
    cross_source_match_score = float(corroboration.get("cross_source_match_score", 0.0) or 0.0)
    source_family_diversity = int(corroboration.get("source_family_diversity", 0) or 0)
    core_source_family_diversity = int(corroboration.get("core_source_family_diversity", 0) or 0)
    generalizability_score = float(corroboration.get("generalizability_score", 0.0) or 0.0)
    generalizability_penalty = float(corroboration.get("generalizability_penalty", 0.0) or 0.0)

    corroborating_sources = sum(1 for count in recurrence_results_by_source.values() if count)

    # Fix 1: Removed cost_hits to avoid double-counting (now only in cost_of_inaction)
    # Fix 2: Removed urgency_hits to avoid triple-counting (now only in urgency_score)
    pain_severity = clamp(0.32 + float(_value(atom, "emotional_intensity", 0.0)) * 0.38)
    # Fix 3: Removed recurrence_score from corroboration_strength and evidence_sufficiency (triple-counted)
    #      Now only in frequency_score
    frequency_score = clamp(0.24 + recurrence_score * 0.3 + frequency_hits * 0.12 + cross_source_match_score * 0.1)
    # Fix 2: Removed urgency_hits to avoid triple-counting (now only in urgency_score)
    cost_of_inaction = clamp(
        0.18
        + value_score * 0.18
        + cost_hits * 0.12
        + operational_buyer * 0.14
        + compliance_burden * 0.16
        + cost_pressure * 0.22
        + wedge_value_lift * 0.08
    )
    workaround_density = clamp(0.2 + workaround_hits * 0.18 + (0.12 if _value(atom, "current_workaround", "") else 0.0))
    urgency_score = clamp(0.22 + urgency_hits * 0.12 + float(_value(atom, "emotional_intensity", 0.0)) * 0.18)
    segment_concentration = clamp(
        0.28
        + min(cluster_summary.get("atom_count", 0), 4) * 0.08
        + min(recurrence_doc_count, 4) * 0.03
        + core_source_family_diversity / 4.0 * 0.08
    )
    reachability = clamp(
        0.42
        + (0.16 if source_type in {"forum", "github_issue", "review"} else 0.05)
        + (0.1 if "small business" in str(_value(atom, "segment", "")).lower() or "seller" in str(_value(atom, "user_role", "")).lower() else 0.0)
    )
    timing_shift = clamp(0.18 + why_now_hits * 0.16 + float(market_gap.get("why_now_strength", 0.0) or 0.0) * 0.25 + trend_score * 0.12)
    buildability = clamp(0.24 + feasibility * 0.5 + (0.08 if "manual" in str(_value(atom, "current_workaround", "")) else 0.0))
    expansion_potential = clamp(
        0.22
        + segment_concentration * 0.16
        + source_family_diversity / 4.0 * 0.12
        + (0.12 if "workflow" in str(_value(atom, "job_to_be_done", "")).lower() else 0.0)
    )
    # Fix 6: Removed operational_buyer to avoid triple-counting (now only in cost_of_inaction)
    # Fix 7: Removed compliance_burden to avoid triple-counting (now only in cost_of_inaction)
    # Fix 8: Removed wedge_value_lift to avoid triple-counting (now only in cost_of_inaction)
    willingness_to_pay_proxy = clamp(
        0.18
        + cost_of_inaction * 0.18
        + buyer_intent_score * 0.14
        + demand_score * 0.08
        + willingness_to_pay_signal * 0.16
    )
    education_burden = clamp(0.2 + (0.18 if recurrence_score < 0.45 else 0.0) + (0.12 if not _value(atom, "user_role", "") else 0.0))
    dependency_risk = clamp(
        0.14
        + (0.2 if "github" in source_type or "platform" in str(_value(atom, "why_now_clues", "")).lower() else 0.0)
        + (0.15 if market_gap["market_gap"] == "already_solved_well" else 0.0)
        + competition_score * 0.12
    )
    adoption_friction = clamp(
        0.16
        + (0.12 if len([p for p in str(_value(atom, "current_tools", "")).split(",") if p.strip()]) > 2 else 0.0)
        + (0.16 if buildability < 0.55 else 0.0)
    )
    recurrence_doc_strength = clamp(min(recurrence_doc_count, 6) / 6 * 0.6 + min(recurrence_domain_count, 4) / 4 * 0.4)
    # Fix 3: Removed recurrence_score to avoid triple-counting (now only in frequency_score)
    corroboration_strength = clamp(
        max(corroboration_score, 0.0) * 0.45
        + recurrence_coverage * 0.12
        + recurrence_doc_strength * 0.08
        + min(corroborating_sources, 4) / 4 * 0.05
        + cross_source_match_score * 0.05
        + core_source_family_diversity / 4.0 * 0.03
    )
    # Fix 3: Removed recurrence_score to avoid triple-counting (now only in frequency_score)
    evidence_sufficiency = clamp(
        max(evidence_sufficiency_hint, 0.0) * 0.5
        + recurrence_coverage * 0.08
        + recurrence_doc_strength * 0.08
        + corroboration_strength * 0.1
        + cluster_summary.get("evidence_quality", 0.15) * 0.05
        - generalizability_penalty * 0.2
    )
    # Fix 4: Removed value_score to avoid triple-counting (now only in cost_of_inaction)
    problem_plausibility = clamp(
        pain_severity * 0.24
        + cost_of_inaction * 0.19
        + workaround_density * 0.14
        + urgency_score * 0.1
        + timing_shift * 0.08
        + clear_trigger * 0.04
        + clear_failure * 0.04
        + generalizability_score * 0.1
    )
    # Fix 5: Removed cost_pressure to avoid double-counting (now only in cost_of_inaction)
    # Fix 6: Removed operational_buyer to avoid triple-counting (now only in cost_of_inaction)
    # Fix 7: Removed compliance_burden to avoid triple-counting (now only in cost_of_inaction)
    operational_value_lift = clamp(
        cross_source_match_score * 0.08
        + multi_source_value_lift * 0.18
    )
    # Fix 1: Reduced cost_of_inaction weight to avoid double-counting cost signals
    # Fix 4: Removed value_score to avoid triple-counting (now only in cost_of_inaction)
    # Fix 8: Removed wedge_value_lift to avoid triple-counting (now only in cost_of_inaction)
    value_support = clamp(
        cost_of_inaction * 0.12
        + willingness_to_pay_proxy * 0.24
        + buyer_intent_score * 0.12
        + demand_score * 0.08
        + trend_score * 0.05
        + review_signal_score * 0.04
        + (1.0 - competition_score) * 0.03
        + operational_value_lift * 0.12
        - generalizability_penalty * 0.2
    )
    evidence_quality = clamp(evidence_sufficiency * 0.72 + corroboration_strength * 0.16 + feasibility * 0.06 + generalizability_score * 0.06)

    # Fix 6: Cap saturated markets - reduce composite if market is saturated
    saturation = float(scores.get("saturation_score", 0.0) or 0.0)
    saturation_multiplier = 0.75 if saturation > 0.7 else 1.0

    positive = (
        pain_severity * 0.14
        + frequency_score * 0.1
        + cost_of_inaction * 0.1
        + workaround_density * 0.08
        + urgency_score * 0.08
        + segment_concentration * 0.07
        + reachability * 0.06
        + timing_shift * 0.08
        + buildability * 0.06
        + expansion_potential * 0.06
        + value_support * 0.17
    )
    # Fix 2: Strengthened penalty weights to better filter bad ideas
    penalty = education_burden * 0.18 + dependency_risk * 0.16 + adoption_friction * 0.16
    # Fix 3: Nonlinear evidence multiplier - weak evidence heavily discounted, strong evidence full weight
    import math
    evidence_multiplier = clamp(0.3 + 0.7 * math.sqrt(evidence_quality), 0.3, 1.0)
    # Apply saturation penalty
    composite_score = clamp(max(0.0, positive - penalty) * evidence_multiplier * saturation_multiplier)
    confidence = clamp(0.35 + evidence_quality * 0.4 + float(_value(atom, "confidence", 0.0)) * 0.25)

    review_feedback_count = 0
    review_feedback_labels: dict[str, int] = {}
    review_feedback_park_bias = 0.0
    review_feedback_kill_bias = 0.0
    if review_feedback:
        review_feedback_count = int(review_feedback.get("total_reviews", 0) or 0)
        review_feedback_labels = dict(review_feedback.get("labels", {}) or {})
        review_feedback_park_bias = float(review_feedback.get("park_bias", 0.0) or 0.0)
        review_feedback_kill_bias = float(review_feedback.get("kill_bias", 0.0) or 0.0)
        composite_score = clamp(composite_score + review_feedback_park_bias - review_feedback_kill_bias)
        value_support = clamp(value_support + max(0.0, review_feedback_park_bias * 0.3) - max(0.0, review_feedback_kill_bias * 0.2))

    # Build scores dict for PTS/RRS computation
    scores_for_rrs = {
        "pain_severity": pain_severity,
        "frequency_score": frequency_score,
        "cost_of_inaction": cost_of_inaction,
        "workaround_density": workaround_density,
        "evidence_quality": evidence_quality,
        "corroboration_strength": corroboration_strength,
        "segment_concentration": segment_concentration,
        "education_burden": education_burden,
        "dependency_risk": dependency_risk,
        "adoption_friction": adoption_friction,
        "value_support": value_support,
        "willingness_to_pay_proxy": willingness_to_pay_proxy,
        "reachability": reachability,
        "buildability": buildability,
        "expansion_potential": expansion_potential,
    }

    # Compute PTS and RRS
    problem_truth_score = compute_problem_truth_score(scores_for_rrs)

    # Build atom text for service-first detection
    atom_text = " ".join([
        str(_value(atom, "problem_description", "")),
        str(_value(atom, "job_to_be_done", "")),
        str(_value(atom, "current_workaround", "")),
    ])
    revenue_readiness_score = compute_revenue_readiness_score(scores_for_rrs, market_enrichment, atom_text)

    # Compute blended decision score
    decision_score = compute_decision_score(problem_truth_score, revenue_readiness_score)

    return {
        "scoring_version": CURRENT_SCORING_VERSION,
        "formula_version": CURRENT_FORMULA_VERSION,
        "threshold_version": CURRENT_THRESHOLD_VERSION,
        "problem_truth_score": round(problem_truth_score, 4),
        "revenue_readiness_score": round(revenue_readiness_score, 4),
        "decision_score": round(decision_score, 4),
        "pain_severity": round(pain_severity, 4),
        "frequency_score": round(frequency_score, 4),
        "cost_of_inaction": round(cost_of_inaction, 4),
        "workaround_density": round(workaround_density, 4),
        "urgency_score": round(urgency_score, 4),
        "segment_concentration": round(segment_concentration, 4),
        "reachability": round(reachability, 4),
        "timing_shift": round(timing_shift, 4),
        "buildability": round(buildability, 4),
        "expansion_potential": round(expansion_potential, 4),
        "willingness_to_pay_proxy": round(willingness_to_pay_proxy, 4),
        "operational_value_lift": round(operational_value_lift, 4),
        "value_support": round(value_support, 4),
        "education_burden": round(education_burden, 4),
        "dependency_risk": round(dependency_risk, 4),
        "adoption_friction": round(adoption_friction, 4),
        "corroboration_strength": round(corroboration_strength, 4),
        "evidence_quality": round(evidence_quality, 4),
        "evidence_sufficiency": round(evidence_sufficiency, 4),
        "problem_plausibility": round(problem_plausibility, 4),
        "composite_score": round(composite_score, 4),
        "confidence": round(confidence, 4),
        "evidence_multiplier": round(clamp(evidence_multiplier), 4),
        "review_feedback_count": review_feedback_count,
        "review_feedback_labels": review_feedback_labels,
        "review_feedback_park_bias": round(review_feedback_park_bias, 4),
        "review_feedback_kill_bias": round(review_feedback_kill_bias, 4),
    }


def build_counterevidence(opportunity_scores: dict[str, Any], market_gap: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {
            "claim": "The pain is rare or isolated.",
            "status": "contradicted" if opportunity_scores["frequency_score"] >= 0.55 else "supported",
            "summary": "Recurring source evidence clears the rarity bar." if opportunity_scores["frequency_score"] >= 0.55 else "Recurrence is still too thin.",
        },
        {
            "claim": "The problem is already solved well enough.",
            "status": "supported" if market_gap["market_gap"] == "already_solved_well" else "contradicted",
            "summary": "Competitor density is high and the gap looks thin." if market_gap["market_gap"] == "already_solved_well" else "Existing tools still leave a visible gap.",
        },
        {
            "claim": "Users tolerate this instead of acting on it.",
            "status": "supported" if opportunity_scores["urgency_score"] < 0.5 else "contradicted",
            "summary": "Urgency is muted." if opportunity_scores["urgency_score"] < 0.5 else "Users are signaling active frustration and urgency.",
        },
        {
            "claim": "The opportunity is too fragmented across segments.",
            "status": "supported" if opportunity_scores["segment_concentration"] < 0.5 and opportunity_scores.get("corroboration_strength", 0.0) < 0.45 else "contradicted",
            "summary": "The segment is still diffuse." if opportunity_scores["segment_concentration"] < 0.5 and opportunity_scores.get("corroboration_strength", 0.0) < 0.45 else "Signals are concentrated enough to target.",
        },
        {
            "claim": "The economics are too weak to matter.",
            "status": "supported" if max(opportunity_scores["cost_of_inaction"], opportunity_scores.get("willingness_to_pay_proxy", 0.0), opportunity_scores.get("value_support", 0.0)) < 0.5 else "contradicted",
            "summary": "Cost-of-inaction evidence is weak." if max(opportunity_scores["cost_of_inaction"], opportunity_scores.get("willingness_to_pay_proxy", 0.0), opportunity_scores.get("value_support", 0.0)) < 0.5 else "Behavior points to real cost or downside.",
        },
    ]
    return checks


def plan_validation_experiment(
    atom: Any,
    cluster_summary: dict[str, Any],
    opportunity_scores: dict[str, Any],
    market_gap: dict[str, Any],
) -> dict[str, Any]:
    if opportunity_scores["frequency_score"] < 0.5 or opportunity_scores["evidence_quality"] < 0.5:
        test_type = "workflow_walkthrough"
        smallest_test = "Run 5 workflow walkthroughs with operators who currently manage the workaround."
        success_signal = "At least 3 participants show the exact workflow and confirm the pain is recurring."
        failure_signal = "Most participants describe the pain as rare, solved, or low-priority."
    elif opportunity_scores["reachability"] >= 0.62 and opportunity_scores["cost_of_inaction"] >= 0.6:
        test_type = "concierge_test"
        smallest_test = "Offer a manual concierge service that removes the workaround for 2-3 target users this week."
        success_signal = "Users hand over the workflow and ask for continued help or offer to pay."
        failure_signal = "Users like the idea but will not share the workflow or do not return after the first run."
    elif opportunity_scores["segment_concentration"] >= 0.6 and market_gap["market_gap"] != "already_solved_well":
        test_type = "fake_door"
        smallest_test = "Launch a narrow landing page for the segment and route interested users to a waitlist plus workflow survey."
        success_signal = "Qualified users convert and describe the same broken workflow in their own words."
        failure_signal = "Traffic is noisy or signups do not describe the underlying pain."
    else:
        test_type = "problem_interviews"
        smallest_test = "Run 6 problem interviews focused on trigger, workaround, and cost of inaction."
        success_signal = "Multiple people report the same trigger, workaround, and downside."
        failure_signal = "Stories vary too widely to support a concentrated wedge."

    cluster_context = cluster_summary.get("summary_json", {}).get("cluster_context", "")
    role = str(_value(atom, "user_role", "") or "Operator").title()
    job = _infer_specific_job(
        str(_value(atom, "job_to_be_done", "")),
        _normalize_problem_fragment(str(_value(atom, "job_to_be_done", "")), fallback="keep a recurring workflow on track", limit=72),
    ).lower()
    trigger = _summarize_context(cluster_context)
    if not trigger:
        trigger = _summarize_context(
            _select_specific_context(
                str(_value(atom, "trigger_event", "")),
                str(_value(atom, "failure_mode", "")),
                str(_value(atom, "pain_statement", "")),
                limit=72,
            )
        )
    if not trigger:
        trigger = _fallback_context_from_atom(atom)
    trigger = trigger.lower() or "the workflow breaks"
    workaround = _normalize_workaround_phrase(
        str(_value(atom, "current_workaround", "")) or cluster_summary.get("summary_json", {}).get("dominant_workaround", ""),
        fallback="manual workarounds",
    )
    cost = _normalize_cost_phrase(str(_value(atom, "cost_consequence_clues", ""))).lower()
    test_label = {
        "workflow_walkthrough": "a workflow walkthrough",
        "concierge_test": "a concierge test",
        "fake_door": "a fake-door test",
        "problem_interviews": "a problem interview",
    }.get(test_type, "a validation test")
    hypothesis = (
        f"{role} teams experiencing {trigger} will engage with {test_label} because they currently patch the workflow with {workaround} "
        f"and risk {cost} while trying to {job}."
    )
    return {
        "test_type": test_type,
        "hypothesis": compact_text(hypothesis, 320),
        "falsifier": "Kill the opportunity if most target users describe the issue as rare, already solved, or not important enough to change their current behavior.",
        "smallest_test": smallest_test,
        "success_signal": success_signal,
        "failure_signal": failure_signal,
        "stage": "research",
        "build_ready": False,
        "cluster_label": cluster_summary.get("label", ""),
    }


def stage_decision(
    opportunity_scores: dict[str, Any],
    market_gap: dict[str, Any],
    counterevidence: list[dict[str, Any]],
    *,
    # v4 thresholds: using decision_score + PTS/RRS floors
    promotion_threshold: float = 0.18,
    park_threshold: float = 0.15,
    review_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supported_count = sum(1 for check in counterevidence if check.get("status") == "supported")
    park_bias = float((review_feedback or {}).get("park_bias", 0.0) or 0.0)
    kill_bias = float((review_feedback or {}).get("kill_bias", 0.0) or 0.0)
    has_v4_scores = any(
        key in opportunity_scores for key in ("decision_score", "problem_truth_score", "revenue_readiness_score")
    )

    if not has_v4_scores:
        composite = float(opportunity_scores.get("composite_score", 0.0) or 0.0) + park_bias - kill_bias
        plausibility = float(opportunity_scores.get("problem_plausibility", 0.0) or 0.0)
        sufficiency = float(opportunity_scores.get("evidence_sufficiency", 0.0) or 0.0)
        value_support = float(opportunity_scores.get("value_support", 0.0) or 0.0)
        evidence_quality = float(opportunity_scores.get("evidence_quality", 0.0) or 0.0)

        hard_kill = (
            market_gap.get("market_gap") == "already_solved_well"
            or supported_count >= 4
            or (market_gap.get("market_gap") == "likely_false_signal" and plausibility < 0.45)
            or (composite < park_threshold and plausibility < 0.38 and sufficiency < 0.35)
            or (kill_bias >= 0.12 and plausibility < 0.45 and sufficiency < 0.32)
        )
        if hard_kill:
            return {
                "status": "killed",
                "recommendation": "kill",
                "reason": "unlikely_or_economically_weak",
                "decision_reason": "unlikely_or_economically_weak",
                "park_subreason": "",
            }

        if (
            composite >= promotion_threshold
            and plausibility >= 0.6
            and evidence_quality >= 0.55
            and supported_count <= 1
            and value_support >= 0.58
        ):
            return {
                "status": "promoted",
                "recommendation": "promote",
                "reason": "validated_selection_gate",
                "decision_reason": "validated_selection_gate",
                "park_subreason": "",
            }

        recurrence_short = market_gap.get("market_gap") == "needs_more_recurrence_evidence" or sufficiency < 0.46
        value_short = value_support < 0.5
        if recurrence_short and value_short:
            subreason = "park_both"
        elif recurrence_short:
            subreason = "park_recurrence"
        elif value_short:
            subreason = "park_value"
        else:
            subreason = "plausible_but_unproven"

        if review_feedback:
            labels = set((review_feedback.get("labels") or {}).keys())
            if "needs_more_evidence" in labels and subreason == "park_recurrence":
                subreason = "plausible_but_unproven"

        return {
            "status": "parked",
            "recommendation": "park",
            "reason": subreason,
            "decision_reason": subreason,
            "park_subreason": subreason,
        }

    # Get new v4 scores
    decision_score = float(opportunity_scores.get("decision_score", 0.0) or 0.0) + park_bias - kill_bias
    problem_truth_score = float(opportunity_scores.get("problem_truth_score", 0.0) or 0.0)
    revenue_readiness_score = float(opportunity_scores.get("revenue_readiness_score", 0.0) or 0.0)

    # Legacy scores still available for diagnostic
    composite = float(opportunity_scores.get("composite_score", 0.0) or 0.0)
    plausibility = float(opportunity_scores.get("problem_plausibility", 0.0) or 0.0)
    sufficiency = float(opportunity_scores.get("evidence_sufficiency", 0.0) or 0.0)
    value_support = float(opportunity_scores.get("value_support", 0.0) or 0.0)
    evidence_quality = float(opportunity_scores.get("evidence_quality", 0.0) or 0.0)
    frequency_score = float(opportunity_scores.get("frequency_score", 0.0) or 0.0)

    # Hard kill conditions
    hard_kill = (
        market_gap.get("market_gap") == "already_solved_well"
        or supported_count >= 4
        or (frequency_score < 0.25)
        or (problem_truth_score < 0.10)  # Hard PTS floor (below P50)
    )
    if hard_kill:
        return {
            "status": "killed",
            "recommendation": "kill",
            "reason": "unlikely_or_economically_weak",
            "decision_reason": "unlikely_or_economically_weak",
            "park_subreason": "",
        }

    # v4 Promote logic: blended score + floors (no longer multi-gate AND)
    # Primary: decision_score >= promotion_threshold AND PTS >= 0.11 AND RRS >= 0.22 AND frequency >= 0.25
    if (decision_score >= promotion_threshold
        and problem_truth_score >= 0.11
        and revenue_readiness_score >= 0.22
        and frequency_score >= 0.25):
        return {
            "status": "promoted",
            "recommendation": "promote",
            "reason": "validated_selection_gate",
            "decision_reason": "validated_selection_gate",
            "park_subreason": "",
        }

    # Override 1: High frequency override
    if decision_score >= 0.40 and frequency_score >= 0.50:
        return {
            "status": "promoted",
            "recommendation": "promote",
            "reason": "high_frequency_override",
            "decision_reason": "high_frequency_override",
            "park_subreason": "",
        }

    # Override 2: Strong evidence override
    if decision_score >= 0.38 and evidence_quality >= 0.70:
        return {
            "status": "promoted",
            "recommendation": "promote",
            "reason": "strong_evidence_override",
            "decision_reason": "strong_evidence_override",
            "park_subreason": "",
        }

    # Park logic: not promoted, not killed, decision_score >= park_threshold
    if decision_score >= park_threshold:
        recurrence_short = market_gap.get("market_gap") == "needs_more_recurrence_evidence" or sufficiency < 0.46
        value_short = value_support < 0.4
        if recurrence_short and value_short:
            subreason = "park_both"
        elif recurrence_short:
            subreason = "park_recurrence"
        elif value_short:
            subreason = "park_value"
        else:
            subreason = "plausible_but_unproven"

        if review_feedback:
            labels = set((review_feedback.get("labels") or {}).keys())
            if "needs_more_evidence" in labels and subreason == "park_recurrence":
                subreason = "plausible_but_unproven"

        return {
            "status": "parked",
            "recommendation": "park",
            "reason": subreason,
            "decision_reason": subreason,
            "park_subreason": subreason,
        }

    # Default: kill
    return {
        "status": "killed",
        "recommendation": "kill",
        "reason": "below_threshold",
        "decision_reason": "below_threshold",
        "park_subreason": "",
    }


def diagnose_stage_decision(
    opportunity_scores: dict[str, Any],
    market_gap: dict[str, Any],
    counterevidence: list[dict[str, Any]],
    *,
    promotion_threshold: float = 0.18,
    park_threshold: float = 0.15,
    review_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Human-readable breakdown of ``stage_decision`` inputs vs floors (for gate debugging)."""
    supported_count = sum(1 for check in counterevidence if check.get("status") == "supported")
    park_bias = float((review_feedback or {}).get("park_bias", 0.0) or 0.0)
    kill_bias = float((review_feedback or {}).get("kill_bias", 0.0) or 0.0)
    composite_raw = float(opportunity_scores.get("composite_score", 0.0) or 0.0)
    composite = composite_raw + park_bias - kill_bias
    plausibility = float(opportunity_scores.get("problem_plausibility", 0.0) or 0.0)
    sufficiency = float(opportunity_scores.get("evidence_sufficiency", 0.0) or 0.0)
    value_support = float(opportunity_scores.get("value_support", 0.0) or 0.0)
    evidence_quality = float(opportunity_scores.get("evidence_quality", 0.0) or 0.0)

    decision = stage_decision(
        opportunity_scores,
        market_gap,
        counterevidence,
        promotion_threshold=promotion_threshold,
        park_threshold=park_threshold,
        review_feedback=review_feedback,
    )

    hard_kill_reasons: list[str] = []
    if market_gap.get("market_gap") == "already_solved_well":
        hard_kill_reasons.append("market_gap=already_solved_well")
    if supported_count >= 4:
        hard_kill_reasons.append(f"counterevidence_supported_count>={supported_count} (>=4)")
    if market_gap.get("market_gap") == "likely_false_signal" and plausibility < 0.45:
        hard_kill_reasons.append("likely_false_signal_and_plausibility<0.45")
    if composite < park_threshold and plausibility < 0.38 and sufficiency < 0.35:
        hard_kill_reasons.append(
            f"weak_triple: composite({composite:.3f})<{park_threshold} & plausibility<0.38 & sufficiency<0.35"
        )
    if kill_bias >= 0.12 and plausibility < 0.45 and sufficiency < 0.32:
        hard_kill_reasons.append("operator_kill_bias_heavy")

    promote_checks: list[dict[str, Any]] = [
        {
            "id": "composite_vs_promotion_threshold",
            "pass": composite >= promotion_threshold,
            "actual": round(composite, 4),
            "floor": promotion_threshold,
            "note": "composite uses scorecard.composite_score ± review biases",
        },
        {
            "id": "problem_plausibility",
            "pass": plausibility >= 0.55,
            "actual": round(plausibility, 4),
            "floor": 0.6,
        },
        {
            "id": "evidence_quality",
            "pass": evidence_quality >= 0.55,
            "actual": round(evidence_quality, 4),
            "floor": 0.55,
        },
        {
            "id": "counterevidence_supported_count",
            "pass": supported_count <= 1,
            "actual": supported_count,
            "ceiling": 1,
        },
        {
            "id": "value_support",
            "pass": value_support >= 0.58,
            "actual": round(value_support, 4),
            "floor": 0.58,
        },
    ]

    all_promotion_numeric_gates_pass = all(check["pass"] for check in promote_checks)

    return {
        "decision": decision,
        "inputs": {
            "composite_raw": round(composite_raw, 4),
            "park_bias": round(park_bias, 4),
            "kill_bias": round(kill_bias, 4),
            "composite_effective": round(composite, 4),
            "supported_counterevidence_hits": supported_count,
            "promotion_threshold": promotion_threshold,
            "park_threshold": park_threshold,
        },
        "hard_kill_reasons": hard_kill_reasons,
        "promote_checks": promote_checks,
        "all_promotion_numeric_gates_pass": all_promotion_numeric_gates_pass,
        "park_subreason_if_parked": decision.get("park_subreason") if decision.get("recommendation") == "park" else "",
    }


class OpportunityEngine:
    """Wrapper around deterministic weak-signal extraction helpers."""

    def build_raw_signal(
        self,
        *,
        finding_id: int,
        source: str,
        source_url: str,
        title: str,
        author: str,
        content_text: str,
        evidence: dict[str, Any] | None = None,
        content_hash: str = "",
    ) -> RawSignal:
        payload = build_raw_signal_payload(
            {
                "source": source,
                "source_url": source_url,
                "product_built": title,
                "entrepreneur": author,
                "outcome_summary": content_text,
                "evidence": evidence or {},
                "finding_kind": "problem_signal",
            }
        )
        return RawSignal(
            finding_id=finding_id,
            source_name=payload["source_name"],
            source_type=payload["source_type"],
            source_url=payload["source_url"],
            title=payload["title"],
            body_excerpt=payload["body_excerpt"],
            content_hash=content_hash or hashlib.sha1(f"{source}|{source_url}|{title}|{content_text}".encode("utf-8")).hexdigest(),
            source_class="pain_signal",
            quote_text=payload.get("quote_text", ""),
            role_hint=payload["role_hint"],
            published_at=payload.get("published_at"),
            timestamp_hint=payload.get("timestamp_hint", ""),
            metadata={"author": author, **payload.get("metadata_json", {})},
        )

    def extract_problem_atom(self, signal: RawSignal, *, finding_kind: str) -> ProblemAtom | None:
        signal_payload = {
            "source_name": signal.source_name,
            "source_type": signal.source_type,
            "source_url": signal.source_url,
            "title": signal.title,
            "body_excerpt": signal.body_excerpt,
            "quote_text": signal.quote_text or signal.title,
            "role_hint": signal.role_hint,
            "metadata_json": signal.metadata or {},
        }
        finding_data = {
            "source": signal.source_name,
            "source_url": signal.source_url,
            "product_built": signal.title,
            "tool_used": "",
            "finding_kind": finding_kind,
            "outcome_summary": signal.body_excerpt,
            "source_class": signal.source_class,
        }
        payload = build_problem_atom(signal_payload, finding_data)
        if finding_kind != "pain_point" and payload["confidence"] < 0.55:
            return None
        return ProblemAtom(
            signal_id=signal.id or 0,
            finding_id=signal.finding_id,
            raw_signal_id=signal.id or 0,
            cluster_key=payload["cluster_key"],
            segment=payload["segment"],
            user_role=payload["user_role"],
            job_to_be_done=payload["job_to_be_done"],
            trigger_event=payload["trigger_event"],
            pain_statement=payload["pain_statement"],
            failure_mode=payload["failure_mode"],
            current_workaround=payload["current_workaround"],
            current_tools=payload["current_tools"],
            source_quote=signal.quote_text or payload["pain_statement"],
            urgency_clues=payload["urgency_clues"],
            frequency_clues=payload["frequency_clues"],
            emotional_intensity=payload["emotional_intensity"],
            cost_consequence_clues=payload["cost_consequence_clues"],
            why_now_clues=payload["why_now_clues"],
            confidence=payload["confidence"],
            confidence_score=payload["confidence"],
            atom_json=json_dumps(payload["atom_json"]),
            metadata={"cluster_key": payload["cluster_key"]},
        )

    def cluster_key_for_atom(self, atom: ProblemAtom) -> str:
        if atom.metadata and atom.metadata.get("cluster_key"):
            return atom.metadata["cluster_key"]
        return infer_recurrence_key(
            f"{atom.segment} {atom.user_role} {atom.job_to_be_done} {atom.failure_mode} {atom.current_workaround}"
        )

    def build_cluster(self, atom: ProblemAtom, cluster_atoms: list[ProblemAtom], source_names: list[str]) -> OpportunityCluster:
        class _Signal:
            def __init__(self, source_type: str, quote_text: str):
                self.source_type = source_type
                self.quote_text = quote_text

        summary = build_cluster_summary(
            cluster_atoms,
            [_Signal(name, member.source_quote) for name, member in zip(source_names, cluster_atoms)],
        )
        return OpportunityCluster(
            label=summary["label"],
            status="candidate",
            summary=summary["summary_json"],
            user_role=summary["user_role"],
            job_to_be_done=summary["job_to_be_done"],
            metadata={
                "cluster_key": self.cluster_key_for_atom(atom),
                "trigger_pattern": summary["trigger_summary"],
                "workaround_pattern": atom.current_workaround,
                "failure_pattern": atom.failure_mode,
                "source_count": len(set(source_names)),
                "signal_count": summary["signal_count"],
                "atom_count": summary["atom_count"],
                "evidence_quality": summary["evidence_quality"],
            },
        )

    def map_market_gap(
        self,
        *,
        cluster: OpportunityCluster,
        recurrence_docs: list[dict[str, Any]],
        competitor_docs: list[dict[str, Any]],
        counter_docs: list[dict[str, Any]],
    ) -> str:
        validation_evidence = {
            "scores": {
                "problem_score": clamp(len(recurrence_docs) / 6.0),
                "solution_gap_score": clamp(1.0 - len(competitor_docs) / 8.0),
                "saturation_score": clamp(1.0 - len(competitor_docs) / 10.0),
            }
        }
        market_gap = assess_market_gap(
            {
                "label": cluster.label,
                "evidence_quality": (cluster.metadata or {}).get("evidence_quality", 0.0),
                "summary_json": cluster.summary or {},
            },
            validation_evidence,
        )
        if counter_docs and len(counter_docs) >= len(recurrence_docs):
            return "likely_false_signal"
        return market_gap["market_gap"]

    def score_opportunity(
        self,
        *,
        cluster: OpportunityCluster,
        atoms: list[ProblemAtom],
        recurrence_docs: list[dict[str, Any]],
        competitor_docs: list[dict[str, Any]],
        counter_docs: list[dict[str, Any]],
        market_gap_state: str,
    ) -> dict[str, Any]:
        atom = atoms[0]
        signal = type("Signal", (), {"source_type": "forum"})()
        validation_evidence = {
            "scores": {
                "problem_score": clamp(len(recurrence_docs) / 6.0),
                "feasibility_score": 0.72 if atom.current_workaround else 0.56,
                "value_score": clamp(0.25 + 0.15 * len([value for value in atom.cost_consequence_clues.split(", ") if value])),
                "solution_gap_score": clamp(1.0 - len(competitor_docs) / 8.0),
                "saturation_score": clamp(1.0 - len(competitor_docs) / 10.0),
            },
            "evidence": {
                "recurrence_query_coverage": clamp(len(recurrence_docs) / 8.0),
                "recurrence_doc_count": len(recurrence_docs),
                "recurrence_domain_count": len({urlparse(doc.get("url", "")).netloc for doc in recurrence_docs if doc.get("url")}),
                "recurrence_results_by_source": {"web": len(recurrence_docs)},
            },
        }
        market_gap = {"market_gap": market_gap_state, "why_now_strength": clamp(0.2 + 0.15 * len([value for value in atom.why_now_clues.split(", ") if value]))}
        raw_scores = score_opportunity(
            atom,
            signal,
            {
                "label": cluster.label,
                "atom_count": (cluster.metadata or {}).get("atom_count", len(atoms)),
                "evidence_quality": (cluster.metadata or {}).get("evidence_quality", 0.0),
            },
            validation_evidence,
            market_gap,
        )
        counterevidence = build_counterevidence(raw_scores, market_gap)
        if counter_docs:
            counterevidence.append(
                {
                    "claim": "Counterevidence search found likely disconfirming signals.",
                    "status": "supported",
                    "summary": f"{len(counter_docs)} contradicting docs were found.",
                }
            )
        decision = stage_decision(raw_scores, market_gap, counterevidence)
        return {
            "positive_dimensions": {
                "pain_severity": raw_scores["pain_severity"],
                "frequency": raw_scores["frequency_score"],
                "cost_of_inaction": raw_scores["cost_of_inaction"],
                "workaround_density": raw_scores["workaround_density"],
                "urgency": raw_scores["urgency_score"],
                "segment_concentration": raw_scores["segment_concentration"],
                "reachability": raw_scores["reachability"],
                "timing_shift": raw_scores["timing_shift"],
                "buildability": raw_scores["buildability"],
                "expansion_potential": raw_scores["expansion_potential"],
            },
            "penalties": {
                "education_burden": raw_scores["education_burden"],
                "dependency_risk": raw_scores["dependency_risk"],
                "adoption_friction": raw_scores["adoption_friction"],
            },
            "evidence_multiplier": raw_scores["evidence_multiplier"],
            "recurrence_count": len(recurrence_docs),
            "competitor_count": len(competitor_docs),
            "counterevidence_count": len(counter_docs),
            "total_score": raw_scores["composite_score"],
            "decision": decision["recommendation"],
            "value_support": raw_scores["value_support"],
            "problem_plausibility": raw_scores["problem_plausibility"],
        }

    def build_experiment_plan(
        self,
        *,
        cluster: OpportunityCluster,
        scorecard: dict[str, Any],
        market_gap_state: str,
    ) -> ValidationExperiment:
        atom = ProblemAtom(
            signal_id=0,
            finding_id=0,
            cluster_key=(cluster.metadata or {}).get("cluster_key", ""),
            segment=cluster.metadata.get("segment", "") if cluster.metadata else "",
            user_role=cluster.user_role,
            job_to_be_done=cluster.job_to_be_done,
            trigger_event=(cluster.metadata or {}).get("trigger_pattern", ""),
            pain_statement=cluster.label,
            failure_mode=(cluster.metadata or {}).get("failure_pattern", ""),
            current_workaround=(cluster.metadata or {}).get("workaround_pattern", ""),
            current_tools="",
            urgency_clues="",
            frequency_clues="",
            emotional_intensity=0.5,
            cost_consequence_clues="",
            why_now_clues="",
            confidence=(cluster.metadata or {}).get("evidence_quality", 0.5),
        )
        plan = plan_validation_experiment(
            atom,
            {
                "label": cluster.label,
                "atom_count": (cluster.metadata or {}).get("atom_count", 1),
                "evidence_quality": (cluster.metadata or {}).get("evidence_quality", 0.5),
            },
            {
                "frequency_score": scorecard["positive_dimensions"]["frequency"],
                "evidence_quality": (cluster.metadata or {}).get("evidence_quality", 0.5),
                "reachability": scorecard["positive_dimensions"]["reachability"],
                "cost_of_inaction": scorecard["positive_dimensions"]["cost_of_inaction"],
                "segment_concentration": scorecard["positive_dimensions"]["segment_concentration"],
            },
            {"market_gap": market_gap_state},
        )
        return ValidationExperiment(
            opportunity_id=0,
            cluster_id=cluster.id or 0,
            test_type=plan["test_type"],
            hypothesis=plan["hypothesis"],
            falsifier=plan["falsifier"],
            smallest_test=plan["smallest_test"],
            success_signal=plan["success_signal"],
            failure_signal=plan["failure_signal"],
        )

    def support_summary(self, atoms: list[ProblemAtom], recurrence_docs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "quotes": [atom.source_quote for atom in atoms if atom.source_quote][:5],
            "workarounds": [atom.current_workaround for atom in atoms if atom.current_workaround][:5],
            "recurrence_docs": recurrence_docs[:5],
        }

    def counter_summary(self, counter_docs: list[dict[str, Any]], competitor_docs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "counterevidence_docs": counter_docs[:5],
            "competitor_docs": competitor_docs[:5],
        }

    def classify_counter_doc(self, title: str, snippet: str) -> bool:
        lowered = _normalized(f"{title} {snippet}")
        return any(term in lowered for term in ["already solved", "good enough", "works fine", "easy workaround"])
