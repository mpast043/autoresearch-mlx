"""Builder output layer - generates commercialization cards for narrow software wedges.

This module takes surviving wedges and outputs standardized commercialization
artifacts optimized for builder decision-making and first revenue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# WEDGE EVALUATION CONSTANTS
SOFTWARE_FIT_WEIGHTS = {
    "platform_native": 0.25,      # Lives in platform user already uses
    "repeat_usage": 0.25,         # User does this regularly
    "data_touchpoint": 0.20,      # Touches business data (high trust requirement)
    "error_prevention": 0.15,     # Prevents costly mistakes
    "manual_to_automated": 0.15,  # Replaces manual work
}

MONETIZATION_FIT_WEIGHTS = {
    "recurring_trigger": 0.30,    # Weekly/monthly action creates subscription logic
    "error_cost": 0.25,           # Errors are costly enough to pay for prevention
    "frequency": 0.20,             # How often user does this
    "buyer_clarity": 0.15,         # Clear first customer profile
    "price_elasticity": 0.10,     # Can justify $9-99/month
}

TRUST_RISK_LEVELS = {
    "low": ["validator", "checker", "scanner", "inspector"],
    "medium": ["fixer", "cleaner", "enricher"],
    "high": ["auto-repair", "rewrite", "AI correction"],
}


@dataclass
class BuilderCard:
    """Standardized commercialization card for builder decision-making."""
    wedge_id: int
    wedge_title: str
    exact_user: str
    exact_workflow: str
    exact_trigger: str
    exact_failure: str
    exact_consequence: str
    host_platform: str
    product_shape: str
    why_this_shape_fits: str
    software_fit_score: float
    monetization_fit_score: float
    trust_risk: str
    mvp_in_scope: list[str] = field(default_factory=list)
    mvp_out_of_scope: list[str] = field(default_factory=list)
    first_paid_offer: str = ""
    pricing_hypothesis: str = ""
    first_customer: str = ""
    first_channel: str = ""
    evidence_summary: str = ""
    why_this_is_narrow: str = ""
    why_this_could_make_money: str = ""
    builder_priority: str = ""
    builder_verdict: str = ""


def evaluate_software_fit(wedge_data: dict[str, Any]) -> float:
    """Evaluate how naturally software-first this wedge is (0-1)."""
    score = 0.0

    workflow = wedge_data.get("workflow", "").lower()
    trigger = wedge_data.get("trigger", "").lower()
    failure = wedge_data.get("failure", "").lower()

    # Platform native indicator
    platforms = ["shopify", "quickbooks", "woocommerce", "excel", "google sheets", "csv", "api"]
    if any(p in workflow or p in trigger for p in platforms):
        score += SOFTWARE_FIT_WEIGHTS["platform_native"]

    # Repeat usage indicator
    repeat_words = ["every week", "monthly", "recurring", "routine", "regular", "daily"]
    if any(w in workflow or w in trigger for w in repeat_words):
        score += SOFTWARE_FIT_WEIGHTS["repeat_usage"]

    # Data touchpoint
    data_words = ["import", "export", "csv", "file", "data", "record", "transaction"]
    if any(w in failure for w in data_words):
        score += SOFTWARE_FIT_WEIGHTS["data_touchpoint"]

    # Error prevention
    error_words = ["error", "fail", "corrupt", "wrong", "mismatch", "break", "reject"]
    if any(w in failure for w in error_words):
        score += SOFTWARE_FIT_WEIGHTS["error_prevention"]

    # Manual to automated
    manual_words = ["manually", "by hand", "check each", "review every"]
    if any(w in workflow for w in manual_words):
        score += SOFTWARE_FIT_WEIGHTS["manual_to_automated"]

    return min(score, 1.0)


def evaluate_monetization_fit(wedge_data: dict[str, Any]) -> float:
    """Evaluate monetization potential (0-1)."""
    score = 0.0

    workflow = wedge_data.get("workflow", "").lower()
    trigger = wedge_data.get("trigger", "").lower()
    consequence = wedge_data.get("consequence", "").lower()

    # Recurring trigger
    if any(w in workflow or w in trigger for w in ["every week", "monthly", "routine", "regular"]):
        score += MONETIZATION_FIT_WEIGHTS["recurring_trigger"]

    # Error cost (consequence indicates pain)
    cost_words = ["hours", "lost", "corrupt", "customer-facing", "thousands", "expensive", "cost"]
    if any(w in consequence for w in cost_words):
        score += MONETIZATION_FIT_WEIGHTS["error_cost"]

    # Frequency
    freq_words = ["daily", "weekly", "many times"]
    if any(w in workflow for w in freq_words):
        score += MONETIZATION_FIT_WEIGHTS["frequency"]

    # Clear buyer
    if "shopify" in workflow or "quickbooks" in workflow:
        score += MONETIZATION_FIT_WEIGHTS["buyer_clarity"]

    return min(score, 1.0)


def assess_trust_risk(product_shape: str, failure: str) -> str:
    """Assess trust risk level based on product shape and failure type."""
    failure_lower = failure.lower()

    # High risk: auto-correction of business data
    high_risk_words = ["auto-fix", "auto-correct", "AI fix", "rewrite", "repair automatically"]
    if any(w in failure_lower for w in high_risk_words):
        return "high"

    # Medium risk: cleaning or enriching data
    if any(w in failure_lower for w in ["clean", "fix", "enrich", "update"]):
        return "medium"

    # Low risk: detection, validation, scanning
    return "low"


def determine_builder_verdict(
    software_fit: float,
    monetization_fit: float,
    trust_risk: str,
    is_narrow: bool,
) -> str:
    """Determine builder verdict based on evaluation scores."""
    if not is_narrow:
        return "reject_for_now"

    if trust_risk == "high":
        return "research_more"

    combined = (software_fit * 0.6) + (monetization_fit * 0.4)

    if combined >= 0.7:
        return "build_now"
    elif combined >= 0.5:
        return "backup_candidate"
    else:
        return "research_more"


def generate_builder_card(
    wedge_id: int,
    wedge_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> BuilderCard:
    """Generate a standardized builder card for a wedge."""

    # Extract core fields
    title = wedge_data.get("title", f"Wedge {wedge_id}")
    cluster_label = wedge_data.get("cluster_label", "")

    # Determine platform from evidence
    host_platform = "csv/file"  # Default
    if evidence:
        for e in evidence:
            source = e.get("source", "")
            if "shopify" in source.lower():
                host_platform = "Shopify"
            elif "quickbooks" in source.lower() or "qbo" in source.lower():
                host_platform = "QuickBooks Online"
            elif "github" in source.lower():
                host_platform = "developer/CI"

    # Build exact user/workflow/trigger/failure from evidence
    user, workflow, trigger, failure, consequence = _extract_wedge_components(
        evidence, cluster_label
    )

    # Determine product shape
    product_shape = _determine_product_shape(host_platform, workflow, failure)

    # Evaluate fit
    wedge_eval = {
        "workflow": workflow,
        "trigger": trigger,
        "failure": failure,
    }
    software_fit = evaluate_software_fit(wedge_eval)
    monetization_fit = evaluate_monetization_fit(wedge_eval)

    # Trust risk
    trust_risk = assess_trust_risk(product_shape, failure)

    # Is narrow?
    is_narrow = _is_narrow_wedge(user, workflow, failure, product_shape)

    # Builder verdict
    builder_verdict = determine_builder_verdict(
        software_fit, monetization_fit, trust_risk, is_narrow
    )

    # MVP scope based on product shape
    mvp_in, mvp_out = _define_mvp_scope(product_shape, failure)

    # Build the card
    card = BuilderCard(
        wedge_id=wedge_id,
        wedge_title=title[:80],
        exact_user=user,
        exact_workflow=workflow,
        exact_trigger=trigger,
        exact_failure=failure,
        exact_consequence=consequence,
        host_platform=host_platform,
        product_shape=product_shape,
        why_this_shape_fits=_why_shape_fits(product_shape, host_platform, workflow),
        software_fit_score=round(software_fit, 2),
        monetization_fit_score=round(monetization_fit, 2),
        trust_risk=trust_risk,
        mvp_in_scope=mvp_in,
        mvp_out_of_scope=mvp_out,
        first_paid_offer=_first_paid_offer(product_shape, host_platform),
        pricing_hypothesis=_pricing_hypothesis(product_shape),
        first_customer=_first_customer(host_platform),
        first_channel=_first_channel(host_platform),
        evidence_summary=_evidence_summary(evidence),
        why_this_is_narrow=_why_narrow(user, failure),
        why_this_could_make_money=_why_money(monetization_fit, consequence),
        builder_priority="primary" if builder_verdict == "build_now" else "secondary",
        builder_verdict=builder_verdict,
    )

    return card


def _extract_wedge_components(
    evidence: list[dict[str, Any]],
    cluster_label: str,
) -> tuple[str, str, str, str, str]:
    """Extract exact user/workflow/trigger/failure from evidence."""

    if not evidence:
        return (
            "Small business operators",
            "Import CSV data into business system",
            "Before clicking import",
            "Single bad value causes import failure or wrong data",
            "Hours debugging, corrupted data, customer-facing errors",
        )

    # Get first evidence for core extraction
    first = evidence[0]
    summary = first.get("outcome_summary", "")

    # Extract user role
    user = first.get("entrepreneur", "Small business operators")

    # Build workflow from source
    source = first.get("source", "")

    if "shopify" in source.lower():
        user = "Shopify store owners"
        workflow = "Import product/vendor CSV into Shopify"
        trigger = "Before clicking 'Import products' in Shopify admin"
    elif "quickbooks" in summary.lower() or "accounting" in source.lower():
        user = "Bookkeepers and accountants"
        workflow = "Import vendor/customer CSV into QuickBooks"
        trigger = "Before monthly reconciliation import"
    elif "inventory" in summary.lower():
        user = "Operations managers"
        workflow = "Import inventory CSV into tracking system"
        trigger = "Weekly inventory update import"
    elif "csv" in summary.lower():
        user = "Small business operators"
        workflow = "Import CSV data into business system"
        trigger = "Before importing file"
    else:
        user = "Business operators"
        workflow = "Import data via CSV"
        trigger = "Before data import"

    # Extract failure mode
    if "single bad" in summary.lower() or "single value" in summary.lower():
        failure = "Single bad value causes import to fail or silently corrupt data"
    elif "mismatch" in summary.lower():
        failure = "Data mismatch between systems causes reconciliation errors"
    elif "duplicate" in summary.lower():
        failure = "Duplicate records break import or create duplicates"
    elif "format" in summary.lower():
        failure = "Format incompatibility causes import to fail"
    else:
        failure = "Data quality issues break the import process"

    # Extract consequence
    if "hours" in summary.lower():
        consequence = "Hours spent debugging failed imports"
    elif "corrupt" in summary.lower():
        consequence = "Wrong data in production system"
    else:
        consequence = "Import fails, operations halt, manual fix required"

    return user, workflow, trigger, failure, consequence


def _determine_product_shape(host_platform: str, workflow: str, failure: str) -> str:
    """Determine best product shape for the wedge."""

    workflow_lower = workflow.lower()

    if "shopify" in workflow_lower:
        return "Shopify App"
    elif "quickbooks" in workflow_lower:
        return "QuickBooks App"
    elif "google sheets" in workflow_lower or "excel" in workflow_lower:
        return "Spreadsheet Add-in"
    elif "import" in workflow_lower and "csv" in workflow_lower:
        return "Web-based CSV Validator"

    return "microSaaS Web App"


def _is_narrow_wedge(user: str, workflow: str, failure: str, product_shape: str) -> bool:
    """Determine if wedge is narrow enough to build."""

    # Check for broad indicators
    broad_words = ["platform", "all systems", "any csv", "universal", "everything"]
    if any(w in workflow.lower() for w in broad_words):
        return False

    # Must have specific platform, workflow, or failure mode
    specific_indicators = [
        "shopify", "quickbooks", "woocommerce", "wordpress",
        "inventory", "product", "vendor", "invoice", "payment",
        "reconciliation", "import", "csv", "spreadsheet", "excel",
        "google sheets", "formula", "budget", "commission", "handoff",
        "bid", "campaign", "billing", "accounting",
    ]
    if not any(p in (workflow + " " + failure).lower() for p in specific_indicators):
        return False

    return True


def _define_mvp_scope(product_shape: str, failure: str) -> tuple[list[str], list[str]]:
    """Define MVP scope - what is in and out."""

    failure_lower = failure.lower()

    mvp_in = [
        "CSV file upload",
        "Schema validation (required columns)",
        "Data type validation (numbers, dates)",
        "Error report display",
        "Cleaned CSV export",
    ]

    mvp_out = [
        "Live system integration (API)",
        "Auto-fix errors",
        "Multiple file formats",
        "Saved templates",
        "User accounts",
        "Team features",
        "Billing (free initially)",
        "Historical reports",
    ]

    # Add auto-fix only if low trust risk
    if "silently" in failure_lower or "corrupt" in failure_lower:
        mvp_in.append("Error highlighting only")

    return mvp_in, mvp_out


def _why_shape_fits(product_shape: str, host_platform: str, workflow: str) -> str:
    """Explain why this product shape fits."""

    if "Shopify" in product_shape:
        return "Direct Shopify admin integration, no install required, discovered in Shopify App Store"
    elif "QuickBooks" in product_shape:
        return "QuickBooks App directory distribution, trusted by QBO users"
    elif "Add-in" in product_shape:
        return "Users already in spreadsheet, no new tool to learn"
    else:
        return "Browser-based, no install, works with any CSV workflow"


def _first_paid_offer(product_shape: str, host_platform: str) -> str:
    """Define the first paid offer."""

    return "Upload CSV before import, see errors, download clean file - $9/month unlimited"


def _pricing_hypothesis(product_shape: str) -> str:
    """Pricing hypothesis."""

    return "$9/month or $99/year - saves 2-4 hours per failed import at $25-50/hr value"


def _first_customer(host_platform: str) -> str:
    """First customer profile."""

    if "Shopify" in host_platform:
        return "Shopify store owner importing products from supplier CSVs"
    elif "QuickBooks" in host_platform:
        return "Bookkeeper with 5-15 SMB clients doing monthly imports"
    else:
        return "Small business ops person doing weekly CSV imports"


def _first_channel(host_platform: str) -> str:
    """First customer acquisition channel."""

    if "Shopify" in host_platform:
        return "Shopify Facebook groups, r/shopify, Product Hunt"
    elif "QuickBooks" in host_platform:
        return "ProAdvisor forums, r/accounting, bookkeeping FB groups"
    else:
        return "Product Hunt, Reddit r/smallbusiness, indie hackers"


def _evidence_summary(evidence: list[dict[str, Any]]) -> str:
    """Summarize evidence supporting this wedge."""

    if not evidence:
        return "Single source, moderate confidence"

    sources = [e.get("source", "") for e in evidence]
    return f"{len(evidence)} sources: {', '.join(sources[:3])}"


def _why_narrow(user: str, failure: str) -> str:
    """Explain why this is narrow."""

    return f"Specific user: {user}. Specific failure: {failure}. One workflow, one platform."


def _why_money(monetization_fit: float, consequence: str) -> str:
    """Explain monetization potential."""

    if monetization_fit >= 0.7:
        return "High frequency, clear error cost justifies $9/month subscription"
    elif monetization_fit >= 0.5:
        return "Moderate frequency, error cost creates one-time purchase opportunity"
    else:
        return "Low monetization fit - consider alternate wedge"


def generate_builder_outputs(db, run_id: str | None = None) -> list[BuilderCard]:
    """Generate builder output cards for all build-ready wedges."""

    cards = []

    # Get all build-ready/launched opportunities via selection_status
    conn = db._get_connection()
    rows = conn.execute("""
        SELECT o.id, o.cluster_id, o.title, o.selection_status, o.problem_truth_score, o.revenue_readiness_score, o.composite_score, c.label as cluster_label
        FROM opportunities o
        LEFT JOIN opportunity_clusters c ON o.cluster_id = c.id
        WHERE o.selection_status IN ('build_ready', 'launched', 'prototype_candidate', 'prototype_ready')
        ORDER BY o.problem_truth_score DESC, o.revenue_readiness_score DESC
    """).fetchall()

    for row in rows:
        opp_id = row[0]
        cluster_id = row[1]
        title = row[2]
        selection_status = row[3]
        pts = row[4] or 0.0
        rrs = row[5] or 0.0
        composite = row[6] or 0.0
        cluster_label = row[7] or ""

        wedge_data = {
            "title": title,
            "cluster_label": cluster_label,
            "problem_truth": pts,
            "revenue_readiness": rrs,
            "selection_status": selection_status,
        }

        # Get evidence for this opportunity
        evidence = _get_opportunity_evidence(db, opp_id)

        card = generate_builder_card(opp_id, wedge_data, evidence)
        cards.append(card)

    return cards


def _get_opportunity_evidence(db, opportunity_id: int) -> list[dict[str, Any]]:
    """Get supporting evidence for an opportunity."""
    opportunity = db.get_opportunity(opportunity_id)
    if not opportunity:
        return []

    conn = db._get_connection()
    member_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cluster_members)").fetchall()}
    atom_column = "problem_atom_id" if "problem_atom_id" in member_columns else "atom_id"
    rows = conn.execute(
        f"""
        SELECT DISTINCT f.source, f.outcome_summary, f.source_url, f.entrepreneur
        FROM cluster_members cm
        JOIN problem_atoms pa ON pa.id = cm.{atom_column}
        JOIN findings f ON f.id = pa.finding_id
        WHERE cm.cluster_id = ?
        ORDER BY pa.id ASC
        """,
        (opportunity.cluster_id,),
    ).fetchall()

    if not rows:
        # Fallback: get evidence directly from the cluster's atoms
        cluster = db.get_cluster(opportunity.cluster_id)
        if cluster:
            rows = conn.execute(
                """
                SELECT DISTINCT f.source, f.outcome_summary, f.source_url, f.entrepreneur
                FROM problem_atoms pa
                JOIN cluster_members cm ON cm.problem_atom_id = pa.id
                JOIN findings f ON f.id = pa.finding_id
                WHERE cm.cluster_id = ?
                ORDER BY pa.id ASC
                """,
                (cluster.id,),
            ).fetchall()

    return [
        {
            "source": row["source"],
            "outcome_summary": row["outcome_summary"],
            "source_url": row["source_url"],
            "entrepreneur": row["entrepreneur"],
        }
        for row in rows
    ]


def save_builder_cards(cards: list[BuilderCard], output_dir: Path) -> None:
    """Save builder cards to JSON output."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual cards
    for card in cards:
        filename = f"wedge_{card.wedge_id}_builder_card.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(asdict(card), f, indent=2)

    # Save summary index
    index = {
        "total_cards": len(cards),
        "build_now": len([c for c in cards if c.builder_verdict == "build_now"]),
        "backup": len([c for c in cards if c.builder_verdict == "backup_candidate"]),
        "research": len([c for c in cards if c.builder_verdict == "research_more"]),
        "cards": [
            {
                "wedge_id": c.wedge_id,
                "title": c.wedge_title,
                "verdict": c.builder_verdict,
                "priority": c.builder_priority,
            }
            for c in cards
        ],
    }

    with open(output_dir / "builder_cards_index.json", "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Saved {len(cards)} builder cards to {output_dir}")


# ---------------------------------------------------------------------------
# WedgeEvaluator — LLM-enhanced wedge evaluation gate
# ---------------------------------------------------------------------------

WEDGE_EVAL_SYSTEM = """\
You are a wedge evaluation engine for a product research pipeline.
Given an opportunity's evidence, evaluate whether it constitutes a
MONETIZABLE NARROW SOFTWARE WEDGE — not a broad problem area.

A monetizable wedge MUST satisfy ALL of:
1. NARROW: One specific user, one specific workflow, one specific failure mode.
   NOT "spreadsheet errors" — rather "Shopify store owners importing vendor
   CSVs where a single bad row silently corrupts inventory counts".
2. SOFTWARE-FIRST: The solution is naturally a plugin, add-in, or microSaaS
   — not a service, consultancy, or training program.
3. RECURRING TRIGGER: The user hits this problem weekly or monthly, creating
   subscription logic ($9-99/month).
4. ERROR COST: The cost of the failure (time, money, trust) justifies paying
   for prevention.
5. TRUST-RIGHT: The solution is a detector/validator (low trust) rather than
   an auto-corrector (high trust), unless the user explicitly delegates.

Output ONLY valid JSON with this structure:
{
  "software_fit": 0.0-1.0,
  "monetization_fit": 0.0-1.0,
  "is_narrow": true/false,
  "trust_risk": "low"/"medium"/"high",
  "verdict": "build_now"/"backup_candidate"/"research_more"/"reject",
  "narrowness_reason": "Why this IS or IS NOT narrow enough",
  "software_fit_reason": "Why software is or isn't the natural solution",
  "monetization_reason": "Why this will or won't make money",
  "suggested_mvp": ["Feature 1", "Feature 2"],
  "first_paid_offer": "Description of first paid version",
  "pricing_hypothesis": "$X/month because...",
  "first_customer": "Who would pay first",
  "first_channel": "Where to find them"
}"""

WEDGE_EVAL_USER = """\
## Opportunity: {title}
Selection Status: {selection_status}
Composite Score: {composite_score}
Problem Plausibility: {problem_plausibility}
Revenue Readiness: {revenue_readiness}
Cost of Inaction: {cost_of_inaction}
Workaround Density: {workaround_density}
Buildability: {buildability}

## Evidence
{evidence_text}

## Current Builder Card (heuristic evaluation)
Software Fit: {heuristic_software_fit}
Monetization Fit: {heuristic_monetization_fit}
Trust Risk: {heuristic_trust_risk}
Product Shape: {heuristic_product_shape}

Evaluate this opportunity as a monetizable narrow software wedge."""


@dataclass
class WedgeEvaluation:
    """Result of wedge evaluation — LLM-enhanced or heuristic fallback."""
    opportunity_id: int
    software_fit: float
    monetization_fit: float
    is_narrow: bool
    trust_risk: str
    verdict: str  # build_now, backup_candidate, research_more, reject
    # Reasons
    narrowness_reason: str = ""
    software_fit_reason: str = ""
    monetization_reason: str = ""
    # Builder details
    suggested_mvp: list[str] = field(default_factory=list)
    first_paid_offer: str = ""
    pricing_hypothesis: str = ""
    first_customer: str = ""
    first_channel: str = ""
    # Meta
    evaluated_by: str = "heuristic"  # heuristic | llm | llm_fallback
    raw_llm_response: str = ""

    # Gate thresholds
    SOFTWARE_FIT_FLOOR = 0.5
    MONETIZATION_FIT_FLOOR = 0.3
    TRUST_RISK_BLOCK = "high"

    @property
    def passes_wedge_gate(self) -> bool:
        """Whether this evaluation passes the wedge gate criteria."""
        if self.trust_risk == self.TRUST_RISK_BLOCK:
            return False
        if self.software_fit < self.SOFTWARE_FIT_FLOOR:
            return False
        if self.monetization_fit < self.MONETIZATION_FIT_FLOOR:
            return False
        if not self.is_narrow:
            return False
        return True

    def gate_failure_reasons(self) -> list[str]:
        """Return human-readable reasons the wedge gate failed."""
        reasons = []
        if self.trust_risk == self.TRUST_RISK_BLOCK:
            reasons.append(f"trust_risk={self.trust_risk} (requires low/medium)")
        if self.software_fit < self.SOFTWARE_FIT_FLOOR:
            reasons.append(f"software_fit={self.software_fit:.2f} < {self.SOFTWARE_FIT_FLOOR}")
        if self.monetization_fit < self.MONETIZATION_FIT_FLOOR:
            reasons.append(f"monetization_fit={self.monetization_fit:.2f} < {self.MONETIZATION_FIT_FLOOR}")
        if not self.is_narrow:
            reasons.append(f"not narrow enough: {self.narrowness_reason}")
        return reasons


class WedgeEvaluator:
    """Evaluates opportunities as monetizable narrow software wedges.

    Uses LLM for deeper analysis when available, falls back to
    heuristic evaluation from builder_output.py.
    """

    def __init__(self, db: Any, config: dict[str, Any]) -> None:
        self.db = db
        self.config = config
        self.llm_client: LLMClient | None = None
        llm_config = config.get("llm", {})
        if llm_config.get("provider") or llm_config.get("model"):
            try:
                from src.llm_discovery_expander import LLMClient
                self.llm_client = LLMClient(config)
            except Exception as e:
                logger.warning(f"Could not create LLMClient for wedge eval: {e}")

    async def evaluate(self, opportunity_id: int) -> WedgeEvaluation:
        """Evaluate an opportunity as a wedge. Tries LLM first, falls back to heuristic."""
        # Always run heuristic first as baseline
        heuristic_eval = self._heuristic_evaluate(opportunity_id)

        # Try LLM evaluation
        if self.llm_client:
            llm_eval = await self._llm_evaluate(opportunity_id, heuristic_eval)
            if llm_eval is not None:
                return llm_eval

        return heuristic_eval

    def evaluate_sync(self, opportunity_id: int) -> WedgeEvaluation:
        """Synchronous evaluation. Tries LLM sync, falls back to heuristic."""
        heuristic_eval = self._heuristic_evaluate(opportunity_id)

        if self.llm_client:
            llm_eval = self._llm_evaluate_sync(opportunity_id, heuristic_eval)
            if llm_eval is not None:
                return llm_eval

        return heuristic_eval

    def _heuristic_evaluate(self, opportunity_id: int) -> WedgeEvaluation:
        """Run the existing heuristic evaluation from builder_output.py."""
        opportunity = self.db.get_opportunity(opportunity_id)
        if opportunity is None:
            return WedgeEvaluation(
                opportunity_id=opportunity_id,
                software_fit=0.0,
                monetization_fit=0.0,
                is_narrow=False,
                trust_risk="high",
                verdict="reject",
                narrowness_reason="Opportunity not found",
            )

        # Generate builder card using existing heuristics
        evidence = _get_opportunity_evidence(self.db, opportunity_id)
        wedge_data = {
            "title": opportunity.title or "",
            "cluster_label": "",
            "problem_truth": getattr(opportunity, "problem_truth_score", 0) or 0,
            "revenue_readiness": getattr(opportunity, "revenue_readiness_score", 0) or 0,
            "selection_status": getattr(opportunity, "selection_status", "") or "",
        }
        card = generate_builder_card(opportunity_id, wedge_data, evidence)

        # Enhance heuristic scores with opportunity-level metrics
        software_fit = card.software_fit_score
        monetization_fit = card.monetization_fit_score

        # Boost monetization_fit from opportunity-level signals
        cost_of_inaction = getattr(opportunity, "cost_of_inaction", 0) or 0
        frequency = getattr(opportunity, "frequency_score", 0) or 0
        workaround_density = getattr(opportunity, "workaround_density", 0) or 0
        revenue_readiness = getattr(opportunity, "revenue_readiness_score", 0) or 0

        # If opportunity has high cost_of_inaction and frequency, it has monetization potential
        if monetization_fit < 0.3:
            opportunity_monetization = (cost_of_inaction * 0.3 + frequency * 0.3 + workaround_density * 0.2 + revenue_readiness * 0.2)
            monetization_fit = max(monetization_fit, opportunity_monetization)

        # Boost software_fit from buildability
        buildability = getattr(opportunity, "buildability", 0) or 0
        if software_fit < 0.5 and buildability >= 0.6:
            software_fit = max(software_fit, buildability * 0.7)

        return WedgeEvaluation(
            opportunity_id=opportunity_id,
            software_fit=software_fit,
            monetization_fit=monetization_fit,
            is_narrow=_is_narrow_wedge(card.exact_user, card.exact_workflow, card.exact_failure, card.product_shape),
            trust_risk=card.trust_risk,
            verdict=card.builder_verdict,
            narrowness_reason=_why_narrow(card.exact_user, card.exact_failure),
            software_fit_reason=card.why_this_shape_fits,
            monetization_reason=_why_money(monetization_fit, card.exact_consequence),
            suggested_mvp=card.mvp_in_scope,
            first_paid_offer=card.first_paid_offer,
            pricing_hypothesis=card.pricing_hypothesis,
            first_customer=card.first_customer,
            first_channel=card.first_channel,
            evaluated_by="heuristic",
        )

    def _build_evidence_text(self, opportunity_id: int) -> str:
        """Build evidence text for the LLM prompt."""
        evidence = _get_opportunity_evidence(self.db, opportunity_id)
        if not evidence:
            return "No evidence available."

        lines = []
        for i, e in enumerate(evidence[:5], 1):
            source = e.get("source", "unknown")
            summary = e.get("outcome_summary", "")[:200]
            lines.append(f"  {i}. [{source}] {summary}")
        return "\n".join(lines)

    async def _llm_evaluate(
        self, opportunity_id: int, heuristic: WedgeEvaluation,
    ) -> WedgeEvaluation | None:
        """Run LLM-based wedge evaluation (async)."""
        opportunity = self.db.get_opportunity(opportunity_id)
        if opportunity is None:
            return None

        system_prompt = WEDGE_EVAL_SYSTEM
        user_prompt = WEDGE_EVAL_USER.format(
            title=(opportunity.title or "")[:200],
            selection_status=getattr(opportunity, "selection_status", ""),
            composite_score=f"{getattr(opportunity, 'composite_score', 0):.3f}",
            problem_plausibility=f"{getattr(opportunity, 'problem_plausibility', 0):.3f}",
            revenue_readiness=f"{getattr(opportunity, 'revenue_readiness_score', 0):.3f}",
            cost_of_inaction=f"{getattr(opportunity, 'cost_of_inaction', 0):.3f}",
            workaround_density=f"{getattr(opportunity, 'workaround_density', 0):.3f}",
            buildability=f"{getattr(opportunity, 'buildability', 0):.3f}",
            evidence_text=self._build_evidence_text(opportunity_id),
            heuristic_software_fit=f"{heuristic.software_fit:.2f}",
            heuristic_monetization_fit=f"{heuristic.monetization_fit:.2f}",
            heuristic_trust_risk=heuristic.trust_risk,
            heuristic_product_shape="",
        )

        raw = await self.llm_client.agenerate(system_prompt, user_prompt)
        if not raw:
            logger.info("LLM async wedge eval failed, trying sync")
            raw = self.llm_client.generate(system_prompt, user_prompt)

        if not raw:
            return None

        return self._parse_llm_evaluation(opportunity_id, raw, "llm")

    def _llm_evaluate_sync(
        self, opportunity_id: int, heuristic: WedgeEvaluation,
    ) -> WedgeEvaluation | None:
        """Run LLM-based wedge evaluation (sync)."""
        opportunity = self.db.get_opportunity(opportunity_id)
        if opportunity is None:
            return None

        system_prompt = WEDGE_EVAL_SYSTEM
        user_prompt = WEDGE_EVAL_USER.format(
            title=(opportunity.title or "")[:200],
            selection_status=getattr(opportunity, "selection_status", ""),
            composite_score=f"{getattr(opportunity, 'composite_score', 0):.3f}",
            problem_plausibility=f"{getattr(opportunity, 'problem_plausibility', 0):.3f}",
            revenue_readiness=f"{getattr(opportunity, 'revenue_readiness_score', 0):.3f}",
            cost_of_inaction=f"{getattr(opportunity, 'cost_of_inaction', 0):.3f}",
            workaround_density=f"{getattr(opportunity, 'workaround_density', 0):.3f}",
            buildability=f"{getattr(opportunity, 'buildability', 0):.3f}",
            evidence_text=self._build_evidence_text(opportunity_id),
            heuristic_software_fit=f"{heuristic.software_fit:.2f}",
            heuristic_monetization_fit=f"{heuristic.monetization_fit:.2f}",
            heuristic_trust_risk=heuristic.trust_risk,
            heuristic_product_shape="",
        )

        raw = self.llm_client.generate(system_prompt, user_prompt)
        if not raw:
            return None

        return self._parse_llm_evaluation(opportunity_id, raw, "llm")

    def _parse_llm_evaluation(
        self, opportunity_id: int, raw: str, evaluated_by: str,
    ) -> WedgeEvaluation | None:
        """Parse LLM response into a WedgeEvaluation."""
        data = _extract_json(raw)
        if not data:
            logger.warning(f"Could not parse LLM wedge evaluation for opp {opportunity_id}")
            return None

        # Validate and clamp scores
        software_fit = max(0.0, min(1.0, float(data.get("software_fit", 0))))
        monetization_fit = max(0.0, min(1.0, float(data.get("monetization_fit", 0))))
        is_narrow = bool(data.get("is_narrow", False))
        trust_risk = data.get("trust_risk", "medium")
        if trust_risk not in ("low", "medium", "high"):
            trust_risk = "medium"

        verdict = data.get("verdict", "research_more")
        if verdict not in ("build_now", "backup_candidate", "research_more", "reject"):
            verdict = "research_more"

        # Override verdict based on gate criteria (LLM verdict is advisory)
        eval_obj = WedgeEvaluation(
            opportunity_id=opportunity_id,
            software_fit=software_fit,
            monetization_fit=monetization_fit,
            is_narrow=is_narrow,
            trust_risk=trust_risk,
            verdict=verdict,
            narrowness_reason=str(data.get("narrowness_reason", "")),
            software_fit_reason=str(data.get("software_fit_reason", "")),
            monetization_reason=str(data.get("monetization_reason", "")),
            suggested_mvp=data.get("suggested_mvp", []),
            first_paid_offer=str(data.get("first_paid_offer", "")),
            pricing_hypothesis=str(data.get("pricing_hypothesis", "")),
            first_customer=str(data.get("first_customer", "")),
            first_channel=str(data.get("first_channel", "")),
            evaluated_by=evaluated_by,
            raw_llm_response=raw[:500],
        )

        # Recalculate verdict from gate criteria
        if eval_obj.passes_wedge_gate:
            if software_fit >= 0.7 and monetization_fit >= 0.5:
                eval_obj.verdict = "build_now"
            else:
                eval_obj.verdict = "backup_candidate"
        elif not is_narrow or trust_risk == "high":
            eval_obj.verdict = "reject"
        else:
            eval_obj.verdict = "research_more"

        return eval_obj


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract JSON from an LLM response — shared with llm_discovery_expander."""
    # Import from the expander module to avoid duplication
    from src.llm_discovery_expander import _extract_json as _expander_extract
    return _expander_extract(raw)


def get_primary_wedge(cards: list[BuilderCard]) -> BuilderCard | None:
    """Get the primary (build_now) wedge."""

    for card in cards:
        if card.builder_verdict == "build_now" and card.builder_priority == "primary":
            return card

    # Fall back to first build_now
    for card in cards:
        if card.builder_verdict == "build_now":
            return card

    return None


def get_backup_wedge(cards: list[BuilderCard]) -> BuilderCard | None:
    """Get the backup candidate wedge."""

    for card in cards:
        if card.builder_verdict == "backup_candidate":
            return card

    return None
