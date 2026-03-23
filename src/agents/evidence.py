"""Evidence enrichment agent for recurrence, competitor, and falsification context."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from agents.base import AgentStatus, BaseAgent
from database import CorroborationRecord, Database, EvidenceLedgerEntry, MarketEnrichment
from messaging import MessageQueue, MessageType
from research_tools import ResearchToolkit

logger = logging.getLogger(__name__)

SOURCE_FAMILY_ALIASES = {
    "reddit": "reddit",
    "forum": "reddit",
    "wordpress-review": "wordpress_review",
    "shopify-review": "shopify_review",
    "review": "review",
    "github": "github",
    "github_issue": "github",
    "stackoverflow": "stackoverflow",
    "etsy": "etsy",
    "forum_fallback": "web",
    "web": "web",
}

SOURCE_FAMILY_GROUPS = {
    "reddit": "reddit",
    "wordpress_review": "review",
    "shopify_review": "review",
    "review": "review",
    "github": "github",
    "stackoverflow": "github",
    "etsy": "forum",
    "web": "web",
}

CORE_CORROBORATION_FAMILIES = {"reddit", "wordpress_review", "shopify_review", "github"}

GENERALIZABLE_WORKFLOW_TERMS = {
    "backup",
    "restore",
    "recovery",
    "sync",
    "manual",
    "compliance",
    "onboarding",
    "template",
    "shipping",
    "device",
    "spreadsheet",
    "pricing",
    "review",
    "email",
}

BACKUP_RESTORE_TERMS = {
    "backup",
    "restore",
    "recovery",
    "snapshot",
    "rollback",
    "disaster recovery",
    "cutover",
    "failover",
}

BACKUP_RESTORE_RISK_TERMS = {
    "unreachable",
    "data loss",
    "corrupt",
    "corrupted",
    "overwrite",
    "ransomware",
    "downtime",
    "blocked",
    "bandwidth",
    "storage",
    "duplicate backup",
    "missed backup",
}

BACKUP_RESTORE_BUYER_TERMS = {
    "operator",
    "operations",
    "platform",
    "infra",
    "infrastructure",
    "sysadmin",
    "admin",
    "it",
    "developer",
    "devops",
    "sre",
}

BACKUP_RESTORE_VENDOR_TERMS = {
    "veeam",
    "acronis",
    "rubrik",
    "commvault",
    "veritas",
    "backblaze",
    "druva",
    "datto",
    "backup",
    "restore",
    "recovery",
}

BACKUP_RESTORE_WEDGE_MIN_FIT = 0.45
BACKUP_RESTORE_WEDGE_RULE_VERSION = "backup_restore_v2"
BACKUP_RESTORE_EXCLUSION_TERMS = {
    "pricing",
    "profitability",
    "p&l",
    "shipping",
    "checkout",
    "reviews",
    "review",
    "reputation",
    "compliance",
    "gdpr",
    "hipaa",
    "soc 2",
    "email",
    "analytics",
    "conversion",
    "cart abandonment",
}

NOISY_SIGNATURE_TERMS = {
    "keep",
    "reliable",
    "workflow",
    "workflows",
    "teams",
    "team",
    "issue",
    "issues",
    "problem",
    "problems",
    "fails",
    "failure",
    "operator",
    "operations",
    "lead",
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _infer_source_family(source_name: str = "", source_url: str = "", source_type: str = "") -> str:
    haystack = _normalized(f"{source_name} {source_url} {source_type}")
    domain = urlparse(source_url).netloc.lower()

    if "apps.shopify.com" in domain or "shopify-review" in haystack:
        return "shopify_review"
    if "wordpress.org" in domain or "wordpress-review" in haystack:
        return "wordpress_review"
    if "github.com" in domain or "github" in haystack:
        return "github"
    if "reddit.com" in domain or "reddit" in haystack:
        return "reddit"
    if "stackoverflow.com" in domain or "stackoverflow" in haystack:
        return "stackoverflow"
    if "etsy.com" in domain or "etsy" in haystack:
        return "etsy"

    source_type_alias = SOURCE_FAMILY_ALIASES.get(source_type.strip().lower())
    if source_type_alias:
        return source_type_alias
    source_name_alias = SOURCE_FAMILY_ALIASES.get(source_name.strip().lower())
    if source_name_alias:
        return source_name_alias
    return "web"


def _source_family_group(source_family: str) -> str:
    return SOURCE_FAMILY_GROUPS.get(source_family, source_family or "web")


def _signature_terms(atom: Optional[Any]) -> list[str]:
    if atom is None:
        return []
    raw = " ".join(
        [
            getattr(atom, "job_to_be_done", "") or "",
            getattr(atom, "failure_mode", "") or "",
            getattr(atom, "trigger_event", "") or "",
            getattr(atom, "current_workaround", "") or "",
            getattr(atom, "cost_consequence_clues", "") or "",
        ]
    ).lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+", raw):
        if len(token) <= 3:
            continue
        if token in NOISY_SIGNATURE_TERMS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:10]


def _doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        title = doc.get("title", "")
        snippet = doc.get("snippet", "")
    else:
        title = getattr(doc, "title", "")
        snippet = getattr(doc, "snippet", "")
    return _normalized(f"{title} {snippet}")


def _doc_url(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("url", "") or "")
    return str(getattr(doc, "url", "") or "")


def _doc_source(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("source", "") or "")
    return str(getattr(doc, "source", "") or "")


def _doc_matches_signature(doc: Any, signature_terms: list[str]) -> bool:
    if not signature_terms:
        return False
    text = _doc_text(doc)
    hits = [term for term in signature_terms if term in text]
    if len(hits) >= 2:
        return True
    if len(hits) == 1 and any(term in text for term in GENERALIZABLE_WORKFLOW_TERMS):
        return True
    return False


def _generalizability_profile(finding: Any, atom: Optional[Any]) -> dict[str, Any]:
    evidence = finding.evidence if hasattr(finding, "evidence") else {}
    source_classification = evidence.get("source_classification", {}) or {}
    review_issue_type = str(source_classification.get("review_issue_type", "") or "").strip()
    github_issue_type = str(source_classification.get("github_issue_type", "") or "").strip()

    reasons: list[str] = []
    score = 0.18
    haystack = _normalized(
        " ".join(
            [
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "trigger_event", "") or "",
                getattr(atom, "current_workaround", "") or "",
                getattr(atom, "cost_consequence_clues", "") or "",
                getattr(atom, "frequency_clues", "") or "",
                getattr(finding, "product_built", "") or "",
                getattr(finding, "outcome_summary", "") or "",
            ]
        )
    )

    if review_issue_type == "reusable_workflow_pain" or github_issue_type == "reusable_workflow_pain":
        score += 0.34
        reasons.append("source_specific_reusable_workflow")
    if review_issue_type == "product_specific_issue" or github_issue_type == "product_specific_issue":
        score -= 0.28
        reasons.append("source_specific_product_issue")
    if any(term in haystack for term in GENERALIZABLE_WORKFLOW_TERMS):
        score += 0.18
        reasons.append("generalizable_workflow_terms")
    if getattr(atom, "current_workaround", ""):
        score += 0.12
        reasons.append("manual_fallback_present")
    if getattr(atom, "cost_consequence_clues", "") or getattr(atom, "frequency_clues", ""):
        score += 0.1
        reasons.append("recurring_cost_or_frequency")

    generalizability_score = _clamp(score)
    generalizability_class = "reusable_workflow_pain" if generalizability_score >= 0.42 else "product_specific_issue"
    generalizability_penalty = 0.0
    if generalizability_class == "product_specific_issue":
        generalizability_penalty = round(min(0.16, 0.06 + max(0.0, 0.42 - generalizability_score) * 0.25), 4)
    return {
        "generalizability_class": generalizability_class,
        "generalizability_score": round(generalizability_score, 4),
        "generalizability_reasons": reasons,
        "generalizability_penalty": generalizability_penalty,
    }


def _empty_wedge_profile(
    *,
    fit_score: float = 0.0,
    activation_reasons: list[str] | None = None,
    block_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "wedge_name": "",
        "fit_score": round(fit_score, 4),
        "operational_risk_score": 0.0,
        "downtime_cost_score": 0.0,
        "buyer_ownership_score": 0.0,
        "alternative_maturity_score": 0.0,
        "value_lift": 0.0,
        "activation_reasons": activation_reasons or [],
        "block_reasons": block_reasons or [],
        "rule_version": BACKUP_RESTORE_WEDGE_RULE_VERSION,
        "active": False,
    }


def _backup_restore_wedge_candidate(
    atom: Optional[Any],
    audience_hint: str,
    competitor_docs: list[dict[str, Any]] | list[Any],
) -> dict[str, Any]:
    if atom is None:
        return {
            "candidate": False,
            "fit_score": 0.0,
            "activation_reasons": [],
            "block_reasons": ["missing_atom"],
            "summary_text": "",
            "core_text": "",
            "competitor_text": "",
            "fit_hits": 0,
            "risk_hits": 0,
            "buyer_hits": 0,
            "vendor_hits": 0,
        }

    summary_text = _normalized(
        " ".join(
            [
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "trigger_event", "") or "",
                getattr(atom, "current_workaround", "") or "",
                getattr(atom, "cost_consequence_clues", "") or "",
                getattr(atom, "why_now_clues", "") or "",
                getattr(atom, "segment", "") or "",
                getattr(atom, "user_role", "") or "",
                audience_hint or "",
            ]
        )
    )
    core_text = _normalized(
        " ".join(
            [
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "trigger_event", "") or "",
            ]
        )
    )
    competitor_text = _normalized(" ".join(_doc_text(doc) for doc in competitor_docs[:8]))

    activation_reasons: list[str] = []
    block_reasons: list[str] = []
    fit_hits = sum(1 for term in BACKUP_RESTORE_TERMS if term in summary_text)
    risk_hits = sum(1 for term in BACKUP_RESTORE_RISK_TERMS if term in summary_text)
    buyer_hits = sum(1 for term in BACKUP_RESTORE_BUYER_TERMS if term in summary_text)
    vendor_hits = sum(1 for term in BACKUP_RESTORE_VENDOR_TERMS if term in competitor_text)
    exclusion_hits = sorted(term for term in BACKUP_RESTORE_EXCLUSION_TERMS if term in summary_text)

    fit_score = _clamp(0.12 + min(fit_hits, 5) * 0.16)
    if "keep backup restore and recovery reliable" in summary_text:
        fit_score = max(fit_score, 0.76)
        activation_reasons.append("explicit_backup_restore_job")
    if fit_hits:
        activation_reasons.append("backup_restore_terms_present")
    if risk_hits:
        activation_reasons.append("operational_risk_context")
    if buyer_hits:
        activation_reasons.append("operator_owner_present")
    if vendor_hits:
        activation_reasons.append("backup_restore_market_context")
    if not any(term in core_text for term in BACKUP_RESTORE_TERMS):
        block_reasons.append("missing_core_backup_restore_signal")
    if exclusion_hits and not any(term in core_text for term in BACKUP_RESTORE_TERMS):
        block_reasons.append("excluded_non_backup_domain_signal")
    if fit_score < BACKUP_RESTORE_WEDGE_MIN_FIT:
        block_reasons.append("fit_below_activation_threshold")
    if not activation_reasons:
        block_reasons.append("no_backup_restore_candidate_signal")

    return {
        "candidate": not block_reasons or (fit_hits > 0 and any(term in core_text for term in BACKUP_RESTORE_TERMS)),
        "fit_score": round(fit_score, 4),
        "activation_reasons": activation_reasons,
        "block_reasons": block_reasons,
        "summary_text": summary_text,
        "core_text": core_text,
        "competitor_text": competitor_text,
        "fit_hits": fit_hits,
        "risk_hits": risk_hits,
        "buyer_hits": buyer_hits,
        "vendor_hits": vendor_hits,
    }


def _validate_backup_restore_wedge(
    candidate: dict[str, Any],
    source_classification: dict[str, Any],
) -> dict[str, Any]:
    fit_score = float(candidate.get("fit_score", 0.0) or 0.0)
    activation_reasons = list(candidate.get("activation_reasons", []) or [])
    block_reasons = list(candidate.get("block_reasons", []) or [])
    summary_text = str(candidate.get("summary_text", "") or "")
    core_text = str(candidate.get("core_text", "") or "")
    competitor_text = str(candidate.get("competitor_text", "") or "")

    if source_classification.get("review_issue_type") == "product_specific_issue":
        block_reasons.append("review_product_specific_issue")
    if source_classification.get("github_issue_type") == "product_specific_issue":
        block_reasons.append("github_product_specific_issue")
    if "manual" in summary_text and not any(term in summary_text for term in BACKUP_RESTORE_TERMS):
        block_reasons.append("manual_work_without_backup_restore_context")
    if any(term in summary_text for term in BACKUP_RESTORE_EXCLUSION_TERMS) and not any(
        term in core_text for term in BACKUP_RESTORE_TERMS
    ):
        block_reasons.append("semantic_exclusion_mismatch")
    if block_reasons:
        return _empty_wedge_profile(
            fit_score=fit_score,
            activation_reasons=activation_reasons,
            block_reasons=list(dict.fromkeys(block_reasons)),
        )

    risk_hits = int(candidate.get("risk_hits", 0) or 0)
    buyer_hits = int(candidate.get("buyer_hits", 0) or 0)
    vendor_hits = int(candidate.get("vendor_hits", 0) or 0)

    operational_risk_score = _clamp(
        0.08
        + min(risk_hits, 5) * 0.13
        + (0.18 if any(term in summary_text for term in ["blocked", "unreachable", "cutover"]) else 0.0)
        + (0.12 if any(term in summary_text for term in ["data loss", "corrupt", "overwrite"]) else 0.0)
    )
    downtime_cost_score = _clamp(
        0.06
        + (0.16 if "time" in summary_text or "hours" in summary_text else 0.0)
        + (0.14 if any(term in summary_text for term in ["bandwidth", "storage", "downtime"]) else 0.0)
        + (0.14 if any(term in summary_text for term in ["every time", "recurring", "daily"]) else 0.0)
        + (0.1 if any(term in summary_text for term in ["cutover", "restore", "backup"]) else 0.0)
    )
    buyer_ownership_score = _clamp(
        0.08
        + min(buyer_hits, 3) * 0.14
        + (0.16 if any(term in summary_text for term in ["sysadmin", "platform", "infrastructure", "devops", "sre"]) else 0.0)
        + (0.1 if any(term in summary_text for term in ["operator", "operations", "admin", "it"]) else 0.0)
    )
    alternative_maturity_score = _clamp(
        0.04
        + min(vendor_hits, 4) * 0.12
        + (0.05 if competitor_text else 0.0)
    )
    value_lift = _clamp(
        fit_score * 0.34
        + operational_risk_score * 0.24
        + downtime_cost_score * 0.18
        + buyer_ownership_score * 0.12
        + alternative_maturity_score * 0.12
    )
    return {
        "wedge_name": "backup_restore_reliability",
        "fit_score": round(fit_score, 4),
        "operational_risk_score": round(operational_risk_score, 4),
        "downtime_cost_score": round(downtime_cost_score, 4),
        "buyer_ownership_score": round(buyer_ownership_score, 4),
        "alternative_maturity_score": round(alternative_maturity_score, 4),
        "value_lift": round(value_lift, 4),
        "activation_reasons": activation_reasons,
        "block_reasons": [],
        "rule_version": BACKUP_RESTORE_WEDGE_RULE_VERSION,
        "active": True,
    }


def _runtime_sanity_check_backup_restore_wedge(
    profile: dict[str, Any],
    atom: Optional[Any],
) -> dict[str, Any]:
    if not profile.get("active"):
        return profile
    core_text = _normalized(
        " ".join(
            [
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "trigger_event", "") or "",
            ]
        )
    )
    block_reasons = list(profile.get("block_reasons", []) or [])
    if float(profile.get("fit_score", 0.0) or 0.0) < BACKUP_RESTORE_WEDGE_MIN_FIT:
        block_reasons.append("runtime_fit_below_threshold")
    if not any(term in core_text for term in BACKUP_RESTORE_TERMS):
        block_reasons.append("runtime_missing_core_backup_restore_signal")
    if any(term in core_text for term in BACKUP_RESTORE_EXCLUSION_TERMS):
        block_reasons.append("runtime_semantic_conflict")
    if block_reasons:
        return _empty_wedge_profile(
            fit_score=float(profile.get("fit_score", 0.0) or 0.0),
            activation_reasons=list(profile.get("activation_reasons", []) or []),
            block_reasons=list(dict.fromkeys(block_reasons)),
        )
    return profile


def _apply_backup_restore_value_lift(base_value: float, profile: dict[str, Any], metric: str) -> float:
    if not profile.get("active"):
        return base_value
    if metric == "cost_pressure":
        return min(1.0, base_value + profile["operational_risk_score"] * 0.18 + profile["downtime_cost_score"] * 0.16)
    if metric == "demand":
        return min(1.0, base_value + profile["value_lift"] * 0.1)
    if metric == "buyer_intent":
        return min(1.0, base_value + profile["buyer_ownership_score"] * 0.1 + profile["alternative_maturity_score"] * 0.08)
    if metric == "willingness_to_pay":
        return min(1.0, base_value + profile["value_lift"] * 0.12)
    return base_value


class EvidenceAgent(BaseAgent):
    """Collects recurrence and market evidence before final validation."""

    def __init__(
        self,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        config: Optional[Dict[str, Any]] = None,
        status_tracker: Optional[Any] = None,
    ):
        super().__init__("evidence", message_queue)
        self.db = db
        self.config = config or {}
        self.toolkit = ResearchToolkit(self.config)
        self.status_tracker = status_tracker
        self.max_concurrency = max(1, int(self.config.get("orchestration", {}).get("evidence_concurrency", 3)))
        configured_timeout = float(self.config.get("orchestration", {}).get("evidence_timeout_seconds", 25))
        minimum_timeout = (
            float(getattr(self.toolkit, "validation_recurrence_budget_seconds", 0.0) or 0.0)
            + float(getattr(self.toolkit, "validation_competitor_budget_seconds", 0.0) or 0.0)
            + 12.0
        )
        self.evidence_timeout_seconds = max(configured_timeout, minimum_timeout)
        self._inflight: set[asyncio.Task] = set()

    async def stop(self) -> None:
        for task in list(self._inflight):
            task.cancel()
        await super().stop()

    async def _run_loop(self) -> None:
        while self.status in (AgentStatus.RUNNING, AgentStatus.PAUSED):
            try:
                await self._pause_event.wait()

                if self.status == AgentStatus.STOPPED:
                    break

                self._inflight = {task for task in self._inflight if not task.done()}
                if len(self._inflight) >= self.max_concurrency:
                    done, _ = await asyncio.wait(self._inflight, return_when=asyncio.FIRST_COMPLETED)
                    self._inflight.difference_update(done)
                    continue

                message = await self._message_queue.get_for_agent(self.name)
                if message is None:
                    if self._inflight:
                        done, _ = await asyncio.wait(self._inflight, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                        self._inflight.difference_update(done)
                    else:
                        await asyncio.sleep(0.1)
                    continue

                task = asyncio.create_task(self._process_with_tracking(message))
                self._inflight.add(task)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._error_count += 1
                logger.exception("evidence loop error: %s", exc)
                if self._error_count >= self._max_errors:
                    self.status = AgentStatus.ERROR
                    break

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type != MessageType.FINDING:
            return {"processed": True, "ignored": message.msg_type.value}
        return await self._enrich_finding(message.payload)

    async def _process_with_tracking(self, message) -> Dict[str, Any]:
        self._processing_count += 1
        try:
            return await self.process(message)
        finally:
            self._processing_count = max(0, self._processing_count - 1)

    def busy_count(self) -> int:
        return int(self._processing_count + len({task for task in self._inflight if not task.done()}))

    def reddit_runtime_summary(self) -> dict[str, Any]:
        return self.toolkit.get_reddit_runtime_metrics()

    async def _enrich_finding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        finding_id = payload.get("finding_id")
        if not finding_id:
            return {"success": False, "error": "No finding_id provided"}

        finding = self.db.get_finding(finding_id)
        if finding is None:
            return {"success": False, "error": f"finding {finding_id} not found"}

        title = finding.product_built or "Untitled"
        summary = finding.outcome_summary or ""
        audience_hint = finding.entrepreneur or ""
        atoms = self.db.get_problem_atoms_by_finding(finding_id)
        anchor_atom = atoms[0] if atoms else None
        logger.info("evidence start finding=%s title=%s", finding_id, title[:80])
        if self.status_tracker:
            self.status_tracker.log(f"evidence_start finding={finding_id}")
        try:
            evidence_scores = await asyncio.wait_for(
                self.toolkit.validate_problem(
                    title=title,
                    summary=summary,
                    finding_kind=finding.finding_kind,
                    audience_hint=audience_hint,
                    atom=anchor_atom,
                ),
                timeout=self.evidence_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("evidence timeout finding=%s after %.1fs", finding_id, self.evidence_timeout_seconds)
            evidence_scores = self._timeout_evidence_scores(anchor_atom)

        corroboration = self._build_corroboration_record(finding, anchor_atom, evidence_scores)
        market_enrichment = self._build_market_enrichment(
            finding,
            evidence_scores,
            anchor_atom,
            audience_hint,
            corroboration,
        )
        self.db.upsert_corroboration(corroboration)
        self.db.upsert_market_enrichment(market_enrichment)

        self.db.insert_ledger_entry(
            EvidenceLedgerEntry(
                entity_type="finding",
                entity_id=finding_id,
                entry_kind="evidence_enrichment",
                stance="neutral",
                source_name="evidence",
                source_url="",
                quote_text="",
                summary="Recurrence and competitor evidence gathered before validation.",
                metadata_json=json.dumps(
                    {
                        "scores": {
                            "problem_score": evidence_scores["problem_score"],
                            "value_score": evidence_scores["value_score"],
                            "feasibility_score": evidence_scores["feasibility_score"],
                            "solution_gap_score": evidence_scores["solution_gap_score"],
                            "saturation_score": evidence_scores["saturation_score"],
                        },
                        "recurrence_docs": evidence_scores["evidence"].get("recurrence_docs", [])[:5],
                        "competitor_docs": evidence_scores["evidence"].get("competitor_docs", [])[:5],
                        "recurrence_queries": evidence_scores["evidence"].get("recurrence_queries", [])[:5],
                        "recurrence_state": evidence_scores["evidence"].get("recurrence_state", ""),
                        "recurrence_query_coverage": evidence_scores["evidence"].get("recurrence_query_coverage", 0.0),
                        "recurrence_results_by_source": evidence_scores["evidence"].get("recurrence_results_by_source", {}),
                        "matched_results_by_source": evidence_scores["evidence"].get("matched_results_by_source", {}),
                        "partial_results_by_source": evidence_scores["evidence"].get("partial_results_by_source", {}),
                        "family_confirmation_count": evidence_scores["evidence"].get("family_confirmation_count", 0),
                        "source_yield": evidence_scores["evidence"].get("source_yield", {}),
                        "reshaped_query_history": evidence_scores["evidence"].get("reshaped_query_history", []),
                        "candidate_meaningful": evidence_scores["evidence"].get("candidate_meaningful", {}),
                        "last_action": evidence_scores["evidence"].get("last_action", ""),
                        "last_transition_reason": evidence_scores["evidence"].get("last_transition_reason", ""),
                        "chosen_family": evidence_scores["evidence"].get("chosen_family", ""),
                        "expected_gain_class": evidence_scores["evidence"].get("expected_gain_class", ""),
                        "source_attempts_snapshot": evidence_scores["evidence"].get("source_attempts_snapshot", {}),
                        "skipped_families": evidence_scores["evidence"].get("skipped_families", {}),
                        "controller_actions": evidence_scores["evidence"].get("controller_actions", []),
                        "budget_snapshot": evidence_scores["evidence"].get("budget_snapshot", {}),
                        "fallback_strategy_used": evidence_scores["evidence"].get("fallback_strategy_used", ""),
                        "decomposed_atom_queries": evidence_scores["evidence"].get("decomposed_atom_queries", []),
                        "routing_override_reason": evidence_scores["evidence"].get("routing_override_reason", ""),
                        "cohort_query_pack_used": evidence_scores["evidence"].get("cohort_query_pack_used", False),
                        "cohort_query_pack_name": evidence_scores["evidence"].get("cohort_query_pack_name", ""),
                        "web_query_strategy_path": evidence_scores["evidence"].get("web_query_strategy_path", []),
                        "specialized_surface_targeting_used": evidence_scores["evidence"].get("specialized_surface_targeting_used", False),
                        "promotion_gap_class": evidence_scores["evidence"].get("promotion_gap_class", ""),
                        "near_miss_enrichment_action": evidence_scores["evidence"].get("near_miss_enrichment_action", ""),
                        "sufficiency_priority_reason": evidence_scores["evidence"].get("sufficiency_priority_reason", ""),
                        "value_enrichment_used": evidence_scores["evidence"].get("value_enrichment_used", False),
                        "value_enrichment_queries": evidence_scores["evidence"].get("value_enrichment_queries", []),
                        "value_enrichment_docs": evidence_scores["evidence"].get("value_enrichment_docs", []),
                        "query": evidence_scores["evidence"].get("query", ""),
                        "competitor_query": evidence_scores["evidence"].get("competitor_query", ""),
                        "corroboration": corroboration.evidence,
                        "market_enrichment": market_enrichment.evidence,
                    }
                ),
            )
        )

        if self.status_tracker:
            self.status_tracker.log(
                "evidence_ready "
                f"finding={finding_id} recurrence={round(evidence_scores['problem_score'], 3)} "
                f"state={evidence_scores['evidence'].get('recurrence_state', 'unknown')} "
                f"value={round(evidence_scores['value_score'], 3)}"
            )

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.EVIDENCE,
            payload={
                **payload,
                "finding_id": finding_id,
                "evidence_scores": evidence_scores,
                "corroboration": corroboration.evidence,
                "market_enrichment": market_enrichment.evidence,
            },
            priority=2,
        )
        return {"success": True, "finding_id": finding_id, "evidence_scores": evidence_scores}

    def _timeout_evidence_scores(self, atom: Optional[Any]) -> Dict[str, Any]:
        has_structure = bool(atom and (getattr(atom, "failure_mode", "") or getattr(atom, "current_workaround", "")))
        value_hint = bool(atom and getattr(atom, "cost_consequence_clues", ""))
        budget_profile = {}
        try:
            budget_profile = self.toolkit._recurrence_budget_profile(atom)
        except Exception:
            budget_profile = {}
        return {
            "problem_score": 0.18 if has_structure else 0.1,
            "value_score": 0.32 if value_hint else 0.24,
            "feasibility_score": 0.52,
            "solution_gap_score": 0.5,
            "saturation_score": 0.5,
            "evidence": {
                "query": "",
                "competitor_query": "",
                "recurrence_queries": [],
                "recurrence_docs": [],
                "competitor_docs": [],
                "recurrence_state": "timeout",
                "recurrence_query_coverage": 0.0,
                "recurrence_doc_count": 0,
                "recurrence_domain_count": 0,
                "recurrence_results_by_query": {},
                "recurrence_results_by_source": {
                    "web": 0,
                    "reddit": 0,
                    "github": 0,
                    "stackoverflow": 0,
                    "etsy": 0,
                    "forum_fallback": 0,
                },
                "matched_results_by_source": {},
                "partial_results_by_source": {},
                "family_confirmation_count": 0,
                "source_yield": {},
                "reshaped_query_history": [],
                "recurrence_gap_reason": "evidence_agent_timeout",
                "queries_considered": [],
                "queries_executed": [],
                "recurrence_budget_profile": budget_profile,
                "candidate_meaningful": self.toolkit._meaningful_candidate_snapshot(atom),
                "last_action": "STOP_FOR_BUDGET",
                "last_transition_reason": "evidence_agent_timeout",
                "chosen_family": "",
                "expected_gain_class": "low",
                "source_attempts_snapshot": {},
                "skipped_families": {},
                "controller_actions": [],
                "budget_snapshot": budget_profile,
                "fallback_strategy_used": "",
                "cohort_query_pack_used": False,
                "cohort_query_pack_name": "",
                "web_query_strategy_path": [],
                "specialized_surface_targeting_used": False,
                "recurrence_timeout": True,
                "competitor_timeout": False,
                "timeout": True,
            },
        }

    def _build_corroboration_record(
        self,
        finding: Any,
        atom: Optional[Any],
        evidence_scores: Dict[str, Any],
    ) -> CorroborationRecord:
        finding_id = int(finding.id)
        evidence = evidence_scores.get("evidence", {})
        recurrence_queries = evidence.get("recurrence_queries", [])
        query_set_hash = hashlib.sha1(json.dumps(recurrence_queries, sort_keys=True).encode("utf-8")).hexdigest()
        recurrence_score = float(evidence_scores.get("problem_score", 0.0) or 0.0)
        query_coverage = float(evidence.get("recurrence_query_coverage", 0.0) or 0.0)
        doc_count = int(evidence.get("recurrence_doc_count", 0) or 0)
        domain_count = int(evidence.get("recurrence_domain_count", 0) or 0)
        results_by_query = evidence.get("recurrence_results_by_query", {}) or {}
        results_by_source = evidence.get("recurrence_results_by_source", {}) or {}
        recurrence_docs = evidence.get("recurrence_docs", []) or []
        source_diversity = sum(1 for count in results_by_source.values() if count)
        query_breadth = sum(1 for count in results_by_query.values() if count)
        confirmation_depth_score = min(doc_count, 8) / 8.0
        domain_depth_score = min(domain_count, 5) / 5.0
        source_diversity_score = min(source_diversity, 4) / 4.0
        query_breadth_score = min(query_breadth, 4) / 4.0
        strongest_source_count = max(results_by_source.values(), default=0)
        source_concentration = (strongest_source_count / doc_count) if doc_count else 0.0
        origin_source_family = _infer_source_family(
            getattr(finding, "source", "") or "",
            getattr(finding, "source_url", "") or "",
            "",
        )
        origin_source_group = _source_family_group(origin_source_family)
        signature_terms = _signature_terms(atom)
        matched_doc_families: list[str] = []
        matched_doc_groups: list[str] = []
        matched_family_counts: dict[str, int] = {}
        cross_source_match_count = 0
        for doc in recurrence_docs:
            if not _doc_matches_signature(doc, signature_terms):
                continue
            cross_source_match_count += 1
            doc_family = _infer_source_family(_doc_source(doc), _doc_url(doc), "")
            matched_doc_families.append(doc_family)
            matched_doc_groups.append(_source_family_group(doc_family))
            matched_family_counts[doc_family] = matched_family_counts.get(doc_family, 0) + 1
        source_families = sorted({origin_source_family, *matched_doc_families})
        source_groups = sorted({origin_source_group, *matched_doc_groups})
        source_family_diversity = len(source_families)
        source_group_diversity = len(source_groups)
        source_family_diversity_score = min(source_family_diversity, 4) / 4.0
        source_group_diversity_score = min(source_group_diversity, 3) / 3.0
        core_source_families = sorted(family for family in source_families if family in CORE_CORROBORATION_FAMILIES)
        core_source_family_diversity = len(core_source_families)
        core_source_family_diversity_score = min(core_source_family_diversity, 4) / 4.0
        cross_source_match_score = 0.0
        if cross_source_match_count:
            cross_source_match_score = min(
                1.0,
                min(cross_source_match_count, 6) / 6.0 * 0.55
                + source_group_diversity_score * 0.3
                + source_family_diversity_score * 0.15,
            )
        generalizability = _generalizability_profile(finding, atom)

        single_source_penalty = 0.0
        cross_source_bonus = 0.0
        core_source_family_bonus = 0.0
        if doc_count >= 2 and source_diversity <= 1:
            single_source_penalty = min(0.12, 0.04 + doc_count * 0.012)
        if source_group_diversity >= 2 and cross_source_match_count >= 2:
            cross_source_bonus = min(0.12, 0.025 + source_group_diversity * 0.02 + cross_source_match_score * 0.04)
        if core_source_family_diversity >= 2 and cross_source_match_count >= 2:
            core_source_family_bonus = min(
                0.14,
                0.03 + core_source_family_diversity * 0.02 + cross_source_match_score * 0.05,
            )

        corroboration_score = min(
            1.0,
            max(
                0.0,
                recurrence_score * 0.34
                + query_coverage * 0.17
                + confirmation_depth_score * 0.18
                + domain_depth_score * 0.1
                + source_diversity_score * 0.12
                + query_breadth_score * 0.09
                + source_group_diversity_score * 0.08
                + cross_source_match_score * 0.08
                + core_source_family_diversity_score * 0.08
                + generalizability["generalizability_score"] * 0.05
                + cross_source_bonus
                + core_source_family_bonus
                - single_source_penalty
                - generalizability["generalizability_penalty"],
            ),
        )
        evidence_sufficiency = min(
            1.0,
            max(
                0.0,
                corroboration_score * 0.48
                + query_coverage * 0.16
                + confirmation_depth_score * 0.14
                + domain_depth_score * 0.08
                + source_diversity_score * 0.1
                + query_breadth_score * 0.08
                + source_group_diversity_score * 0.08
                + cross_source_match_score * 0.08
                + core_source_family_diversity_score * 0.08
                + cross_source_bonus * 0.7
                + core_source_family_bonus * 0.75
                - single_source_penalty * 0.7
                - generalizability["generalizability_penalty"] * 0.8,
            ),
        )
        corroboration_json = {
            "finding_id": finding_id,
            "run_id": self.db.active_run_id,
            "recurrence_state": evidence.get("recurrence_state", "weak"),
            "recurrence_score": recurrence_score,
            "corroboration_score": round(corroboration_score, 4),
            "evidence_sufficiency": round(evidence_sufficiency, 4),
            "query_coverage": query_coverage,
            "independent_confirmations": doc_count,
            "source_diversity": source_diversity,
            "source_family_diversity": source_family_diversity,
            "source_families": source_families,
            "source_family_match_counts": matched_family_counts,
            "source_family_diversity_score": round(source_family_diversity_score, 4),
            "core_source_families": core_source_families,
            "core_source_family_diversity": core_source_family_diversity,
            "core_source_family_diversity_score": round(core_source_family_diversity_score, 4),
            "source_group_diversity": source_group_diversity,
            "source_groups": source_groups,
            "source_group_diversity_score": round(source_group_diversity_score, 4),
            "origin_source_family": origin_source_family,
            "origin_source_group": origin_source_group,
            "cross_source_match_count": cross_source_match_count,
            "cross_source_match_score": round(cross_source_match_score, 4),
            "query_set_hash": query_set_hash,
            "results_by_source": results_by_source,
            "results_by_query": results_by_query,
            "recurrence_doc_count": doc_count,
            "recurrence_domain_count": domain_count,
            "confirmation_depth_score": round(confirmation_depth_score, 4),
            "domain_depth_score": round(domain_depth_score, 4),
            "source_diversity_score": round(source_diversity_score, 4),
            "query_breadth_score": round(query_breadth_score, 4),
            "source_concentration": round(source_concentration, 4),
            "cross_source_bonus": round(cross_source_bonus, 4),
            "core_source_family_bonus": round(core_source_family_bonus, 4),
            "single_source_penalty": round(single_source_penalty, 4),
            **generalizability,
        }
        return CorroborationRecord(
            finding_id=finding_id,
            recurrence_state=evidence.get("recurrence_state", "weak"),
            recurrence_score=round(recurrence_score, 4),
            corroboration_score=round(corroboration_score, 4),
            evidence_sufficiency=round(evidence_sufficiency, 4),
            query_coverage=round(query_coverage, 4),
            independent_confirmations=doc_count,
            source_diversity=source_diversity,
            query_set_hash=query_set_hash,
            evidence_json=json.dumps(corroboration_json),
            run_id=self.db.active_run_id,
        )

    def _build_market_enrichment(
        self,
        finding: Any,
        evidence_scores: Dict[str, Any],
        atom: Optional[Any],
        audience_hint: str,
        corroboration: CorroborationRecord,
    ) -> MarketEnrichment:
        finding_id = finding.id
        evidence = evidence_scores.get("evidence", {})
        competitor_docs = evidence.get("competitor_docs", [])
        competitor_domains = evidence.get("competitor_domains", [])
        pain_hits = int(evidence.get("pain_hits", 0) or 0)
        value_hits = int(evidence.get("value_hits", 0) or 0)
        review_metadata = (finding.evidence or {}).get("review_metadata", {}) or {}
        source_classification = (finding.evidence or {}).get("source_classification", {}) or {}
        review_issue_type = str(source_classification.get("review_issue_type", "") or "").strip()
        review_generalizability_score = float(source_classification.get("review_generalizability_score", 0.0) or 0.0)
        review_generalizability_reasons = list(source_classification.get("review_generalizability_reasons", []) or [])
        source_review_rating = float(review_metadata.get("review_rating", 0.0) or 0.0)
        aggregate_rating = float(review_metadata.get("aggregate_rating", 0.0) or 0.0)
        review_count = int(review_metadata.get("review_count", 0) or 0)
        active_installs = int(review_metadata.get("active_installs", 0) or 0)
        popularity_proxy = int(review_metadata.get("popularity_proxy", 0) or 0)
        pricing = str(review_metadata.get("pricing", "") or "").strip().lower()
        version = str(review_metadata.get("version", "") or "").strip()
        last_updated = str(review_metadata.get("last_updated", "") or "").strip()
        category = str(review_metadata.get("category", "") or "").strip()
        listing_url = str(review_metadata.get("listing_url", "") or "").strip()
        developer_name = str(review_metadata.get("developer_name", "") or "").strip()
        launched_at = str(review_metadata.get("launched_at", "") or "").strip()
        built_for_shopify = bool(review_metadata.get("built_for_shopify", False))
        corroboration_evidence = corroboration.evidence if corroboration else {}
        source_family_diversity = int(corroboration_evidence.get("source_family_diversity", 0) or 0)
        source_group_diversity = int(corroboration_evidence.get("source_group_diversity", 0) or 0)
        cross_source_match_score = float(corroboration_evidence.get("cross_source_match_score", 0.0) or 0.0)
        generalizability_class = str(corroboration_evidence.get("generalizability_class", "") or "").strip()
        generalizability_score = float(corroboration_evidence.get("generalizability_score", 0.0) or 0.0)
        generalizability_penalty = float(corroboration_evidence.get("generalizability_penalty", 0.0) or 0.0)
        wedge_candidate = _backup_restore_wedge_candidate(atom, audience_hint, competitor_docs)
        wedge_profile = _validate_backup_restore_wedge(wedge_candidate, source_classification)
        wedge_profile = _runtime_sanity_check_backup_restore_wedge(wedge_profile, atom)
        popularity_score = min(popularity_proxy / 500.0, 1.0) if popularity_proxy else 0.0
        active_install_score = min(active_installs / 500_000, 1.0) if active_installs else popularity_score
        review_count_score = min(review_count / 250.0, 1.0) if review_count else 0.0
        review_pain_score = min(max((3.2 - source_review_rating) / 2.2, 0.0), 1.0) if source_review_rating else 0.0
        product_maturity_score = 0.12 if version else 0.0
        fresh_update_score = 0.08 if last_updated or launched_at else 0.0
        summary_text = " ".join(
            [
                getattr(atom, "cost_consequence_clues", "") or "",
                getattr(atom, "why_now_clues", "") or "",
                getattr(atom, "current_workaround", "") or "",
                getattr(atom, "failure_mode", "") or "",
                getattr(atom, "job_to_be_done", "") or "",
                getattr(atom, "user_role", "") or "",
                getattr(atom, "segment", "") or "",
                audience_hint or "",
            ]
        ).lower()
        operational_buyer_score = min(
            1.0,
            0.1
            + (0.25 if any(term in summary_text for term in ["operations", "operator", "seller", "support", "finance", "accounting"]) else 0.0)
            + (0.18 if any(term in summary_text for term in ["shipping", "payout", "device setup", "inventory", "reconcile", "pricing"]) else 0.0),
        )
        compliance_burden_score = min(
            1.0,
            0.0
            + (0.42 if any(term in summary_text for term in ["compliance", "gdpr", "hipaa", "soc 2", "audit", "evidence"]) else 0.0)
            + (0.12 if any(term in summary_text for term in ["risk", "policy", "monitoring"]) else 0.0),
        )
        cost_pressure_score = min(
            1.0,
            0.1
            + min(value_hits, 4) * 0.06
            + min(pain_hits, 4) * 0.04
            + (0.18 if any(term in summary_text for term in ["manual", "spreadsheet", "time", "hours", "late", "missed"]) else 0.0)
            + (0.16 if any(term in summary_text for term in ["risk", "audit", "consequence", "penalty", "compliance"]) else 0.0)
            + (0.1 if any(term in summary_text for term in ["every week", "daily", "recurring"]) else 0.0),
        )
        cost_pressure_score = _apply_backup_restore_value_lift(cost_pressure_score, wedge_profile, "cost_pressure")
        review_signal_score = min(
            1.0,
            sum(1 for doc in competitor_docs if "review" in json.dumps(doc).lower()) / 3.0
            + review_count_score * 0.28
            + active_install_score * 0.2
            + review_pain_score * 0.22
            + product_maturity_score
            + fresh_update_score,
        )
        if review_issue_type == "product_specific_issue":
            review_signal_score = max(review_signal_score - 0.08, 0.0)
        elif review_issue_type == "reusable_workflow_pain":
            review_signal_score = min(review_signal_score + 0.05, 1.0)
        trend_score = min(
            1.0,
            0.18
            + (0.2 if any(term in summary_text for term in ["new regulation", "compliance", "gdpr", "hipaa", "soc 2"]) else 0.0)
            + (0.12 if any(term in summary_text for term in ["staff cut", "ai", "policy", "platform"]) else 0.0),
        )
        multi_source_value_lift = min(
            1.0,
            cross_source_match_score * 0.42
            + min(source_group_diversity, 3) / 3.0 * 0.28
            + min(source_family_diversity, 4) / 4.0 * 0.12
            + generalizability_score * 0.18
            - generalizability_penalty * 0.5,
        )
        demand_score = min(
            1.0,
            float(evidence_scores.get("value_score", 0.0) or 0.0) * 0.48
            + min(value_hits, 4) * 0.05
            + min(pain_hits, 4) * 0.04
            + review_signal_score * 0.12
            + cost_pressure_score * 0.16
            + operational_buyer_score * 0.08
            + compliance_burden_score * 0.07
            + active_install_score * 0.1
            + review_count_score * 0.08
            + multi_source_value_lift * 0.08
        )
        demand_score = _apply_backup_restore_value_lift(demand_score, wedge_profile, "demand")
        if review_issue_type == "product_specific_issue":
            demand_score = max(demand_score - 0.07, 0.0)
        elif review_issue_type == "reusable_workflow_pain":
            demand_score = min(demand_score + 0.05, 1.0)
        if generalizability_class == "product_specific_issue":
            demand_score = max(demand_score - generalizability_penalty * 0.45, 0.0)
        buyer_intent_score = min(
            1.0,
            demand_score * 0.44
            + cost_pressure_score * 0.22
            + operational_buyer_score * 0.12
            + compliance_burden_score * 0.16
            + (0.08 if pricing and pricing != "free" else 0.0)
            + (0.1 if audience_hint else 0.0)
            + multi_source_value_lift * 0.08
        )
        buyer_intent_score = _apply_backup_restore_value_lift(buyer_intent_score, wedge_profile, "buyer_intent")
        if review_issue_type == "product_specific_issue":
            buyer_intent_score = max(buyer_intent_score - 0.05, 0.0)
        competition_score = min(
            1.0,
            (1.0 - float(evidence_scores.get("saturation_score", 0.0) or 0.0)) * 0.55
            + (1.0 - float(evidence_scores.get("solution_gap_score", 0.0) or 0.0)) * 0.15
            + min(len(competitor_domains), 6) / 6 * 0.3,
        )
        willingness_to_pay_signal = min(
            1.0,
            demand_score * 0.35
            + buyer_intent_score * 0.3
            + cost_pressure_score * 0.18
            + operational_buyer_score * 0.08
            + compliance_burden_score * 0.09
            + multi_source_value_lift * 0.1
        )
        willingness_to_pay_signal = _apply_backup_restore_value_lift(willingness_to_pay_signal, wedge_profile, "willingness_to_pay")
        market_json = {
            "finding_id": finding_id,
            "run_id": self.db.active_run_id,
            "demand_score": round(demand_score, 4),
            "buyer_intent_score": round(buyer_intent_score, 4),
            "competition_score": round(competition_score, 4),
            "trend_score": round(trend_score, 4),
            "review_signal_score": round(review_signal_score, 4),
            "value_signal_score": round(float(evidence_scores.get("value_score", 0.0) or 0.0), 4),
            "cost_pressure_score": round(cost_pressure_score, 4),
            "operational_buyer_score": round(operational_buyer_score, 4),
            "compliance_burden_score": round(compliance_burden_score, 4),
            "willingness_to_pay_signal": round(willingness_to_pay_signal, 4),
            "multi_source_value_lift": round(multi_source_value_lift, 4),
            "wedge_name": wedge_profile["wedge_name"],
            "wedge_fit_score": wedge_profile["fit_score"],
            "wedge_operational_risk_score": wedge_profile["operational_risk_score"],
            "wedge_downtime_cost_score": wedge_profile["downtime_cost_score"],
            "wedge_buyer_ownership_score": wedge_profile["buyer_ownership_score"],
            "wedge_alternative_maturity_score": wedge_profile["alternative_maturity_score"],
            "wedge_value_lift": wedge_profile["value_lift"],
            "wedge_activation_reasons": wedge_profile["activation_reasons"],
            "wedge_block_reasons": wedge_profile["block_reasons"],
            "wedge_rule_version": wedge_profile["rule_version"],
            "wedge_active": wedge_profile["active"],
            "source_family_diversity": source_family_diversity,
            "source_group_diversity": source_group_diversity,
            "cross_source_match_score": round(cross_source_match_score, 4),
            "generalizability_class": generalizability_class,
            "generalizability_score": round(generalizability_score, 4),
            "generalizability_penalty": round(generalizability_penalty, 4),
            "competitor_domains": competitor_domains,
            "competitor_doc_count": len(competitor_docs),
            "review_product_name": review_metadata.get("product_name", ""),
            "source_review_rating": round(source_review_rating, 4) if source_review_rating else 0.0,
            "review_issue_type": review_issue_type,
            "review_generalizability_score": round(review_generalizability_score, 4),
            "review_generalizability_reasons": review_generalizability_reasons,
            "aggregate_rating": round(aggregate_rating, 4) if aggregate_rating else 0.0,
            "review_count": review_count,
            "active_installs": active_installs,
            "popularity_proxy": popularity_proxy,
            "pricing": pricing,
            "version": version,
            "last_updated": last_updated,
            "category": category,
            "listing_url": listing_url,
            "developer_name": developer_name,
            "launched_at": launched_at,
            "built_for_shopify": built_for_shopify,
        }
        return MarketEnrichment(
            finding_id=finding_id,
            demand_score=round(demand_score, 4),
            buyer_intent_score=round(buyer_intent_score, 4),
            competition_score=round(competition_score, 4),
            trend_score=round(trend_score, 4),
            review_signal_score=round(review_signal_score, 4),
            value_signal_score=round(float(evidence_scores.get("value_score", 0.0) or 0.0), 4),
            evidence_json=json.dumps(market_json),
            run_id=self.db.active_run_id,
        )
