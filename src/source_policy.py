"""First-class source policy and routing rules for the evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_CLASSES = {
    "pain_signal",
    "success_signal",
    "demand_signal",
    "competition_signal",
    "meta_guidance",
    "low_signal_summary",
}


@dataclass(frozen=True)
class SourcePolicy:
    source_class: str
    atom_eligible: bool
    discovery_status: str
    use_for_search_seed: bool
    use_for_market_enrichment: bool
    use_for_prompt_eval: bool
    exclude_from_active_path: bool


SOURCE_POLICY_MAP: dict[str, SourcePolicy] = {
    "pain_signal": SourcePolicy(
        source_class="pain_signal",
        atom_eligible=True,
        discovery_status="qualified",
        use_for_search_seed=False,
        use_for_market_enrichment=True,
        use_for_prompt_eval=False,
        exclude_from_active_path=False,
    ),
    "success_signal": SourcePolicy(
        source_class="success_signal",
        atom_eligible=False,
        discovery_status="screened_out",
        use_for_search_seed=True,
        use_for_market_enrichment=True,
        use_for_prompt_eval=False,
        exclude_from_active_path=True,
    ),
    "demand_signal": SourcePolicy(
        source_class="demand_signal",
        atom_eligible=False,
        discovery_status="screened_out",
        use_for_search_seed=False,
        use_for_market_enrichment=True,
        use_for_prompt_eval=False,
        exclude_from_active_path=True,
    ),
    "competition_signal": SourcePolicy(
        source_class="competition_signal",
        atom_eligible=False,
        discovery_status="screened_out",
        use_for_search_seed=False,
        use_for_market_enrichment=True,
        use_for_prompt_eval=False,
        exclude_from_active_path=True,
    ),
    "meta_guidance": SourcePolicy(
        source_class="meta_guidance",
        atom_eligible=False,
        discovery_status="screened_out",
        use_for_search_seed=False,
        use_for_market_enrichment=False,
        use_for_prompt_eval=True,
        exclude_from_active_path=True,
    ),
    "low_signal_summary": SourcePolicy(
        source_class="low_signal_summary",
        atom_eligible=False,
        discovery_status="screened_out",
        use_for_search_seed=False,
        use_for_market_enrichment=False,
        use_for_prompt_eval=False,
        exclude_from_active_path=True,
    ),
}


def normalize_source_class(raw_source_class: str | None, finding_kind: str | None = None) -> str:
    """Resolve source-class values into the enforced policy vocabulary."""
    source_class = (raw_source_class or "").strip() or ""
    if source_class in SOURCE_CLASSES:
        return source_class
    if (finding_kind or "").strip() == "success_signal":
        return "success_signal"
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Unknown source class '%s' normalized to 'low_signal_summary'; finding_kind=%s", raw_source_class, finding_kind)
    return "low_signal_summary"


def policy_for(source_class: str | None, finding_kind: str | None = None) -> SourcePolicy:
    normalized = normalize_source_class(source_class, finding_kind)
    return SOURCE_POLICY_MAP[normalized]


def atom_generation_allowed(source_class: str | None, finding_kind: str | None = None) -> bool:
    return policy_for(source_class, finding_kind).atom_eligible


def discovery_status_for(source_class: str | None, finding_kind: str | None = None) -> str:
    return policy_for(source_class, finding_kind).discovery_status


def source_policy_summary(source_class: str | None, finding_kind: str | None = None) -> dict[str, Any]:
    policy = policy_for(source_class, finding_kind)
    return {
        "source_class": policy.source_class,
        "atom_eligible": policy.atom_eligible,
        "discovery_status": policy.discovery_status,
        "use_for_search_seed": policy.use_for_search_seed,
        "use_for_market_enrichment": policy.use_for_market_enrichment,
        "use_for_prompt_eval": policy.use_for_prompt_eval,
        "exclude_from_active_path": policy.exclude_from_active_path,
    }
