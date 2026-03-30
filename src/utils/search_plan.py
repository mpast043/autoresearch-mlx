"""Search plan dataclasses extracted from research_tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CorroborationPlan:
    signature_terms: list[str]
    role_terms: list[str]
    segment_terms: list[str]
    job_phrase: str
    failure_phrase: str
    workaround_phrase: str
    cost_terms: list[str]
    ecosystem_hints: list[str]
    family_queries: dict[str, list[str]]
    max_attempts_per_family: int = 2
    source_priority: tuple[str, ...] = ("reddit", "web", "github", "stackoverflow", "etsy")


@dataclass(frozen=True)
class DiscoveryQueryPlan:
    source_name: str
    queries: list[str]
    slice_size: int
    cycle_index: int = 0
    query_offset: int = 0
    rotation_applied: bool = False
    rotated_queries_used: list[str] = field(default_factory=list)


@dataclass
class CorroborationAction:
    action: str
    target_family: str = ""
    reason: str = ""
    expected_gain_class: str = ""
    skipped_families: dict[str, str] = field(default_factory=dict)
    budget_snapshot: dict[str, Any] = field(default_factory=dict)
    fallback_strategy: str = ""
    promotion_gap_class: str = ""
    sufficiency_priority_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_family": self.target_family,
            "reason": self.reason,
            "expected_gain_class": self.expected_gain_class,
            "skipped_families": dict(self.skipped_families),
            "budget_snapshot": dict(self.budget_snapshot),
            "fallback_strategy": self.fallback_strategy,
            "promotion_gap_class": self.promotion_gap_class,
            "sufficiency_priority_reason": self.sufficiency_priority_reason,
        }


@dataclass
class SkillAudit:
    name: str
    path: str
    approved: bool
    score: float
    runnable: bool
    capabilities: list[str]
    reasons: list[str]