"""Resolve promotion/park thresholds from config (multiple legacy shapes)."""

from __future__ import annotations

from typing import Any


def resolve_promotion_park_thresholds(config: dict[str, Any] | None) -> tuple[float, float]:
    """Return (promotion_threshold, park_threshold) used by ValidationAgent / stage_decision.

    Precedence (first hit wins):
    1. ``validation.decisions.promote_score`` / ``park_score``
    2. ``orchestration.promotion_threshold`` / ``park_threshold``
    3. ``validation.promotion_threshold`` / ``park_threshold`` (top-level under validation)
    4. Defaults ``0.62`` / ``0.48``
    """
    cfg = config or {}
    vc = cfg.get("validation", {}) or {}
    decision = vc.get("decisions", {}) or {}
    orch = cfg.get("orchestration", {}) or {}

    promote = decision.get("promote_score")
    if promote is None:
        promote = orch.get("promotion_threshold")
    if promote is None:
        promote = vc.get("promotion_threshold", 0.62)

    park = decision.get("park_score")
    if park is None:
        park = orch.get("park_threshold")
    if park is None:
        park = vc.get("park_threshold", 0.48)

    return float(promote), float(park)
