"""Problem space lifecycle management.

Mirrors the pattern from discovery_term_lifecycle.py but for
ProblemSpace entities.  Tracks state transitions, computes yield
scores, and determines when a space should be promoted, demoted, or
exhausted.
"""

from __future__ import annotations

from typing import Any

from src.problem_space import (
    EXPLORING,
    VALIDATED,
    EXHAUSTED,
    ARCHIVED,
    SOURCE_LLM,
    SOURCE_MANUAL,
    SOURCE_THEME_MIGRATION,
    ProblemSpace,
)


# State transition rules: current_state -> set of allowed next states
TRANSITIONS: dict[str, set[str]] = {
    EXPLORING: {VALIDATED, EXHAUSTED, ARCHIVED},
    VALIDATED: {EXHAUSTED, ARCHIVED},
    EXHAUSTED: {ARCHIVED, EXPLORING},  # can be reactivated
    ARCHIVED: {EXPLORING},  # manual reactivation only
}

# Default thresholds
DEFAULT_MIN_VALIDATED_FOR_PROMOTED = 1
DEFAULT_MIN_PROTOTYPE_CANDIDATES = 1
DEFAULT_MAX_IDLE_CYCLES_EXHAUSTED = 3
DEFAULT_MAX_IDLE_CYCLES_ARCHIVED = 10


class ProblemSpaceLifecycleManager:
    """Manages ProblemSpace lifecycle transitions and metrics."""

    def __init__(
        self,
        db: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.config = config or {}
        lc_config = self.config.get("discovery", {}).get("llm_expansion", {})
        self.min_validated_for_promoted = lc_config.get(
            "min_validated_for_promoted", DEFAULT_MIN_VALIDATED_FOR_PROMOTED,
        )
        self.min_prototype_candidates = lc_config.get(
            "min_prototype_candidates", DEFAULT_MIN_PROTOTYPE_CANDIDATES,
        )
        self.max_idle_cycles_exhausted = lc_config.get(
            "max_idle_cycles_for_exhausted", DEFAULT_MAX_IDLE_CYCLES_EXHAUSTED,
        )
        self.max_idle_cycles_archived = lc_config.get(
            "max_idle_cycles_for_archived", DEFAULT_MAX_IDLE_CYCLES_ARCHIVED,
        )

    def compute_next_state(
        self,
        space: ProblemSpace,
        idle_cycles: int = 0,
    ) -> str:
        """Determine the next state for a problem space.

        Args:
            space: Current problem space.
            idle_cycles: Number of consecutive cycles with no new findings.

        Returns:
            The next state (may be the same as current).
        """
        if space.status == EXPLORING:
            if self._should_promote_to_validated(space):
                return VALIDATED
            if idle_cycles >= self.max_idle_cycles_exhausted:
                return EXHAUSTED
            return EXPLORING

        if space.status == VALIDATED:
            if idle_cycles >= self.max_idle_cycles_exhausted:
                return EXHAUSTED
            return VALIDATED

        if space.status == EXHAUSTED:
            if idle_cycles >= self.max_idle_cycles_archived:
                return ARCHIVED
            return EXHAUSTED

        # ARCHIVED stays archived unless manually reactivated
        return ARCHIVED

    def _should_promote_to_validated(self, space: ProblemSpace) -> bool:
        """Check if an exploring space should be promoted to validated."""
        if space.total_prototype_candidates >= self.min_prototype_candidates:
            return True
        if space.total_validations >= self.min_validated_for_promoted * 2:
            return True
        return False

    def compute_yield_score(self, space: ProblemSpace) -> float:
        """Compute a composite yield score for a problem space.

        Weighted similarly to the term-level wedge_quality but applied at
        the space level: prototype_candidates and build_briefs are the
        strongest signals, followed by validations and findings.
        """
        if space.total_findings == 0:
            return 0.0

        score = 0.0
        # Strongest signals
        score += 0.35 * min(space.total_prototype_candidates, 5) / 5
        score += 0.25 * min(space.total_build_briefs, 3) / 3
        # Moderate signals
        score += 0.20 * min(space.total_validations, 10) / 10
        score += 0.10 * min(space.total_findings, 20) / 20
        # Yield ratio
        if space.total_findings > 0:
            validation_rate = space.total_validations / space.total_findings
            score += 0.10 * min(validation_rate, 1.0)

        return round(score, 4)

    def transition_space(
        self,
        space: ProblemSpace,
        new_status: str,
    ) -> ProblemSpace:
        """Validate and apply a state transition.

        Args:
            space: The problem space to transition.
            new_status: The target status.

        Returns:
            The updated space.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = TRANSITIONS.get(space.status, set())
        if new_status not in allowed and new_status != space.status:
            raise ValueError(
                f"Cannot transition ProblemSpace '{space.space_key}' "
                f"from '{space.status}' to '{new_status}'. "
                f"Allowed: {allowed}"
            )
        space.status = new_status
        return space

    def get_active_spaces(self, limit: int = 25) -> list[ProblemSpace]:
        """Get all problem spaces in exploring or validated state."""
        from src.database import Database
        if not isinstance(self.db, Database):
            return []
        return self.db.list_problem_spaces(status=None, limit=limit)

    def get_exploring_spaces(self, limit: int = 25) -> list[ProblemSpace]:
        """Get spaces currently being explored."""
        from src.database import Database
        if not isinstance(self.db, Database):
            return []
        return self.db.list_problem_spaces(status=EXPLORING, limit=limit)

    def get_validated_spaces(self, limit: int = 25) -> list[ProblemSpace]:
        """Get spaces that have been validated."""
        from src.database import Database
        if not isinstance(self.db, Database):
            return []
        return self.db.list_problem_spaces(status=VALIDATED, limit=limit)

    def get_exhausted_space_keys(self) -> list[str]:
        """Get keys of exhausted and archived spaces (for LLM prompt context)."""
        from src.database import Database
        if not isinstance(self.db, Database):
            return []
        exhausted = self.db.list_problem_spaces(status=EXHAUSTED, limit=100)
        archived = self.db.list_problem_spaces(status=ARCHIVED, limit=100)
        return [s.space_key for s in exhausted + archived]

    def update_space_metrics(self, space_key: str) -> None:
        """Recalculate and persist metrics for a problem space.

        Aggregates metrics from linked discovery_search_terms that were
        derived from this problem space.
        """
        from src.database import Database
        if not isinstance(self.db, Database):
            return

        terms = self.db.get_problem_space_terms(space_key)
        total_findings = 0
        total_validations = 0
        total_prototype_candidates = 0
        total_build_briefs = 0

        for term in terms:
            # Look up this term in discovery_search_terms for metrics
            row = self.db._get_connection().execute(
                "SELECT findings_emitted, validations, prototype_candidates, build_briefs "
                "FROM discovery_search_terms WHERE term_type = ? AND term_value = ?",
                (term.term_type, term.term_value),
            ).fetchone()
            if row:
                total_findings += row[0] or 0
                total_validations += row[1] or 0
                total_prototype_candidates += row[2] or 0
                total_build_briefs += row[3] or 0

        space = self.db.get_problem_space(space_key)
        if not space:
            return

        space.total_findings = total_findings
        space.total_validations = total_validations
        space.total_prototype_candidates = total_prototype_candidates
        space.total_build_briefs = total_build_briefs
        space.yield_score = self.compute_yield_score(space)

        self.db.update_problem_space_metrics(
            space_key=space_key,
            total_findings=total_findings,
            total_validations=total_validations,
            total_prototype_candidates=total_prototype_candidates,
            total_build_briefs=total_build_briefs,
            yield_score=space.yield_score,
        )