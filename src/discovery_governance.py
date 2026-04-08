"""Automatic governance for discovery terms and source families.

Manages:
- Locked default term updates based on live performance
- Challenger replacement decisions
- Source family enable/disable/pause states
- Durable state persistence with explainability
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.database import Database
from src.discovery_next_wave import (
    LOCKED_KEYWORDS,
    LOCKED_SUBREDDITS,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/governance_state.json")

# Governance configuration thresholds (can be overridden)
DEFAULT_CONFIG = {
    # Locked default retention thresholds
    "min_findings_to_retain": 5,
    "min_qualified_to_retain": 1,
    "min_prototype_candidates_to_retain": 1,
    "max_waves_without_output": 3,

    # Challenger replacement thresholds
    "challenger_margin": 0.2,
    "consecutive_waves_to_replace": 2,
    "challenger_outperformance_factor": 1.5,

    # Source family thresholds
    "min_findings_for_evaluation": 20,
    "max_screened_out_rate": 0.85,
    "consecutive_empty_waves_to_disable": 3,
    "consecutive_empty_waves_to_pause": 2,
    "min_build_briefs_to_promote": 1,
    "min_promoted_to_promote": 2,

    # Wave tracking
    "waves_to_track": 10,
}


@dataclass
class TermGovernance:
    """Governance state for a single term."""
    term_value: str
    term_type: str  # 'keyword' or 'subreddit'
    state: str = "active"  # active, locked, challenger, replaced, exhausted, banned
    locked_at: float | None = None
    replaced_by: str | None = None
    replacement_reason: str | None = None
    waves_at_locked: int = 0
    consecutive_empty_waves: int = 0
    total_findings: int = 0
    total_qualified: int = 0
    total_promoted: int = 0
    total_prototype_candidates: int = 0
    total_build_briefs: int = 0
    last_updated: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class SourceFamilyGovernance:
    """Governance state for a source family."""
    family: str  # e.g., 'reddit-problem', 'shopify-review'
    state: str = "active"  # active, probation, paused, disabled, manual_override
    locked_at: float | None = None
    paused_at: float | None = None
    disabled_at: float | None = None
    reactivated_at: float | None = None
    consecutive_empty_waves: int = 0
    total_findings: int = 0
    total_qualified: int = 0
    total_promoted: int = 0
    total_build_briefs: int = 0
    total_screened_out: int = 0
    screened_out_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    manual_override_reason: str | None = None
    notes: str = ""


@dataclass
class GovernanceState:
    """Full governance state."""
    terms: dict[str, TermGovernance] = field(default_factory=dict)
    source_families: dict[str, SourceFamilyGovernance] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    last_wave_ts: float = 0
    wave_count: int = 0

    def to_dict(self) -> dict:
        # Custom serialization to handle required fields
        terms_dict = {}
        for k, v in self.terms.items():
            terms_dict[k] = {
                "term_value": v.term_value,
                "term_type": v.term_type,
                "state": v.state,
                "locked_at": v.locked_at,
                "replaced_by": v.replaced_by,
                "replacement_reason": v.replacement_reason,
                "waves_at_locked": v.waves_at_locked,
                "consecutive_empty_waves": v.consecutive_empty_waves,
                "total_findings": v.total_findings,
                "total_qualified": v.total_qualified,
                "total_promoted": v.total_promoted,
                "total_prototype_candidates": v.total_prototype_candidates,
                "total_build_briefs": v.total_build_briefs,
                "last_updated": v.last_updated,
                "notes": v.notes,
            }

        sources_dict = {}
        for k, v in self.source_families.items():
            sources_dict[k] = {
                "family": v.family,
                "state": v.state,
                "locked_at": v.locked_at,
                "paused_at": v.paused_at,
                "disabled_at": v.disabled_at,
                "reactivated_at": v.reactivated_at,
                "consecutive_empty_waves": v.consecutive_empty_waves,
                "total_findings": v.total_findings,
                "total_qualified": v.total_qualified,
                "total_promoted": v.total_promoted,
                "total_build_briefs": v.total_build_briefs,
                "total_screened_out": v.total_screened_out,
                "screened_out_rate": v.screened_out_rate,
                "last_updated": v.last_updated,
                "manual_override_reason": v.manual_override_reason,
                "notes": v.notes,
            }

        return {
            "terms": terms_dict,
            "source_families": sources_dict,
            "config": self.config,
            "last_wave_ts": self.last_wave_ts,
            "wave_count": self.wave_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GovernanceState:
        state = cls()
        state.config = data.get("config", DEFAULT_CONFIG.copy())
        state.last_wave_ts = data.get("last_wave_ts", 0)
        state.wave_count = data.get("wave_count", 0)

        for k, v in data.get("terms", {}).items():
            state.terms[k] = TermGovernance(**v)

        for k, v in data.get("source_families", {}).items():
            state.source_families[k] = SourceFamilyGovernance(**v)

        return state


class DiscoveryGovernance:
    """Main governance manager."""

    def __init__(
        self,
        db: Database,
        config: dict[str, Any] | None = None,
        state_path: Path | None = None,
    ):
        self.db = db
        self.conn = db._get_connection()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        db_path = Path(getattr(db, "db_path", "") or "")
        default_state_path = (
            db_path.with_name(f"{db_path.stem}_governance_state.json")
            if db_path.name
            else DEFAULT_STATE_PATH
        )
        self.state_path = state_path or default_state_path or DEFAULT_STATE_PATH
        self.state = self._load_state()

        # Initialize source families if not present
        self._ensure_source_families()

    def _load_state(self) -> GovernanceState:
        """Load governance state from file."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                logger.info(f"Loaded governance state from {self.state_path}")
                return GovernanceState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load governance state: {e}, creating new")

        return GovernanceState(config=self.config.copy())

    def _save_state(self) -> None:
        """Save governance state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        logger.info(f"Saved governance state to {self.state_path}")

    def _ensure_source_families(self) -> None:
        """Initialize source families if not present."""
        known_families = [
            "reddit-problem",
            "web-problem",
            "web-success",
            "shopify-review",
            "wordpress-review",
            "github-issue",
            "market-problem",
        ]

        for family in known_families:
            if family not in self.state.source_families:
                self.state.source_families[family] = SourceFamilyGovernance(family=family)

    def _get_term_key(self, term_value: str, term_type: str) -> str:
        return f"{term_type}:{term_value}"

    def _normalize_source_family(self, source_name: str) -> str:
        raw = str(source_name or "").strip().lower()
        family = raw.split("/", 1)[0] if "/" in raw else raw
        aliases = {
            "reddit": "reddit-problem",
            "reddit-problem": "reddit-problem",
            "web": "web-problem",
            "web-problem": "web-problem",
            "web-success": "web-success",
            "github": "github-issue",
            "github-problem": "github-issue",
            "github-issue": "github-issue",
            "shopify_reviews": "shopify-review",
            "shopify-review": "shopify-review",
            "wordpress_reviews": "wordpress-review",
            "wordpress-review": "wordpress-review",
            "market": "market-problem",
            "market-problem": "market-problem",
        }
        return aliases.get(family, family)

    def evaluate_wave(self) -> dict[str, Any]:
        """Evaluate the most recent wave and update governance.

        Returns a governance summary with all changes made.
        """
        summary = {
            "terms_locked": [],
            "terms_replaced": [],
            "terms_exhausted": [],
            "terms_banned": [],
            "challengers_promoted": [],
            "sources_promoted": [],
            "sources_paused": [],
            "sources_disabled": [],
            "sources_reactivated": [],
            "wave_number": self.state.wave_count + 1,
        }

        # Update wave count
        self.state.wave_count += 1
        self.state.last_wave_ts = time.time()

        # Get current locked terms
        locked_keywords = set(LOCKED_KEYWORDS)
        locked_subs = set(LOCKED_SUBREDDITS)

        # Evaluate each locked keyword
        for kw in locked_keywords:
            result = self._evaluate_term(kw, "keyword")
            if result:
                if result == "retained":
                    summary["terms_locked"].append(kw)
                elif result == "replaced":
                    summary["terms_replaced"].append(kw)
                elif result == "exhausted":
                    summary["terms_exhausted"].append(kw)
                elif result == "banned":
                    summary["terms_banned"].append(kw)

        # Evaluate each locked subreddit
        for sub in locked_subs:
            result = self._evaluate_term(sub, "subreddit")
            if result:
                if result == "retained":
                    summary["terms_locked"].append(f"r/{sub}")
                elif result == "replaced":
                    summary["terms_replaced"].append(f"r/{sub}")

        # Evaluate source families
        self._evaluate_source_families(summary)

        # Save state
        self._save_state()

        return summary

    def _evaluate_term(self, term_value: str, term_type: str) -> str | None:
        """Evaluate a single term for retention/replacement."""
        key = self._get_term_key(term_value, term_type)

        row = self.conn.execute(
            """
            SELECT
                findings_emitted as findings,
                validations,
                passes,
                prototype_candidates as pc,
                build_briefs as bf,
                screened_out as screened
            FROM discovery_search_terms
            WHERE term_type = ? AND term_value = ?
            """,
            (term_type, term_value),
        ).fetchone()

        if not row and term_type == "subreddit":
            row = self.conn.execute(
                """
                SELECT
                    COUNT(*) as findings,
                    SUM(CASE WHEN status = 'qualified' THEN 1 ELSE 0 END) as validations,
                    SUM(CASE WHEN status = 'promoted' THEN 1 ELSE 0 END) as passes,
                    0 as pc,
                    0 as bf,
                    SUM(CASE WHEN status = 'screened_out' THEN 1 ELSE 0 END) as screened
                FROM findings
                WHERE source = ?
                """,
                (f"reddit-problem/{term_value}",),
            ).fetchone()

        if not row:
            return None

        findings = int(row["findings"] or 0)
        validations = int(row["validations"] or 0)
        passes = int(row["passes"] or 0)
        pc = int(row["pc"] or 0)
        bf = int(row["bf"] or 0)

        # Get or create governance record
        if key not in self.state.terms:
            self.state.terms[key] = TermGovernance(
                term_value=term_value,
                term_type=term_type,
                state="locked",
                locked_at=time.time(),
            )

        term = self.state.terms[key]
        term.total_findings = findings
        term.total_qualified = validations
        term.total_promoted = passes
        term.total_prototype_candidates = pc
        term.total_build_briefs = bf
        term.last_updated = time.time()

        # Check for replacement by checking if there are better challengers
        challengers = self._find_challengers(term_type, term_value)

        if challengers:
            best_challenger = challengers[0]
            current_output_score = (pc * 0.5) + (bf * 0.5)

            # If challenger significantly outperforms, consider replacement
            if (
                best_challenger["output_score"]
                > current_output_score * self.config.get("challenger_outperformance_factor", 1.5)
                and best_challenger["output_score"] >= self.config.get("min_prototype_candidates_to_retain", 1)
            ):

                # Replace the term
                term.state = "replaced"
                term.replaced_by = best_challenger["term_value"]
                term.replacement_reason = (
                    f"challenger {best_challenger['term_value']} outperformed by "
                    f"{best_challenger['output_score']} vs {current_output_score}"
                )
                term.waves_at_locked = self.state.wave_count

                return "replaced"

        # Check for exhaustion
        if findings < self.config.get("min_findings_to_retain", 5):
            term.consecutive_empty_waves += 1
            if term.consecutive_empty_waves >= self.config.get("max_waves_without_output", 3):
                term.state = "exhausted"
                return "exhausted"
        else:
            term.consecutive_empty_waves = 0

        return "retained"

    def _find_challengers(self, term_type: str, exclude_term: str) -> list[dict]:
        """Find challenger terms that could replace the current locked term."""
        challengers = []

        rows = self.conn.execute(f'''
            SELECT term_value,
                   COALESCE(prototype_candidates, 0) as pc,
                   COALESCE(build_briefs, 0) as bf,
                   COALESCE(findings_emitted, 0) as findings
            FROM discovery_search_terms
            WHERE term_type = ?
              AND term_value != ?
              AND state NOT IN ('exhausted', 'banned', 'paused')
              AND (prototype_candidates > 0 OR build_briefs > 0)
            ORDER BY (prototype_candidates * 0.5 + build_briefs * 0.5) DESC
            LIMIT 5
        ''', (term_type, exclude_term)).fetchall()

        for r in rows:
            output_score = (r["pc"] or 0) * 0.5 + (r["bf"] or 0) * 0.5
            challengers.append({
                "term_value": r["term_value"],
                "output_score": output_score,
                "prototype_candidates": r["pc"],
                "build_briefs": r["bf"],
            })

        return challengers

    def _evaluate_source_families(self, summary: dict) -> None:
        """Evaluate source families and update their states."""
        source_rows = self.conn.execute(
            """
            SELECT source, status, COUNT(*) as total
            FROM findings
            GROUP BY source, status
            """
        ).fetchall()
        family_metrics: dict[str, dict[str, int]] = {}
        for row in source_rows:
            family = self._normalize_source_family(row["source"])
            if not family:
                continue
            bucket = family_metrics.setdefault(
                family,
                {"total": 0, "qualified": 0, "promoted": 0, "screened": 0, "build_briefs": 0},
            )
            count = int(row["total"] or 0)
            status = str(row["status"] or "")
            bucket["total"] += count
            if status == "qualified":
                bucket["qualified"] += count
            elif status == "promoted":
                bucket["promoted"] += count
            elif status == "screened_out":
                bucket["screened"] += count

        build_brief_rows = self.conn.execute(
            """
            SELECT f.source, COUNT(*) as total
            FROM build_briefs bb
            JOIN validations v ON v.id = bb.validation_id
            JOIN findings f ON f.id = v.finding_id
            GROUP BY f.source
            """
        ).fetchall()
        for row in build_brief_rows:
            family = self._normalize_source_family(row["source"])
            if not family:
                continue
            bucket = family_metrics.setdefault(
                family,
                {"total": 0, "qualified": 0, "promoted": 0, "screened": 0, "build_briefs": 0},
            )
            bucket["build_briefs"] += int(row["total"] or 0)

        for family, metrics in family_metrics.items():
            total = metrics["total"]
            qualified = metrics["qualified"]
            promoted = metrics["promoted"]
            screened = metrics["screened"]
            build_briefs = metrics["build_briefs"]
            screened_rate = screened / total if total > 0 else 0

            # Get or create governance record
            if family not in self.state.source_families:
                self.state.source_families[family] = SourceFamilyGovernance(family=family)

            source = self.state.source_families[family]
            source.total_findings = total
            source.total_qualified = qualified
            source.total_promoted = promoted
            source.total_build_briefs = build_briefs
            source.total_screened_out = screened
            source.screened_out_rate = screened_rate
            source.last_updated = time.time()

            # Skip if manual override
            if source.state == "manual_override":
                continue

            # Check for disable (repeated empty waves with enough findings)
            min_findings = self.config.get("min_findings_for_evaluation", 20)
            max_screened = self.config.get("max_screened_out_rate", 0.85)
            empty_waves_limit = self.config.get("consecutive_empty_waves_to_disable", 3)

            if qualified == 0 and promoted == 0 and total >= min_findings:
                source.consecutive_empty_waves += 1

                if source.consecutive_empty_waves >= empty_waves_limit:
                    if source.state != "disabled":
                        source.state = "disabled"
                        source.disabled_at = time.time()
                        source.notes = f"Disabled after {source.consecutive_empty_waves} waves with 0 qualified/promoted despite {total} findings"
                        summary["sources_disabled"].append(family)

                elif source.consecutive_empty_waves >= self.config.get("consecutive_empty_waves_to_pause", 2):
                    if source.state not in ("paused", "disabled"):
                        source.state = "paused"
                        source.paused_at = time.time()
                        source.notes = f"Paused after {source.consecutive_empty_waves} waves with 0 qualified/promoted"
                        summary["sources_paused"].append(family)
            else:
                source.consecutive_empty_waves = 0

                # Check for promotion (consistently producing results)
                min_promoted = self.config.get("min_promoted_to_promote", 2)
                min_briefs = self.config.get("min_build_briefs_to_promote", 1)

                if promoted >= min_promoted or source.total_build_briefs >= min_briefs:
                    if source.state == "paused":
                        source.state = "active"
                        source.reactivated_at = time.time()
                        source.notes = "Reactivated due to strong performance"
                        summary["sources_reactivated"].append(family)
                    elif source.state == "active" and family not in summary["sources_promoted"]:
                        summary["sources_promoted"].append(family)

    def get_active_source_families(self) -> list[str]:
        """Get list of active source families for next wave."""
        active = []
        for family, state in self.state.source_families.items():
            if state.state == "active":
                active.append(family)

        # Always include reddit-problem as primary unless explicitly paused/disabled.
        reddit_state = self.state.source_families.get("reddit-problem")
        if (
            "reddit-problem" not in active
            and reddit_state
            and reddit_state.state not in {"paused", "disabled", "manual_override"}
        ):
            active.insert(0, "reddit-problem")

        return active

    def get_next_wave_terms(self) -> dict[str, Any]:
        """Generate next wave with governance."""
        # Get locked terms that passed evaluation
        locked_kw = []
        locked_sub = []

        for key, term in self.state.terms.items():
            if term.state == "locked":
                if term.term_type == "keyword":
                    locked_kw.append(term.term_value)
                else:
                    locked_sub.append(term.term_value)

        # Fall back to defaults if no governance state
        if not locked_kw:
            locked_kw = LOCKED_KEYWORDS
        if not locked_sub:
            locked_sub = LOCKED_SUBREDDITS

        return {
            "keywords": locked_kw[:5],
            "subreddits": locked_sub[:5],
            "active_sources": self.get_active_source_families(),
        }

    def get_governance_summary(self) -> dict:
        """Get current governance state summary."""
        terms_summary = {}
        for key, term in self.state.terms.items():
            terms_summary[key] = {
                "state": term.state,
                "total_findings": term.total_findings,
                "total_qualified": term.total_qualified,
                "total_promoted": term.total_promoted,
                "replaced_by": term.replaced_by,
                "notes": term.notes,
            }

        sources_summary = {}
        for family, source in self.state.source_families.items():
            sources_summary[family] = {
                "state": source.state,
                "total_findings": source.total_findings,
                "total_qualified": source.total_qualified,
                "total_promoted": source.total_promoted,
                "screened_out_rate": round(source.screened_out_rate, 2),
                "notes": source.notes,
            }

        return {
            "terms": terms_summary,
            "source_families": sources_summary,
            "wave_count": self.state.wave_count,
            "last_wave_ts": self.state.last_wave_ts,
        }


def run_governance_cycle(db: Database) -> dict:
    """Run a complete governance cycle and return summary."""
    governance = DiscoveryGovernance(db)

    # Evaluate current wave
    summary = governance.evaluate_wave()

    # Add next wave terms
    next_wave = governance.get_next_wave_terms()
    summary["next_wave"] = next_wave

    # Add governance summary
    summary["governance_state"] = governance.get_governance_summary()

    return summary
