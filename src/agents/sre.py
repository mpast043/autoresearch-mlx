"""SRE agent for wedge health monitoring and regression detection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.agents.base import BaseAgent


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RegressionType(Enum):
    """Types of regressions."""

    VALIDATION_DROP = "validation_drop"
    SCORE_DECAY = "score_decay"
    ATOM_DRIFT = "atom_drift"
    CORROBORATION_LOSS = "corroboration_loss"
    OPPORTUNITY_STALL = "opportunity_stall"


@dataclass
class WedgeHealth:
    """Health metrics for a wedge."""

    wedge_id: int
    wedge_title: str
    status: HealthStatus
    score: float
    previous_score: float
    validation_count: int
    last_validation: str
    atom_count: int
    corroboration_depth: float
    opportunity_count: int
    build_ready_count: int
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class Regression:
    """Detected regression in a wedge."""

    wedge_id: int
    wedge_title: str
    regression_type: RegressionType
    severity: str  # critical, warning, info
    current_value: float
    previous_value: float
    change_percent: float
    detected_at: str
    cause: str
    recommended_action: str


@dataclass
class SREReport:
    """SRE health and regression report."""

    generated_at: float
    wedges_checked: int
    healthy_count: int
    degraded_count: int
    critical_count: int
    regressions: list[Regression] = field(default_factory=list)
    wedge_health: dict[int, WedgeHealth] = field(default_factory=dict)
    summary: str = ""


class SREAgent(BaseAgent):
    """SRE agent for monitoring wedge health and detecting regressions.

    Tracks wedge metrics over time, detects score decay and validation
    drops, and recommends recovery actions.
    """

    def __init__(self, db, config: dict[str, Any] | None = None):
        super().__init__("SREAgent")
        self.db = db
        self.config = config or {}

        # Thresholds from config
        self.score_drop_threshold = self.config.get("score_drop_threshold", 0.15)
        self.validation_drop_threshold = self.config.get("validation_drop_threshold", 0.30)
        self.corroboration_min = self.config.get("corroboration_min", 0.3)
        self.stall_days = self.config.get("stall_days", 7)

    async def process(self, message) -> dict[str, Any]:
        """Process an SRE health check request."""
        payload = message.payload if hasattr(message, "payload") else message
        wedge_id = payload.get("wedge_id")

        if wedge_id:
            health = self.check_wedge_health(wedge_id)
            regressions = self.detect_regressions(wedge_id)
            return {"health": health, "regressions": regressions, "formatted": self.format_wedge_status(wedge_id)}
        else:
            report = self.generate_report()
            return {"report": report, "summary": report.summary}

    def check_wedge_health(self, wedge_id: int) -> WedgeHealth:
        """Check health metrics for a single wedge.

        Args:
            wedge_id: Wedge ID to check

        Returns:
            WedgeHealth with current metrics
        """
        # Get opportunity data - filter by wedge_id from all opportunities
        all_opportunities = self.db.get_opportunities(limit=500)
        opportunities = [o for o in all_opportunities if o.wedge_id == wedge_id]
        if not opportunities:
            return WedgeHealth(
                wedge_id=wedge_id,
                wedge_title="Unknown",
                status=HealthStatus.UNKNOWN,
                score=0.0,
                previous_score=0.0,
                validation_count=0,
                last_validation="",
                atom_count=0,
                corroboration_depth=0.0,
                opportunity_count=0,
                build_ready_count=0,
                issues=["No opportunities found"],
            )

        # Get current and previous scores
        current_opp = opportunities[0]
        current_score = current_opp.composite_score or 0.0

        # Get previous score from history (would need to track in DB)
        previous_score = self._get_previous_score(wedge_id)

        # Count validations
        validation_count = len([o for o in opportunities if o.validation_status])

        # Get last validation time
        last_validation = current_opp.last_validated_at or ""

        # Get atom count
        atoms = self.db.get_problem_atoms(limit=1000)
        atom_count = len([a for a in atoms if a.wedge_id == wedge_id])

        # Get corroboration depth
        corroboration_depth = self._get_corroboration_depth(wedge_id)

        # Count build-ready opportunities
        build_ready_count = len(
            [o for o in opportunities if o.selection_status == "build_ready"]
        )

        # Determine health status
        status = self._determine_health(
            current_score, previous_score, validation_count, corroboration_depth
        )

        # Generate issues and recommendations
        issues = self._generate_issues(
            current_score, previous_score, validation_count, corroboration_depth
        )
        recommendations = self._generate_recommendations(
            current_score, previous_score, validation_count, corroboration_depth
        )

        return WedgeHealth(
            wedge_id=wedge_id,
            wedge_title=current_opp.title or "Unknown",
            status=status,
            score=current_score,
            previous_score=previous_score,
            validation_count=validation_count,
            last_validation=last_validation,
            atom_count=atom_count,
            corroboration_depth=corroboration_depth,
            opportunity_count=len(opportunities),
            build_ready_count=build_ready_count,
            issues=issues,
            recommendations=recommendations,
        )

    def detect_regressions(self, wedge_id: int) -> list[Regression]:
        """Detect regressions for a wedge by comparing to historical data.

        Args:
            wedge_id: Wedge ID to check

        Returns:
            List of detected regressions
        """
        regressions = []
        health = self.check_wedge_health(wedge_id)

        # Score decay
        if health.previous_score > 0:
            score_change = (health.score - health.previous_score) / health.previous_score
            if score_change <= -self.score_drop_threshold:
                regressions.append(
                    Regression(
                        wedge_id=wedge_id,
                        wedge_title=health.wedge_title,
                        regression_type=RegressionType.SCORE_DECAY,
                        severity="critical" if score_change <= -0.3 else "warning",
                        current_value=health.score,
                        previous_value=health.previous_score,
                        change_percent=score_change * 100,
                        detected_at=datetime.now().isoformat(),
                        cause=f"Score dropped from {health.previous_score:.2f} to {health.score:.2f}",
                        recommended_action="Re-run validation with fresh evidence",
                    )
                )

        # Validation drop (if we had historical validation counts)
        # This would require storing validation history in DB

        # Corroboration loss
        if health.corroboration_depth < self.corroboration_min:
            regressions.append(
                Regression(
                    wedge_id=wedge_id,
                    wedge_title=health.wedge_title,
                    regression_type=RegressionType.CORROBORATION_LOSS,
                    severity="warning",
                    current_value=health.corroboration_depth,
                    previous_value=self.corroboration_min,
                    change_percent=((health.corroboration_depth - self.corroboration_min) / self.corroboration_min) * 100,
                    detected_at=datetime.now().isoformat(),
                    cause=f"Corroboration depth ({health.corroboration_depth:.2f}) below minimum ({self.corroboration_min})",
                    recommended_action="Gather additional evidence sources",
                )
            )

        # Opportunity stall
        if health.last_validation:
            try:
                last_val = datetime.fromisoformat(health.last_validation.replace("Z", "+00:00"))
                days_stale = (datetime.now() - last_val).days
                if days_stale >= self.stall_days:
                    regressions.append(
                        Regression(
                            wedge_id=wedge_id,
                            wedge_title=health.wedge_title,
                            regression_type=RegressionType.OPPORTUNITY_STALL,
                            severity="warning",
                            current_value=days_stale,
                            previous_value=0,
                            change_percent=days_stale * 100,
                            detected_at=datetime.now().isoformat(),
                            cause=f"No validation activity in {days_stale} days",
                            recommended_action="Trigger re-validation or mark as stale",
                        )
                    )
            except (ValueError, TypeError):
                pass

        return regressions

    def generate_report(self) -> SREReport:
        """Generate comprehensive SRE report for all wedges.

        Returns:
            SREReport with all health metrics and regressions
        """
        # Get all opportunities
        opportunities = self.db.get_all_opportunities()
        wedge_ids = set(o.wedge_id for o in opportunities if o.wedge_id)

        health_map = {}
        regressions = []
        healthy = degraded = critical = 0

        for wedge_id in wedge_ids:
            health = self.check_wedge_health(wedge_id)
            health_map[wedge_id] = health

            if health.status == HealthStatus.HEALTHY:
                healthy += 1
            elif health.status == HealthStatus.DEGRADED:
                degraded += 1
            elif health.status == HealthStatus.CRITICAL:
                critical += 1

            # Check for regressions
            wedge_regressions = self.detect_regressions(wedge_id)
            regressions.extend(wedge_regressions)

        report = SREReport(
            generated_at=time.time(),
            wedges_checked=len(wedge_ids),
            healthy_count=healthy,
            degraded_count=degraded,
            critical_count=critical,
            regressions=regressions,
            wedge_health=health_map,
        )

        report.summary = self._format_summary(report)
        return report

    def _get_previous_score(self, wedge_id: int) -> float:
        """Get previous score from historical data."""
        # Would need to track in DB - for now return current
        # TODO: Implement score history table
        return 0.0

    def _get_corroboration_depth(self, wedge_id: int) -> float:
        """Calculate corroboration depth for a wedge."""
        conn = self.db._get_connection()
        row = conn.execute(
            """
            SELECT AVG(corroboration_count) as avg_corr
            FROM opportunities
            WHERE wedge_id = ?
            """,
            (wedge_id,),
        ).fetchone()

        if row and row[0]:
            return float(row[0])
        return 0.0

    def _determine_health(
        self,
        current_score: float,
        previous_score: float,
        validation_count: int,
        corroboration_depth: float,
    ) -> HealthStatus:
        """Determine health status from metrics."""
        if validation_count == 0:
            return HealthStatus.UNKNOWN

        if current_score >= 0.7 and corroboration_depth >= self.corroboration_min:
            return HealthStatus.HEALTHY

        if current_score >= 0.4 or corroboration_depth >= 0.2:
            return HealthStatus.DEGRADED

        return HealthStatus.CRITICAL

    def _generate_issues(
        self,
        current_score: float,
        previous_score: float,
        validation_count: int,
        corroboration_depth: float,
    ) -> list[str]:
        """Generate list of issues."""
        issues = []

        if validation_count == 0:
            issues.append("No validations recorded")

        if current_score < 0.4:
            issues.append(f"Low score: {current_score:.2f}")

        if corroboration_depth < self.corroboration_min:
            issues.append(f"Weak corroboration: {corroboration_depth:.2f}")

        if previous_score > 0 and current_score < previous_score:
            issues.append(f"Score dropped from {previous_score:.2f} to {current_score:.2f}")

        return issues

    def _generate_recommendations(
        self,
        current_score: float,
        previous_score: float,
        validation_count: int,
        corroboration_depth: float,
    ) -> list[str]:
        """Generate recommendations."""
        recs = []

        if validation_count == 0:
            recs.append("Trigger initial validation")

        if current_score < 0.4:
            recs.append("Re-gather evidence with new sources")

        if corroboration_depth < self.corroboration_min:
            recs.append("Add more corroboration sources")

        if previous_score > current_score:
            recs.append("Review recent changes that may have caused score drop")

        return recs

    def _format_summary(self, report: SREReport) -> str:
        """Format SRE report summary."""
        lines = [
            "# Wedge Health Report",
            "",
            f"Wedges checked: {report.wedges_checked}",
            f"  ✅ Healthy: {report.healthy_count}",
            f"  ⚠️  Degraded: {report.degraded_count}",
            f"  ❌ Critical: {report.critical_count}",
            "",
        ]

        if report.regressions:
            lines.append(f"## Regressions Detected: {len(report.regressions)}")
            lines.append("")

            # Group by severity
            critical = [r for r in report.regressions if r.severity == "critical"]
            warning = [r for r in report.regressions if r.severity == "warning"]

            if critical:
                lines.append("### Critical")
                for r in critical:
                    lines.append(f"- 🔴 Wedge {r.wedge_id}: {r.regression_type.value} ({r.change_percent:.1f}%)")
                    lines.append(f"  Cause: {r.cause}")

            if warning:
                lines.append("### Warning")
                for r in warning:
                    lines.append(f"- ⚠️ Wedge {r.wedge_id}: {r.regression_type.value} ({r.change_percent:.1f}%)")
                    lines.append(f"  Cause: {r.cause}")

        # Health by wedge
        lines.append("")
        lines.append("## Wedge Status")
        for wedge_id, health in report.wedge_health.items():
            icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.CRITICAL: "❌",
                HealthStatus.UNKNOWN: "❓",
            }.get(health.status, "?")

            lines.append(f"{icon} Wedge {wedge_id}: {health.status.value} (score: {health.score:.2f})")

        return "\n".join(lines)

    def format_wedge_status(self, wedge_id: int) -> str:
        """Format status for a specific wedge."""
        health = self.check_wedge_health(wedge_id)
        regressions = self.detect_regressions(wedge_id)

        lines = [
            f"# Wedge {wedge_id} Status",
            "",
            f"Title: {health.wedge_title}",
            f"Status: {health.status.value}",
            "",
            f"Score: {health.score:.2f} (was {health.previous_score:.2f})",
            f"Validations: {health.validation_count}",
            f"Atoms: {health.atom_count}",
            f"Corroboration: {health.corroboration_depth:.2f}",
            "",
        ]

        if health.issues:
            lines.append("## Issues")
            for issue in health.issues:
                lines.append(f"- {issue}")
            lines.append("")

        if health.recommendations:
            lines.append("## Recommendations")
            for rec in health.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        if regressions:
            lines.append("## Regressions")
            for reg in regressions:
                lines.append(f"- {reg.severity.upper()}: {reg.regression_type.value}")
                lines.append(f"  {reg.cause}")
                lines.append(f"  Action: {reg.recommended_action}")
            lines.append("")

        return "\n".join(lines)