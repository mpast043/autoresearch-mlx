"""SRE agent for opportunity health monitoring and regression detection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
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
class OpportunityHealth:
    """Health metrics for an opportunity."""

    opportunity_id: int
    cluster_id: int
    title: str
    status: HealthStatus
    score: float
    previous_score: float
    validation_count: int
    last_validation: str
    atom_count: int
    corroboration_depth: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class Regression:
    """Detected regression in an opportunity."""

    opportunity_id: int
    title: str
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
    opportunities_checked: int
    healthy_count: int
    degraded_count: int
    critical_count: int
    regressions: list[Regression] = field(default_factory=list)
    opportunity_health: dict[int, OpportunityHealth] = field(default_factory=dict)
    summary: str = ""


class SREAgent(BaseAgent):
    """SRE agent for monitoring opportunity health and detecting regressions.

    Tracks opportunity metrics over time, detects score decay and validation
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
        opportunity_id = payload.get("opportunity_id")

        if opportunity_id:
            health = self.check_opportunity_health(opportunity_id)
            regressions = self.detect_regressions(opportunity_id)
            return {"health": health, "regressions": regressions, "formatted": self.format_opportunity_status(opportunity_id)}
        else:
            report = self.generate_report()
            return {"report": report, "summary": report.summary}

    def check_opportunity_health(self, opportunity_id: int) -> OpportunityHealth:
        """Check health metrics for a single opportunity.

        Args:
            opportunity_id: Opportunity ID to check

        Returns:
            OpportunityHealth with current metrics
        """
        opp = self.db.get_opportunity(opportunity_id)
        if not opp:
            return OpportunityHealth(
                opportunity_id=opportunity_id,
                cluster_id=0,
                title="Unknown",
                status=HealthStatus.UNKNOWN,
                score=0.0,
                previous_score=0.0,
                validation_count=0,
                last_validation="",
                atom_count=0,
                corroboration_depth=0.0,
                issues=["Opportunity not found"],
            )

        current_score = opp.composite_score or 0.0
        previous_score = self._get_previous_score(opportunity_id)

        # Count validations via corroborations for this opportunity's cluster
        validation_count = 0
        corroboration_depth = 0.0
        try:
            corrs = self.db.get_corroborations(cluster_id=opp.cluster_id, limit=100)
            validation_count = len(corrs)
            if corrs:
                depths = [c.corroboration_depth for c in corrs if hasattr(c, "corroboration_depth") and c.corroboration_depth]
                corroboration_depth = sum(depths) / len(depths) if depths else 0.0
        except Exception:
            pass

        # Count atoms in the same cluster
        atom_count = 0
        try:
            atoms = self.db.get_problem_atoms(limit=500)
            atom_count = len([a for a in atoms if getattr(a, "cluster_key", None) == str(opp.cluster_id)])
        except Exception:
            pass

        # Last validation time
        last_validation = opp.evaluated_at or opp.last_rescored_at or ""

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

        return OpportunityHealth(
            opportunity_id=opp.id or opportunity_id,
            cluster_id=opp.cluster_id,
            title=opp.title or "Unknown",
            status=status,
            score=current_score,
            previous_score=previous_score,
            validation_count=validation_count,
            last_validation=last_validation,
            atom_count=atom_count,
            corroboration_depth=corroboration_depth,
            issues=issues,
            recommendations=recommendations,
        )

    def detect_regressions(self, opportunity_id: int) -> list[Regression]:
        """Detect regressions for an opportunity by comparing to historical data.

        Args:
            opportunity_id: Opportunity ID to check

        Returns:
            List of detected regressions
        """
        regressions = []
        health = self.check_opportunity_health(opportunity_id)

        # Score decay
        if health.previous_score > 0:
            score_change = (health.score - health.previous_score) / health.previous_score
            if score_change <= -self.score_drop_threshold:
                regressions.append(
                    Regression(
                        opportunity_id=opportunity_id,
                        title=health.title,
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

        # Corroboration loss
        if health.corroboration_depth < self.corroboration_min:
            regressions.append(
                Regression(
                    opportunity_id=opportunity_id,
                    title=health.title,
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
                            opportunity_id=opportunity_id,
                            title=health.title,
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
        """Generate comprehensive SRE report for all opportunities.

        Returns:
            SREReport with all health metrics and regressions
        """
        # Get opportunities using the actual Database API
        opportunities = self.db.get_opportunities(limit=500)

        health_map = {}
        regressions = []
        healthy = degraded = critical = 0

        for opp in opportunities:
            if not opp.id:
                continue
            health = self.check_opportunity_health(opp.id)

            health_map[opp.id] = health

            if health.status == HealthStatus.HEALTHY:
                healthy += 1
            elif health.status == HealthStatus.DEGRADED:
                degraded += 1
            elif health.status == HealthStatus.CRITICAL:
                critical += 1

            # Check for regressions
            opp_regressions = self.detect_regressions(opp.id)
            regressions.extend(opp_regressions)

        report = SREReport(
            generated_at=time.time(),
            opportunities_checked=len(opportunities),
            healthy_count=healthy,
            degraded_count=degraded,
            critical_count=critical,
            regressions=regressions,
            opportunity_health=health_map,
        )

        report.summary = self._format_summary(report)
        return report

    def _get_previous_score(self, opportunity_id: int) -> float:
        """Get previous score from historical data."""
        # Would need to track in DB - for now return 0
        # TODO: Implement score history table
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
            "# Opportunity Health Report",
            "",
            f"Opportunities checked: {report.opportunities_checked}",
            f"  Healthy: {report.healthy_count}",
            f"  Degraded: {report.degraded_count}",
            f"  Critical: {report.critical_count}",
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
                    lines.append(f"- Opportunity {r.opportunity_id}: {r.regression_type.value} ({r.change_percent:.1f}%)")
                    lines.append(f"  Cause: {r.cause}")

            if warning:
                lines.append("### Warning")
                for r in warning:
                    lines.append(f"- Opportunity {r.opportunity_id}: {r.regression_type.value} ({r.change_percent:.1f}%)")
                    lines.append(f"  Cause: {r.cause}")

        # Health by opportunity
        lines.append("")
        lines.append("## Opportunity Status")
        for opp_id, health in report.opportunity_health.items():
            icon = {
                HealthStatus.HEALTHY: "OK",
                HealthStatus.DEGRADED: "WARN",
                HealthStatus.CRITICAL: "CRIT",
                HealthStatus.UNKNOWN: "???",
            }.get(health.status, "?")

            lines.append(f"[{icon}] Opportunity {opp_id}: {health.status.value} (score: {health.score:.2f})")

        return "\n".join(lines)

    def format_opportunity_status(self, opportunity_id: int) -> str:
        """Format status for a specific opportunity."""
        health = self.check_opportunity_health(opportunity_id)
        regressions = self.detect_regressions(opportunity_id)

        lines = [
            f"# Opportunity {opportunity_id} Status",
            "",
            f"Title: {health.title}",
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