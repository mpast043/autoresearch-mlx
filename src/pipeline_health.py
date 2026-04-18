"""One-screen explanation of why discovery/validation may appear idle."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.database import Database


def compute_pipeline_health(db: Database) -> dict[str, Any]:
    """Summarize finding lifecycle + whether ``run_once`` has work to do."""
    findings = db.get_findings(limit=5000)
    by_status = Counter((f.status or "") for f in findings)

    actionable = len(db.get_backlog_workbench(limit=5000)) if hasattr(db, "get_backlog_workbench") else 0

    n_val = len(db.get_validation_review(limit=10000, run_id=None))

    blockers: list[str] = []
    if actionable == 0:
        blockers.append(
            "No qualified pain_signal findings with raw_signal + problem_atom — run_once will not dispatch evidence. "
            "Either discovery is deduping against existing content hashes, or all findings are already parked/killed/screened_out."
        )
    hints: list[str] = []
    if by_status.get("qualified", 0) == 0 and len(findings) > 0:
        hints.append(
            "All findings are in terminal states (parked/killed/screened_out). To get fresh runs, add new sources/queries "
            "or use a fresh DB for development."
        )
    if actionable == 0 and by_status.get("qualified", 0) == 0:
        hints.append(
            "Content hash dedup now allows re-discovery of screened-out findings. "
            "Seed cache auto-bypasses when pipeline is idle. If still stuck, try: python cli.py run-once --fresh"
        )

    return {
        "finding_count": len(findings),
        "findings_by_status": dict(by_status),
        "actionable_qualified_for_pipeline": actionable,
        "validation_rows_total": n_val,
        "interpretation": {
            "run_once_will_process_backlog": actionable > 0,
            "likely_blockers": blockers,
            "hints": hints,
        },
    }
