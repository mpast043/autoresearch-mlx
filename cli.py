"""CLI for the evidence-first discovery and validation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

from src.runtime.paths import DEFAULT_CONFIG_PATH, resolve_project_path  # noqa: E402
from run import AutoResearcher  # noqa: E402
from src.behavior_eval import run_behavior_eval  # noqa: E402
from src.database import ReviewFeedback  # noqa: E402
from src.reddit_relay import run_relay_server  # noqa: E402
from src.reddit_seed import RedditSeeder  # noqa: E402


def run_backup_db(config_path: str | Path) -> dict:
    """Copy SQLite DB to data/backups/ (see docs/RECOVERY.md)."""
    import shutil
    from datetime import datetime

    import yaml

    from src.runtime.env import load_local_env

    load_local_env()
    cfg_file = resolve_project_path(config_path, default=DEFAULT_CONFIG_PATH)
    with cfg_file.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    rel = (cfg.get("database") or {}).get("path", "data/autoresearch.db")
    db_path = resolve_project_path(rel, default=Path(rel))
    if not db_path.exists():
        return {"ok": False, "error": f"database not found: {db_path}"}
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = backup_dir / f"autoresearch-{stamp}.db"
    shutil.copy2(db_path, dest)
    return {"ok": True, "from": str(db_path.resolve()), "to": str(dest.resolve())}


async def run_check_bridge(config_path: str | Path) -> dict:
    """Call the hosted relay `/api/health` using `reddit_bridge` config (validates URL + token)."""
    import yaml

    from src.runtime.env import load_local_env

    load_local_env()
    cfg_file = resolve_project_path(config_path, default=DEFAULT_CONFIG_PATH)
    with cfg_file.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    bridge_cfg = dict(cfg.get("reddit_bridge") or {})
    from src.reddit_bridge import BridgeError, RedditBridgeClient

    client = RedditBridgeClient(bridge_cfg)
    try:
        if not client.enabled:
            return {
                "ok": False,
                "error": "reddit_bridge disabled or base_url empty after env resolution",
                "hint": "Set REDDIT_BRIDGE_BASE_URL and enable reddit_bridge in config.yaml",
            }
        health = await client.health()
        return {
            "ok": True,
            "base_url": client.base_url,
            "token_configured": bool(client.auth_token),
            "health": health,
        }
    except BridgeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "code": exc.code,
            "base_url": getattr(client, "base_url", ""),
            "hint": "If code is auth_failed, REDDIT_BRIDGE_AUTH_TOKEN must match the relay token on Render.",
        }
    finally:
        await client.close()


def print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_discovery_sort_diagnostics(db, *, limit: int = 500, run_id: str = "") -> dict:
    """Summarize which Reddit sort modes are yielding findings."""
    findings = db.get_findings(limit=limit) if db else []
    sort_counts: dict[str, int] = collections.Counter()
    sort_status_counts: dict[str, dict[str, int]] = {}
    sub_counts: dict[str, int] = collections.Counter()
    examined = 0

    for finding in findings:
        source = str(getattr(finding, "source", "") or "")
        if not source.startswith("reddit-problem/"):
            continue
        evidence = getattr(finding, "evidence", None) or {}
        if run_id and str(evidence.get("run_id", "")) != run_id:
            continue
        sort_mode = str(evidence.get("discovery_sort", "unknown") or "unknown").lower()
        status = str(getattr(finding, "status", "") or "unknown")
        subreddit = source.split("/", 1)[1] if "/" in source else "unknown"
        examined += 1
        sort_counts[sort_mode] += 1
        sub_counts[subreddit] += 1
        bucket = sort_status_counts.setdefault(sort_mode, {})
        bucket[status] = int(bucket.get(status, 0)) + 1

    return {
        "rows_examined": examined,
        "sort_counts": dict(sorted(sort_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sort_status_counts": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in sort_status_counts.items()},
        "top_subreddits": [
            {"subreddit": subreddit, "count": count}
            for subreddit, count in sorted(sub_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        ],
        "hint": "Use discovery.reddit.search_sorts/per_sort_limit/max_docs_per_pair to tune sort coverage.",
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def build_source_health_summary(db, *, limit: int = 8) -> dict:
    if not db or not hasattr(db, "get_discovery_feedback"):
        return {}

    rows = db.get_discovery_feedback() or []
    if not rows:
        return {"rows": 0, "source_count": 0, "status_mix": {}, "top_sources": []}

    status_mix: dict[str, int] = collections.Counter()
    by_source: dict[str, dict] = {}
    for row in rows:
        source_name = str(row.get("source_name") or "unknown")
        bucket = by_source.setdefault(
            source_name,
            {
                "source_name": source_name,
                "queries": 0,
                "runs": 0,
                "docs_seen": 0,
                "findings_emitted": 0,
                "validations": 0,
                "passes": 0,
                "prototype_candidates": 0,
                "build_briefs": 0,
                "_latency_total": 0.0,
                "_latency_samples": 0,
                "_status_counts": collections.Counter(),
            },
        )
        bucket["queries"] += 1
        bucket["runs"] += int(row.get("runs") or 0)
        bucket["docs_seen"] += int(row.get("docs_seen") or 0)
        bucket["findings_emitted"] += int(row.get("findings_emitted") or 0)
        bucket["validations"] += int(row.get("validations") or 0)
        bucket["passes"] += int(row.get("passes") or 0)
        bucket["prototype_candidates"] += int(row.get("prototype_candidates") or 0)
        bucket["build_briefs"] += int(row.get("build_briefs") or 0)
        latency_ms = float(row.get("last_latency_ms") or 0.0)
        if latency_ms > 0:
            bucket["_latency_total"] += latency_ms
            bucket["_latency_samples"] += 1
        last_status = str(row.get("last_status") or "unknown")
        status_mix[last_status] = status_mix.get(last_status, 0) + 1
        bucket["_status_counts"][last_status] += 1

    top_sources = []
    for bucket in by_source.values():
        status_counts = bucket["_status_counts"]
        dominant_status = max(status_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if status_counts else "unknown"
        top_sources.append(
            {
                "source_name": bucket["source_name"],
                "queries": bucket["queries"],
                "runs": bucket["runs"],
                "docs_seen": bucket["docs_seen"],
                "findings_emitted": bucket["findings_emitted"],
                "validations": bucket["validations"],
                "passes": bucket["passes"],
                "prototype_candidates": bucket["prototype_candidates"],
                "build_briefs": bucket["build_briefs"],
                "yield_per_run": _safe_ratio(bucket["findings_emitted"], bucket["runs"]),
                "pass_rate": _safe_ratio(bucket["passes"], bucket["validations"]),
                "avg_latency_ms": round(bucket["_latency_total"] / bucket["_latency_samples"], 2) if bucket["_latency_samples"] else 0.0,
                "dominant_status": dominant_status,
                "status_mix": dict(sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        )
    top_sources.sort(
        key=lambda item: (
            -int(item["build_briefs"]),
            -int(item["prototype_candidates"]),
            -int(item["passes"]),
            -int(item["findings_emitted"]),
            item["source_name"],
        )
    )
    return {
        "rows": len(rows),
        "source_count": len(by_source),
        "status_mix": dict(sorted(status_mix.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_sources": top_sources[:limit],
    }


def build_builder_jobs_view(db, *, run_id: str = "", limit: int = 25) -> list[dict]:
    if not db or not hasattr(db, "list_build_briefs") or not hasattr(db, "list_build_prep_outputs"):
        return []
    briefs = db.list_build_briefs(run_id=run_id or None, limit=max(limit * 4, 100))
    prep_outputs = db.list_build_prep_outputs(run_id=run_id or None, limit=max(limit * 8, 200))
    prep_by_brief: dict[int, list] = {}
    for item in prep_outputs:
        prep_by_brief.setdefault(int(getattr(item, "build_brief_id", 0) or 0), []).append(item)

    state_map = {
        "prototype_candidate": "queued",
        "prototype_ready": "scoped",
        "build_ready": "ready_to_build",
        "launched": "shipped",
        "iterate": "reviewed",
        "expand": "shipped",
        "archive": "failed",
    }
    rows: list[dict] = []
    for brief in briefs:
        outputs = prep_by_brief.get(int(getattr(brief, "id", 0) or 0), [])
        output_statuses = collections.Counter(str(getattr(item, "status", "") or "") for item in outputs)
        output_stages = [str(getattr(item, "prep_stage", "") or "") for item in outputs]
        rows.append(
            {
                "build_brief_id": int(getattr(brief, "id", 0) or 0),
                "run_id": getattr(brief, "run_id", "") or "",
                "opportunity_id": int(getattr(brief, "opportunity_id", 0) or 0),
                "validation_id": int(getattr(brief, "validation_id", 0) or 0),
                "recommended_output_type": getattr(brief, "recommended_output_type", "") or "",
                "build_brief_status": getattr(brief, "status", "") or "",
                "builder_status": state_map.get(str(getattr(brief, "status", "") or ""), "queued"),
                "ready_for_build": str(getattr(brief, "status", "") or "") == "build_ready",
                "prep_output_count": len(outputs),
                "prep_stage_status_mix": dict(sorted(output_statuses.items(), key=lambda kv: (-kv[1], kv[0]))),
                "prep_stages": sorted([stage for stage in output_stages if stage]),
                "updated_at": getattr(brief, "updated_at", None),
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item["ready_for_build"] else 1,
            item["builder_status"],
            -(item["build_brief_id"] or 0),
        )
    )
    return rows[:limit]


def build_operator_report(app: AutoResearcher, *, limit: int = 10) -> dict:
    from src.pipeline_health import compute_pipeline_health

    if not app.db:
        return {
            "runtime": app.runtime_paths(),
            "current_run_id": app.current_run_id,
            "pipeline_health": {},
            "source_health": {},
            "money_surface": {},
            "operator_focus": {"recommended_focus": "initialize_pipeline", "primary_blocker": "database_unavailable", "notes": []},
            "recent_logs": app.status_tracker.status.get("logs", [])[-12:],
        }

    decision_surface = app.db.get_candidate_workbench(limit=max(limit * 3, 25), run_id=app.current_run_id) if hasattr(app.db, "get_candidate_workbench") else []
    builder_jobs = build_builder_jobs_view(app.db, run_id=app.current_run_id, limit=max(limit * 3, 25))
    validation_rows = app.validation_report(limit=1000)
    action_mix = collections.Counter(
        str(item.get("next_recommended_action") or "")
        for item in decision_surface
        if str(item.get("next_recommended_action") or "")
    )
    builder_status_mix = collections.Counter(
        str(item.get("builder_status") or "")
        for item in builder_jobs
        if str(item.get("builder_status") or "")
    )
    prototype_now_count = sum(1 for item in decision_surface if item.get("next_recommended_action") == "prototype_now")
    build_ready_count = sum(1 for item in builder_jobs if item.get("ready_for_build"))

    pipeline_health = compute_pipeline_health(app.db)
    source_health = build_source_health_summary(app.db, limit=limit)
    blockers = list((pipeline_health.get("interpretation") or {}).get("likely_blockers") or [])
    hints = list((pipeline_health.get("interpretation") or {}).get("hints") or [])
    notes = [*blockers[:1], *hints[:2]]

    if prototype_now_count > 0:
        recommended_focus = "prototype_now"
    elif build_ready_count > 0:
        recommended_focus = "prepare_build_queue"
    elif pipeline_health.get("actionable_qualified_for_pipeline", 0) > 0:
        recommended_focus = "continue_validation"
    else:
        recommended_focus = "refresh_discovery"

    return {
        "runtime": app.runtime_paths(),
        "current_run_id": app.current_run_id,
        "pipeline_health": pipeline_health,
        "artifact_coverage": {
            "validations_in_run": len(validation_rows),
            "build_briefs": len(app.db.list_build_briefs(run_id=app.current_run_id, limit=1000)) if hasattr(app.db, "list_build_briefs") else 0,
            "build_prep_outputs": len(app.db.list_build_prep_outputs(run_id=app.current_run_id, limit=1000)) if hasattr(app.db, "list_build_prep_outputs") else 0,
            "builder_jobs": len(builder_jobs),
        },
        "source_health": source_health,
        "money_surface": {
            "prototype_now_count": prototype_now_count,
            "build_ready_count": build_ready_count,
            "action_mix": dict(sorted(action_mix.items(), key=lambda kv: (-kv[1], kv[0]))),
            "builder_job_status_mix": dict(sorted(builder_status_mix.items(), key=lambda kv: (-kv[1], kv[0]))),
            "top_ranked_opportunities": decision_surface[:limit],
            "build_queue": builder_jobs[:limit],
        },
        "operator_focus": {
            "recommended_focus": recommended_focus,
            "primary_blocker": blockers[0] if blockers else "",
            "notes": notes[:4],
        },
        "recent_logs": app.status_tracker.status.get("logs", [])[-12:],
    }


def build_verbose_report(app: AutoResearcher, summary: dict) -> dict:
    return {
        "runtime": app.runtime_paths(),
        "counts": app.summary_counts(),
        "decisions": app.decision_summary(),
        "reddit_runtime": app.reddit_runtime_summary(),
        "screening": app.db.get_finding_status_counts(run_id=app.current_run_id) if app.db else {},
        "screening_all_time": app.db.get_finding_status_counts() if app.db else {},
        "actionable_screening": app.db.get_finding_status_counts(run_id=app.current_run_id, actionable_only=True) if app.db else {},
        "screening_summary": app.db.get_screening_summary(limit=10, run_id=app.current_run_id) if app.db else {},
        "validation": app.validation_report(limit=10),
        "run_diff": app.run_diff(limit=10),
        "build_prep": {
            "build_briefs": [item.__dict__ for item in app.db.list_build_briefs(run_id=app.current_run_id, limit=10)],
            "outputs": [item.__dict__ for item in app.db.list_build_prep_outputs(run_id=app.current_run_id, limit=20)],
        } if app.db and hasattr(app.db, "list_build_briefs") and hasattr(app.db, "list_build_prep_outputs") else {},
        "candidate_workbench": app.db.get_candidate_workbench(limit=10, run_id=app.current_run_id)
        if app.db and hasattr(app.db, "get_candidate_workbench")
        else [],
        "decision_surface": app.db.get_candidate_workbench(limit=10, run_id=app.current_run_id)
        if app.db and hasattr(app.db, "get_candidate_workbench")
        else [],
        "builder_jobs": build_builder_jobs_view(app.db, run_id=app.current_run_id, limit=10),
        "operator_report": build_operator_report(app, limit=10),
        "review": app.review_report(limit=10),
        "recent_logs": app.status_tracker.status.get("logs", [])[-12:],
        "summary": summary,
    }


def render_watch_snapshot(status: dict, runtime_paths: dict[str, str]) -> str:
    lines = [
        f"stage={status.get('stage', 'idle')}",
        f"status={status.get('status', 'idle')}",
        f"discoveries={status.get('discoveries', 0)}",
        f"raw_signals={status.get('rawSignals', 0)}",
        f"runtime={runtime_paths}",
    ]
    logs = status.get("logs", [])[-8:]
    if logs:
        lines.append("recent_logs:")
        lines.extend(f"- {entry}" for entry in logs)
    return "\n".join(lines)


async def watch_status(interval: float = 1.0, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    app = AutoResearcher(config_path=config_path)
    await app.initialize(start_new_run=False)
    try:
        while True:
            snapshot = app.status_tracker.status
            print(render_watch_snapshot(snapshot, app.runtime_paths()))
            await asyncio.sleep(interval)
    finally:
        await app.shutdown()


async def main() -> None:
    parser = argparse.ArgumentParser("AutoResearcher CLI")
    parser.add_argument(
        "command",
        choices=[
            "run",
            "run-once",
            "run-unseeded",
            "deep-research",
            "watch",
            "search",
            "signals",
            "atoms",
            "clusters",
            "opportunities",
            "experiments",
            "ledger",
            "findings",
            "build-briefs",
            "build-prep",
            "workbench",
            "decision-surface",
            "operator-report",
            "builder-jobs",
            "report",
            "gate-diagnostics",
            "pipeline-health",
            "discovery-sort-diagnostics",
            "eval",
            "review-queue",
            "review-mark",
            "ideas",
            "products",
            "reddit-relay",
            "reddit-seed",
            "check-bridge",
            "backup-db",
            "suggest-discovery",
            "patterns",
            "scoring-report",
            "revalidate",
        ],
        help="Command to execute",
    )
    parser.add_argument("query", nargs="?", help="Search query for the search command")
    parser.add_argument("--verbose", action="store_true", help="Print operator-facing runtime details")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval for watch mode")
    parser.add_argument("--host", help="Host override for reddit-relay")
    parser.add_argument("--port", type=int, help="Port override for reddit-relay")
    parser.add_argument("--eval-path", default="evals/behavior_gold.json", help="Path to behavior eval fixture set")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to runtime config")
    parser.add_argument("--finding-id", type=int, help="Finding id for review-mark")
    parser.add_argument("--cluster-id", type=int, help="Cluster id for review-mark")
    parser.add_argument("--opportunity-id", type=int, help="Opportunity id for review-mark")
    parser.add_argument("--validation-id", type=int, help="Validation id for review-mark")
    parser.add_argument("--vertical", default="devtools", help="Vertical for unseeded/run-unseeded")
    parser.add_argument("--max-findings", type=int, default=20, help="Max raw signals for unseeded/run-unseeded")
    parser.add_argument("--label", default="", help="Review label for review-mark")
    parser.add_argument("--note", default="", help="Optional review note")
    parser.add_argument("--run-id", default="", help="Scope gate-diagnostics to a specific pipeline run id")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="gate-diagnostics: max validation rows; suggest-discovery: max clusters scanned (default: 25)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="suggest-discovery: only clusters with at least this many atoms (default: 2)",
    )
    parser.add_argument(
        "--money-claims-min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="suggest-discovery: minimum confidence tier for money claims (default: low)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="run/run-once: focus discovery on specific pattern (e.g., spreadsheet_versioning, bank_reconciliation)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="run-once: bypass signal cache, force fresh discovery (ignores cached results)",
    )
    args = parser.parse_args()

    if args.command == "watch":
        await watch_status(interval=args.interval, config_path=args.config)
        return

    if args.command == "eval":
        print_json(run_behavior_eval(str(resolve_project_path(args.eval_path))))
        return

    if args.command == "reddit-relay":
        app = AutoResearcher(config_path=args.config)
        host = args.host or app.config.get("reddit_relay", {}).get("host", "127.0.0.1")
        port = args.port or int(app.config.get("reddit_relay", {}).get("port", 8787))
        await run_relay_server(app.config, host=host, port=port)
        return

    if args.command == "reddit-seed":
        app = AutoResearcher(config_path=args.config)
        await app.initialize(start_new_run=False)
        try:
            seeder = RedditSeeder(config=app.config)
            summary = await seeder.seed()
            print_json(summary.__dict__ if hasattr(summary, "__dict__") else summary)
        finally:
            await app.shutdown()
        return

    if args.command == "check-bridge":
        print_json(await run_check_bridge(args.config))
        return

    if args.command == "backup-db":
        print_json(run_backup_db(args.config))
        return

    if args.command == "suggest-discovery":
        app = AutoResearcher(config_path=args.config)
        await app.initialize(start_new_run=False)
        try:
            from src.discovery_suggestions import build_discovery_suggestions

            print_json(
                build_discovery_suggestions(
                    app.db,
                    min_atoms=args.min_atoms,
                    limit_clusters=args.limit,
                    limit_atoms=max(args.limit * 4, 40),
                    limit_findings=max(args.limit * 16, 200),
                    max_keywords=min(28, max(args.limit, 12)),
                    theme_keywords=((app.config.get("discovery", {}) or {}).get("reddit", {}) or {}).get(
                        "theme_keywords", {}
                    ),
                    money_claim_min_confidence=args.money_claims_min_confidence,
                )
            )
        finally:
            await app.shutdown()
        return

    if args.command == "patterns":
        # Show emerging specific patterns from signals
        from src.opportunity_engine import get_patterns_for_discovery

        db_path = "data/autoresearch.db"
        patterns = get_patterns_for_discovery(db_path, min_atoms=1)

        print("=== EMERGING PATTERNS ===")
        print("Specific problems detected from signals:\n")
        for p in patterns:
            print(f"  {p['pattern']}")
            print(f"    Signals: {p['signal_count']}, Urgency: {p['urgency']}")
            if 'tools' in p:
                print(f"    Tools: {p['tools']}")
            print()

        if not patterns:
            print("  No specific patterns detected yet.")
            print("  Run more discovery to identify specific integration problems.")
        return

    if args.command == "scoring-report":
        # Show scoring percentile monitor
        import sqlite3

        db_path = Path("data/autoresearch.db")
        if not db_path.exists():
            print("No database found at data/autoresearch.db")
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get scores
        cursor.execute("SELECT composite_score, frequency_score, evidence_quality, education_burden, adoption_friction FROM opportunities ORDER BY composite_score DESC")
        rows = cursor.fetchall()
        scores = [r[0] for r in rows]
        total = len(scores)

        if total == 0:
            print("No opportunities in database")
            return

        sorted_scores = sorted(scores)
        p99 = sorted_scores[int(total * 0.99)] if total > 0 else 0
        p95 = sorted_scores[int(total * 0.95)] if total > 0 else 0
        p90 = sorted_scores[int(total * 0.90)] if total > 0 else 0
        median = sorted_scores[int(total * 0.50)] if total > 0 else 0

        above_50 = sum(1 for s in scores if s >= 0.50)
        above_45 = sum(1 for s in scores if s >= 0.45)
        above_40 = sum(1 for s in scores if s >= 0.40)
        above_35 = sum(1 for s in scores if s >= 0.35)

        print("=== SCORING PERCENTILE MONITOR ===\n")
        print(f"Total opportunities: {total}\n")
        print("PERCENTILES:")
        print(f"  Max:   {max(scores):.4f}")
        print(f"  P99:   {p99:.4f}")
        print(f"  P95:   {p95:.4f}")
        print(f"  P90:   {p90:.4f}")
        print(f"  Median:{median:.4f}")
        print(f"\nDECISION BUCKETS (thresholds: promote=0.50, park=0.35):")
        print(f"  Promote (≥0.50): {above_50} ({above_50/total*100:.1f}%)")
        print(f"  Park (0.35-0.50): {above_35-above_50} ({(above_35-above_50)/total*100:.1f}%)")
        print(f"  Kill (<0.35): {total-above_35} ({(total-above_35)/total*100:.1f}%)")

        # Get parked analysis
        cursor.execute("""
            SELECT
                AVG(frequency_score) as avg_freq,
                AVG(evidence_quality) as avg_eq,
                AVG(education_burden) as avg_edu,
                AVG(adoption_friction) as avg_fric
            FROM opportunities
            WHERE composite_score >= 0.35 AND composite_score < 0.50
        """)
        parked = cursor.fetchone()
        if parked and parked[0]:
            print(f"\nPARKED OPPORTUNITY ANALYSIS (n={above_35-above_50}):")
            print(f"  Avg frequency:        {parked[0]:.3f}")
            print(f"  Avg evidence_quality: {parked[1]:.3f}")
            print(f"  Avg education_burden: {parked[2]:.3f} ⚠️")
            print(f"  Avg adoption_friction:{parked[3]:.3f} ⚠️")

        conn.close()
        return

    if args.command == "revalidate":
        # Re-run validation for all opportunities with new formula (v2)
        import sqlite3

        db_path = Path("data/autoresearch.db")
        if not db_path.exists():
            print("No database found")
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get unvalidated opportunities (scoring_version = '0')
        cursor.execute("""
            SELECT o.id, o.cluster_id
            FROM opportunities o
            WHERE o.scoring_version = '0'
        """)
        unvalidated_opps = cursor.fetchall()

        if not unvalidated_opps:
            print("No opportunities need revalidation.")
            print("All have been scored with current formula.")
            conn.close()
            return

        print(f"=== REVALIDATING {len(unvalidated_opps)} OPPORTUNITIES ===")
        print("This will re-run validation with v2 formula (new weights/thresholds)")
        print("Each opportunity will be marked with scoring_version='v2' after revalidation.\n")

        # For now, just show what would happen
        # Actual revalidation would require running the full pipeline

        for opp_id, cluster_id in unvalidated_opps[:5]:
            print(f"  Would revalidate opportunity #{opp_id} (cluster={cluster_id})")

        print(f"\n  ... and {len(unvalidated_opps) - 5} more")

        # Check current status
        cursor.execute("SELECT COUNT(*), COUNT(CASE WHEN scoring_version = '0' THEN 1 END), COUNT(CASE WHEN scoring_version = 'v2' THEN 1 END) FROM opportunities")
        total, legacy, current = cursor.fetchone()

        print(f"\n=== CURRENT STATUS ===")
        print(f"  Total opportunities: {total}")
        print(f"  Not validated (needs revalidate): {legacy}")
        print(f"  v2 (current formula): {current}")

        conn.close()
        return

    app = AutoResearcher(config_path=args.config)

    if args.command == "run":
        await app.run()
        return

    # Pattern-based focused discovery
    if args.pattern:
        from src.opportunity_engine import PATTERN_TO_DISCOVERY_QUERIES

        if args.pattern not in PATTERN_TO_DISCOVERY_QUERIES:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available patterns: {list(PATTERN_TO_DISCOVERY_QUERIES.keys())}")
            return

        queries = PATTERN_TO_DISCOVERY_QUERIES[args.pattern]
        print(f"=== FOCUSED DISCOVERY: {args.pattern} ===")
        print(f"Queries: {queries[:3]}...")

        # Temporarily override config with focused queries (both web and reddit)
        original_web = app.config.get("discovery", {}).get("web", {}).get("keywords", [])
        original_reddit = app.config.get("discovery", {}).get("reddit", {}).get("problem_keywords", [])

        if "web" in app.config.get("discovery", {}):
            app.config["discovery"]["web"]["keywords"] = queries

        # Also override reddit problem_keywords for seed queries
        if "reddit" in app.config.get("discovery", {}):
            app.config["discovery"]["reddit"]["problem_keywords"] = queries

        print(f"Running focused discovery with {len(queries)} queries...\n")

    # Set bypass_cache flag if --fresh is specified
    if args.fresh and "discovery" in app.agents:
        print("=== FRESH MODE: Bypassing signal cache ===")
        app.agents["discovery"].bypass_cache = True

    if args.command == "run-once":
        summary = await app.run_once()
        if args.verbose:
            print_json(build_verbose_report(app, summary))
        else:
            print_json(summary)

        # Restore original config after focused run
        if args.pattern:
            if "web" in app.config.get("discovery", {}):
                app.config["discovery"]["web"]["keywords"] = original_web
            print(f"\n=== Focused discovery complete ===")
            print(f"Use 'python3 cli.py patterns' to see updated pattern counts")
        return

    if args.command == "run-unseeded":
        await app.initialize(start_new_run=True)
        try:
            summary = await app.run_unseeded(
                vertical=args.vertical,
                max_findings=args.max_findings,
            )
            print_json(summary)
        finally:
            await app.shutdown()
        return

    if args.command == "deep-research":
        from src.agents.deep_research import DeepResearchAgent
        await app.initialize(start_new_run=True)
        try:
            agent = DeepResearchAgent(
                name="deep_research",
                db=app.db,
                vertical=args.vertical,
            )
            summary = await agent.run_deep_research(
                max_signals_per_source=args.max_findings,
            )
            print_json(summary)
        finally:
            await app.shutdown()
        return

    await app.initialize(start_new_run=False)
    try:
        if args.command == "search":
            if not args.query:
                print("search requires a query", file=sys.stderr)
                sys.exit(1)
            print_json(app.search_opportunities(args.query))
        elif args.command == "signals":
            print_json([item.__dict__ for item in app.db.get_raw_signals(limit=100)] if app.db else [])
        elif args.command == "atoms":
            print_json([item.__dict__ for item in app.db.get_problem_atoms(limit=100)] if app.db else [])
        elif args.command == "clusters":
            print_json([item.__dict__ for item in app.db.get_clusters(limit=100)] if app.db else [])
        elif args.command == "opportunities":
            print_json([item.__dict__ for item in app.db.get_opportunities(limit=100)] if app.db else [])
        elif args.command == "experiments":
            print_json([item.__dict__ for item in app.db.get_experiments(limit=100)] if app.db else [])
        elif args.command == "ledger":
            print_json([item.__dict__ for item in app.db.list_ledger_entries(limit=100)] if app.db else [])
        elif args.command == "findings":
            print_json([item.__dict__ for item in app.db.get_findings(limit=100)] if app.db else [])
        elif args.command == "build-briefs":
            print_json([item.__dict__ for item in app.db.list_build_briefs(limit=100)] if app.db else [])
        elif args.command == "build-prep":
            print_json([item.__dict__ for item in app.db.list_build_prep_outputs(limit=100)] if app.db else [])
        elif args.command == "workbench":
            print_json(app.db.get_candidate_workbench(limit=100, run_id=app.current_run_id) if app.db else [])
        elif args.command == "decision-surface":
            print_json(app.db.get_candidate_workbench(limit=100, run_id=app.current_run_id) if app.db else [])
        elif args.command == "operator-report":
            print_json(build_operator_report(app, limit=min(max(args.limit, 5), 25)))
        elif args.command == "builder-jobs":
            print_json(build_builder_jobs_view(app.db, run_id=app.current_run_id, limit=100) if app.db else [])
        elif args.command == "report":
            print_json(build_verbose_report(app, app.snapshot()))
        elif args.command == "gate-diagnostics":
            from src.gate_diagnostics import build_gate_diagnostics_report

            report = build_gate_diagnostics_report(
                app.db,
                config=app.config,
                run_id=args.run_id or None,
                limit=args.limit,
                finding_id=args.finding_id,
            )
            print_json(report)
        elif args.command == "pipeline-health":
            from src.pipeline_health import compute_pipeline_health

            print_json(compute_pipeline_health(app.db))
        elif args.command == "discovery-sort-diagnostics":
            print_json(
                build_discovery_sort_diagnostics(
                    app.db,
                    limit=max(args.limit * 20, 200),
                    run_id=args.run_id or "",
                )
            )
        elif args.command == "review-queue":
            print_json(app.db.get_review_queue(limit=50, run_id=app.current_run_id) if app.db else [])
        elif args.command == "review-mark":
            if not args.finding_id or not args.label:
                print("review-mark requires --finding-id and --label", file=sys.stderr)
                sys.exit(1)
            review_id = app.db.insert_review_feedback(
                ReviewFeedback(
                    finding_id=args.finding_id,
                    cluster_id=args.cluster_id,
                    opportunity_id=args.opportunity_id,
                    validation_id=args.validation_id,
                    review_label=args.label,
                    note=args.note,
                    run_id=app.current_run_id,
                )
            )
            print_json({"review_id": review_id, "finding_id": args.finding_id, "label": args.label})
        elif args.command == "ideas":
            print_json([item.__dict__ for item in app.db.get_ideas(limit=100)] if app.db else [])
        elif args.command == "products":
            print_json(app.db.get_products(limit=100) if app.db else [])
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
