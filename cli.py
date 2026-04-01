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
            from discovery_suggestions import build_discovery_suggestions

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

    app = AutoResearcher(config_path=args.config)

    if args.command == "run":
        await app.run()
        return

    if args.command == "run-once":
        summary = await app.run_once()
        if args.verbose:
            print_json(build_verbose_report(app, summary))
        else:
            print_json(summary)
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
        from agents.deep_research import DeepResearchAgent
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
        elif args.command == "report":
            print_json(build_verbose_report(app, app.snapshot()))
        elif args.command == "gate-diagnostics":
            from gate_diagnostics import build_gate_diagnostics_report

            report = build_gate_diagnostics_report(
                app.db,
                config=app.config,
                run_id=args.run_id or None,
                limit=args.limit,
                finding_id=args.finding_id,
            )
            print_json(report)
        elif args.command == "pipeline-health":
            from pipeline_health import compute_pipeline_health

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
