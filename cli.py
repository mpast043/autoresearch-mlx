"""CLI for the evidence-first discovery and validation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime.paths import DEFAULT_CONFIG_PATH, resolve_project_path  # noqa: E402
from run import AutoResearcher  # noqa: E402
from behavior_eval import run_behavior_eval  # noqa: E402
from database import ReviewFeedback  # noqa: E402
from reddit_relay import run_relay_server  # noqa: E402
from reddit_seed import RedditSeeder  # noqa: E402


def print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


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
            "eval",
            "review-queue",
            "review-mark",
            "ideas",
            "products",
            "reddit-relay",
            "reddit-seed",
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
    parser.add_argument("--label", default="", help="Review label for review-mark")
    parser.add_argument("--note", default="", help="Optional review note")
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
