"""CLI for the evidence-first discovery and validation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from src.runtime.paths import DEFAULT_CONFIG_PATH, DEFAULT_EVAL_PATH, resolve_project_path  # noqa: E402
from src.validation_thresholds import resolve_promotion_park_thresholds  # noqa: E402
from run import AutoResearcher  # noqa: E402
from src.behavior_eval import run_behavior_eval  # noqa: E402
from src.database import ReviewFeedback  # noqa: E402
from src.reddit_relay import run_relay_server  # noqa: E402
from src.reddit_seed import RedditSeeder  # noqa: E402


def load_runtime_config(config_path: str | Path) -> tuple[dict, Path]:
    import yaml

    from src.runtime.env import load_local_env

    load_local_env()
    cfg_file = resolve_project_path(config_path, default=DEFAULT_CONFIG_PATH)
    with cfg_file.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}, cfg_file


def resolve_database_path_from_config(config_path: str | Path) -> Path:
    cfg, _ = load_runtime_config(config_path)
    rel = (cfg.get("database") or {}).get("path", "data/autoresearch.db")
    return resolve_project_path(rel, default=Path(rel))


def run_backup_db(config_path: str | Path) -> dict:
    """Copy SQLite DB to data/backups/ (see docs/RECOVERY.md)."""
    import shutil
    from datetime import datetime
    db_path = resolve_database_path_from_config(config_path)
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
    cfg, _ = load_runtime_config(config_path)
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


def _normalize_revalidation_notes(
    *,
    existing_notes: dict | None,
    scorecard: dict[str, float | int | str],
    cluster_summary: dict | None,
    market_gap: dict | None,
    counterevidence: list[dict] | None,
    scoring_version: str,
    formula_version: str,
    threshold_version: str,
    recommendation: str,
    selection_status: str,
    selection_reason: str,
) -> dict:
    notes = dict(existing_notes or {})
    normalized_cluster_summary = dict(cluster_summary or {})
    if scorecard.get("cluster_signal_count") is not None:
        normalized_cluster_summary["signal_count"] = int(scorecard.get("cluster_signal_count") or 0)
    if scorecard.get("cluster_atom_count") is not None:
        normalized_cluster_summary["atom_count"] = int(scorecard.get("cluster_atom_count") or 0)

    prior_scorecard = dict(notes.get("scorecard") or {})
    normalized_scorecard = {
        **prior_scorecard,
        **scorecard,
        "scoring_version": scoring_version,
        "formula_version": formula_version,
        "threshold_version": threshold_version,
    }

    notes["scorecard"] = normalized_scorecard
    notes["cluster_summary"] = normalized_cluster_summary
    notes["market_gap"] = dict(market_gap or {})
    notes["counterevidence"] = list(counterevidence or [])
    notes["revalidation"] = {
        "recommendation": recommendation,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "scoring_version": scoring_version,
        "formula_version": formula_version,
        "threshold_version": threshold_version,
    }
    return notes


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
    backlog_workbench = app.db.get_backlog_workbench(limit=max(limit * 3, 25)) if hasattr(app.db, "get_backlog_workbench") else []
    builder_jobs = build_builder_jobs_view(app.db, run_id=app.current_run_id, limit=max(limit * 3, 25))
    high_leverage = app.high_leverage_report(limit=limit)
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
            "top_ranked_backlog": backlog_workbench[:limit],
            "build_queue": builder_jobs[:limit],
        },
        "high_leverage": high_leverage,
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
        "backlog_workbench": app.db.get_backlog_workbench(limit=10)
        if app.db and hasattr(app.db, "get_backlog_workbench")
        else [],
        "decision_surface": app.db.get_candidate_workbench(limit=10, run_id=app.current_run_id)
        if app.db and hasattr(app.db, "get_candidate_workbench")
        else [],
        "builder_jobs": build_builder_jobs_view(app.db, run_id=app.current_run_id, limit=10),
        "high_leverage": app.high_leverage_report(limit=10),
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


# ---------------------------------------------------------------------------
# Async context manager for app lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_context(args: argparse.Namespace, *, start_new_run: bool = False):
    """Yield an initialised AutoResearcher, shutting it down on exit."""
    app = AutoResearcher(config_path=args.config)
    await app.initialize(start_new_run=start_new_run)
    try:
        yield app
    finally:
        await app.shutdown()


# ---------------------------------------------------------------------------
# Command handlers — each receives (args, app) where app is already
# initialised, unless the command manages its own lifecycle.
# ---------------------------------------------------------------------------

async def cmd_watch(args: argparse.Namespace, _app: AutoResearcher) -> None:
    await watch_status(interval=args.interval, config_path=args.config)


async def cmd_eval(args: argparse.Namespace, _app: AutoResearcher) -> None:
    print_json(run_behavior_eval(str(resolve_project_path(args.eval_path))))


async def cmd_reddit_relay(args: argparse.Namespace, _app: AutoResearcher) -> None:
    app = AutoResearcher(config_path=args.config)
    host = args.host or app.config.get("reddit_relay", {}).get("host", "127.0.0.1")
    port = args.port or int(app.config.get("reddit_relay", {}).get("port", 8787))
    await run_relay_server(app.config, host=host, port=port)


async def cmd_reddit_seed(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        seeder = RedditSeeder(config=app.config)
        summary = await seeder.seed()
        print_json(summary.__dict__ if hasattr(summary, "__dict__") else summary)


async def cmd_check_bridge(args: argparse.Namespace, _app: AutoResearcher) -> None:
    print_json(await run_check_bridge(args.config))


async def cmd_backup_db(args: argparse.Namespace, _app: AutoResearcher) -> None:
    print_json(run_backup_db(args.config))


async def cmd_suggest_discovery(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
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


async def cmd_term_lifecycle(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        from src.discovery_term_lifecycle import TermLifecycleManager
        from src.database import get_traceability_stats

        _ = TermLifecycleManager(app.db)

        if args.term_type:
            terms = app.db.list_search_terms(term_type=args.term_type, limit=args.limit or 100)
        else:
            terms = app.db.list_search_terms(limit=args.limit or 100)

        # Include traceability stats so the user can see if the feedback loop is working
        trace_stats = get_traceability_stats()

        print_json({
            "terms": terms,
            "counts": {
                "total": len(terms),
                "by_state": collections.Counter(t.get("state") for t in terms),
            },
            "traceability": trace_stats,
        })


async def cmd_term_state(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        from src.discovery_term_lifecycle import TermLifecycleManager

        lifecycle = TermLifecycleManager(app.db)
        action = args.action or "list"

        if action == "list":
            terms = app.db.list_search_terms(
                term_type=args.term_type if args.term_type else None,
                state=args.state if args.state else None,
                limit=args.limit or 100,
            )
            print_json({"terms": terms})
        elif action == "ban":
            if not args.term_value:
                print("term-state ban requires --term-value", file=sys.stderr)
                sys.exit(1)
            lifecycle.ban_term(args.term_type or "keyword", args.term_value, reason=args.note or "manual ban")
            print_json({"ok": True, "action": "ban", "term_type": args.term_type, "term_value": args.term_value})
        elif action == "reactivate":
            if not args.term_value:
                print("term-state reactivate requires --term-value", file=sys.stderr)
                sys.exit(1)
            lifecycle.reactivate_term(args.term_type or "keyword", args.term_value, reason=args.note or "manual reactivation")
            print_json({"ok": True, "action": "reactivate", "term_type": args.term_type, "term_value": args.term_value})
        elif action == "complete":
            if not args.term_value:
                print("term-state complete requires --term-value", file=sys.stderr)
                sys.exit(1)
            lifecycle.complete_term(args.term_type or "keyword", args.term_value, reason=args.note or "manual completion")
            print_json({"ok": True, "action": "complete", "term_type": args.term_type, "term_value": args.term_value})
        elif action == "reset":
            if not args.term_value:
                print("term-state reset requires --term-value", file=sys.stderr)
                sys.exit(1)
            lifecycle.reset_term(args.term_type or "keyword", args.term_value)
            print_json({"ok": True, "action": "reset", "term_type": args.term_type, "term_value": args.term_value})
        elif action == "high-performers":
            high_kw = lifecycle.get_terms_for_expansion("keyword", limit=args.limit or 20)
            high_sub = lifecycle.get_terms_for_expansion("subreddit", limit=args.limit or 20)
            print_json({
                "high_performing_keywords": high_kw,
                "high_performing_subreddits": high_sub,
            })
        elif action == "exhausted":
            exh_kw = app.db.get_exhausted_terms("keyword", limit=args.limit or 50)
            exh_sub = app.db.get_exhausted_terms("subreddit", limit=args.limit or 50)
            print_json({
                "exhausted_keywords": exh_kw,
                "exhausted_subreddits": exh_sub,
            })
        elif action == "wedge-quality":
            wedge_kw = lifecycle.get_terms_for_expansion_by_wedge_quality("keyword", limit=args.limit or 20)
            wedge_sub = lifecycle.get_terms_for_expansion_by_wedge_quality("subreddit", limit=args.limit or 20)
            print_json({
                "wedge_quality_keywords": wedge_kw,
                "wedge_quality_subreddits": wedge_sub,
            })
        elif action == "specificity":
            spec_kw = lifecycle.get_terms_by_specificity("keyword", limit=args.limit or 20)
            spec_sub = lifecycle.get_terms_by_specificity("subreddit", limit=args.limit or 20)
            print_json({
                "specificity_keywords": spec_kw,
                "specificity_subreddits": spec_sub,
            })
        elif action == "platform-native":
            plat_kw = lifecycle.get_terms_by_platform_native("keyword", limit=args.limit or 20)
            plat_sub = lifecycle.get_terms_by_platform_native("subreddit", limit=args.limit or 20)
            print_json({
                "platform_native_keywords": plat_kw,
                "platform_native_subreddits": plat_sub,
            })
        elif action == "abstraction-collapse":
            collapse_kw = lifecycle.get_abstraction_collapse_terms("keyword", limit=args.limit or 20)
            collapse_sub = lifecycle.get_abstraction_collapse_terms("subreddit", limit=args.limit or 20)
            print_json({
                "abstraction_collapse_keywords": collapse_kw,
                "abstraction_collapse_subreddits": collapse_sub,
            })
        elif action == "buildable":
            build_kw = lifecycle.get_buildable_terms("keyword", limit=args.limit or 20)
            build_sub = lifecycle.get_buildable_terms("subreddit", limit=args.limit or 20)
            print_json({
                "buildable_keywords": build_kw,
                "buildable_subreddits": build_sub,
            })
        else:
            print(f"Unknown action: {action}", file=sys.stderr)
            sys.exit(1)


async def cmd_security_scan(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        from src.agents.security import SecurityAgent
        config = app.config.get("security", {})
        security = SecurityAgent(config)

        wedge_id = args.wedge_id if hasattr(args, "wedge_id") else None
        code = args.code if hasattr(args, "code") else None

        if code:
            report = security.scan_code(code, file_name=args.file_name if hasattr(args, "file_name") else "stdin")
            print(security.format_report(report))
        elif wedge_id:
            opp = app.db.get_opportunity(wedge_id)
            if opp:
                spec = {"id": str(opp.id), "cluster_id": opp.cluster_id, "title": opp.title}
                report = security.scan_solution_spec(spec)
                print(security.format_report(report))
            else:
                print(f"Opportunity {wedge_id} not found", file=sys.stderr)
                sys.exit(1)
        else:
            opportunities = app.db.get_opportunities(status="build_ready", limit=10)
            for opp in opportunities:
                spec = {"id": str(opp.id), "cluster_id": opp.cluster_id, "title": opp.title}
                report = security.scan_solution_spec(spec)
                print(f"\n--- Opportunity {opp.id}: {opp.title} ---")
                print(security.format_report(report))


async def cmd_generate_docs(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        from src.agents.technical_writer import TechnicalWriterAgent
        from pathlib import Path

        config = app.config.get("technical_writer", {})
        writer = TechnicalWriterAgent(config)

        output_dir = Path(config.get("output_dir", "output/docs"))

        opportunities = app.db.get_opportunities(status="build_ready", limit=5)
        for opp in opportunities:
            spec = app.db.get_opportunity(opp.id)
            if spec:
                spec_dict = {"id": str(spec.id), "cluster_id": spec.cluster_id, "title": spec.title}
                bundle = writer.generate_docs(spec_dict)
                opp_dir = output_dir / f"opportunity_{spec.id}"
                files = writer.save_docs(bundle, opp_dir)
                print(f"Generated docs for opportunity {spec.id}: {list(files.keys())}")


async def cmd_sre_health(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args) as app:
        from src.agents.sre import SREAgent

        config = app.config.get("sre", {})
        sre = SREAgent(app.db, config)

        opportunity_id = args.wedge_id if hasattr(args, "wedge_id") else None

        if opportunity_id:
            print(sre.format_opportunity_status(opportunity_id))
        else:
            report = sre.generate_report()
            print(report.summary)


async def cmd_patterns(args: argparse.Namespace, _app: AutoResearcher) -> None:
    from src.opportunity_engine import get_patterns_for_discovery

    db_path = str(resolve_database_path_from_config(args.config))
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


async def cmd_scoring_report(args: argparse.Namespace, _app: AutoResearcher) -> None:
    import sqlite3

    db_path = resolve_database_path_from_config(args.config)
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    config, _ = load_runtime_config(args.config)
    promote_thresh, park_thresh = resolve_promotion_park_thresholds(config)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT scoring_version, COUNT(*) FROM opportunities GROUP BY scoring_version")
    version_dist = cursor.fetchall()
    print("=== SCORING VERSION DISTRIBUTION ===")
    has_legacy = False
    for ver, count in version_dist:
        marker = ""
        if ver in ('0', 'v1', 'v2', 'v3'):
            marker = " (LEGACY - needs rescore)"
            has_legacy = True
        elif ver == 'v4':
            marker = " <-- CURRENT"
        print(f"  {ver}: {count}{marker}")
    if has_legacy:
        print("\n⚠️  WARNING: Legacy-scored rows detected. Run revalidation for accurate analysis.")
    print()

    cursor.execute("SELECT composite_score, frequency_score, evidence_quality, education_burden, adoption_friction FROM opportunities ORDER BY composite_score DESC")
    rows = cursor.fetchall()
    scores = [r[0] for r in rows]
    total = len(scores)

    if total == 0:
        print("No opportunities in database")
        conn.close()
        return

    sorted_scores = sorted(scores)
    p99 = sorted_scores[int(total * 0.99)] if total > 0 else 0
    p95 = sorted_scores[int(total * 0.95)] if total > 0 else 0
    p90 = sorted_scores[int(total * 0.90)] if total > 0 else 0
    median = sorted_scores[int(total * 0.50)] if total > 0 else 0

    above_promote = sum(1 for s in scores if s >= promote_thresh)
    above_park = sum(1 for s in scores if s >= park_thresh)

    print("=== SCORING PERCENTILE MONITOR ===\n")
    print(f"Total opportunities: {total}\n")
    print("PERCENTILES:")
    print(f"  Max:   {max(scores):.4f}")
    print(f"  P99:   {p99:.4f}")
    print(f"  P95:   {p95:.4f}")
    print(f"  P90:   {p90:.4f}")
    print(f"  Median:{median:.4f}")
    print(f"\nDECISION BUCKETS (thresholds: promote={promote_thresh}, park={park_thresh}):")
    print(f"  Promote (≥{promote_thresh}): {above_promote} ({above_promote/total*100:.1f}%)")
    print(f"  Park ({park_thresh}-{promote_thresh}): {above_park-above_promote} ({(above_park-above_promote)/total*100:.1f}%)")
    print(f"  Kill (<{park_thresh}): {total-above_park} ({(total-above_park)/total*100:.1f}%)")

    cursor.execute("""
        SELECT
            AVG(frequency_score) as avg_freq,
            AVG(evidence_quality) as avg_eq,
            AVG(education_burden) as avg_edu,
            AVG(adoption_friction) as avg_fric
        FROM opportunities
        WHERE composite_score >= ? AND composite_score < ?
    """, (park_thresh, promote_thresh))
    parked = cursor.fetchone()
    park_count = above_park - above_promote
    if parked and parked[0]:
        print(f"\nPARKED OPPORTUNITY ANALYSIS (n={park_count}):")
        print(f"  Avg frequency:        {parked[0]:.3f}")
        print(f"  Avg evidence_quality: {parked[1]:.3f}")
        print(f"  Avg education_burden: {parked[2]:.3f} ⚠️")
        print(f"  Avg adoption_friction:{parked[3]:.3f} ⚠️")

    conn.close()


async def cmd_revalidate(args: argparse.Namespace, _app: AutoResearcher) -> None:
    from src.build_prep import determine_selection_state
    from src.database import Database
    from src.opportunity_engine import (
        CURRENT_FORMULA_VERSION,
        CURRENT_SCORING_VERSION,
        CURRENT_THRESHOLD_VERSION,
        stage_decision,
    )
    from src.research_tools import ResearchToolkit

    config, _ = load_runtime_config(args.config)
    db_path = resolve_database_path_from_config(args.config)
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    db = Database(str(db_path))
    db.init_schema()
    conn = db._get_connection()
    stale_rows = conn.execute(
        """
        SELECT id
        FROM opportunities
        WHERE COALESCE(scoring_version, '') != ?
           OR COALESCE(formula_version, '') != ?
           OR COALESCE(threshold_version, '') != ?
        ORDER BY id ASC
        """,
        (CURRENT_SCORING_VERSION, CURRENT_FORMULA_VERSION, CURRENT_THRESHOLD_VERSION),
    ).fetchall()
    all_rows = conn.execute("SELECT id FROM opportunities ORDER BY id ASC").fetchall()

    if not all_rows:
        print("No opportunities found.")
        db.close()
        return

    promote_thresh, park_thresh = resolve_promotion_park_thresholds(config)
    toolkit = ResearchToolkit(config)
    print(f"=== REVALIDATING {len(all_rows)} OPPORTUNITIES ===")
    print(
        f"Using scoring={CURRENT_SCORING_VERSION}, formula={CURRENT_FORMULA_VERSION}, "
        f"threshold_version={CURRENT_THRESHOLD_VERSION}"
    )
    print(f"Thresholds: promote={promote_thresh:.2f}, park={park_thresh:.2f}\n")
    print(f"Version-stale before pass: {len(stale_rows)}\n")

    changed = 0
    promote_count = 0
    park_count = 0
    kill_count = 0
    samples: list[str] = []

    with db.batch():
        for row in all_rows:
            opportunity = db.get_opportunity(int(row["id"]))
            if not opportunity:
                continue

            notes = opportunity.notes or {}
            cluster_summary = notes.get("cluster_summary", {}) or {}
            market_gap = notes.get("market_gap") or {"market_gap": opportunity.market_gap}
            counterevidence = notes.get("counterevidence", []) or []
            cluster_record = db.get_cluster_record(opportunity.cluster_id)
            cluster_signal_count = (
                cluster_summary.get("signal_count")
                or getattr(cluster_record, "signal_count", 0)
                or 0
            )
            cluster_atom_count = (
                cluster_summary.get("atom_count")
                or getattr(cluster_record, "atom_count", 0)
                or 0
            )
            scorecard = {
                "decision_score": opportunity.decision_score,
                "problem_truth_score": opportunity.problem_truth_score,
                "revenue_readiness_score": opportunity.revenue_readiness_score,
                "composite_score": opportunity.composite_score,
                "problem_plausibility": opportunity.problem_plausibility,
                "value_support": opportunity.value_support,
                "corroboration_strength": opportunity.corroboration_strength,
                "evidence_sufficiency": opportunity.evidence_sufficiency,
                "willingness_to_pay_proxy": opportunity.willingness_to_pay_proxy,
                "pain_severity": opportunity.pain_severity,
                "frequency_score": opportunity.frequency_score,
                "cost_of_inaction": opportunity.cost_of_inaction,
                "workaround_density": opportunity.workaround_density,
                "urgency_score": opportunity.urgency_score,
                "segment_concentration": opportunity.segment_concentration,
                "reachability": opportunity.reachability,
                "timing_shift": opportunity.timing_shift,
                "buildability": opportunity.buildability,
                "expansion_potential": opportunity.expansion_potential,
                "education_burden": opportunity.education_burden,
                "dependency_risk": opportunity.dependency_risk,
                "adoption_friction": opportunity.adoption_friction,
                "evidence_quality": opportunity.evidence_quality,
                "cluster_signal_count": cluster_signal_count,
                "cluster_atom_count": cluster_atom_count,
            }
            cluster_atoms = db.get_problem_atoms_by_cluster_key(cluster_record.cluster_key) if cluster_record and cluster_record.cluster_key else []
            cluster_signals = db.get_raw_signals_by_ids([atom.signal_id for atom in cluster_atoms if atom.signal_id]) if cluster_atoms else []
            resolved_signals = [signal for signal in cluster_signals if signal is not None]

            web_only_cluster = bool(resolved_signals) and all(str(signal.source_type or "") == "web" for signal in resolved_signals)
            low_quality_web_cluster = web_only_cluster and all(
                toolkit._is_low_quality_web_problem_page(
                    title=str(signal.title or ""),
                    snippet=str(signal.body_excerpt or ""),
                    body=str(signal.quote_text or signal.body_excerpt or ""),
                    url=str(signal.source_url or ""),
                )
                for signal in resolved_signals
            )

            if low_quality_web_cluster:
                decision = {
                    "recommendation": "kill",
                    "status": "killed",
                    "reason": "source_policy_rejected_web_page",
                    "decision_reason": "source_policy_rejected_web_page",
                    "park_subreason": "",
                }
            else:
                decision = stage_decision(
                    scorecard,
                    market_gap,
                    counterevidence,
                    promotion_threshold=promote_thresh,
                    park_threshold=park_thresh,
                )

            brief = db.get_build_brief_for_opportunity(opportunity.id or 0)
            validation = db.get_validation(brief.validation_id) if brief and brief.validation_id else None
            validation_evidence = validation.evidence_dict if validation else {}
            corroboration = validation_evidence.get("corroboration", {}) or {}
            if not corroboration:
                corroboration = notes.get("corroboration", {}) or {}
            # Also check latest corroboration record from DB — it may have
            # been updated after the opportunity was first validated.
            if not corroboration and cluster_atoms:
                latest_cor = db._get_connection().execute(
                    "SELECT evidence_json FROM corroborations WHERE finding_id = ? ORDER BY id DESC LIMIT 1",
                    (cluster_atoms[0].finding_id,),
                ).fetchone()
                if latest_cor and latest_cor[0]:
                    import json as _json
                    try:
                        corroboration = _json.loads(latest_cor[0])
                    except Exception:
                        pass
            market_enrichment = validation_evidence.get("market_enrichment", {}) or {}
            if not market_enrichment:
                market_enrichment = notes.get("market_enrichment", {}) or {}

            if validation or corroboration:
                selection_status, selection_reason, _ = determine_selection_state(
                    decision=decision["recommendation"],
                    scorecard=scorecard,
                    corroboration=corroboration,
                    market_enrichment=market_enrichment,
                )
            else:
                selection_status = (
                    "archive"
                    if decision["recommendation"] == "kill"
                    else ("prototype_candidate" if decision["recommendation"] == "promote" else "research_more")
                )
                selection_reason = "revalidated_without_full_validation_context"

            if low_quality_web_cluster:
                selection_status = "archive"
                selection_reason = "revalidated_source_policy_reject"

            prev_signature = (
                opportunity.recommendation,
                opportunity.status,
                opportunity.selection_status,
                opportunity.threshold_version,
            )

            opportunity.recommendation = decision["recommendation"]
            opportunity.status = decision["status"]
            opportunity.selection_status = selection_status
            opportunity.selection_reason = selection_reason
            opportunity.scoring_version = CURRENT_SCORING_VERSION
            opportunity.formula_version = CURRENT_FORMULA_VERSION
            opportunity.threshold_version = CURRENT_THRESHOLD_VERSION
            opportunity.notes = _normalize_revalidation_notes(
                existing_notes=notes,
                scorecard=scorecard,
                cluster_summary=cluster_summary,
                market_gap=market_gap,
                counterevidence=counterevidence,
                scoring_version=CURRENT_SCORING_VERSION,
                formula_version=CURRENT_FORMULA_VERSION,
                threshold_version=CURRENT_THRESHOLD_VERSION,
                recommendation=decision["recommendation"],
                selection_status=selection_status,
                selection_reason=selection_reason,
            )
            opportunity.notes_json = json.dumps(opportunity.notes, default=str)
            db.upsert_opportunity(opportunity)

            if brief:
                if selection_status in {"research_more", "archive"}:
                    db.update_build_brief_status(brief.id or 0, "archive")
                elif selection_status == "prototype_candidate" and str(brief.status or "") != "prototype_candidate":
                    db.update_build_brief_status(brief.id or 0, "prototype_candidate")

            next_signature = (
                opportunity.recommendation,
                opportunity.status,
                opportunity.selection_status,
                opportunity.threshold_version,
            )
            if prev_signature != next_signature:
                changed += 1
                if len(samples) < 8:
                    samples.append(
                        f"opp #{opportunity.id} cluster={opportunity.cluster_id} -> "
                        f"{decision['recommendation']}/{selection_status}"
                    )

            if decision["recommendation"] == "promote":
                promote_count += 1
            elif decision["recommendation"] == "park":
                park_count += 1
            else:
                kill_count += 1

    print("=== REVALIDATION COMPLETE ===")
    print(f"  Revalidated: {len(all_rows)}")
    print(f"  Changed: {changed}")
    print(f"  Promote: {promote_count}")
    print(f"  Park: {park_count}")
    print(f"  Kill: {kill_count}")
    if samples:
        print("\nSample changes:")
        for sample in samples:
            print(f"  - {sample}")
    db.close()


async def cmd_rescreen(args: argparse.Namespace, _app: AutoResearcher) -> None:
    from datetime import datetime, timezone

    from src.agents.discovery import _serialize_atom_json, is_wedge_ready_signal
    from src.database import Database, ProblemAtom, RawSignal
    from src.high_leverage import score_high_leverage_finding
    from src.opportunity_engine import (
        build_problem_atom,
        build_raw_signal_payload,
        classify_source_signal,
        qualify_problem_signal,
    )

    db_path = resolve_database_path_from_config(args.config)
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    db = Database(str(db_path))
    db.init_schema()

    candidates = []
    if args.finding_id:
        finding = db.get_finding(args.finding_id)
        if not finding:
            print_json({"requested": [args.finding_id], "processed": 0, "error": "finding_not_found"})
            db.close()
            return
        candidates = [finding]
    else:
        candidates = db.get_findings(status="screened_out", limit=max(args.limit, 1))

    summary = {
        "processed": 0,
        "qualified": [],
        "still_screened_out": [],
        "skipped": [],
    }

    with db.batch():
        for finding in candidates:
            finding_id = int(finding.id or 0)
            if finding_id <= 0:
                continue

            raw_signals = db.get_raw_signals_by_finding(finding_id)
            problem_atoms = db.get_problem_atoms_by_finding(finding_id)
            if raw_signals or problem_atoms:
                summary["skipped"].append({"finding_id": finding_id, "reason": "existing_artifacts"})
                continue

            evidence = dict(finding.evidence or {})
            finding_data = {
                "source": finding.source,
                "source_url": finding.source_url,
                "entrepreneur": finding.entrepreneur,
                "tool_used": finding.tool_used,
                "product_built": finding.product_built,
                "monetization_method": finding.monetization_method,
                "outcome_summary": finding.outcome_summary,
                "finding_kind": finding.finding_kind,
                "source_class": finding.source_class,
                "recurrence_key": finding.recurrence_key,
                "evidence": evidence,
            }

            is_ready, reject_reason = is_wedge_ready_signal(finding_data)
            if not is_ready:
                evidence["pre_atom_filter"] = {"accepted": False, "reason": reject_reason}
                evidence["screening"] = {
                    "accepted": False,
                    "score": 0.0,
                    "positive_signals": [],
                    "negative_signals": [f"pre_atom_filter:{reject_reason}"],
                    "source_class": "low_signal_summary",
                }
                evidence["rescreen"] = {
                    "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "accepted": False,
                    "reason": reject_reason,
                }
                db.update_finding_screening(
                    finding_id,
                    status="screened_out",
                    source_class="low_signal_summary",
                    evidence=evidence,
                )
                summary["still_screened_out"].append({"finding_id": finding_id, "reason": reject_reason})
                summary["processed"] += 1
                continue

            signal_payload = build_raw_signal_payload(finding_data)
            atom_payload = build_problem_atom(signal_payload, finding_data)
            source_classification = classify_source_signal(finding_data, signal_payload, atom_payload)
            finding_data["source_class"] = source_classification["source_class"]
            signal_payload.setdefault("metadata_json", {})["source_class"] = source_classification["source_class"]
            screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)
            screening["source_class"] = source_classification["source_class"]

            evidence["pre_atom_filter"] = {"accepted": True, "reason": "passed"}
            evidence["screening"] = screening
            evidence["source_classification"] = source_classification

            temp_signal = RawSignal(
                finding_id=finding_id,
                source_name=signal_payload["source_name"],
                source_type=signal_payload["source_type"],
                source_class=source_classification["source_class"],
                source_url=signal_payload["source_url"],
                title=signal_payload["title"],
                body_excerpt=signal_payload["body_excerpt"],
                quote_text=signal_payload["quote_text"],
                role_hint=signal_payload["role_hint"],
                published_at=signal_payload["published_at"],
                timestamp_hint=signal_payload["timestamp_hint"],
                content_hash=finding.content_hash,
                metadata=signal_payload["metadata_json"],
            )
            temp_atom = ProblemAtom(
                signal_id=0,
                finding_id=finding_id,
                cluster_key=atom_payload["cluster_key"],
                segment=atom_payload["segment"],
                user_role=atom_payload["user_role"],
                job_to_be_done=atom_payload["job_to_be_done"],
                trigger_event=atom_payload["trigger_event"],
                pain_statement=atom_payload["pain_statement"],
                failure_mode=atom_payload["failure_mode"],
                current_workaround=atom_payload["current_workaround"],
                current_tools=atom_payload["current_tools"],
                urgency_clues=atom_payload["urgency_clues"],
                frequency_clues=atom_payload["frequency_clues"],
                emotional_intensity=atom_payload["emotional_intensity"],
                cost_consequence_clues=atom_payload["cost_consequence_clues"],
                why_now_clues=atom_payload["why_now_clues"],
                confidence=atom_payload["confidence"],
                platform=atom_payload.get("platform", ""),
                specificity_score=atom_payload.get("specificity_score", 0.0),
                consequence_score=atom_payload.get("consequence_score", 0.0),
                atom_extraction_method=atom_payload.get("atom_extraction_method", "heuristic"),
                atom_json=_serialize_atom_json(atom_payload),
            )
            high_leverage = score_high_leverage_finding(finding, temp_signal, temp_atom, evidence)
            evidence["high_leverage"] = high_leverage
            evidence["rescreen"] = {
                "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "accepted": bool(screening["accepted"]),
                "reason": "qualified" if screening["accepted"] else ",".join(screening.get("negative_signals", [])),
            }

            if screening["accepted"]:
                signal_payload.setdefault("metadata_json", {})["high_leverage"] = high_leverage
                signal = RawSignal(
                    finding_id=finding_id,
                    source_name=signal_payload["source_name"],
                    source_type=signal_payload["source_type"],
                    source_class=source_classification["source_class"],
                    source_url=signal_payload["source_url"],
                    title=signal_payload["title"],
                    body_excerpt=signal_payload["body_excerpt"],
                    quote_text=signal_payload["quote_text"],
                    role_hint=signal_payload["role_hint"],
                    published_at=signal_payload["published_at"],
                    timestamp_hint=signal_payload["timestamp_hint"],
                    content_hash=finding.content_hash,
                    metadata=signal_payload["metadata_json"],
                )
                signal_id = db.insert_raw_signal(signal)
                atom = ProblemAtom(
                    signal_id=signal_id,
                    raw_signal_id=signal_id,
                    finding_id=finding_id,
                    cluster_key=atom_payload["cluster_key"],
                    segment=atom_payload["segment"],
                    user_role=atom_payload["user_role"],
                    job_to_be_done=atom_payload["job_to_be_done"],
                    trigger_event=atom_payload["trigger_event"],
                    pain_statement=atom_payload["pain_statement"],
                    failure_mode=atom_payload["failure_mode"],
                    current_workaround=atom_payload["current_workaround"],
                    current_tools=atom_payload["current_tools"],
                    urgency_clues=atom_payload["urgency_clues"],
                    frequency_clues=atom_payload["frequency_clues"],
                    emotional_intensity=atom_payload["emotional_intensity"],
                    cost_consequence_clues=atom_payload["cost_consequence_clues"],
                    why_now_clues=atom_payload["why_now_clues"],
                    confidence=atom_payload["confidence"],
                    platform=atom_payload.get("platform", ""),
                    specificity_score=atom_payload.get("specificity_score", 0.0),
                    consequence_score=atom_payload.get("consequence_score", 0.0),
                    atom_extraction_method=atom_payload.get("atom_extraction_method", "heuristic"),
                    atom_json=_serialize_atom_json(atom_payload),
                )
                db.insert_problem_atom(atom)
                db.update_finding_screening(
                    finding_id,
                    status="qualified",
                    source_class=source_classification["source_class"],
                    evidence=evidence,
                )
                summary["qualified"].append(
                    {
                        "finding_id": finding_id,
                        "source_class": source_classification["source_class"],
                        "screening_score": screening.get("score", 0.0),
                    }
                )
            else:
                db.update_finding_screening(
                    finding_id,
                    status="screened_out",
                    source_class=source_classification["source_class"],
                    evidence=evidence,
                )
                summary["still_screened_out"].append(
                    {
                        "finding_id": finding_id,
                        "source_class": source_classification["source_class"],
                        "negative_signals": screening.get("negative_signals", []),
                    }
                )

            summary["processed"] += 1

    print_json(summary)
    db.close()


async def cmd_rescore_v4(args: argparse.Namespace, _app: AutoResearcher) -> None:
    import sqlite3
    from datetime import datetime
    from src.opportunity_engine import (
        score_opportunity,
        stage_decision,
        build_counterevidence,
        assess_market_gap,
        CURRENT_SCORING_VERSION,
        CURRENT_FORMULA_VERSION,
        CURRENT_THRESHOLD_VERSION,
    )

    config, _ = load_runtime_config(args.config)
    db_path = resolve_database_path_from_config(args.config)
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    print("=== V4 RESCORING ===")
    print(f"Formula: {CURRENT_FORMULA_VERSION}")
    print(f"Scoring: {CURRENT_SCORING_VERSION}")
    print(f"Thresholds: {CURRENT_THRESHOLD_VERSION}")
    print()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    def _has_column(table: str, column: str) -> bool:
        return any(row["name"] == column for row in cursor.execute(f"PRAGMA table_info({table})").fetchall())

    atom_cluster_expr = "a.cluster_key" if _has_column("problem_atoms", "cluster_key") else "json_extract(a.metadata_json, '$.cluster_key')"
    cluster_key_expr = "c.cluster_key" if _has_column("opportunity_clusters", "cluster_key") else "json_extract(c.metadata_json, '$.cluster_key')"

    cursor.execute(f"""
        SELECT c.id as cluster_id, c.label, c.summary_json,
               GROUP_CONCAT(a.id) as atom_ids
        FROM opportunity_clusters c
        LEFT JOIN problem_atoms a ON {atom_cluster_expr} = {cluster_key_expr}
        GROUP BY c.id
        ORDER BY c.id
    """)
    clusters = cursor.fetchall()

    if not clusters:
        print("No clusters found")
        conn.close()
        return

    print(f"Found {len(clusters)} clusters to rescore")
    print()

    promote_thresh, park_thresh = resolve_promotion_park_thresholds(config)
    promote_count = 0
    park_count = 0
    kill_count = 0

    for i, cluster in enumerate(clusters):
        cluster_id = cluster["cluster_id"]

        atom_ids = cluster["atom_ids"]
        if not atom_ids:
            print(f"  Skipping cluster {cluster_id}: no atoms")
            continue

        cursor.execute("SELECT * FROM problem_atoms WHERE id = ?", (int(atom_ids.split(",")[0]),))
        atom_row = cursor.fetchone()
        if not atom_row:
            continue

        atom = dict(atom_row)

        signal_id = atom.get("raw_signal_id") or atom.get("signal_id")
        signal = {}
        if signal_id:
            cursor.execute("SELECT * FROM raw_signals WHERE id = ?", (signal_id,))
            signal_row = cursor.fetchone()
            if signal_row:
                signal = dict(signal_row)

        import json
        summary = json.loads(cluster["summary_json"]) if cluster["summary_json"] else {}

        validation_evidence = {
            "scores": {},
            "evidence": {},
            "corroboration": {},
            "market_enrichment": {},
        }

        try:
            scorecard = score_opportunity(
                atom=atom,
                signal=signal,
                cluster_summary=summary,
                validation_evidence=validation_evidence,
                market_gap=assess_market_gap(summary, {}),
            )
        except Exception as e:
            print(f"  Error scoring cluster {cluster_id}: {e}")
            continue

        market_gap = assess_market_gap(summary, {})
        counterevidence = build_counterevidence(scorecard, market_gap)
        decision = stage_decision(
            scorecard,
            market_gap,
            counterevidence,
            promotion_threshold=promote_thresh,
            park_threshold=park_thresh,
        )

        now = datetime.now().isoformat()

        cursor.execute("""
            UPDATE opportunities
            SET recommendation = ?, status = ?,
                scoring_version = ?,
                problem_truth_score = ?,
                revenue_readiness_score = ?,
                decision_score = ?,
                problem_plausibility = ?,
                value_support = ?,
                corroboration_strength = ?,
                evidence_sufficiency = ?,
                willingness_to_pay_proxy = ?,
                formula_version = ?,
                threshold_version = ?,
                evaluated_at = ?,
                last_rescored_at = ?,
                composite_score = ?,
                pain_severity = ?,
                frequency_score = ?,
                evidence_quality = ?
            WHERE cluster_id = ?
        """, (
            decision["recommendation"],
            decision["status"],
            CURRENT_SCORING_VERSION,
            scorecard.get("problem_truth_score", 0.0),
            scorecard.get("revenue_readiness_score", 0.0),
            scorecard.get("decision_score", 0.0),
            scorecard.get("problem_plausibility", 0.0),
            scorecard.get("value_support", 0.0),
            scorecard.get("corroboration_strength", 0.0),
            scorecard.get("evidence_sufficiency", 0.0),
            scorecard.get("willingness_to_pay_proxy", 0.0),
            CURRENT_FORMULA_VERSION,
            CURRENT_THRESHOLD_VERSION,
            now,
            now,
            scorecard.get("composite_score", 0.0),
            scorecard.get("pain_severity", 0.0),
            scorecard.get("frequency_score", 0.0),
            scorecard.get("evidence_quality", 0.0),
            cluster_id,
        ))

        if decision["recommendation"] == "promote":
            promote_count += 1
        elif decision["recommendation"] == "park":
            park_count += 1
        else:
            kill_count += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(clusters)} processed...")

    conn.commit()

    cursor.execute("""
        INSERT INTO scoring_runs (run_id, formula_version, threshold_version, scoring_version,
            opportunity_count, promote_count, park_count, kill_count, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        f"rescore-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        CURRENT_FORMULA_VERSION,
        CURRENT_THRESHOLD_VERSION,
        CURRENT_SCORING_VERSION,
        len(clusters),
        promote_count,
        park_count,
        kill_count,
        datetime.now().isoformat(),
    ))
    conn.commit()

    print()
    print("=== RESCORING COMPLETE ===")
    print(f"  Total: {len(clusters)}")
    print(f"  Promote: {promote_count}")
    print(f"  Park: {park_count}")
    print(f"  Kill: {kill_count}")
    print(f"  Version: {CURRENT_SCORING_VERSION}")
    print(f"  Formula: {CURRENT_FORMULA_VERSION}")

    conn.close()


async def cmd_run(args: argparse.Namespace, _app: AutoResearcher) -> None:
    app = AutoResearcher(config_path=args.config)
    original_focus: dict[str, Any] | None = None

    if args.pattern:
        from src.opportunity_engine import PATTERN_TO_DISCOVERY_QUERIES

        if args.pattern not in PATTERN_TO_DISCOVERY_QUERIES:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available patterns: {list(PATTERN_TO_DISCOVERY_QUERIES.keys())}")
            raise SystemExit(1)

        queries = PATTERN_TO_DISCOVERY_QUERIES[args.pattern]
        print(f"=== FOCUSED DISCOVERY: {args.pattern} ===")
        print(f"Queries: {queries[:3]}...")
        discovery_cfg = app.config.get("discovery", {})
        original_focus = {
            "sources": list(discovery_cfg.get("sources", []) or []),
            "web_keywords": list(discovery_cfg.get("web", {}).get("keywords", []) or []),
            "web_problem_keywords": list(discovery_cfg.get("web", {}).get("problem_keywords", []) or []),
            "web_success_keywords": list(discovery_cfg.get("web", {}).get("success_keywords", []) or []),
            "web_market_keywords": list(discovery_cfg.get("web", {}).get("market_keywords", []) or []),
            "reddit_keywords": list(discovery_cfg.get("reddit", {}).get("problem_keywords", []) or []),
            "github_keywords": list(discovery_cfg.get("github", {}).get("problem_keywords", []) or []),
            "focused_problem_only": bool(discovery_cfg.get("focused_problem_only", False)),
        }
        focused_sources = [source for source in ["reddit", "web", "github"] if source in (discovery_cfg.get("sources", []) or [])]
        if focused_sources:
            discovery_cfg["sources"] = focused_sources
        discovery_cfg["focused_problem_only"] = True
        if "web" in discovery_cfg:
            discovery_cfg["web"]["keywords"] = queries
            discovery_cfg["web"]["problem_keywords"] = queries
            discovery_cfg["web"]["success_keywords"] = []
            discovery_cfg["web"]["market_keywords"] = []
        if "reddit" in discovery_cfg:
            discovery_cfg["reddit"]["problem_keywords"] = queries
        if "github" in discovery_cfg:
            discovery_cfg["github"]["problem_keywords"] = queries

        print(f"Running focused discovery with {len(queries)} queries...\n")

    if args.fresh:
        print("=== FRESH MODE: Bypassing signal cache ===")
        app.discovery_bypass_cache = True

    await app.run()


async def cmd_run_once(args: argparse.Namespace, _app: AutoResearcher) -> None:
    app = AutoResearcher(config_path=args.config)
    original_focus: dict[str, Any] | None = None

    try:
        if args.pattern:
            from src.opportunity_engine import PATTERN_TO_DISCOVERY_QUERIES

            if args.pattern not in PATTERN_TO_DISCOVERY_QUERIES:
                print(f"Unknown pattern: {args.pattern}")
                print(f"Available patterns: {list(PATTERN_TO_DISCOVERY_QUERIES.keys())}")
                raise SystemExit(1)

            queries = PATTERN_TO_DISCOVERY_QUERIES[args.pattern]
            print(f"=== FOCUSED DISCOVERY: {args.pattern} ===")
            print(f"Queries: {queries[:3]}...")
            discovery_cfg = app.config.get("discovery", {})
            original_focus = {
                "sources": list(discovery_cfg.get("sources", []) or []),
                "web_keywords": list(discovery_cfg.get("web", {}).get("keywords", []) or []),
                "web_problem_keywords": list(discovery_cfg.get("web", {}).get("problem_keywords", []) or []),
                "web_success_keywords": list(discovery_cfg.get("web", {}).get("success_keywords", []) or []),
                "web_market_keywords": list(discovery_cfg.get("web", {}).get("market_keywords", []) or []),
                "reddit_keywords": list(discovery_cfg.get("reddit", {}).get("problem_keywords", []) or []),
                "github_keywords": list(discovery_cfg.get("github", {}).get("problem_keywords", []) or []),
                "focused_problem_only": bool(discovery_cfg.get("focused_problem_only", False)),
            }
            focused_sources = [source for source in ["reddit", "web", "github"] if source in (discovery_cfg.get("sources", []) or [])]
            if focused_sources:
                discovery_cfg["sources"] = focused_sources
            discovery_cfg["focused_problem_only"] = True
            if "web" in discovery_cfg:
                discovery_cfg["web"]["keywords"] = queries
                discovery_cfg["web"]["problem_keywords"] = queries
                discovery_cfg["web"]["success_keywords"] = []
                discovery_cfg["web"]["market_keywords"] = []
            if "reddit" in discovery_cfg:
                discovery_cfg["reddit"]["problem_keywords"] = queries
            if "github" in discovery_cfg:
                discovery_cfg["github"]["problem_keywords"] = queries

            print(f"Running focused discovery with {len(queries)} queries...\n")

        if args.fresh:
            print("=== FRESH MODE: Bypassing signal cache ===")
            app.discovery_bypass_cache = True

        summary = await app.run_once(skip_backlog=bool(args.skip_backlog))
        if args.verbose:
            print_json(build_verbose_report(app, summary))
        else:
            print_json(summary)
    finally:
        if args.pattern and original_focus is not None:
            discovery_cfg = app.config.get("discovery", {})
            if original_focus.get("sources"):
                discovery_cfg["sources"] = original_focus["sources"]
            discovery_cfg["focused_problem_only"] = bool(original_focus.get("focused_problem_only", False))
            if "web" in discovery_cfg:
                discovery_cfg["web"]["keywords"] = original_focus.get("web_keywords", [])
                discovery_cfg["web"]["problem_keywords"] = original_focus.get("web_problem_keywords", [])
                discovery_cfg["web"]["success_keywords"] = original_focus.get("web_success_keywords", [])
                discovery_cfg["web"]["market_keywords"] = original_focus.get("web_market_keywords", [])
            if "reddit" in discovery_cfg:
                discovery_cfg["reddit"]["problem_keywords"] = original_focus.get("reddit_keywords", [])
            if "github" in discovery_cfg:
                discovery_cfg["github"]["problem_keywords"] = original_focus.get("github_keywords", [])
        close_app = getattr(app, "shutdown", None)
        if callable(close_app):
            await close_app()

    if args.pattern:
        print(f"\n=== Focused discovery complete ===")
        print(f"Use 'python3 cli.py patterns' to see updated pattern counts")


async def cmd_run_unseeded(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args, start_new_run=True) as app:
        summary = await app.run_unseeded(
            vertical=args.vertical,
            max_findings=args.max_findings,
        )
        print_json(summary)


async def cmd_deep_research(args: argparse.Namespace, _app: AutoResearcher) -> None:
    async with app_context(args, start_new_run=True) as app:
        from src.agents.deep_research import DeepResearchAgent

        agent = DeepResearchAgent(
            name="deep_research",
            db=app.db,
            vertical=args.vertical,
        )
        summary = await agent.run_deep_research(
            max_signals_per_source=args.max_findings,
        )
        print_json(summary)


async def cmd_simple_list(args: argparse.Namespace, app: AutoResearcher, attr: str, method: str, limit: int = 100) -> None:
    items = getattr(app.db, method)(limit=limit) if app.db else []
    print_json([item.__dict__ for item in items])


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Callable[..., Awaitable[None]]] = {
    "watch": cmd_watch,
    "eval": cmd_eval,
    "reddit-relay": cmd_reddit_relay,
    "reddit-seed": cmd_reddit_seed,
    "check-bridge": cmd_check_bridge,
    "backup-db": cmd_backup_db,
    "suggest-discovery": cmd_suggest_discovery,
    "term-lifecycle": cmd_term_lifecycle,
    "term-state": cmd_term_state,
    "security-scan": cmd_security_scan,
    "generate-docs": cmd_generate_docs,
    "sre-health": cmd_sre_health,
    "patterns": cmd_patterns,
    "scoring-report": cmd_scoring_report,
    "revalidate": cmd_revalidate,
    "rescreen": cmd_rescreen,
    "rescore-v4": cmd_rescore_v4,
    "run": cmd_run,
    "run-once": cmd_run_once,
    "run-unseeded": cmd_run_unseeded,
    "deep-research": cmd_deep_research,
}


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
            "high-leverage-report",
            "wedge-eval",
            "report",
            "gate-diagnostics",
            "pipeline-health",
            "backlog-workbench",
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
            "rescreen",
            "rescore-v4",
            "term-lifecycle",
            "term-state",
            "security-scan",
            "generate-docs",
            "sre-health",
        ],
        help="Command to execute",
    )
    parser.add_argument("query", nargs="?", help="Search query for the search command")
    parser.add_argument("--verbose", action="store_true", help="Print operator-facing runtime details")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval for watch mode")
    parser.add_argument("--host", help="Host override for reddit-relay")
    parser.add_argument("--port", type=int, help="Port override for reddit-relay")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="Path to behavior eval fixture set")
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
    parser.add_argument("--wedge-id", type=int, help="Specific opportunity ID for security-scan, sre-health")
    parser.add_argument("--code", type=str, help="Code to scan (for security-scan)")
    parser.add_argument("--file-name", type=str, default="stdin", help="File name for security-scan context")
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
    parser.add_argument(
        "--skip-backlog",
        action="store_true",
        help="run-once: discovery-only mode; do not requeue existing qualified findings for evidence",
    )
    # term-lifecycle and term-state arguments
    parser.add_argument("--term-type", default=None, help="Term type: keyword or subreddit")
    parser.add_argument("--term-value", default=None, help="Term value to modify")
    parser.add_argument("--action", default=None, help="term-state action: list, ban, reactivate, complete, reset, high-performers, exhausted")
    parser.add_argument("--state", default=None, help="Filter by state")
    args = parser.parse_args()

    # --- Dispatch to extracted command handlers ---
    handler = COMMANDS.get(args.command)
    if handler:
        await handler(args, None)
        return

    # --- Commands that share a common app context ---
    async with app_context(args) as app:
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
        elif args.command == "high-leverage-report":
            print_json(app.high_leverage_report(limit=args.limit))
        elif args.command == "wedge-eval":
            from src.builder_output import WedgeEvaluator

            # Use heuristic-only for CLI batch evaluation (LLM is too slow for batch)
            evaluator = WedgeEvaluator(app.db, {})
            import sqlite3 as _sq
            _conn = _sq.connect(str(resolve_database_path_from_config(args.config)))
            statuses = ["build_ready", "prototype_candidate", "prototype_ready"]
            placeholders = ",".join("?" * len(statuses))
            _rows = _conn.execute(
                f"SELECT id FROM opportunities WHERE selection_status IN ({placeholders}) ORDER BY composite_score DESC",
                statuses,
            ).fetchall()
            _conn.close()
            results = []
            for r in _rows:
                ev = evaluator.evaluate_sync(r[0])
                results.append({
                    "opportunity_id": ev.opportunity_id,
                    "verdict": ev.verdict,
                    "passes_gate": ev.passes_wedge_gate,
                    "software_fit": round(ev.software_fit, 3),
                    "monetization_fit": round(ev.monetization_fit, 3),
                    "is_narrow": ev.is_narrow,
                    "trust_risk": ev.trust_risk,
                    "narrowness_reason": ev.narrowness_reason,
                    "monetization_reason": ev.monetization_reason,
                    "suggested_mvp": ev.suggested_mvp,
                    "first_paid_offer": ev.first_paid_offer,
                    "pricing_hypothesis": ev.pricing_hypothesis,
                    "first_customer": ev.first_customer,
                    "first_channel": ev.first_channel,
                    "evaluated_by": ev.evaluated_by,
                })
            print_json(results)
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
        elif args.command == "backlog-workbench":
            print_json(app.db.get_backlog_workbench(limit=args.limit) if app.db else [])
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
            requalified = False
            if args.label == "needs_more_evidence":
                requalified = bool(
                    app.db.requalify_finding_for_evidence(
                        args.finding_id,
                        review_label=args.label,
                        note=args.note,
                    )
                )
            print_json(
                {
                    "review_id": review_id,
                    "finding_id": args.finding_id,
                    "label": args.label,
                    "requalified": requalified,
                }
            )
        elif args.command == "ideas":
            print_json([item.__dict__ for item in app.db.get_ideas(limit=100)] if app.db else [])
        elif args.command == "products":
            print_json(app.db.get_products(limit=100) if app.db else [])


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
