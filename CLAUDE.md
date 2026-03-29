# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoResearch MLX is an evidence-first weak-signal problem discovery and validation pipeline. It harvests raw pain signals from multiple sources (Reddit, GitHub, WordPress Plugin Directory, Shopify App Store, YouTube, web search), screens and classifies them, extracts structured problem atoms, clusters recurring patterns, scores market opportunities, plans falsifiable experiments, and gates validated opportunities toward build briefs.

The default pipeline path is: `discovery -> evidence -> validation -> build_prep`. **Auto-ideate** can be enabled in `config.yaml` (`orchestration.auto_ideate_after_validation`); it runs on **promote** (`passed`) or on **prototype_candidate** when a **build brief** exists (see `docs/PRODUCT_LOOP.md`). **Auto-build** remains off by default (`builder.auto_build`).

## Commands

```bash
# Run the pipeline
python cli.py run                    # discovery waves until SIGINT; optional stop_on_hit (see config)
python cli.py run-once              # single discovery run
python cli.py run-once --verbose    # verbose output with artifact counts and decision logs
python cli.py watch                  # continuous watch mode (renders pipeline_status.json)

# Deep research (targeted vertical exploration)
python cli.py deep-research --vertical devtools   # multi-source synthesis for devtools vertical
python cli.py deep-research --vertical ecommerce # ecommerce vertical

# Behavioral evaluation
python cli.py eval                   # runs gold-set behavioral eval harness

# Operator review
python cli.py review-queue           # shows borderline parked/killed cases
python cli.py review-mark --finding-id 10 --label needs_more_evidence --note "plausible but thin"

# Reddit relay
python cli.py check-bridge         # hosted relay health (Render URL + token from .env)
python cli.py backup-db            # copy data/autoresearch.db → data/backups/ (see docs/RECOVERY.md)
python cli.py suggest-discovery [--min-atoms N] [--limit N]   # JSON hints for discovery.reddit keywords/subs from DB
python cli.py reddit-relay --host 127.0.0.1 --port 8787
python cli.py reddit-seed

# End-to-end loop (discovery → validation) — see docs/LOOP.md

# Data inspection
python cli.py findings
python cli.py signals
python cli.py atoms
python cli.py clusters
python cli.py opportunities
python cli.py experiments
python cli.py ledger
python cli.py build-briefs
python cli.py build-prep
python cli.py report     # includes run_diff, decision_reason_mix, corroboration_depth, wedge provenance
python cli.py gate-diagnostics   # why promote/park/kill + selection_status (see docs/gates.md)
python cli.py pipeline-health      # why run_once may show 0 new validations (backlog + dedupe)

# Dashboard (Next.js, port 3001)
cd dashboard/
npm install
npm run dev
npm run build
npm run lint

# Tests
pytest tests/ -v
```

## Architecture

### Pipeline Stages
```
discovery -> evidence -> validation -> build_prep
```
Messages flow through an async priority queue (`src/messaging.py`). Lower priority number = higher precedence (1 is highest).

### Agents
- **DiscoveryAgent** (`src/agents/discovery.py`) — harvests from configured sources, runs source policy screening, stores `RawSignal` + `ProblemAtom`
- **EvidenceAgent** (`src/agents/evidence.py`) — gathers corroboration and market enrichment per run
- **ValidationAgent** (`src/agents/validation.py`) — clusters atoms, scores opportunities, searches counterevidence, plans experiments, makes promote/park/kill decisions
- **Build-prep chain** (`src/agents/build_prep.py`) — `solution_framing -> experiment_design -> spec_generation` only runs for `prototype_candidate` opportunities
- **Optional** (`src/agents/ideation.py`, `src/agents/builder.py`) — disabled by default; enable via `orchestration.auto_ideate_after_validation` and `builder.auto_build` in config

### Key Files
| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point, all commands |
| `run.py` | `AutoResearcher` application class, async runtime |
| `config.yaml` | All configuration (sources, weights, thresholds, API keys via `${ENV_VAR}`) |
| `src/database.py` | SQLite schema and CRUD — source of truth at runtime |
| `src/orchestrator.py` | Message routing between agents |
| `src/messaging.py` | `MessageQueue`, `MessageType`, async message protocol |
| `src/opportunity_engine.py` | Clustering, scoring, falsification (~110K file) |
| `src/research_tools.py` | Reddit, GitHub, YouTube, web scraping, DuckDuckGo integrations (~115K file) |
| `src/source_policy.py` | Signal classification: `pain_signal`, `success_signal`, `demand_signal`, `competition_signal`, `meta_guidance`, `low_signal_summary` — only `pain_signal` enters atom generation |

### Data Model
The authoritative schema is in `src/database.py`. Key runtime tables:
- `findings` — source intake + lifecycle status (`new -> qualified -> screened_out/parked/killed/promoted`)
- `raw_signals` — normalized accepted evidence
- `problem_atoms` — structured pain extraction
- `corroborations` / `market_enrichments` — run-scoped, keyed by `(run_id, finding_id)`, idempotent within a run
- `clusters` / `cluster_members` — atom pattern grouping
- `opportunities` — latest-state scored opportunity per cluster, with `selection_status` (`research_more -> prototype_candidate -> prototype_ready -> build_ready -> launched`)
- `experiments` — proposed validation experiments
- `evidence_ledger` — per-run evidence history for audit trail
- `review_feedback` — operator labels for calibration (`correct`, `false_positive`, `bad_extraction`, `should_park`, `should_kill`, `needs_more_evidence`)
- `build_briefs` / `build_prep_outputs` — post-validation handoffs

State model documentation: `docs/state-model.md`

Full architecture / DB alignment review: `docs/CODE_REVIEW_FULL.md`

## Configuration

All settings are in `config.yaml`. Adding a source to `discovery.sources` activates it:
```yaml
discovery:
  sources:
    - "reddit"
    - "github"
    - "wordpress_reviews"
    - "shopify_reviews"
```

Supported `discovery.sources` values are implemented in `DiscoveryAgent._check_source` (`reddit`, `github`, `youtube`, `web`, `wordpress_reviews`, `shopify_reviews`). **`web`** runs DuckDuckGo-style web discovery (success + problem queries) and marketplace problem threads; tune optional `discovery.web.keywords`. **`wordpress_reviews`** / **`shopify_reviews`** are explicit review-only lanes (1–2★ pain). Reddit: **`discovery.reddit.search_time_filter`** (`all`, `year`, …) for historic search; **`discovery.reddit.use_r_all`** to search **r/all** instead of only `problem_subreddits`. **`max_subreddits_per_wave`** / **`max_keywords_per_wave`**: cap sub×query pairs per wave; use **`0`** for **no cap** (every sub × every keyword in config). Tune **`pair_concurrency`** if rate-limited.

Gates span discovery filtering, `stage_decision` (composite vs promotion/park thresholds), and build-prep `determine_selection_state`. **Authoritative reference:** `docs/gates.md`.

`config.yaml` examples:
- `discovery.candidate_filter` — discovery-time heuristic (not validation composite)
- `validation.promotion_threshold` / `validation.park_threshold` — passed to `stage_decision` (see `src/validation_thresholds.py` for resolution order vs `validation.decisions.*` and `orchestration.*`)

Default validation weights (if unset): `market: 0.40, technical: 0.35, distribution: 0.25`

**Continuous waves (`orchestration.continuous_waves`):** set `true` so `python cli.py run` repeats discovery after each drained pass. `stop_on_hit.retry_interval_seconds` sets the pause between waves — use **`0`** for immediate next wave (only the pipeline drain wait applies). Ctrl+C stops anytime. Does not require `stop_on_hit.enabled`.

**Stop after a “hit” (`orchestration.stop_on_hit`):** set `enabled: true` to exit `python cli.py run` when validation reports a matching `selection_status` (default includes `prototype_candidate`) or `decision` (default includes `promote`). Set `exit_on_hit: false` to log matches but keep running (useful with `continuous_waves`). Between waves, `retry_interval_seconds` sleeps before the next discovery pass.

API keys use `${ENV_VAR}` syntax and are loaded from `.env` at startup via `load_local_env()` in `run.py`.

## Patterns and Conventions

1. **Async/await throughout** — all agents use `asyncio`; never use blocking I/O in agent code
2. **Dataclasses for data models** — `Finding`, `RawSignal`, `ProblemAtom`, `Opportunity`, etc.
3. **SQLite-first persistence** — all state to `data/autoresearch.db`
4. **Run-scoped history** — `corroborations`, `market_enrichments`, `validations`, `evidence_ledger` are keyed by `(run_id, entity_id)`; repeated writes in the same run update in place
5. **Source policy routing** — `src/source_policy.py` classifies all signals; only `pain_signal` enters atom generation
6. **Reddit relay fallback** — `reddit_bridge.enabled: true` calls external relay; falls back to local scraping on failure (timeout, auth failure, bad shape, unavailable)
7. **Wedge scoring default-off** — activates only after candidate detection, fit validation, and runtime sanity check all pass
8. **Build prep gated** — only runs for `prototype_candidate` opportunities; never auto-routes from validation directly to build

## Gotchas

- Reddit relay requires a public HTTPS URL; set `REDDIT_BRIDGE_BASE_URL` + `REDDIT_BRIDGE_AUTH_TOKEN` env vars
- `src/opportunity_engine.py` and `src/research_tools.py` are large (~110K and ~115K respectively) — contains core business logic
- Review feedback calibration requires repeated consistent labels and age-based decay; single labels don't bypass staged evidence model
- `shopify_reviews` lane uses bounded popularity proxies (review count) since install counts are not public
- GitHub issues with generic feature wishlists or thin product noise are screened out before atom generation
- Live status file: `output/pipeline_status.json`; runtime log: `output/autoresearcher.log`
