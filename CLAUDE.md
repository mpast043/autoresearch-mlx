# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoResearch MLX is an evidence-first weak-signal problem discovery and validation pipeline. It harvests raw pain signals from multiple sources (Reddit, GitHub, WordPress Plugin Directory, Shopify App Store, YouTube, web search), screens and classifies them, extracts structured problem atoms, clusters recurring patterns, scores market opportunities, plans falsifiable experiments, and gates validated opportunities toward build briefs.

The default pipeline path is: `discovery -> evidence -> validation -> build_prep`. Auto-ideate and auto-build are disabled by default.

## Commands

```bash
# Run the pipeline
python cli.py run-once              # single discovery run
python cli.py run-once --verbose    # verbose output with artifact counts and decision logs
python cli.py watch                  # continuous watch mode (renders pipeline_status.json)

# Behavioral evaluation
python cli.py eval                   # runs gold-set behavioral eval harness

# Operator review
python cli.py review-queue           # shows borderline parked/killed cases
python cli.py review-mark --finding-id 10 --label needs_more_evidence --note "plausible but thin"

# Reddit relay
python cli.py reddit-relay --host 127.0.0.1 --port 8787
python cli.py reddit-seed

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

Stage gates in `config.yaml`:
- `signal_min_confidence: 0.35` — minimum confidence to persist a signal
- `cluster_min_atoms: 2` — minimum atoms to form a cluster
- `promote_min_score: 0.66` — gate to `prototype_candidate`
- `park_min_score: 0.42` — gate to `parked`

Validation weights: `market: 0.40, technical: 0.35, distribution: 0.25`

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
