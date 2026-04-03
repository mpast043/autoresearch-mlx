# AutoResearch MLX

AutoResearch MLX is an evidence-first weak-signal discovery and validation pipeline. It pulls candidate pain signals from public sources, classifies them through an explicit source-policy layer, extracts structured problem atoms, validates recurrence and commercial value, and only then prepares build-facing artifacts.

The active runtime path is:

`discovery -> evidence -> validation -> build_prep`

Optional downstream stages exist for ideation and code generation, but they sit behind validation and build-prep gates.

## What The System Does

- Discovers candidate pain signals from Reddit, web search, GitHub, WordPress reviews, Shopify reviews, and YouTube lanes.
- Routes source material through source-policy classification so only `pain_signal` evidence can become `raw_signals` and `problem_atoms`.
- Clusters similar atoms into recurring opportunities.
- Runs recurrence, corroboration, and market/value enrichment before making `promote` / `park` / `kill` decisions.
- Persists build briefs and build-prep outputs only for `prototype_candidate` opportunities.
- Learns from both good yield and bad yield to adjust discovery scope over time.
- Supports autonomous Reddit keyword/subreddit expansion while still preserving curated practitioner lanes.

## Quick Start

```bash
# Install editable package + console script
python -m pip install -e .

# Single pass
autoresearch run-once

# Single pass with operator detail
autoresearch run-once --verbose

# Discovery-only sample (skip backlog replay)
autoresearch run-once --skip-backlog --verbose

# Continuous discovery waves
autoresearch run

# Live status snapshot loop
autoresearch watch
```

The repo-root entrypoint still works for local development via `python cli.py ...`, but the packaged console script is now the preferred surface.

## Key Commands

```bash
# Core runtime
autoresearch run
autoresearch run-once
autoresearch run-once --skip-backlog --verbose
autoresearch watch

# Diagnostics and operator review
autoresearch report
autoresearch gate-diagnostics
autoresearch pipeline-health
autoresearch backlog-workbench --limit 20
autoresearch review-queue
autoresearch review-mark --finding-id 10 --label needs_more_evidence --note "plausible but thin"

# Data inspection
autoresearch findings
autoresearch signals
autoresearch atoms
autoresearch clusters
autoresearch opportunities
autoresearch experiments
autoresearch ledger
autoresearch build-briefs
autoresearch build-prep

# Discovery support
autoresearch suggest-discovery --min-atoms 2 --limit 25
autoresearch patterns

# Relay and recovery
autoresearch check-bridge
autoresearch reddit-seed
autoresearch backup-db

# Evaluation
autoresearch eval
```

## Installation

```bash
# Editable local install
python -m pip install -e .

# Verify the console script works outside the repo root
cd /tmp
autoresearch eval
```

The package now ships:

- a console entrypoint: `autoresearch`
- the Python modules needed for `import cli`, `import run`, and `import src...`
- bundled fallback defaults for `config.yaml` and `evals/behavior_gold.json`

That means imports and `autoresearch eval` work correctly even when the current working directory is not the repository root.

## Runtime Model

The runtime uses:

- `run.py` for process orchestration and lifecycle
- `src/orchestrator.py` for message routing
- `src/messaging.py` for the async per-agent message bus
- `src/database.py` for the SQLite schema and CRUD
- `src/status_tracker.py` for `output/pipeline_status.json`

The important persisted dataflow is:

`finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> experiment + validation -> build_brief + build_prep_outputs -> ledger`

Run-scoped history is preserved for validations, experiments, corroborations, market enrichments, evidence ledger entries, build briefs, and build-prep outputs.

## Discovery And Expansion

Reddit discovery is now practitioner-first by default. Curated operator-heavy subreddits are front-loaded in capped waves:

- `accounting`
- `smallbusiness`
- `ecommerce`
- `shopify`
- `EtsySellers`

Broader communities such as `projectmanagement`, `automation`, and `indiehackers` can still participate, but they are pushed later in rotation.

Autonomous expansion is still enabled through `discovery.auto_expand`. Expanded subreddits and keywords are merged into the runtime pool, then discovery planning ranks and rotates them alongside the curated base set. The result is:

- curated practitioner lanes stay available every run
- expanded lanes can enter future waves
- low-yield pairs can be cooled down instead of permanently polluting the front of the queue

## Configuration

Primary runtime config lives in [config.yaml](/Users/meganpastore/Projects/autoresearch-mlx/config.yaml).

Important sections:

- `database.path`
- `output_dir`
- `discovery.sources`
- `discovery.source_selection`
- `discovery.expansion`
- `discovery.reddit`
- `discovery.web`
- `discovery.shopify_reviews`
- `discovery.wordpress_reviews`
- `orchestration`
- `validation`
- `builder`
- `llm`
- `reddit_bridge`
- `reddit_relay`

Environment variables are loaded from `.env` at startup via `load_local_env()` in [run.py](/Users/meganpastore/Projects/autoresearch-mlx/run.py).

## Observability

Important runtime artifacts:

- DB: [data/autoresearch.db](/Users/meganpastore/Projects/autoresearch-mlx/data/autoresearch.db)
- Log: [output/autoresearcher.log](/Users/meganpastore/Projects/autoresearch-mlx/output/autoresearcher.log)
- Status JSON: [output/pipeline_status.json](/Users/meganpastore/Projects/autoresearch-mlx/output/pipeline_status.json)

Use these together with `python cli.py report`, `python cli.py gate-diagnostics`, and `python cli.py pipeline-health` when tuning discovery or validation.

## Documentation Map

- Engineering spec: [docs/ENGINEERING_SPEC.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/ENGINEERING_SPEC.md)
- State model: [docs/state-model.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/state-model.md)
- Gates: [docs/gates.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/gates.md)
- Loop operations: [docs/LOOP.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/LOOP.md)
- Product loop: [docs/PRODUCT_LOOP.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/PRODUCT_LOOP.md)

## Tests

```bash
pytest tests/ -v
```
