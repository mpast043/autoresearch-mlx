# AutoResearch MLX

AutoResearch MLX is an evidence-first weak-signal discovery and validation pipeline. It pulls candidate pain signals from public sources, classifies them through an explicit source-policy layer, extracts structured problem atoms, validates recurrence and commercial value, and only then prepares build-facing artifacts.

The active runtime path is:

`discovery -> evidence -> validation -> build_prep`

Optional downstream stages exist for ideation and code generation, but they sit behind validation and build-prep gates. Optional agents (SecurityAgent, SREAgent, TechnicalWriterAgent, DeepResearchAgent) can be enabled in `config.yaml`.

## Branch Focus

This branch, `codex/reconciliation-narrow-mode`, is intentionally narrower than the broader multi-source system described in some older docs and commits.

Current branch defaults:

- `discovery.sources = ["reddit"]`
- `discovery.auto_expand = false`
- `discovery.llm_expansion.enabled = false`
- `discovery.reddit.use_r_all = false`
- `discovery.reddit.search_sorts = ["new"]`
- discovery is centered on reconciliation / payout mismatch / spreadsheet-heavy finance and seller reporting workflows

The default practitioner subreddit pack on this branch is:

- `accounting`
- `Bookkeeping`
- `quickbooksonline`
- `Netsuite`
- `smallbusiness`

The default keyword pack on this branch is also intentionally narrow:

- `manual reconciliation`
- `invoice reconciliation`
- `payment reconciliation`
- `shopify payout reconciliation`
- `stripe payout reconciliation`
- `quickbooks reconciliation`
- `bank deposit reconciliation`
- `month end close spreadsheet`
- `invoice does not match payment`
- `bank deposits not matching invoices`
- `partial payment reconciliation`
- `csv cleanup before import`
- `accounts receivable follow up spreadsheet`
- `credit memo tracking spreadsheet`

## What The System Does

- Discovers candidate pain signals from Reddit, web search, GitHub, WordPress reviews, Shopify reviews, and YouTube lanes.
- Routes source material through source-policy classification so only `pain_signal` evidence can become `raw_signals` and `problem_atoms`.
- Clusters similar atoms into recurring opportunities.
- Runs recurrence, corroboration, and market/value enrichment before making `promote` / `park` / `kill` decisions.
- Persists build briefs and build-prep outputs only for `prototype_candidate` opportunities.
- Learns from both good yield and bad yield to adjust discovery scope over time.
- Supports autonomous Reddit keyword/subreddit expansion while still preserving curated practitioner lanes.
- Scans generated solutions for OWASP Top 10 vulnerabilities (SecurityAgent).
- Monitors wedge health and detects regressions (SREAgent).
- Auto-generates documentation for build-ready opportunities (TechnicalWriterAgent).
- Supports targeted deep research across multiple sources for specific verticals (DeepResearchAgent).

On this branch, the most important runtime capability is narrower:

- replaying screened-out or parked reconciliation-style findings through `review-mark` and `rescreen`
- validating whether practitioner-style process questions survive recurrence and corroboration
- keeping discovery disciplined inside a small boring-money lane instead of broad adjacent-space exploration

## Quick Start

```bash
# Install editable package + console script
python -m pip install -e .

# Install dev dependencies (pytest, pytest-asyncio)
python -m pip install -r requirements-dev.txt

# Copy environment template and fill in API keys
cp .env.example .env

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

**Prerequisites:** Python >=3.10, [Ollama](https://ollama.ai) running locally for builder_v2 and build-prep classification (defaults to `gemma4:latest`).

The repo-root entrypoint still works for local development via `python cli.py ...`, but the packaged console script is now the preferred surface.

## Key Commands

```bash
# Core runtime
autoresearch run
autoresearch run-once
autoresearch run-once --skip-backlog --verbose
autoresearch watch

# Deep research (targeted vertical exploration)
autoresearch deep-research --vertical devtools
autoresearch deep-research --vertical ecommerce
autoresearch run-unseeded --vertical devtools

# Diagnostics and operator review
autoresearch report
autoresearch gate-diagnostics
autoresearch pipeline-health
autoresearch scoring-report
autoresearch backlog-workbench --limit 20
autoresearch review-queue
autoresearch review-mark --finding-id 10 --label needs_more_evidence --note "plausible but thin"
autoresearch rescreen --finding-id 10
autoresearch rescreen --limit 1000

# Operator workbenches
autoresearch workbench
autoresearch decision-surface
autoresearch operator-report
autoresearch builder-jobs

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
autoresearch ideas
autoresearch products
autoresearch patterns
autoresearch search <query>

# Discovery support
autoresearch suggest-discovery --min-atoms 2 --limit 25
autoresearch discovery-sort-diagnostics

# Re-scoring and term management
autoresearch revalidate
autoresearch rescore-v4
autoresearch term-lifecycle
autoresearch term-state <action>   # ban/reactivate/complete/reset/high-performers/exhausted/wedge-quality/specificity/platform-native/abstraction-collapse/buildable

# Security and SRE
autoresearch security-scan
autoresearch sre-health
autoresearch generate-docs

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

# Optional: Anthropic SDK, scikit-learn, sentence-transformers
python -m pip install -r requirements-optional.txt

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

Key agents:

| Agent | Module | Purpose |
|-------|--------|---------|
| DiscoveryAgent | `src/agents/discovery.py` | Harvests pain signals from configured sources |
| EvidenceAgent | `src/agents/evidence.py` | Gathers corroboration and market enrichment |
| ValidationAgent | `src/agents/validation.py` | Clusters, scores, promotes/parks/kills |
| Build-prep chain | `src/agents/build_prep.py` | Solution framing, experiment design, spec generation |
| IdeationAgent | `src/agents/ideation.py` | Turns promoted opportunities into research briefs |
| BuilderAgent | `src/agents/builder.py` | Deterministic local product artifacts |
| BuilderAgentV2 | `src/agents/builder_v2.py` | LLM code-generating builder (Ollama) |
| DeepResearchAgent | `src/agents/deep_research.py` | Multi-source vertical synthesis |
| SecurityAgent | `src/agents/security.py` | OWASP Top 10 vulnerability scanning |
| SREAgent | `src/agents/sre.py` | Wedge health monitoring and regression detection |
| TechnicalWriterAgent | `src/agents/technical_writer.py` | Auto-generate documentation |

The important persisted dataflow is:

`finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> experiment + validation -> build_brief + build_prep_outputs -> ledger`

Run-scoped history is preserved for validations, experiments, corroborations, market enrichments, evidence ledger entries, build briefs, and build-prep outputs.

## Discovery And Expansion

On this branch, Reddit discovery is intentionally narrow and practitioner-first. Curated operator-heavy subreddits are front-loaded in capped waves:

- `accounting`
- `Bookkeeping`
- `quickbooksonline`
- `Netsuite`
- `smallbusiness`

Broad Reddit exploration is intentionally disabled here:

- `use_r_all = false`
- `auto_expand = false`
- `llm_expansion.enabled = false`

This branch is meant to answer a much narrower question:

- can the system repeatedly find and confirm reconciliation / payout / close-cleanup pain from practitioner communities?

That means the discovery loop is optimized for:

- named incumbent systems
- recurring finance or seller operations workflows
- spreadsheet-heavy manual cleanup
- operator-style “how are you handling this?” threads

It is not optimized here for:

- broad adjacent-market discovery
- generic startup-idea generation
- multi-source novelty hunting across web / GitHub / review lanes

## Configuration

Primary runtime config lives in [config.yaml](/Users/meganpastore/Projects/autoresearch-mlx/config.yaml).

For isolated fresh verification runs, start from [configs/fresh-verify.ollama.example.yaml](/Users/meganpastore/Projects/autoresearch-mlx/configs/fresh-verify.ollama.example.yaml) and run:

```bash
python cli.py run-once --config configs/fresh-verify.ollama.example.yaml --fresh --verbose
python cli.py run-once --config configs/fresh-verify.ollama.example.yaml --pattern stripe_to_quickbooks --fresh --verbose
```

Important sections:

- `database.path`
- `output_dir`
- `discovery.sources`
- `discovery.reddit`
- `discovery.candidate_filter`
- `discovery.llm_expansion`
- `orchestration`
- `validation`
- `builder`
- `security`
- `technical_writer`
- `sre`
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

For this branch specifically, the most useful operator loop is:

1. `autoresearch run-once --verbose`
2. `autoresearch pipeline-health`
3. `autoresearch review-queue`
4. `autoresearch review-mark --finding-id ... --label needs_more_evidence`
5. `autoresearch rescreen --finding-id ...` or `autoresearch rescreen --limit 1000`
6. `autoresearch run-once --verbose` again to replay the newly qualified backlog

`rescreen` matters on this branch because many of the fixes are about admitting or replaying older practitioner-style findings under newer source-policy and recurrence rules. `rescore-v4` and `revalidate` do not replace that workflow; they operate on later-stage opportunity state.

## Documentation Map

- Engineering spec: [docs/ENGINEERING_SPEC.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/ENGINEERING_SPEC.md)
- State model: [docs/state-model.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/state-model.md)
- Gates: [docs/gates.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/gates.md)
- Loop operations: [docs/LOOP.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/LOOP.md)
- Product loop: [docs/PRODUCT_LOOP.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/PRODUCT_LOOP.md)
- Calibration: [docs/CALIBRATION_v4.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/CALIBRATION_v4.md)
- Build brief contract: [docs/build_brief_contract.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/build_brief_contract.md)
- Recovery: [docs/RECOVERY.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/RECOVERY.md)
- Production runtime: [docs/production-runtime.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/production-runtime.md)
- Render deploy: [docs/render-deploy.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/render-deploy.md)
- Codespaces setup: [docs/codespaces.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/codespaces.md)

## Tests

```bash
pytest tests/ -v
```

Test configuration: `pytest.ini` sets `asyncio_mode = auto` (no `@pytest.mark.asyncio` decorator needed).

**Note:** No test coverage exists for `ideation.py`, `builder.py`, `deep_research.py`, or `competitor_intel.py`.
