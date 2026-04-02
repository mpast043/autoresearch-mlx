# AutoResearch MLX

AutoResearch MLX is an evidence-first weak-signal discovery system. It is designed to pull in potential pain signals from multiple public sources, route them through an explicit source-policy layer, enrich them with corroboration and market context, validate them, and only then prepare narrow build briefs for promising opportunities.

The current pipeline shell is:

`discovery -> evidence -> validation -> build_prep`

This repository currently contains the policy, source adapters, evaluation fixtures, state-model documentation, build-prep helpers, tests, data artifacts, and the Reddit bridge scaffold that support that flow.

## What The System Does

- Collects source items from multiple public lanes.
- Separates pain evidence from low-signal summaries, success chatter, demand context, competition context, and meta guidance.
- Extracts structured problem atoms only from eligible pain signals.
- Persists corroboration and market enrichment separately.
- Validates opportunities before any build-prep work happens.
- Gates validated opportunities into explicit post-validation selection states instead of allowing direct promotion into building.
- **Self-expanding discovery**: Automatically adds new keywords and subreddits based on what patterns yield validated opportunities.
- **Idea generation**: Creates product ideas from validated opportunities.
- **Code generation**: Generates functional prototypes using Ollama (local LLM).

## Full Pipeline

The complete pipeline is:

`discovery -> evidence -> validation -> build_prep -> ideation -> builder`

- **Discovery**: Finds pain signals from configured sources
- **Evidence**: Gathers corroboration and market enrichment
- **Validation**: Clusters atoms, scores opportunities, makes promote/park/kill decisions
- **Build Prep**: Solution framing, experiment design, spec generation (for prototype_candidate)
- **Ideation**: Creates product ideas from validated opportunities (optional, via `auto_ideate_after_validation`)
- **Builder**: Generates functional code from ideas using Ollama (optional, via `auto_build`)

Enable optional stages in `config.yaml`:
```yaml
orchestration:
  auto_ideate_after_validation: true  # Enable idea generation

builder:
  auto_build: true  # Enable code generation
```

## Source Families In Scope

The current source-policy-aware lanes include:

- Reddit (problem posts from configured subreddits + r/all)
- GitHub issues and discussions
- WordPress Plugin Directory reviews (1-2 star pain)
- Shopify App Store reviews (1-2 star pain)
- Web search (DuckDuckGo - problem + success queries)
- YouTube (comments)

**Deep Research** - Multi-source synthesis agent for targeted vertical exploration:
```bash
python cli.py deep-research --vertical <name>  # devtools, ecommerce, etc.
```

Important policy rules:
- Reddit `search_time_filter: year` - only recent posts (configurable)
- review text or issue text can become `pain_signal` only when specific enough
- ratings, counts, pricing, popularity proxies, and listing metadata are enrichment inputs only
- vague summaries, marketing copy, generic praise, and thin product-specific noise are screened out

The source-policy implementation lives in [src/source_policy.py](./src/source_policy.py).

## Self-Expanding Discovery

The system can autonomously expand its search scope based on what patterns perform well:

```yaml
discovery:
  auto_expand: true
  expansion:
    max_keywords_per_wave: 3      # Max new keywords per expansion
    max_subreddits_per_wave: 2    # Max new subreddits per expansion
    min_validation_score: 0.5     # Only add from queries with avg_score >= 0.5
    cooldown_hours: 24            # Hours between expansions
```

When enabled:
1. After each discovery wave, the system analyzes validation feedback
2. Finds keywords/subreddits that yielded prototype_candidate or promote decisions
3. Uses `discovery_suggestions` logic to find similar patterns
4. Adds new candidates to the active discovery scope

Expanded state is stored in `data/discovery_expansion.json` and merged with base config at startup.

## Pattern-Based Discovery

The system detects **specific integration problems** from signals and can focus discovery on them:

```bash
# View emerging patterns with signal counts
python3 cli.py patterns
# Output:
# === EMERGING PATTERNS ===
#   spreadsheet_versioning (16 signals, high)
#   bank_reconciliation (12 signals, high)
#   stripe_to_quickbooks (1 signal, low)
```

Detected patterns include:
- `spreadsheet_versioning` - Excel/Google Sheets version control issues
- `bank_reconciliation` - Manual bank reconciliation pain
- `stripe_to_quickbooks` - Stripe ↔ accounting software sync
- `multi_channel_ecom` - Amazon/Shopify/Etsy sales reconciliation

### Focused Discovery

Run discovery focused on a specific pattern:

```bash
# Focused discovery on spreadsheet versioning
python3 cli.py run-once --pattern spreadsheet_versioning

# Focused + fresh (bypass cache)
python3 cli.py run-once --pattern bank_reconciliation --fresh

# Fresh discovery only (no pattern)
python3 cli.py run-once --fresh
```

The `--pattern` flag overrides discovery queries with pattern-specific keywords.
The `--fresh` flag bypasses the signal cache to force new discovery.

Pattern detection is implemented in [src/opportunity_engine.py](./src/opportunity_engine.py) via:
- `SPECIFIC_INTEGRATION_PATTERNS` - Regex patterns for tool pairs
- `_extract_specific_patterns()` - Extracts patterns from signal text
- `get_patterns_for_discovery()` - Returns patterns sorted by signal count

## Code Generation (BuilderV2)

When `builder.auto_build: true`, the system uses [BuilderV2Agent](./src/agents/builder_v2.py) to generate functional code:

```yaml
llm:
  provider: ollama  # or 'anthropic'
  model: qwen2.5:7b-instruct
  base_url: http://localhost:11434
  max_tokens: 8000
```

Output types:
- `workflow_reliability_console` - Python CLI tools
- `workflow_diagnostic_prototype` - React + FastAPI web apps
- `operator_evidence_workspace` - Evidence collection tools

Generated projects go to `data/generated_projects/{slug}/`.

## Active Runtime Model

The active evidence path is:

`finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> experiment + validation -> build_brief + build_prep_outputs -> ledger`

The most important state-model reference is:

- [docs/state-model.md](./docs/state-model.md)

That document defines:

- source classes and routing rules
- source-of-truth tables
- compatibility tables
- finding lifecycle states
- post-validation selection states
- build-prep handoff records

## Post-Validation Build Prep

The current build-prep layer is intentionally narrow and explicit.

Selection states currently include:

- `research_more`
- `prototype_candidate` - eligible for build_prep and ideation
- `prototype_ready`
- `build_ready` - eligible for builder
- `launched`
- `iterate`
- `expand`
- `archive`

There is no direct `validated -> build_ready` path.

Current build-prep components:

- [src/build_prep.py](./src/build_prep.py)
- [src/agents/build_prep.py](./src/agents/build_prep.py)

The three build-prep agents are:

- `solution_framing`
- `experiment_design`
- `spec_generation`

These agents consume a canonical persisted build brief and write traceable outputs back to the runtime store.

## Ideation

When `orchestration.auto_ideate_after_validation: true`, the [IdeationAgent](./src/agents/ideation.py) creates product ideas from validated opportunities (promote decisions or prototype_candidate status).

Ideas are stored in the `ideas` table and can trigger code generation via the builder.

## Key Files

- [CLAUDE.md](./CLAUDE.md): internal operator/developer guide
- [docs/state-model.md](./docs/state-model.md): runtime state model and active table definitions
- [docs/production-runtime.md](./docs/production-runtime.md): packaging, environment, and Docker runtime notes
- [docs/render-deploy.md](./docs/render-deploy.md): Render relay deployment notes
- [src/source_policy.py](./src/source_policy.py): first-class source routing rules
- [src/review_sources.py](./src/review_sources.py): review-source adapters and parsing
- [src/github_sources.py](./src/github_sources.py): GitHub issue/discussion adapter
- [src/behavior_eval.py](./src/behavior_eval.py): behavioral eval harness
- [src/build_prep.py](./src/build_prep.py): selection transitions and build-brief helpers
- [src/agents/build_prep.py](./src/agents/build_prep.py): solution-framing, experiment-design, and spec-generation agents
- [src/agents/builder_v2.py](./src/agents/builder_v2.py): LLM-powered code generator
- [src/agents/ideation.py](./src/agents/ideation.py): idea generation from validated opportunities
- [src/discovery_expander.py](./src/discovery_expander.py): self-expanding discovery logic
- [src/discovery_suggestions.py](./src/discovery_suggestions.py): keyword/subreddit suggestions from DB
- [evals/behavior_gold.json](./evals/behavior_gold.json): gold-set behavioral fixtures
- [bridges/reddit-devvit](./bridges/reddit-devvit): Devvit bridge scaffold for the Reddit relay path

## Repository Layout

- [src](./src): core source adapters, policy modules, and build-prep helpers
- [tests](./tests): regression and behavior coverage
- [docs](./docs): system documentation
- [evals](./evals): behavioral evaluation fixtures
- [data](./data): runtime databases and backups
- [output](./output): logs and generated runtime artifacts
- [dashboard](./dashboard): dashboard/frontend workspace
- [bridges](./bridges): external bridge integrations, including Reddit Devvit

## Packaging And Runtime

The Python runtime can now be installed from [requirements.txt](./requirements.txt) and packaged with:

- [Dockerfile](./Dockerfile)
- [docker-compose.yml](./docker-compose.yml)
- [render.yaml](./render.yaml)

For environment variables, runtime paths, and startup steps, use:

- [docs/production-runtime.md](./docs/production-runtime.md)
- [docs/render-deploy.md](./docs/render-deploy.md)
- [docs/codespaces.md](./docs/codespaces.md)

Codespaces support is now included through:

- [.devcontainer/devcontainer.json](./.devcontainer/devcontainer.json)
- [.env.example](./.env.example)

## Current Guardrails

- Source policy is explicit, not implicit.
- Wedge activation is default-off and requires fit validation plus runtime sanity checks.
- Review metadata is enrichment-only.
- Generic listing copy and vague summaries do not generate atoms.
- Validated opportunities do not flow directly into build.
- Build-prep output remains narrow and traceable; broad product generation is intentionally deferred.

## Current Limitations

- Source quality and corroboration depth still matter more than volume.
- Some runtime entrypoints referenced in internal notes may not be present at the repo root in this snapshot; use [CLAUDE.md](./CLAUDE.md) and [docs/state-model.md](./docs/state-model.md) as the authoritative local references for workflow and state.
- Public review sources do not always expose install counts or rich reviewer metadata, so some market fields use bounded proxies when necessary.

## Recommended Next Reference

If you are trying to understand how the system behaves at runtime, start with:

1. [docs/state-model.md](./docs/state-model.md)
2. [src/source_policy.py](./src/source_policy.py)
3. [src/build_prep.py](./src/build_prep.py)
4. [evals/behavior_gold.json](./evals/behavior_gold.json)
