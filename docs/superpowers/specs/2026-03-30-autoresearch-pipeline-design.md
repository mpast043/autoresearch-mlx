# AutoResearch MLX Pipeline Design

## Overview

AutoResearch MLX is an evidence-first weak-signal discovery system that continuously finds pain signals from public sources, validates them, and generates functional code solutions.

## Pipeline Architecture

```
discovery → evidence → validation → build_prep → ideation → builder
```

### Stages

1. **Discovery**: Harvests pain signals from Reddit, GitHub, web search, YouTube comments, WordPress/Shopify reviews
2. **Evidence**: Gathers corroboration and market enrichment per finding
3. **Validation**: Clusters problem atoms, scores opportunities, makes promote/park/kill decisions
4. **Build Prep**: Solution framing, experiment design, spec generation (runs for `prototype_candidate` opportunities)
5. **Ideation**: Creates product ideas from validated opportunities (triggered by `promote` or `prototype_candidate`)
6. **Builder**: Generates functional code from ideas using Ollama (local LLM)

### Message Flow

The pipeline uses an async message queue:
- `MessageType.FINDING` → Discovery → Evidence
- `MessageType.EVIDENCE` → Evidence → Validation
- `MessageType.VALIDATION` → Validation → Build Prep / Ideation
- `MessageType.BUILD_REQUEST` → Ideation → Builder

## Validation Thresholds

### Configurable Thresholds

| Threshold | Config Path | Default | Recommended | Purpose |
|-----------|-------------|---------|-------------|---------|
| `promotion_threshold` | `validation.promotion_threshold` | 0.62 | 0.50 | Score needed to promote |
| `park_threshold` | `validation.park_threshold` | 0.48 | 0.30 | Below this = park |

### Config Precedence (first wins)

1. `validation.decisions.promote_score` / `park_score`
2. `orchestration.promotion_threshold` / `park_threshold`
3. `validation.promotion_threshold` / `park_threshold`
4. Defaults: 0.62 / 0.48

### Hardcoded Gates

In `src/opportunity_engine.py` line ~1324, promotion requires ALL of:

```python
if (composite >= promotion_threshold and
    plausibility >= 0.6 and
    evidence_quality >= 0.55 and
    supported_count <= 1 and
    value_support >= 0.58):
    return {"status": "promoted", ...}
```

**Problem**: These are hardcoded. Lowering `promotion_threshold` doesn't help if `plausibility`, `evidence_quality`, or `value_support` are too high.

**Recommendation**: Make these configurable via config.yaml:

```yaml
validation:
  gates:
    plausibility_min: 0.6
    evidence_quality_min: 0.55
    value_support_min: 0.58
    supported_count_max: 1
```

## Self-Expanding Discovery

### How It Works

After each discovery wave, the system analyzes validation feedback to expand search scope:

1. **Feedback Analysis**: Queries `discovery_feedback` table for keywords/subreddits with `prototype_candidates > 0`
2. **Candidate Generation**: Uses `src/discovery_suggestions.py` to find similar language patterns
3. **Expansion State**: Stores in `data/discovery_expansion.json`
4. **Config Merge**: At startup, merges base config + expanded keywords/subreddits

### Configuration

```yaml
discovery:
  auto_expand: true              # Enable self-expansion
  use_r_all: true                # Search r/all for broader reach
  expansion:
    max_keywords_per_wave: 5     # Max new keywords per expansion
    max_subreddits_per_wave: 5   # Max new subreddits per expansion
    min_validation_score: 0.4    # Only add from queries with avg_score >= 0.4
    cooldown_hours: 12           # Hours between expansions
```

### Key Files

- `src/discovery_expander.py`: Expansion logic
- `src/discovery_suggestions.py`: Keyword extraction from DB
- `data/discovery_expansion.json`: Runtime state

## Code Generation (BuilderV2)

### How It Works

When `builder.auto_build: true`, the BuilderV2Agent generates functional code:

1. Fetches idea from database by `idea_id`
2. Creates spec from idea fields + `spec_json`
3. Calls Ollama with prompt template for output type
4. Parses JSON response for file list
5. Writes files to `data/generated_projects/{slug}/`

### Configuration

```yaml
llm:
  provider: ollama              # or 'anthropic'
  model: qwen2.5:7b-instruct    # or 'codellama', 'llama3.1:8b'
  base_url: http://localhost:11434
  max_tokens: 8000

builder:
  auto_build: true
```

### Output Types

| Type | Description |
|------|-------------|
| `workflow_reliability_console` | Python CLI tools |
| `workflow_diagnostic_prototype` | React + FastAPI web apps |
| `operator_evidence_workspace` | Evidence collection tools |

### Generated Project Structure

```
data/generated_projects/{slug}/
├── package.json        # frontend dependencies
├── requirements.txt   # backend dependencies
├── README.md          # setup instructions
├── Makefile           # or run.sh
├── .env.example       # secrets placeholder
├── SPEC.md            # source spec
├── server.py          # backend entry
├── main.py            # frontend/main entry
└── src/               # source code
```

## Current System State

### Database Stats (as of 2026-03-30)

| Table | Count |
|-------|-------|
| findings | 241 |
| qualified findings | 31 |
| opportunities | 86 |
| build_ready | 3 |
| ideas | 2 |
| built | 2 |

### Finding Status Distribution

| Status | Count |
|--------|-------|
| killed | 29 |
| parked | 105 |
| qualified | 31 |
| screened_out | 76 |

### Opportunity Selection Status

| Status | Count |
|--------|-------|
| archive | 23 |
| build_ready | 3 |
| research_more | 60 |

## Recommended Changes

### 1. Lower Validation Thresholds

```yaml
validation:
  promotion_threshold: 0.50
  park_threshold: 0.30
```

### 2. Make Hardcoded Gates Configurable

In `src/opportunity_engine.py`, refactor line ~1324:

```python
plausibility_min = config.get("validation", {}).get("gates", {}).get("plausibility_min", 0.6)
```

### 3. Enable Full Auto-Expand

```yaml
discovery:
  auto_expand: true
  use_r_all: true
  expansion:
    cooldown_hours: 6  # Faster learning
```

### 4. Add New Subreddits

Current list is small. Add more:
```yaml
problem_subreddits:
  - accounting
  - projectmanagement
  - automation
  - notion
  - airtable
```

## Running the Pipeline

```bash
# Continuous mode (loops forever)
python cli.py run

# Single wave
python cli.py run-once

# Watch mode
python cli.py watch
```

## Architecture Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Discovery  │───▶│  Evidence   │───▶│ Validation  │───▶│ Build Prep  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                                       │                    │
      ▼                                       ▼                    ▼
┌─────────────┐                         ┌─────────────┐    ┌─────────────┐
│  Expander   │                         │  Ideation   │    │   Builder  │
└─────────────┘                         └─────────────┘    └─────────────┘
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `cli.py` | Entry point |
| `run.py` | Main application |
| `config.yaml` | Configuration |
| `src/orchestrator.py` | Message routing |
| `src/agents/discovery.py` | Discovery agent |
| `src/agents/validation.py` | Validation agent |
| `src/agents/ideation.py` | Ideation agent |
| `src/agents/builder_v2.py` | Code generator |
| `src/discovery_expander.py` | Self-expansion |
| `src/opportunity_engine.py` | Scoring logic |
| `src/database.py` | SQLite schema |
| `data/autoresearch.db` | Runtime database |