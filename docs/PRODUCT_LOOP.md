# Waves → overlap → products → build

This pipeline is designed around **repeated discovery waves** so weak signals can **accumulate** into **clusters** (overlapping problem atoms / recurring patterns), then **score** and **gate** toward actionable work.

## Cross-subreddit / cross-source “common problems”

Clustering is **not** scoped to one subreddit. Atoms with the same **`cluster_key`** (derived from segment, role, JTBD, failure, workarounds — see `build_problem_atom` in `opportunity_engine.py`) are grouped **wherever** they were found (Reddit, Shopify reviews, GitHub, etc.). Reddit **`source`** used to include `reddit-problem/<subreddit>`, which could skew **segment** matching; **segment rules for Reddit now use post text** so the same pain in different communities is more likely to share a **cluster_key** and evolve together.

## What runs automatically

| Stage | What it does |
|--------|----------------|
| **Discovery waves** (`python cli.py run`, `orchestration.continuous_waves`) | New findings per cycle; dedupe by `content_hash`. |
| **Atoms & clusters** | Atoms share a `cluster_key`; validation pulls **all atoms in the cluster** — that is the “overlap” signal. |
| **Opportunity + validation** | Composite score, corroboration, recurrence — `docs/gates.md`. |
| **Build prep** | For `prototype_candidate`, agents can produce build briefs / spec handoff (`src/agents/build_prep.py`). |
| **Ideation** (`orchestration.auto_ideate_after_validation`) | Emits **`ideas`** when **`decision == promote`** (validation `passed`) **or** when **`selection_status == prototype_candidate`** and a **build brief** was created. `prototype_candidate` is now a promoted checkpoint path, not a `park` override. Inspect with `python cli.py ideas`. |
| **Builder** (`builder.auto_build`) | **Off by default.** If enabled, can consume idea payloads — review `BuilderAgent` before turning on. |

## What you should expect

- **Overlap takes time**: clusters strengthen as **more waves** add atoms with similar `cluster_key`; one-off posts rarely promote.
- **“Meaningful product”** in code terms is usually **`prototype_candidate`** or **`promote`** plus passing gates — not every parked finding.
- **Auto-build** is intentionally conservative: enable only when you trust gates and cost/risk.

## Commands

```bash
python cli.py run              # continuous waves (see continuous_waves + retry_interval_seconds)
python cli.py clusters
python cli.py opportunities
python cli.py ideas
python cli.py build-briefs
python cli.py gate-diagnostics
```

See also: `docs/LOOP.md`, `docs/gates.md`.
