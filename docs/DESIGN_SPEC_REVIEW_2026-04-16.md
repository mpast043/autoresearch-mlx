# AutoResearch MLX Design Spec and Review

Date: 2026-04-16

Scope: repository-wide architecture review and implementation design spec for the current AutoResearch MLX runtime. This is a design baseline and operating contract, not a rewrite plan. It intentionally references the existing code paths so future changes can be judged against concrete invariants.

## 1. Product Intent

AutoResearch MLX is an evidence-first weak-signal discovery system. Its job is to find repeated, commercially meaningful operator pain before product ideation or building begins.

The system should optimize for:

- specific workflow pain over broad market chatter
- repeatable evidence over one-off anecdotes
- falsifiable build briefs over speculative ideas
- operator-readable diagnostics over opaque model judgment
- conservative downstream routing, especially before build automation

The default runtime path is:

```text
discovery -> evidence -> validation -> build_prep
```

Optional paths exist for ideation, product generation, security review, SRE health checks, and documentation generation. Those paths must remain gated by validation and configuration.

## 2. Users and Jobs

Primary users:

- Operator reviewing whether the pipeline is finding useful opportunities.
- Developer tuning discovery, source policy, validation, and build-prep gates.
- Builder workflow consuming build briefs after a validated `prototype_candidate`.

Core jobs:

- Run discovery waves safely against public sources.
- Preserve source provenance and decision traces.
- Reject low-signal, vendor, meta, and product-specific noise before atom generation.
- Confirm recurrence and market value across independent families.
- Promote, park, or kill opportunities with explainable reasons.
- Produce build-prep artifacts only for opportunities that pass selection gates.

## 3. Non-Goals

The system should not:

- auto-build from raw discovery or thin validation
- treat marketplace listings, vendor copy, or generic comparison pages as pain evidence
- use success stories as direct atom sources
- collapse all source lanes into a single generic scoring rule
- hide promotion decisions behind unexplained score blends
- require a dashboard to operate the CLI pipeline

## 4. Runtime Components

### CLI

`cli.py` is the operator surface. It routes commands into either standalone handlers or an initialized `AutoResearcher` context.

Required command categories:

- runtime: `run`, `run-once`, `run-unseeded`, `watch`
- inspection: `findings`, `signals`, `atoms`, `clusters`, `opportunities`, `experiments`, `ledger`
- diagnostics: `report`, `gate-diagnostics`, `pipeline-health`, `operator-report`, `backlog-workbench`
- calibration: `eval`, `review-queue`, `review-mark`, `revalidate`, `rescore-v4`
- discovery support: `reddit-seed`, `check-bridge`, `suggest-discovery`, `term-state`

Design contract:

- Commands that inspect or mutate persisted state should use `app_context`.
- Commands that run network services may manage their own lifecycle.
- CLI diagnostics must report the same effective gates used by runtime scoring.

### AutoResearcher

`run.py` owns process-level orchestration:

- load `.env`
- load config
- resolve project/runtime paths
- configure logging
- initialize SQLite
- set the active run id
- configure opportunity and build-prep modules
- create agents
- start/stop the orchestrator
- wait for queue and agent drain

Design contract:

- `run_once` must not exit while evidence or validation agents are still busy.
- `run` may repeat discovery waves only through `continuous_waves` or `stop_on_hit`.
- `snapshot`, `report`, and status output must label run-scoped versus DB-wide counts.

### Orchestrator

`src/orchestrator.py` routes messages through agents:

- `FINDING` -> evidence
- `EVIDENCE` -> validation
- `VALIDATION` -> build-prep and optional ideation
- `BUILD_PREP` -> next build-prep stage or optional builder
- `IDEA` -> optional builder
- `RESULT` -> status completion

Design contract:

- Build prep should only start from `selection_status == "prototype_candidate"` and an existing build brief.
- Ideation may run for promoted validations or prototype candidates, but only when enabled.
- Auto-build must remain behind build-ready checks.
- `stop_on_hit` should observe validation payloads after persistence, not before.

### Message Bus

`src/messaging.py` uses per-agent priority queues.

Design contract:

- Lower priority numbers are higher precedence.
- Per-agent queues isolate backpressure.
- Agent failures must be surfaced rather than silently dropped.

## 5. Agent Design

### Discovery Agent

`src/agents/discovery.py` discovers candidate findings, runs pre-atom filtering, classifies source signals, persists accepted findings, and emits qualified findings to the orchestrator.

Inputs:

- configured sources
- query plans
- discovery feedback
- learned problem spaces
- Reddit relay cache/seed data

Outputs:

- `findings`
- `raw_signals`
- `problem_atoms`
- discovery feedback rows
- `FINDING` messages for qualified findings

Critical invariants:

- Only `pain_signal` material can produce problem atoms.
- Success, demand, competition, meta, and low-signal material may inform search or market context, but must not enter active validation as pain atoms.
- Screened-out/killed duplicate rediscovery may be re-evaluated under newer policy, but accepted duplicates should remain deduped.
- Source-family-specific rejection should be deterministic for obvious non-wedge material.

Source lane expectations:

- Reddit: practitioner-first pain threads, query/subreddit rotation, relay fallback.
- GitHub: transferable external workflow pain only, not internal backlog hygiene.
- WordPress/Shopify reviews: transferable 1-2 star workflow pain only, not product-specific support.
- YouTube comments: concrete operator pain in comments, not roundup/recommendation chatter.
- Web: problem threads and forums, not listicles, marketing copy, or trend pages.

### Evidence Agent

`src/agents/evidence.py` gathers recurrence and market enrichment for qualified findings, writes run-scoped evidence rows, writes ledger entries, and emits `EVIDENCE` messages.

Inputs:

- qualified finding
- anchor atom
- `ResearchToolkit.validate_problem`

Outputs:

- `corroborations`
- `market_enrichments`
- evidence ledger entries
- evidence payload to validation

Critical invariants:

- Evidence writes are run-scoped.
- Timeouts should degrade to explicit timeout evidence, not block the run.
- Concurrent enrichment tasks must propagate exceptions into agent status, logs, and diagnostics.
- Generic prompt breadth should be an explicit product choice, with tests aligned to the desired exploration policy.

### Validation Agent

`src/agents/validation.py` clusters atoms, scores opportunities, persists validations, plans experiments, updates finding lifecycle, creates build briefs when selected, and emits validation messages.

Inputs:

- evidence payload
- finding, raw signal, and problem atoms
- corroboration and market enrichment
- review feedback

Outputs:

- clusters and cluster members
- opportunities
- validation experiments
- validations
- evidence ledger entries
- optional build briefs

Critical invariants:

- Promotion should be based on the current stage-decision formula, not stale legacy gates.
- Diagnostics must explain exactly why the same inputs promoted, parked, or killed.
- `selection_status` is separate from `decision`; a promoted item may still be `research_more`.
- Build briefs are created only for `prototype_candidate`.

### Build Prep

`src/agents/build_prep.py` runs:

```text
solution_framing -> experiment_design -> spec_generation
```

Inputs:

- persisted build brief
- validation evidence
- cluster and opportunity context

Outputs:

- build-prep rows keyed by build brief and agent stage
- status transitions toward prototype readiness/build readiness

Critical invariants:

- Build-ready status requires specific product shape, host/platform fit, concrete failure mode, and corroboration.
- Generic build-ready language must be rejected even when phrased fluently.
- Auto-build can only run after spec generation and build-ready status.

## 6. Source Policy Design

`src/source_policy.py` is the vocabulary and routing contract.

Allowed classes:

- `pain_signal`
- `success_signal`
- `demand_signal`
- `competition_signal`
- `meta_guidance`
- `low_signal_summary`

Only `pain_signal` is atom eligible.

Policy responsibilities:

- normalize unknown source classes to `low_signal_summary`
- map each class to discovery status
- declare whether a class may seed search, enrich market context, or enter prompt evaluation

Required source-family guards:

- hard-reject listing or marketing copy
- hard-reject help pages and generic summaries
- route trend/search-result pages to demand signal
- route alternative/comparison/vendor chatter to competition signal
- route internal methods, prompt templates, and implementation guidance to meta guidance
- reject broad buying prompts without a specific workflow slice
- reject product-specific review/support complaints without transferability

## 7. Evidence Confirmation Design

Evidence confirmation is the main quality bottleneck.

The confirmation layer should separate:

- recurrence evidence: repeated pain in independent sources
- value evidence: cost, urgency, willingness-to-pay, buyer intensity
- counterevidence: already solved, weak value, low plausibility, vendor saturation
- corroboration quality: source diversity and match strength

Budgeting rules:

- Specific operational pain gets broader query, subreddit, and site coverage.
- Generic manual-work prompts may keep exploratory breadth when the operator wants wider discovery; regression tests should encode that choice explicitly.
- Source family expansion should be justified by expected information gain.
- Partial matches must not be treated as strong recurrence.
- Matched documents must be inspectable by source family.

Minimum metadata:

- queries considered and executed
- recurrence budget profile
- probe summary
- branch actions
- source attempts snapshot
- matched and partial docs by source
- family confirmation count
- recurrence gap reason
- recurrence failure class

## 8. Scoring and Gates

There are three independent gates:

1. Discovery candidate filtering.
2. Stage decision: `promote`, `park`, or `kill`.
3. Build-prep selection: `prototype_candidate`, `research_more`, `archive`, etc.

Current v4 stage decision uses:

- `decision_score`
- `problem_truth_score`
- `revenue_readiness_score`
- `frequency_score`
- hard-kill floors
- override branches for high frequency or strong evidence
- sharp-but-thin research candidate logic

Design contract:

- Operator diagnostics must distinguish v4 gates from legacy composite gates.
- `promotion_threshold` and `park_threshold` must be resolved once and displayed consistently.
- The docs, tests, diagnostics, and config defaults must agree on the active formula.
- Counterevidence should remain visible even when v4 promotes via decision score.

## 9. Persistence Design

SQLite remains the source of truth.

Core entity flow:

```text
finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> experiment + validation -> build_brief -> build_prep_outputs
```

Run-scoped tables:

- `validations`
- `corroborations`
- `market_enrichments`
- `evidence_ledger`
- `review_feedback`
- `build_briefs`
- `build_prep_outputs`
- `experiments`
- `validation_experiments`

Design contract:

- Run-scoped writes should have unique keys or idempotent upserts where repeated runs can collide.
- Mirror tables must be documented and kept in sync.
- Foreign keys should be enabled for app-managed connections.
- Hot query fields should either be denormalized or indexed when data grows.
- App paths should use `Database` rather than ad hoc SQLite connections unless a read-only diagnostic has a strong reason.

## 10. Configuration Contract

Runtime config lives in `config.yaml`; packaged fallback config lives in `src/resources/config.default.yaml`.

Required config sections:

- `database`
- `output_dir`
- `discovery`
- `orchestration`
- `validation`
- `build_prep`
- `builder`
- `reddit_bridge`
- `reddit_relay`
- optional `security`, `technical_writer`, `sre`, `llm`

Design contract:

- Repo config and packaged fallback config should be intentionally versioned together.
- Any intentional divergence must be called out in docs.
- Secrets must use environment substitution or app-secret settings, never literals.
- Threshold changes must include tests and calibration notes.

## 11. Observability

Required operator surfaces:

- `output/pipeline_status.json`
- `output/autoresearcher.log`
- `python3 cli.py report`
- `python3 cli.py gate-diagnostics`
- `python3 cli.py pipeline-health`
- `python3 cli.py discovery-sort-diagnostics`
- `python3 cli.py operator-report`

Design contract:

- Status should show current stage, recent logs, and high-level counts.
- Gate diagnostics should report the same branch that made the decision.
- Pipeline health should distinguish empty backlog from blocked queue from discovery yield failure.
- Logs should include run id.

## 12. Security Design

Security requirements:

- No tokens or API keys committed to source, generated JavaScript, or fixtures.
- Reddit bridge and relay tokens must come from `.env`, deployment environment, or Devvit secret settings.
- Relay endpoints that ingest or fetch data must require bearer auth unless explicitly running in local no-auth mode.
- Generated product code should be scanned before any launch-oriented handoff.

Immediate security concern:

- `bridges/reddit-devvit/src/server/index.ts` currently hardcodes a relay token. Rotate that token and move the value into Devvit secret settings.

## 13. Reliability Design

Failure behavior:

- Source timeout: record source error and continue other sources.
- Evidence timeout: write explicit timeout evidence with low recurrence confidence.
- Agent task exception: log, increment agent error state, expose in status, and avoid silent message loss.
- Pipeline drain timeout: report completion state and leave enough diagnostics to resume.
- Duplicate finding: preserve accepted dedupe, allow re-evaluation of terminal low-quality rows when policy changes.

Concurrency expectations:

- Discovery runs source checks under an API semaphore.
- Evidence runs bounded concurrent enrichment.
- SQLite writes occur through the shared `Database` manager.
- Fire-and-forget tasks must be awaited, harvested, or explicitly supervised.

## 14. Testing Strategy

Required test layers:

- Unit tests for source classification and qualification.
- Unit tests for recurrence budgeting and branching.
- Unit tests for `stage_decision` and diagnostics.
- Database migration and CRUD tests against fresh and legacy schemas.
- Agent integration tests for message routing and build-prep handoff.
- Behavioral eval fixture suite for source policy, labels, decisions, and calibration.
- TypeScript build for Devvit relay.

Current verification snapshot:

- `pytest tests/ -q`: 891 passed, 2 failed.
- `python3 cli.py eval`: 15/15 passed.
- `npm run build` in `bridges/reddit-devvit`: passed.

The two failing Python tests encode a narrower generic-manual recurrence policy than the operator currently wants; update those tests if broad exploration remains the desired behavior.

## 15. Review Findings Summary

High-priority findings:

- Hardcoded relay bearer token in Devvit server source.
- Evidence-agent concurrent task exceptions can be dropped without being observed.
- Stage-decision diagnostics report legacy promotion checks even when v4 promoted through `decision_score`.

Medium-priority findings:

- Repo config and packaged fallback config diverge in source breadth, thresholds, LLM limits, and optional agent sections.
- The AGENTS dashboard instructions describe a Next.js dashboard, but the `dashboard/` directory contains no app.

## 16. Target Fix Order

1. Harvest evidence-agent task exceptions and route failures into logs/status.
2. Align generic-manual recurrence tests with the desired broader exploration policy.
3. Align `diagnose_stage_decision` and `docs/gates.md` with v4 scoring.
4. Decide whether `src/resources/config.default.yaml` should mirror `config.yaml` or explicitly serve as a safer packaged default.
5. Remove or replace dashboard instructions unless a real dashboard app is restored.
6. Add regression tests for the diagnostic mismatch and evidence task exception path.
