# Scoring and gates

This branch is moving the repo away from "many competing score blobs" and toward
one canonical post-validation snapshot: `OpportunityEvaluationV1`.

There are still multiple gates in the pipeline, but after validation they should
be read as:

`raw inputs -> canonical evaluation -> policy -> selection/build-prep routing`

So a finding can still pass one layer and fail another, but the downstream
layers should now consume the same evaluated snapshot instead of rebuilding
their own partial truth.

## 1) Discovery candidate filter

Configured under `discovery.candidate_filter` (see `config.yaml`). Used when deciding whether a raw item is worth structured extraction. **Not** the same as validation promotion scores.

## 2) Evidence / `validate_problem` scores

`ResearchToolkit.validate_problem` produces `problem_score`, `feasibility_score`,
`value_score`, etc. These feed into `score_opportunity` and become the reusable
measures written into `OpportunityEvaluationV1`.

The current canonical snapshot still carries legacy transition fields such as
`composite_score`, `problem_plausibility`, and `evidence_sufficiency`, but they
are now **diagnostic only**. The primary operator-facing score language is:

- `decision_score`
- `problem_truth_score`
- `revenue_readiness_score`

## 3) `stage_decision` (promote / park / kill)

Implemented in `src/opportunity_engine.py` (`stage_decision`).

Current runtime behavior is the newer v4 path:

- primary promote language:
  - `decision_score`
  - `problem_truth_score`
  - `revenue_readiness_score`
- supporting floors:
  - `frequency_score`
  - evidence / value / counterevidence checks

`stage_decision` is still implemented in `src/opportunity_engine.py`, but the
result is now written into `OpportunityEvaluationV1.policy.*` so downstream
consumers do not need to recompute it.

- **Kill** on hard-kill conditions such as overwhelming counterevidence,
  extremely weak problem truth, or obviously dead opportunity shape.
- **Promote** when the decision path clears the decision score and support
  floors.
- Otherwise **park** with a subreason (`park_recurrence`, `park_value`, …).

### Promotion / park thresholds in config

Resolved by `src/validation_thresholds.py` (`resolve_promotion_park_thresholds`), **first match wins**:

1. `validation.decisions.promote_score` / `park_score`
2. `orchestration.promotion_threshold` / `park_threshold`
3. `validation.promotion_threshold` / `park_threshold`
4. Defaults **0.62** / **0.48**

The repo `config.yaml` uses (3) — `validation.promotion_threshold` and `validation.park_threshold`.

## 4) Build-prep selection (`prototype_candidate` vs `research_more`)

Implemented in `src/build_prep.py` (`determine_selection_state`).

Selection is stricter than promote/park/kill, but it should now be understood
as a **policy layer on top of the canonical evaluation**, not a second
independent scorer.

It reads:

- canonical policy decision (`promote` / `park` / `kill`)
- corroboration structure (`source_family_diversity`, `corroboration_score`,
  `generalizability_class`, `recurrence_state`, `cross_source_match_score`)
- canonical measures (`evidence_quality`, `value_support`, `frequency_score`,
  `buildability`, `cost_of_inaction`, `workaround_density`)

Selection may still admit exploratory `prototype_candidate` cases when strict
gates miss but explicit checkpoint rules pass, but only after the canonical
decision is already `promote`.

**Hard mapping:**

- `kill -> archive`
- `park -> research_more`
- `promote -> prototype_candidate` or `research_more`

So selection is stricter than policy, but it is not a second scorer.

## Canonical evaluation snapshot

The canonical post-validation store now lives at:

`validations.evidence["opportunity_evaluation"]`

It owns these top-level groups:

- `inputs`
- `measures`
- `evidence`
- `policy`
- `selection`
- `shadow`

Downstream artifact builders such as build briefs and research specs should
prefer this snapshot over reconstructing evaluation facts from scattered fields.

## `run_once` completion

`run.AutoResearcher.run_once` waits until `completion_state()["drained"]` is stable (not only an empty message queue). The queue can be momentarily empty while **evidence** or **validation** agents still have in-flight work (`BaseAgent` processing count). If there are **no** qualified findings to process and discovery emits nothing new, the run will still exit quickly — that is expected.

## Debugging

```bash
# After at least one validation exists in the DB:
python cli.py gate-diagnostics --limit 10

# Focus one finding:
python cli.py gate-diagnostics --finding-id 42 --limit 5

# Specific run:
python cli.py gate-diagnostics --run-id <run_id> --limit 20

# Shadow comparison report:
python cli.py scoring-report --limit 20

# Canonical backfill (dry run by default):
python cli.py backfill-evaluations --limit 5000

# Apply canonical backfill:
python cli.py backfill-evaluations --limit 5000 --apply
```

Output includes:

- Effective **promotion_threshold** / **park_threshold**
- `canonical_evaluation`: stored decision/selection snapshot when available
- `stage_decision` diagnostics (`diagnose_stage_decision`): which promotion
  floors failed, hard-kill reasons
- `selection_gate` (`explain_selection_gate_detail`): strict vs exploratory
  checks from the canonical evaluation when present

Programmatic helpers:

- `opportunity_engine.diagnose_stage_decision`
- `build_prep.explain_selection_gate_detail`
- `gate_diagnostics.explain_validation_evidence`
- `opportunity_evaluation_report.build_shadow_scoring_report`

## `summary_counts` semantics

`run.AutoResearcher.summary_counts()` labels which totals are **DB-wide** vs **scoped to `current_run_id`** (see `count_semantics` in the returned dict).
