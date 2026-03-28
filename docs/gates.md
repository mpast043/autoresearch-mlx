# Scoring and gates

This pipeline applies **several independent layers**. A finding can pass one layer and still fail another.

## 1) Discovery candidate filter

Configured under `discovery.candidate_filter` (see `config.yaml`). Used when deciding whether a raw item is worth structured extraction. **Not** the same as validation promotion scores.

## 2) Evidence / `validate_problem` scores

`ResearchToolkit.validate_problem` produces `problem_score`, `feasibility_score`, `value_score`, etc. These feed into `score_opportunity` → **composite_score**, `evidence_quality`, `value_support`, `problem_plausibility`, …

## 3) `stage_decision` (promote / park / kill)

Implemented in `src/opportunity_engine.py` (`stage_decision`).

- **Kill** (“hard kill”) if the market is already solved, counterevidence is overwhelming, or weak composite + plausibility + sufficiency, etc.
- **Promote** only if **all** hold:
  - `composite_score` (± review biases) ≥ **promotion_threshold**
  - `problem_plausibility` ≥ 0.6
  - `evidence_quality` ≥ 0.55
  - supported counterevidence hits ≤ 1
  - `value_support` ≥ 0.58
- Otherwise **park** with a subreason (`park_recurrence`, `park_value`, …).

### Promotion / park thresholds in config

Resolved by `src/validation_thresholds.py` (`resolve_promotion_park_thresholds`), **first match wins**:

1. `validation.decisions.promote_score` / `park_score`
2. `orchestration.promotion_threshold` / `park_threshold`
3. `validation.promotion_threshold` / `park_threshold`
4. Defaults **0.62** / **0.48**

The repo `config.yaml` uses (3) — `validation.promotion_threshold` and `validation.park_threshold`.

## 4) Build-prep selection (`prototype_candidate` vs `research_more`)

Implemented in `src/build_prep.py` (`determine_selection_state`). Uses **corroboration** (families, `corroboration_score`, `generalizability_class`, `recurrence_state`) plus scorecard fields. Can admit **exploratory** `prototype_candidate` when strict gates miss but “near-miss” thresholds pass.

**Important:** If `stage_decision` returns **park** or **kill**, you will not get `validated_selection_gate` from promotion — selection uses the recommendation string; kills map to `archive`.

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
```

Output includes:

- Effective **promotion_threshold** / **park_threshold**
- `stage_decision` diagnostics (`diagnose_stage_decision`): which promotion floors failed, hard-kill reasons
- `selection_gate` (`explain_selection_gate_detail`): strict vs exploratory checks

Programmatic helpers:

- `opportunity_engine.diagnose_stage_decision`
- `build_prep.explain_selection_gate_detail`
- `gate_diagnostics.explain_validation_evidence`

## `summary_counts` semantics

`run.AutoResearcher.summary_counts()` labels which totals are **DB-wide** vs **scoped to `current_run_id`** (see `count_semantics` in the returned dict).
