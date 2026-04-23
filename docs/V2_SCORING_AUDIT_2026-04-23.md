# Version 2.0 Scoring Audit

Date: 2026-04-23
Branch: `codex/v2-scoring-audit`
Scope: audit-first review only; no runtime scoring behavior changed in this pass

## Executive Summary

The repo currently has multiple independent scoring and recommendation systems, not one:

1. discovery candidate filtering
2. source-policy screening
3. high-leverage scoring
4. `validate_problem()` evidence scoring
5. market enrichment scoring
6. opportunity scorecard scoring
7. `stage_decision()` promote/park/kill gating
8. `determine_selection_state()` build-prep gating
9. discovery term lifecycle and next-wave ranking
10. workbench recommendation logic

That is the core overengineering problem. The system does not have a single canonical evaluation contract; it has a chain of partially overlapping score vocabularies that feed different tables and different outputs.

Live DB snapshot from `data/autoresearch.db` during this audit:

- `validations`: 535
- `opportunities`: 43 latest-state rows
- `build_briefs`: 29
- `build_prep_outputs`: 79
- `ideas`: 36
- `selection_status`: `research_more` 31, `archive` 9, `prototype_candidate` 3
- validation `market_gap_state`: `needs_more_recurrence_evidence` in 526/535 rows
- promoted validations: 47 total
  - 34 ended as `research_more`
  - 13 ended as `prototype_candidate`

The current runtime does work, but the scoring stack is too layered and too redundant for how much real discrimination it provides. The most important architectural issue is that `stage_decision()` uses the newer `decision_score` / `problem_truth_score` / `revenue_readiness_score`, while build-prep selection still leans on older `composite_score` / `value_support` / `evidence_quality` thresholds plus special-case shortcuts.

Version 2.0 is directionally better as a scoring philosophy because it is simpler, more explainable, and separates upside, risk, and evidence more cleanly. But the literal V2 model is not fully automatable from the current data contract. Several required inputs are missing or only weakly proxied. If implemented naively, it would become one more derived layer on top of the existing stack.

Verdict: **B. Improvement with simplification needed**

The right move is not "replace current scoring with full V2 immediately." The right move is:

1. collapse the current evaluation contract into one canonical artifact
2. build a shadow V2-lite scorer from existing fields only
3. compare it against current outcomes before changing runtime gates

## Current Architecture: Where Scoring Actually Happens

### Primary runtime decision path

| Layer | Main code | Inputs | Stored fields | Used for |
| --- | --- | --- | --- | --- |
| discovery candidate filter | `src/research_tools.py:_is_problem_candidate` | title/body/url text + config | `discovery_feedback.avg_screening_score` aggregate | keep/drop at discovery |
| source policy screening | `src/source_policy.py`, discovery agent | scraped/source metadata, classifier output | finding evidence + discovery aggregates | `pain_signal` vs excluded classes |
| problem atom extraction | `build_problem_atom()` | raw signal text/metadata | `problem_atoms.*` | all downstream scoring |
| validation evidence scoring | `ResearchToolkit.validate_problem()` | title, summary, atom, scraped recurrence docs, competitor docs | `validations.evidence.scores.*` | raw validation inputs |
| corroboration scoring | `src/agents/evidence.py` | recurrence matches, source families, source groups, query coverage | `corroborations.*`, `corroborations.evidence_json` | recurrence/evidence reuse |
| market enrichment scoring | `src/agents/evidence.py` | value hints, review metadata, wedge logic, corroboration evidence | `market_enrichments.*`, `market_enrichments.evidence_json` | buyer/value/review context |
| opportunity scorecard | `src/opportunity_engine.py:score_opportunity()` | atoms + validation scores + corroboration + market enrichment + review feedback | `opportunities` columns, `opportunities.notes_json.scorecard`, `validations.evidence.opportunity_scorecard` | primary scoring surface |
| stage decision | `src/opportunity_engine.py:stage_decision()` | scorecard + market gap + counterevidence + review bias | `opportunities.recommendation/status`, `validations.evidence.decision` | promote / park / kill |
| selection gate | `src/build_prep.py:determine_selection_state()` | decision + scorecard + corroboration + market enrichment | `opportunities.selection_status`, `validations.evidence.selection_status`, `build_briefs.status` | `research_more` vs `prototype_candidate` vs `archive` |
| build-prep/spec routing | validation agent + orchestrator | validation evidence + brief payload | `build_briefs`, `build_prep_outputs`, `ideas.spec_json` | downstream product/spec work |

### Secondary ranking / recommendation systems

| Layer | Main code | Stored fields | Used for |
| --- | --- | --- | --- |
| high leverage | `src/high_leverage.py` | finding evidence + raw signal metadata | discovery triage/reporting |
| discovery term lifecycle | `src/discovery_term_lifecycle.py` | `discovery_search_terms.*`, `discovery_feedback.*` | term promotion/demotion/exhaustion |
| next-wave ranking | `src/discovery_next_wave.py` | `wedge_quality_score` + output counts | choosing next discovery terms |
| workbench next action | `src/database_views.py:next_recommended_action()` | `decision`, `selection_status`, build-brief presence | operator UX recommendation |

## Phase 1: Data Flow Map

### Primary validation path

| Input source | Transformation logic | Stored field | Scoring use | Final output |
| --- | --- | --- | --- | --- |
| scraped title/body/url, source metadata | discovery candidate filter + source policy screening | finding evidence, `discovery_feedback.avg_screening_score` | early keep/drop and term quality | discovery acceptance, later term ranking |
| accepted raw signal | `build_problem_atom()` heuristic extraction | `problem_atoms.segment`, `user_role`, `job_to_be_done`, `trigger_event`, `pain_statement`, `failure_mode`, `current_workaround`, `frequency_clues`, `urgency_clues`, `cost_consequence_clues`, etc. | all later score components are derived from these atom fields | cluster summary, opportunity scoring, build brief context |
| title + summary + atom + recurrence queries + competitor query | `ResearchToolkit.validate_problem()` | `validations.evidence.scores.problem_score`, `solution_gap_score`, `saturation_score`, `feasibility_score`, `value_score`; plus large `validations.evidence.*` meta block | seed inputs to `assess_market_gap()` and `score_opportunity()` | validation evidence payload |
| recurrence documents and match metadata | evidence agent corroboration builder | `corroborations.recurrence_state`, `recurrence_score`, `corroboration_score`, `evidence_sufficiency`, `query_coverage`, `independent_confirmations`, `source_diversity`, `evidence_json.*` | `frequency_score`, `corroboration_strength`, `evidence_sufficiency`, selection gate family checks | recurrence posture, source diversity, selection eligibility |
| competitor docs, review metadata, wedge logic, corroboration evidence | evidence agent market enrichment | `market_enrichments.demand_score`, `buyer_intent_score`, `competition_score`, `trend_score`, `review_signal_score`, `value_signal_score`, `evidence_json.*` | `cost_of_inaction`, `value_support`, `willingness_to_pay_proxy`, `revenue_readiness_score`, build brief profitability fields | buyer/value context, wedge framing |
| cluster summary + validation scores | `assess_market_gap()` | `validations.evidence.market_gap_state`, `market_gap.*`, `opportunities.market_gap` | hard-kill / park reasoning, validation-plan branch | market gap summary |
| atom fields + validation scores + corroboration + market enrichment + review feedback | `score_opportunity()` | 36-field scorecard in `validations.evidence.opportunity_scorecard`, duplicated into `opportunities` columns and `opportunities.notes_json.scorecard` | main numeric decision surface | opportunity row, validation evidence, build brief inputs |
| scorecard + market gap + counterevidence | `stage_decision()` | `validations.evidence.decision`, `decision_reason`, `park_subreason`; `opportunities.recommendation`, `status` | promote / park / kill | validation outcome, finding lifecycle update |
| decision + scorecard + corroboration + market enrichment | `determine_selection_state()` | `selection_status`, `selection_reason`, `selection_gate` in validation evidence, opportunity row, build brief | prototype routing and build-prep gate | `research_more`, `prototype_candidate`, `archive` |
| validation evidence + atom/cluster context | `build_brief_payload()` | `build_briefs.brief_json` | build-prep chain and spec generation | build-prep handoff |
| validation evidence + ideation output | `build_research_spec()` | `ideas.spec_json` | idea/spec rendering | research spec artifact |

### Proposed Version 2.0 factor coverage

The practical question is whether the current system already collects the data needed for the V2 score.

| V2 factor | Current equivalent | Coverage | Notes |
| --- | --- | --- | --- |
| `P` pain severity | `pain_severity` | direct | already computed and stored in `opportunities` |
| `F` frequency | `frequency_score` | direct | already computed and stored |
| `B` buyer budget / ability to pay | `buyer_intent_score`, `willingness_to_pay_proxy`, `willingness_to_pay_signal` | proxy | good proxy, but not a clean budget field |
| `U` urgency | `urgency_score` | direct-ish | current score exists, but is partly heuristic |
| `D` distribution ease | `reachability` | proxy | close enough for a first shadow model |
| `M` buildability | `buildability` | direct | already computed and stored |
| `X` expansion potential | `expansion_potential` | direct | already computed and stored |
| `C` competition intensity | `competition_score` | direct-ish | available, though non-zero only in a subset of rows |
| `S` support burden | none | missing | would need new extraction or explicit heuristic |
| `T` trust barrier | partial via `compliance_burden_score`, `dependency_risk`, `adoption_friction` | proxy/missing | not explicitly modeled as one field |
| `I` integration complexity | none explicit | missing | could maybe be inferred later, but not currently collected |
| `Q` evidence quality | `evidence_quality` | direct | already computed and stored |
| `K` independent confirmation strength | `family_confirmation_count`, `corroboration_strength` | direct-ish | enough for a shadow scorer |
| `Y` source diversity | `source_family_diversity`, `core_source_family_diversity` | direct | already present in corroboration evidence |
| `buyer_identifiable` gate | `user_role`, `segment`, `operational_buyer_score`, audience hints | proxy | not explicit boolean |
| `target_user_clear` gate | `segment_concentration`, atom role/JTBD fields | proxy | good enough for shadow evaluation, not for a clean hard gate |

Bottom line: V2 is only **partially** supported by current data.

Directly supported: `P, F, M, X, Q, Y`

Supported by acceptable proxy: `B, U, D, C, K`

Weak or missing: `S, T, I, buyer_identifiable, target_user_clear`

## Current Model vs Version 2.0

### Current scoring model

#### Formulas and gates

- `validate_problem()` produces:
  - `problem_score`
  - `solution_gap_score`
  - `saturation_score`
  - `feasibility_score`
  - `value_score`
- `score_opportunity()` turns those plus corroboration and market enrichment into:
  - legacy fields such as `composite_score`, `problem_plausibility`, `value_support`, `evidence_sufficiency`
  - newer v4 fields: `problem_truth_score`, `revenue_readiness_score`, `decision_score`
- `stage_decision()` now promotes mainly from:
  - `decision_score >= promotion_threshold`
  - `problem_truth_score >= 0.11`
  - `revenue_readiness_score >= 0.22`
  - `frequency_score >= 0.25`
  - plus overrides for high frequency or strong evidence
- `determine_selection_state()` still relies heavily on:
  - `composite_score`
  - `value_support`
  - `evidence_quality`
  - corroboration/source-family thresholds
  - plus a separate promote requalification shortcut

#### Strengths

- uses real scraped evidence, not just idea descriptions
- already captures recurrence, corroboration, and some market/value context
- has an audit trail through `validations`, `corroborations`, `market_enrichments`, and ledger rows
- can explain individual pieces of the scorecard at runtime

#### Weaknesses

- too many derived layers for one decision
- legacy and v4 score vocabularies coexist
- several dimensions are nested inside other derived dimensions, which creates pseudo-precision
- selection/build-prep does not run on the same score vocabulary as promote/park/kill
- operator-facing docs and diagnostics still reflect older logic in places
- many thresholds are calibrated to the current distribution rather than to an externally grounded success label

### Version 2.0 scoring model

#### Intended logic

- use a weighted geometric mean for upside:
  - pain severity
  - frequency
  - buyer budget
  - urgency
  - distribution
  - buildability
  - expansion
- apply separate risk and evidence multipliers
- apply hard gates and soft caps
- produce a final 0-100 score with decision bands

#### Why it is attractive

- much easier to explain
- separates commercial value from evidence quality
- punishes one weak leg hard instead of letting averages hide it
- closer to an operator rubric than to an opaque scoring lattice

#### Complexity cost

The formula itself is simpler than the current system. The complexity is in getting trustworthy inputs:

- risk factors `S/T/I` are not currently explicit
- buyer clarity and target-user clarity are not explicit hard-gate booleans
- if those are filled by LLM judgment without a grounded extraction contract, V2 becomes a cleaner-looking but less reliable score

#### Expected gains

- cleaner explanation surface
- easier workbench comparison
- clearer reasoning about why something is weak
- easier to delete redundant current layers

#### Risks and failure modes

- becoming a second score layered on top of the first instead of a replacement
- introducing new subjectivity through missing factors
- overfitting to "commercial attractiveness" and underweighting discovery uncertainty unless evidence remains first-class
- lane logic and agent swarm logic becoming additional architecture before the core contract is simplified

## What The Live Data Says

### The current stack is over-layered

- 47 validations were promoted.
- Only 13 promoted validations became `prototype_candidate`.
- 34 promoted validations stayed `research_more`.

That means promote/park/kill and build-prep selection are materially different systems, not two views of the same decision.

### The strict build-prep gate appears effectively unreachable in current data

In the latest `opportunities` snapshot:

- `composite_score >= 0.5`: 0 rows
- `value_support >= 0.55`: 0 rows
- `evidence_quality >= 0.6`: 6 rows
- `corroboration_strength >= 0.6`: 2 rows

Yet prototype candidates still exist in validation history. They are coming through the promote requalification shortcut, not the strict path. That is a strong sign that the build-prep gate carries thresholds from an older regime.

### Market-gap output is not discriminating enough

Validation history by `market_gap_state`:

- `needs_more_recurrence_evidence`: 526
- `partially_solved`: 6
- `underserved_edge_case`: 3

If one category accounts for almost everything, it is not adding much decision clarity.

### Historical state is not fully clean

The latest `opportunities` table still contains rows with:

- `recommendation = kill`
- `selection_status = prototype_candidate`

That suggests historical lifecycle drift or stale rows. It does not necessarily mean current runtime logic is broken, but it does mean any serious scoring comparison needs a rescore/backfill pass first.

## Duplicate Fields, Dead Fields, Hidden Dependencies, And Stale Inputs

### Duplicate fields

The same evaluation facts are duplicated across several artifacts:

- `opportunities` columns
- `opportunities.notes_json.scorecard`
- `validations.evidence.opportunity_scorecard`
- `build_briefs.brief_json`
- `ideas.spec_json`

This is the main source of drift risk. We already started a better direction with `research_spec_v1`, but the validation/build-brief path is still separate.

### Dead or mostly dead runtime pieces

- `src/research/scoring.py` is effectively orphaned from the runtime path; it appears in tests, not in the main validation flow.
- `ValidationAgent.market_weight`, `technical_weight`, and `distribution_weight` are loaded and can be updated, but the main validation path does not use them to make the stored `overall_score`.
- `ValidationAgent.overall_threshold` is largely legacy.
- `compute_cluster_corroboration()` in `src/opportunity_engine.py` appears unused in the live path.
- `docs/gates.md` still describes older composite/plausibility-based promotion logic instead of the current v4 decision path.

### Hidden dependencies

- thresholds come from multiple config shapes through `resolve_promotion_park_thresholds()`
- build-prep selection depends on both scorecard fields and corroboration structure
- `overall_score` in validations is the scorecard `composite_score`, not a weighted combination of `market_score`, `technical_score`, and `distribution_score`
- build-brief creation depends on `selection_status` and on a special promoted-but-not-build-ready draft path

### Circular or overlapping logic

The code is acyclic, but the evidence reuse is heavily overlapping:

- `value_score` feeds `cost_of_inaction`
- `cost_of_inaction` feeds `willingness_to_pay_proxy`
- both feed `value_support`
- `value_support` then feeds `revenue_readiness_score`

Similarly:

- recurrence signals feed `frequency_score`
- corroboration signals feed `corroboration_strength`
- both contribute to `evidence_sufficiency`
- that then contributes to `evidence_quality`

The comments inside `score_opportunity()` already show repeated attempts to remove double-counting. That is usually a sign that the model has too many intermediate synthetic dimensions.

### Stale or unreliable inputs

- `competition_score` is non-zero in only 206/572 `market_enrichments`
- `review_signal_score` is non-zero in only 55/572 `market_enrichments`
- many corroboration rows in history have zeroed fields, which means historical comparisons need careful filtering or rescoring

## Feasibility Verdict

### Verdict

**B. Improvement with simplification needed**

### Why not A

I do not think V2 is ready to replace the current scorer as-is because:

- too many required factors are missing or weakly inferred
- it would be easy to accidentally build V2 as a new layer instead of a replacement
- the current historical dataset needs cleanup/rescoring before any "better than current" claim is credible

### Why not C or D

The current system is more complicated than the ROI justifies. V2 has a better decision shape:

- clearer
- easier to explain
- easier to reason about
- better separation between upside, risk, and evidence

So this is not a worse direction. It is just not ready to be dropped in whole.

## Overengineering Cleanup Plan

### Immediate (1 day)

1. Freeze one canonical evaluation contract.
   - One artifact should own the scorecard, market gap, counterevidence, selection state, and validation plan.
   - `research_spec_v1` is the right direction; validation/build-brief should converge toward the same contract.

2. Stop carrying two score vocabularies as if both are primary.
   - Pick one public scoring language for operators.
   - Either keep `decision_score / problem_truth_score / revenue_readiness_score` or replace them.
   - Do not let build-prep continue to use a separate legacy language.

3. Remove or quarantine dead scoring code.
   - `src/research/scoring.py`
   - unused validation weights / thresholds if they are no longer operational
   - unused helper paths like `compute_cluster_corroboration()` if confirmed dead

4. Fix the docs to match runtime.
   - especially `docs/gates.md`

### Short-term (1 week)

1. Build a shadow `v2_lite` scorer from existing fields only.
   - Use direct/proxy fields already available:
     - `pain_severity`
     - `frequency_score`
     - `urgency_score`
     - `reachability`
     - `buildability`
     - `expansion_potential`
     - `competition_score`
     - `evidence_quality`
     - `family_confirmation_count` / `corroboration_strength`
     - `source_family_diversity`
     - `buyer_intent_score` / `willingness_to_pay_proxy`
   - Do not invent support burden / trust barrier / integration complexity yet.

2. Run shadow scoring on historical validation rows.
   - Compare rank order against:
     - prototype-candidate history
     - build-brief creation
     - operator-reviewed survivors
   - Start workbench-only. No gate changes yet.

3. Collapse build-prep selection onto the same core evaluation contract.
   - remove the strict legacy composite/value gate if the live data shows it is unreachable
   - express prototype readiness as a policy on the canonical scorecard, not a second scoring system

4. Simplify `market_gap_state`.
   - if one state dominates almost everything, reduce or redefine the taxonomy

### Medium-term (1 month)

1. Replace the current layered score braid with one canonical scorer and one policy layer.
   - scorer: evaluate opportunity quality
   - policy: decide promote / prototype / archive

2. Separate lane classification from the score itself.
   - if you want "solo founder", "SMB SaaS", "enterprise", etc., make that a separate classification output
   - do not bake lanes into the main score until the main score is stable

3. Add any truly missing inputs explicitly.
   - support burden
   - trust barrier
   - integration complexity
   - buyer clarity
   - target-user clarity

4. Backfill or rescore historical rows after the contract is settled.
   - otherwise old rows will continue to poison comparisons

## Recommended Next Implementation Steps

1. **Create `OpportunityEvaluationV1`**
   - canonical post-validation contract
   - source of truth for scorecard, market gap, counterevidence, selection, validation plan

2. **Implement `v2_lite_shadow_score()`**
   - no new scraping
   - no new LLM factors
   - only existing stored fields and safe proxies

3. **Build a comparison harness**
   - current decision score vs shadow V2-lite
   - compare against prototype candidates, build briefs, and operator-accepted outcomes

4. **Choose one runtime decision language**
   - either keep v4 labels or replace them
   - update docs and operator views accordingly

5. **Delete redundant layers**
   - especially legacy scoring helpers and duplicated artifact payloads

## Candid Bottom Line

Yes: the current system is overbuilt for the amount of stable decision signal it is getting.

Also yes: Version 2.0 is a better direction, but only if it replaces complexity instead of joining it.

If we implement V2 literally right now, we will probably end up with:

- the current scorecard
- plus a V2 score
- plus the current selection gate
- plus lane logic
- plus swarm logic

That would make the repo worse.

If we implement V2 as a simplification program instead, it can help a lot.
