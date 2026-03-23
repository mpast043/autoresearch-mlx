# State Model

This document defines the active runtime state model for the evidence-first pipeline.

## Active Runtime Flow

`finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> experiment + validation -> build_brief + build_prep_outputs -> ledger`

The live pipeline flow is:

1. `discovery`
2. `evidence`
3. `validation`
4. `build_prep`

## Source Policy

The first-class source policy is implemented in [`src/source_policy.py`](/Users/meganpastore/Projects/autoresearch-mlx/src/source_policy.py).

Supported source classes:

- `pain_signal`
- `success_signal`
- `demand_signal`
- `competition_signal`
- `meta_guidance`
- `low_signal_summary`

Routing rules:

- `pain_signal`: eligible for `raw_signals` and `problem_atoms`
- `success_signal`: excluded from atom generation, but can be used for search seeding and market context
- `demand_signal`: excluded from atom generation, used for market enrichment only
- `competition_signal`: excluded from atom generation, used for market-gap enrichment only
- `meta_guidance`: excluded from active discovery, used for prompt/eval improvement only
- `low_signal_summary`: excluded from the active path

The explicit review lanes currently use WordPress Plugin Directory reviews and Shopify App Store reviews through [`src/review_sources.py`](/Users/meganpastore/Projects/autoresearch-mlx/src/review_sources.py):

- detailed review text can become `pain_signal`
- vague review praise stays `low_signal_summary`
- listing/marketing copy is explicitly excluded from atom generation
- rating, review count, installs or popularity proxies, pricing, category, developer/vendor, and version/update metadata are persisted for market enrichment and reporting, not atom generation
- Shopify-specific limitations:
  - install counts are not public, so the lane uses bounded popularity proxies when available
  - review titles are often absent, so a compact derived title may be used for the finding record while raw review text remains the actual evidence body

The first explicit technical issue lane currently uses GitHub issues/discussions through [`src/github_sources.py`](/Users/meganpastore/Projects/autoresearch-mlx/src/github_sources.py):

- issue/discussion text can become `pain_signal` when it includes enough workflow detail
- generic feature wishlists and thin product-specific issue noise are routed to `low_signal_summary`
- GitHub metadata is persisted explicitly for:
  - `trigger`
  - `failure_mode`
  - `workaround`
  - `reproduction_context`
  - `cost_friction`

## Source Of Truth Tables

These tables are the active runtime model:

- `findings`
- `raw_signals`
- `problem_atoms`
- `corroborations`
- `market_enrichments`
- `clusters`
- `cluster_members`
- `opportunities`
- `experiments`
- `validations`
- `evidence_ledger`
- `review_feedback`
- `build_briefs`
- `build_prep_outputs`

Meaning:

- `findings`: source intake record and lifecycle state
- `raw_signals`: normalized accepted source evidence
- `problem_atoms`: structured pain extraction
- `corroborations`: recurrence/corroboration evidence for a finding in one run
  - `evidence_json` carries depth and diversity details such as:
    - `confirmation_depth_score`
    - `query_breadth_score`
    - `source_concentration`
    - `single_source_penalty`
    - `source_families`
    - `source_family_match_counts`
    - `core_source_families`
    - `core_source_family_diversity`
    - `core_source_family_bonus`
    - `source_group_diversity`
    - `cross_source_match_score`
    - `generalizability_class`
- `market_enrichments`: market/demand evidence for a finding in one run
  - `evidence_json` carries value-calibration details such as:
    - `cost_pressure_score`
    - `operational_buyer_score`
    - `compliance_burden_score`
    - `multi_source_value_lift`
    - `willingness_to_pay_signal`
    - wedge provenance and activation details:
      - `wedge_fit_score`
      - `wedge_activation_reasons`
      - `wedge_block_reasons`
      - `wedge_rule_version`
      - `wedge_active`
    - review-source metadata when present:
      - `review_product_name`
      - `source_review_rating`
      - `aggregate_rating`
      - `review_count`
      - `active_installs`
      - `popularity_proxy`
      - `pricing`
      - `category`
      - `listing_url`
      - `developer_name`
      - `launched_at`
      - `version`
      - `last_updated`
- `clusters`: cluster projection used for operator review and traceability
- `cluster_members`: atom-to-cluster traceability
- `opportunities`: latest-state scored opportunity per cluster
- `experiments`: latest proposed experiment per `(run_id, opportunity_id, plan_hash)`
- `validations`: per-run validation history keyed by `(run_id, finding_id)`
- `evidence_ledger`: per-run evidence history keyed by `(run_id, entity_type, entity_id, entry_kind)`
- `review_feedback`: operator review history for borderline, screened-out, parked, or killed cases
  - supported review labels:
    - `correct`
    - `false_positive`
    - `bad_extraction`
    - `should_park`
    - `should_kill`
    - `needs_more_evidence`
  - this table is also used as a bounded calibration input for later validation runs against the same finding or cluster
  - calibration strength is limited by:
    - repeated consistent labels before full-strength influence
    - age-based decay so older feedback gradually weakens
  - operator-facing reports surface both raw review counts and effective decayed review strength
  - run reports also surface `decision_reason_mix` and `run_diff` against the prior run
- `build_briefs`: canonical post-validation handoff records for selection/build-prep
  - generated only for opportunities that pass the `validated -> prototype_candidate` gate
  - `brief_json` is the source of truth for:
    - linked finding ids
    - problem summary
    - job to be done
    - pain/workaround
    - evidence provenance
    - source family corroboration
    - screening summary
    - wedge/profitability/relevance fields
    - recommended narrow output type
    - first experiment hypothesis
    - launch artifact plan
    - open questions / risks
- `build_prep_outputs`: run-scoped outputs from:
  - `solution_framing`
  - `experiment_design`
  - `spec_generation`

## Compatibility / Mirror Tables

These are retained for compatibility or projection purposes:

- `opportunity_clusters`
- `validation_experiments`
- `ideas`
- `products`
- `resources`

Notes:

- `opportunity_clusters` mirrors the active `clusters` row with the same `id` for existing call sites.
- `validation_experiments` is legacy-compatible; the active validation path writes `experiments`.
- `ideas` and `products` are downstream optional layers and are not part of the core discovery/validation truth model.

## Status Model

`findings.status` is the operator-facing lifecycle state for the active path.

Allowed active statuses:

- `new`
- `qualified`
- `screened_out`
- `parked`
- `killed`
- `promoted`
- `reviewed`

Expected transitions:

- `screened_out`: no `raw_signal` or `problem_atom`
- `qualified`: accepted `pain_signal` awaiting evidence/validation completion
- `parked`: validation recommendation is `park`
- `killed`: validation recommendation is `kill`
- `promoted`: validation recommendation is `promote`

`finding_kind` is retained as a compatibility/source-descriptor field. `source_class` is the enforcement field used for routing.

`opportunities.status` remains the validation recommendation-facing field (`parked`, `killed`, `promoted`).

`opportunities.selection_status` is the post-validation lifecycle field for build preparation.

Allowed selection states:

- `research_more`
- `prototype_candidate`
- `prototype_ready`
- `build_ready`
- `launched`
- `iterate`
- `expand`
- `archive`

Allowed selection transitions:

- `research_more -> prototype_candidate`
- `research_more -> archive`
- `prototype_candidate -> prototype_ready`
- `prototype_candidate -> research_more`
- `prototype_candidate -> archive`
- `prototype_ready -> build_ready`
- `prototype_ready -> research_more`
- `prototype_ready -> archive`
- `build_ready -> launched`
- `build_ready -> iterate`
- `build_ready -> archive`
- `launched -> iterate`
- `launched -> expand`
- `launched -> archive`
- `iterate -> build_ready`
- `iterate -> expand`
- `iterate -> archive`
- `expand -> iterate`
- `expand -> archive`

## Run-Scoped History

The following tables are run-scoped and idempotent within a run:

- `validations`
- `experiments`
- `corroborations`
- `market_enrichments`
- `evidence_ledger`
- `build_briefs`
- `build_prep_outputs`

The runtime assigns a `run_id` at startup. Repeated writes in the same run update in place. Writes in a later run create a new history row.

## Wedge Activation

Wedge assignment is default-off.

The active wedge path currently uses a guarded backup/restore profile in [`src/agents/evidence.py`](/Users/meganpastore/Projects/autoresearch-mlx/src/agents/evidence.py):

1. candidate detection
2. fit validation
3. runtime sanity check
4. score-lift application

If any validation or sanity rule fails, the wedge remains inactive and only provenance fields are persisted.
