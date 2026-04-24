# AutoResearch MLX Vault-Grounded Design Spec and Review

Date: 2026-04-18

Scope: repository review grounded in the local Obsidian notes under
`/Users/meganpastore/Documents/Obsidian Vault/Autoagent thing/` plus nearby
project notes on PainOnSocial, SaaS evidence quality, and source lists.

This document is an operating contract for the current system. It separates
what the product is trying to become from the code paths that currently enforce
or violate that intent.

## 1. Note-Derived Product Intent

The project is no longer just an idea finder. The intended product is a local,
evidence-first venture studio operating system:

```text
discover -> evidence -> validation -> selection -> build brief -> build prep -> build -> launch -> learn
```

The research engine should become "good enough" before the build half is allowed
to act. In note terms, good enough means:

- high junk rejection
- valid problem atoms
- clean cross-source corroboration
- clear park/kill/promote reasons
- stable operator reports across runs
- bounded review feedback
- no source leakage or wedge contamination

The system should prefer narrow, operator-visible pain over broad SaaS chatter.
The SaaS review notes specifically distinguish weak headline claims from useful
underlying patterns. The code should preserve that distinction: a good thread can
contribute a strict evidence row and candidate atom, while unverifiable success
claims should not become proof.

## 2. Current Architecture Contract

The core runtime path remains:

```text
discovery -> evidence -> validation -> build_prep
```

Primary source of truth:

- `src/database.py` for schema and persistence
- `src/source_policy.py` for source-class routing vocabulary
- `src/agents/discovery.py` for source intake and atom eligibility
- `src/agents/evidence.py` for recurrence, corroboration, and market enrichment
- `src/agents/validation.py` for score persistence, lifecycle state, and build brief creation
- `src/opportunity_engine.py` for scoring, counterevidence, and stage decision
- `src/build_prep.py` for selection and build handoff gates
- `src/orchestrator.py` for runtime message routing

The first-class state chain is:

```text
finding -> raw_signal -> problem_atom -> corroboration + market_enrichment -> cluster -> opportunity -> validation -> build_brief -> build_prep_outputs
```

`decision` and `selection_status` are distinct:

- `decision`: validation recommendation, one of promote / park / kill.
- `selection_status`: post-validation portfolio/build state such as research_more,
  prototype_candidate, prototype_ready, build_ready, launched, iterate, expand, archive.

## 3. Source Policy Rules

The target source classes from the notes are implemented in `src/source_policy.py`:

- `pain_signal`
- `success_signal`
- `demand_signal`
- `competition_signal`
- `meta_guidance`
- `low_signal_summary`

The required routing contract is:

- `pain_signal` may generate problem atoms.
- `success_signal` can seed search or market motion, but must not become a pain atom.
- `demand_signal` belongs in validation/market enrichment.
- `competition_signal` belongs in market gap enrichment.
- `meta_guidance` belongs in prompt/eval improvement.
- `low_signal_summary` should be excluded or down-ranked.

The repository now has the correct module shape for this. The main review concern
is preserving the policy when non-Reddit lanes expand, especially Shopify,
WordPress, GitHub, and YouTube comments.

## 4. Evidence And Corroboration Contract

The notes repeatedly call out corroboration depth and source diversity as the
main trust bottleneck. Evidence must therefore distinguish:

- origin provenance: where the original finding came from
- corroborating families: independent sources that matched the pain signature
- raw recurrence counts: total docs, domains, and query breadth
- evidence sufficiency: whether the evidence is enough to act
- market value: buyer intent, cost pressure, competition, and trend context

Important invariant: origin provenance should not inflate corroborating source
families. A Shopify review that is confirmed by Reddit and WordPress has review
origin plus two corroborating families; a web-origin item confirmed only by Reddit
has one corroborating family, not web plus Reddit.

Current focused tests show this invariant is violated. See Review Finding 1.

## 5. Validation And Gate Contract

`stage_decision` is currently v4-style:

- `decision_score`
- `problem_truth_score`
- `revenue_readiness_score`
- frequency floor
- high-frequency override
- strong-evidence override

Operator diagnostics must explain the same formula that runtime uses. If the
runtime promotes through v4 scores but `gate-diagnostics` reports failed legacy
composite/plausibility/value checks, the operator surface becomes misleading.

Current focused repro confirms this mismatch. See Review Finding 2.

## 6. Build Handoff Contract

The P3 notes define the canonical `build_brief` handoff. It should include:

- opportunity id
- source-family support summary
- target segment
- user role
- JTBD
- normalized pain statement
- corroboration summary
- market/value summary
- generalizability assessment
- proposed wedge
- recommended product type
- recommended experiment type
- initial scope
- assumptions
- constraints
- non-goals
- success metric
- kill criteria

The current code has `build_brief_payload` and build-prep agents for:

```text
solution_framing -> experiment_design -> spec_generation
```

The right design posture is narrow and diagnostic. `prototype_candidate` can be
exploratory, but `build_ready` must require a concrete product shape, host/platform
fit, failure mode, and enough corroboration.

## 7. Runtime Configuration Contract

Repo-root execution and packaged execution should not silently diverge. The path
resolver uses repo `config.yaml` when present and falls back to
`src/resources/config.default.yaml` when installed outside the repo.

That fallback config currently differs materially from repo config in discovery
breadth, Reddit settings, candidate penalties, threshold values, LLM limits, and
optional agent sections. If intentional, the safer packaged policy needs to be
documented and tested. If not intentional, the default resource should be synced.

See Review Finding 3.

## 8. Review Findings

### Finding 1: Corroboration counts origin provenance as confirming family

File: `src/agents/evidence.py`

`_build_corroboration_record` appends `origin_source_family` into
`source_families` whenever any recurrence doc matches the signature. That turns
provenance into corroboration and inflates `source_family_diversity`,
`core_source_family_diversity`, corroboration bonuses, market value lift, and
selection eligibility.

Focused tests failing:

```text
tests/test_agents/test_evidence.py::TestEvidenceAgent::test_cross_source_corroboration_counts_reddit_and_review_families
tests/test_agents/test_evidence.py::TestEvidenceAgent::test_source_families_report_only_corroborating_families_not_origin_provenance
```

Expected fix direction:

- Keep `origin_source_family` as separate provenance.
- Compute `source_families` only from matched recurrence docs.
- If origin participation is useful, expose it through a separate
  `origin_plus_corroborating_family_diversity` field rather than overloading
  corroborating families.

### Finding 2: Gate diagnostics still report legacy promotion checks

File: `src/opportunity_engine.py`

`stage_decision` promotes through v4 fields, but `diagnose_stage_decision`
builds `promote_checks` from legacy composite/plausibility/evidence/value gates.
The result can be a promoted decision with all displayed promotion checks failing.

Expected fix direction:

- Branch diagnostics based on whether v4 score fields are present, mirroring
  `stage_decision`.
- Include primary v4 checks and override checks separately.
- Update `docs/gates.md` so operator docs match runtime behavior.

### Finding 3: Packaged fallback config drifts from repo config

File: `src/resources/config.default.yaml`

Installed execution can load the packaged fallback instead of repo `config.yaml`.
The fallback currently differs on discovery settings, Reddit breadth, candidate
penalties, validation thresholds, LLM token limits, and optional agent sections.

Expected fix direction:

- Decide whether fallback is a deliberately safer packaged profile.
- If yes, document it and add a config parity/drift test for intentional deltas.
- If no, sync `src/resources/config.default.yaml` from repo `config.yaml`.

## 9. Verification Performed

CodeRabbit CLI was requested via the CodeRabbit skill but is not installed:

```text
coderabbit --version -> command not found
coderabbit auth status --agent -> command not found
```

Focused local verification:

```text
python3 -m pytest tests/test_source_policy.py tests/test_orchestrator.py -q
12 passed
```

Focused evidence/build/gate verification:

```text
python3 -m pytest tests/test_opportunity_engine_stage_decision.py tests/test_agents/test_evidence.py tests/test_validation_thresholds.py tests/test_build_prep.py -q
57 passed, 2 failed
```

The two failures are the corroboration-family accounting issue in Finding 1.

Diagnostic repro for Finding 2:

```text
decision: promote
all_promotion_numeric_gates_pass: false
promote_checks: legacy composite/plausibility/evidence/value checks
```

## 10. Next Implementation Order

1. Fix evidence-family accounting without narrowing generic recurrence breadth.
2. Align `diagnose_stage_decision` and `docs/gates.md` with v4 stage decision.
3. Decide and test config fallback policy.
4. Re-run the focused suite above, then full `python3 -m pytest tests/ -q`.
5. After gates and evidence accounting are stable, continue calibration depth and
   source expansion rather than broad new architecture.

