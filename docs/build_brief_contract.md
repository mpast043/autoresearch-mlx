# Build Brief Certainty Contract

This note defines the certainty fields that downstream code must preserve when consuming build briefs.

## Fields

- `selection_reason`
  - Source gate that admitted the item.
  - Important values:
    - `validated_selection_gate`
    - `prototype_candidate_gate`

- `selection_gate.gate_version`
  - Rule version for the admitting gate.
  - Expected values:
    - `build_prep_v1`
    - `prototype_candidate_v1`

- `prototype_gate.prototype_gate_mode`
  - High-level certainty mode for downstream handling.
  - Expected values:
    - `strict_validated`
    - `prototype_candidate_exception`

- `prototype_gate.prototype_gate_basis`
  - Why the item was admitted.
  - Common values:
    - `full_market_proof`
    - `supported_single_family_workflow_pain`
    - `multifamily_near_miss`

- `prototype_gate.prototype_gate_family_count`
  - Core corroborating source-family count used at admission time.

- `prototype_gate.prototype_gate_evidence_strength`
  - Snapshot of admission-time evidence signals.
  - This is diagnostic context, not the primary runtime score language.
  - Primary operator-facing scores live in embedded `opportunity_evaluation.measures.scores`.
  - Includes:
    - `recurrence_state`
    - `corroboration_score`
    - `family_count`
    - `value_support`
    - `problem_plausibility`
    - `composite_score`
    - `wedge_active`

- `prototype_gate.market_confidence_level`
  - Downstream confidence label.
  - Expected values:
    - `market_confirmed`
    - `prototype_checkpoint`

- `prototype_gate.validation_certainty`
  - Human-readable certainty summary.
  - Expected values:
    - `validated_selection_gate_met`
    - `credible_prototype_candidate_not_market_confirmed`

- `prototype_gate.overclaim_guardrail`
  - Required downstream messaging constraint.

- `prototype_spec_posture`
  - Default framing for first prototype outputs.
  - Includes:
    - `recommended_output_type`
    - `confidence_label`
    - `build_scope_rule`
    - `messaging_rule`

## Interpretation Rules

- Market-confirmed path:
  - `selection_reason == validated_selection_gate`
  - `prototype_gate.prototype_gate_mode == strict_validated`
  - `prototype_gate.market_confidence_level == market_confirmed`

- Prototype-candidate checkpoint path:
  - `selection_reason == prototype_candidate_gate`
  - `prototype_gate.prototype_gate_mode == prototype_candidate_exception`
  - `prototype_gate.market_confidence_level == prototype_checkpoint`

## Consumer Rules

- Downstream consumers must not overstate market proof when `prototype_gate.prototype_gate_mode == prototype_candidate_exception`.
- Prototype-candidate cases should default to narrow, diagnostic, uncertainty-aware prototype framing using `prototype_spec_posture`.
- Consumers may use stronger validated language only when `prototype_gate.prototype_gate_mode == strict_validated`.
