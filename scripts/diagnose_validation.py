#!/usr/bin/env python3
"""
diagnose_validation.py
Run a single validation end-to-end in-process and print every intermediate score
so you can see exactly which gate threshold is failing.

Usage:
    python scripts/diagnose_validation.py [finding_id]
    python scripts/diagnose_validation.py 42
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root AND src/ to path (mimics run.py)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for p in [str(SRC_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.database import Database
from src.agents.validation import ValidationAgent
from src.opportunity_engine import (
    assess_market_gap,
    build_cluster_summary,
    build_counterevidence,
    build_problem_atom,
    score_opportunity,
    stage_decision,
)
from src.build_prep import determine_selection_state
from src.research_tools import ResearchToolkit
from src.messaging import MessageType


def load_config():
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


async def diagnose(finding_id: int | None = None):
    db_path = Path(__file__).parent.parent / "data" / "autoresearch.db"
    db = Database(str(db_path))
    config = load_config()

    # Pick a finding — prefer 'parked' (has been through validation attempt)
    if finding_id is None:
        row = db._get_connection().execute(
            "SELECT id FROM findings WHERE status IN ('parked','killed') LIMIT 1"
        ).fetchone()
        if row is None:
            print("No findings to diagnose."); return
        finding_id = row["id"]

    finding = db.get_finding(finding_id)
    if finding is None:
        print(f"Finding {finding_id} not found."); return

    print(f"\n{'='*60}")
    print(f"DIAGNOSING FINDING {finding_id}")
    print(f"{'='*60}")
    print(f"  source:        {finding.source}")
    print(f"  status:       {finding.status}")
    print(f"  finding_kind: {finding.finding_kind}")
    print(f"  product_built:{finding.product_built}")
    print(f"  outcome_sum:  {finding.outcome_summary[:80] if finding.outcome_summary else '(empty)'}...")

    # ── Step 1: Problem atoms ──────────────────────────────────────
    print(f"\n[1] Problem atoms...")
    agent = ValidationAgent(db=db, config=config)
    atoms_result = agent._ensure_problem_atoms(finding_id, {
        "evidence": json.loads(finding.evidence_json or "{}"),
        "source_class": finding.source_class,
        "finding_kind": finding.finding_kind,
    }, finding.finding_kind)
    if not atoms_result:
        print("  → FAIL: _ensure_problem_atoms returned empty (finding not eligible)")
        return
    anchor_atom = atoms_result[0]
    print(f"  → anchor_atom: {anchor_atom.signal_id} — '{anchor_atom.pain_statement[:60]}'")

    # ── Step 2: Evidence scoring ───────────────────────────────────
    print(f"\n[2] Evidence scoring (LLM call — validate_problem)...")
    toolkit = ResearchToolkit(config)
    evidence_scores = await toolkit.validate_problem(
        title=finding.product_built or "Untitled",
        summary=finding.outcome_summary or "",
        finding_kind=finding.finding_kind,
        audience_hint=finding.entrepreneur or "",
    )
    # Cast to float (LLM may return strings)
    ps = float(evidence_scores.get('problem_score', 0) or 0)
    vs = float(evidence_scores.get('value_score', 0) or 0)
    fs = float(evidence_scores.get('feasibility_score', 0) or 0)
    sgs = float(evidence_scores.get('solution_gap_score', 0) or 0)
    ss = float(evidence_scores.get('saturation_score', 0) or 0)

    print(f"  problem_score:       {ps}")
    print(f"  value_score:         {vs}")
    print(f"  feasibility_score:   {fs}")
    print(f"  solution_gap_score: {sgs}")
    print(f"  saturation_score:    {ss}")

    evidence_scores = {
        "problem_score": ps,
        "value_score": vs,
        "feasibility_score": fs,
        "solution_gap_score": sgs,
        "saturation_score": ss,
        "evidence": evidence_scores.get("evidence", []),
    }
    market_score = (ps + vs) / 2
    technical_score = fs
    distribution_score = (sgs + ss) / 2
    print(f"\n  → market_score:       {round(market_score,4)}")
    print(f"  → technical_score:    {round(technical_score,4)}")
    print(f"  → distribution_score: {round(distribution_score,4)}")

    # ── Step 3: Clustering ─────────────────────────────────────────
    print(f"\n[3] Clustering...")
    cluster_id, cluster, cluster_atoms = agent._cluster_atoms(anchor_atom)
    print(f"  cluster_id: {cluster_id}")
    print(f"  cluster:    {cluster.label}")
    print(f"  atom_count: {len(cluster_atoms)}")

    # ── Step 4: Market gap ─────────────────────────────────────────
    print(f"\n[4] Market gap assessment...")
    validation_payload = {
        "scores": evidence_scores,  # already cast to floats above
        "evidence": evidence_scores.get("evidence", []),
        "corroboration": {},
        "market_enrichment": {},
        "review_feedback": {},
    }
    cluster_context = agent._cluster_context(cluster)
    market_gap = assess_market_gap(cluster_context, validation_payload)
    print(f"  → market_gap: {market_gap['market_gap']}")
    for k, v in market_gap.items():
        if k != "market_gap":
            print(f"     {k}: {v}")

    # ── Step 5: Score card ─────────────────────────────────────────
    print(f"\n[5] Opportunity scoring...")
    anchor_signal = db.get_raw_signal(anchor_atom.signal_id)
    scorecard = score_opportunity(
        anchor_atom, anchor_signal, cluster_context,
        validation_payload, market_gap,
        review_feedback=None,
    )
    print(f"\n  KEY SCORES:")
    print(f"  ─{'─'*40}")
    print(f"  {'composite_score':<30} {scorecard['composite_score']:.4f}  (promote threshold: 0.66)")
    print(f"  {'evidence_quality':<30} {scorecard['evidence_quality']:.4f}  (promote threshold: 0.55)")
    print(f"  {'problem_plausibility':<30} {scorecard['problem_plausibility']:.4f}  (promote threshold: 0.60)")
    print(f"  {'value_support':<30} {scorecard['value_support']:.4f}  (promote threshold: 0.58)")
    print(f"  {'corroboration_strength':<30} {scorecard['corroboration_strength']:.4f}")
    print(f"  {'evidence_sufficiency':<30} {scorecard['evidence_sufficiency']:.4f}")
    print(f"  {'evidence_multiplier':<30} {scorecard['evidence_multiplier']:.4f}")
    print(f"  {'pain_severity':<30} {scorecard['pain_severity']:.4f}")
    print(f"  {'frequency_score':<30} {scorecard['frequency_score']:.4f}")
    print(f"  {'urgency_score':<30} {scorecard['urgency_score']:.4f}")
    print(f"  {'timing_shift':<30} {scorecard['timing_shift']:.4f}")
    print(f"  {'buildability':<30} {scorecard['buildability']:.4f}")
    print(f"  {'adoption_friction':<30} {scorecard['adoption_friction']:.4f}")
    print(f"  {'dependency_risk':<30} {scorecard['dependency_risk']:.4f}")

    # ── 5b: Why is market_score so low? ──────────────────────────
    print(f"\n[5b] market_score breakdown (ROOT CAUSE):")
    print(f"  market_score = (problem_score + value_score) / 2")
    print(f"               = ({evidence_scores['problem_score']} + {evidence_scores['value_score']}) / 2")
    print(f"               = {market_score:.4f}")
    if evidence_scores["problem_score"] == 0.0:
        print(f"  ⚠️  problem_score=0.0 — validate_problem LLM could not extract a valid problem from this content!")
        print(f"      title used: '{(finding.product_built or 'N/A')[:80]}'")
        print(f"      summary used: '{(finding.outcome_summary or 'N/A')[:80]}'")
    if scorecard["corroboration_strength"] == 0.0:
        print(f"  ⚠️  corroboration_strength=0.0 — NO multi-source corroboration for this finding!")
        print(f"      This finding has no related_signals from other sources.")
        print(f"      Reddit single-source signals almost always fail at this gate.")
    if scorecard["evidence_sufficiency"] < 0.1:
        print(f"  ⚠️  evidence_sufficiency={scorecard['evidence_sufficiency']:.4f} — very thin evidence")

    # ── Step 6: Counterevidence ────────────────────────────────────
    print(f"\n[6] Counterevidence...")
    counterevidence = build_counterevidence(scorecard, market_gap)
    supported_count = sum(1 for c in counterevidence if c.get("status") == "supported")
    print(f"  total checks: {len(counterevidence)}")
    print(f"  supported:    {supported_count}  (promote allows ≤1)")
    for c in counterevidence:
        status = c.get("status", "?")
        marker = "✅" if status == "supported" else "❌" if status == "refuted" else "⚠️"
        print(f"  {marker} [{status}] {c.get('check','')[:60]}")

    # ── Step 7: Stage decision ─────────────────────────────────────
    print(f"\n[7] Stage decision...")
    promotion_threshold = 0.66
    park_threshold = 0.42
    decision = stage_decision(
        scorecard, market_gap, counterevidence,
        promotion_threshold=promotion_threshold,
        park_threshold=park_threshold,
    )
    print(f"\n  {'─'*40}")
    print(f"  RECOMMENDATION: {decision['recommendation'].upper()}")
    print(f"  STATUS:          {decision['status']}")
    print(f"  REASON:          {decision.get('reason','')}")
    print(f"  {'─'*40}")

    # ── Step 8: Gate summary ──────────────────────────────────────
    print(f"\n[8] Gate thresholds:")
    diag_result = stage_decision(
        scorecard, market_gap, counterevidence,
        promotion_threshold=promotion_threshold,
        park_threshold=park_threshold,
    )
    print(f"  recommendation: {diag_result['recommendation'].upper()}")
    print(f"  reason:         {diag_result.get('reason', '')}")

    # ── Step 9: Selection state ───────────────────────────────────
    print(f"\n[9] Selection state...")
    sel_status, sel_reason, sel_gate = determine_selection_state(
        decision=decision["recommendation"],
        scorecard=scorecard,
        corroboration={},
        market_enrichment={},
    )
    print(f"  selection_status: {sel_status}")
    print(f"  selection_reason: {sel_reason}")
    print(f"  selection_gate:   {sel_gate}")
    print(f"\n  → passed: {decision['recommendation'] == 'promote'}")
    if sel_status == "prototype_candidate":
        print(f"  → WOULD REACH IDEATION ✅")
    else:
        print(f"  → WOULD NOT REACH IDEATION ❌")

    # ── Summary of what's failing ─────────────────────────────────
    print(f"\n{'='*60}")
    print("FAILING GATES:")
    print(f"  composite_score >= 0.66      : {scorecard['composite_score']:.4f} → {'✅' if scorecard['composite_score'] >= 0.66 else '❌ FAIL'}")
    print(f"  plausibility >= 0.6          : {scorecard['problem_plausibility']:.4f} → {'✅' if scorecard['problem_plausibility'] >= 0.6 else '❌ FAIL'}")
    print(f"  evidence_quality >= 0.55     : {scorecard['evidence_quality']:.4f} → {'✅' if scorecard['evidence_quality'] >= 0.55 else '❌ FAIL'}")
    print(f"  value_support >= 0.58        : {scorecard['value_support']:.4f} → {'✅' if scorecard['value_support'] >= 0.58 else '❌ FAIL'}")
    print(f"  supported_count <= 1         : {supported_count} → {'✅' if supported_count <= 1 else '❌ FAIL'}")
    print(f"  market_gap != already_solved  : {market_gap['market_gap']} → {'✅' if market_gap['market_gap'] != 'already_solved_well' else '❌ FAIL'}")


if __name__ == "__main__":
    finding_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(diagnose(finding_id))
