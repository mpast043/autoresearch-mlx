import cli


def test_normalize_revalidation_notes_refreshes_versions_and_cluster_counts():
    notes = {
        "scorecard": {
            "scoring_version": "v3",
            "formula_version": "older",
            "threshold_version": "2025_q2",
            "composite_score": 0.18,
        },
        "cluster_summary": {
            "signal_count": 1,
            "atom_count": 1,
            "source_types": {"web": 1},
        },
    }

    normalized = cli._normalize_revalidation_notes(
        existing_notes=notes,
        scorecard={
            "composite_score": 0.12,
            "evidence_quality": 0.2,
            "cluster_signal_count": 3,
            "cluster_atom_count": 2,
        },
        cluster_summary=notes["cluster_summary"],
        market_gap={"market_gap": "needs_more_recurrence_evidence"},
        counterevidence=[{"claim": "example", "status": "supported"}],
        scoring_version="v4",
        formula_version="pts_rrs_v1",
        threshold_version="2026_q2",
        recommendation="kill",
        selection_status="archive",
        selection_reason="revalidated_source_policy_reject",
    )

    assert normalized["scorecard"]["scoring_version"] == "v4"
    assert normalized["scorecard"]["formula_version"] == "pts_rrs_v1"
    assert normalized["scorecard"]["threshold_version"] == "2026_q2"
    assert normalized["scorecard"]["composite_score"] == 0.12
    assert normalized["cluster_summary"]["signal_count"] == 3
    assert normalized["cluster_summary"]["atom_count"] == 2
    assert normalized["cluster_summary"]["source_types"] == {"web": 1}
    assert normalized["revalidation"] == {
        "recommendation": "kill",
        "selection_status": "archive",
        "selection_reason": "revalidated_source_policy_reject",
        "scoring_version": "v4",
        "formula_version": "pts_rrs_v1",
        "threshold_version": "2026_q2",
    }
