"""pipeline_health read model."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import Database, Finding
from pipeline_health import compute_pipeline_health


def test_pipeline_health_counts_actionable_qualified():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        assert compute_pipeline_health(db)["actionable_qualified_for_pipeline"] == 0

        fid = db.insert_finding(
            Finding(
                source="t",
                source_url="https://example.com/a",
                product_built="p",
                outcome_summary="o",
                content_hash="hash-a",
                status="qualified",
                source_class="pain_signal",
                finding_kind="problem_signal",
            )
        )
        from database import ProblemAtom, RawSignal

        sid = db.insert_raw_signal(
            RawSignal(
                source_name="t",
                source_type="forum",
                source_url="https://example.com/a",
                title="t",
                body_excerpt="b",
                quote_text="q",
                role_hint="r",
                content_hash="s1",
                finding_id=fid,
            )
        )
        db.insert_problem_atom(
            ProblemAtom(
                signal_id=sid,
                finding_id=fid,
                cluster_key="k",
                segment="s",
                user_role="u",
                job_to_be_done="j",
                trigger_event="",
                pain_statement="p",
                failure_mode="",
                current_workaround="",
                current_tools="",
                urgency_clues="",
                frequency_clues="",
                emotional_intensity=0.5,
                cost_consequence_clues="",
                why_now_clues="",
                confidence=0.5,
                atom_json="{}",
            )
        )
        h = compute_pipeline_health(db)
        assert h["actionable_qualified_for_pipeline"] == 1
        assert h["interpretation"]["run_once_will_process_backlog"] is True
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)
