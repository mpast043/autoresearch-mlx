"""Small gold-set behavioral evaluation harness for the discovery pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from database import ProblemAtom, RawSignal
from opportunity_engine import (
    build_cluster_summary,
    build_problem_atom,
    build_raw_signal_payload,
    classify_source_signal,
    qualify_problem_signal,
    score_opportunity,
    stage_decision,
)


def _load_gold(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def run_behavior_eval(path: str | Path) -> dict[str, Any]:
    gold = _load_gold(path)
    failures: list[dict[str, Any]] = []
    counts = {
        "source_policy": 0,
        "cluster_labels": 0,
        "decisions": 0,
        "feedback_calibration": 0,
        "scoring_calibration": 0,
    }

    for case in gold.get("source_policy", []):
        counts["source_policy"] += 1
        finding_data = dict(case["finding_data"])
        signal_payload = build_raw_signal_payload(finding_data)
        atom_payload = build_problem_atom(signal_payload, finding_data)
        classification = classify_source_signal(finding_data, signal_payload, atom_payload)
        finding_data["source_class"] = classification["source_class"]
        screening = qualify_problem_signal(finding_data, signal_payload, atom_payload)
        if (
            classification["source_class"] != case["expected_source_class"]
            or screening["accepted"] != case["expected_accepted"]
        ):
            failures.append(
                {
                    "suite": "source_policy",
                    "name": case["name"],
                    "expected": {
                        "source_class": case["expected_source_class"],
                        "accepted": case["expected_accepted"],
                    },
                    "actual": {
                        "source_class": classification["source_class"],
                        "accepted": screening["accepted"],
                    },
                }
            )

    for case in gold.get("cluster_labels", []):
        counts["cluster_labels"] += 1
        atoms = [
            ProblemAtom(
                signal_id=index + 1,
                cluster_key="gold-cluster",
                segment=item["segment"],
                user_role=item["user_role"],
                job_to_be_done=item["job_to_be_done"],
                trigger_event=item.get("trigger_event", ""),
                pain_statement=item["pain_statement"],
                failure_mode=item.get("failure_mode", ""),
                current_workaround=item.get("current_workaround", ""),
                current_tools=item.get("current_tools", ""),
                urgency_clues="",
                frequency_clues="",
                emotional_intensity=0.4,
                cost_consequence_clues="",
                why_now_clues="",
                confidence=item.get("confidence", 0.7),
                atom_json="{}",
            )
            for index, item in enumerate(case["atoms"])
        ]
        signals = [
            RawSignal(
                source_name="gold",
                source_type=item.get("source_type", "forum"),
                source_url="https://example.com",
                title=item["title"],
                body_excerpt=item["body_excerpt"],
                content_hash=f"gold-{case['name']}-{index}",
            )
            for index, item in enumerate(case["signals"])
        ]
        cluster = build_cluster_summary(atoms, signals)
        missing = [needle for needle in case["label_contains"] if needle.lower() not in cluster["label"].lower()]
        if missing:
            failures.append(
                {
                    "suite": "cluster_labels",
                    "name": case["name"],
                    "expected_contains": case["label_contains"],
                    "actual_label": cluster["label"],
                }
            )

    for case in gold.get("decisions", []):
        counts["decisions"] += 1
        decision = stage_decision(
            case["opportunity_scores"],
            case["market_gap"],
            case["counterevidence"],
        )
        if decision["recommendation"] != case["expected_recommendation"]:
            failures.append(
                {
                    "suite": "decisions",
                    "name": case["name"],
                    "expected": case["expected_recommendation"],
                    "actual": decision["recommendation"],
                }
            )

    for case in gold.get("feedback_calibration", []):
        counts["feedback_calibration"] += 1
        decision = stage_decision(
            case["opportunity_scores"],
            case["market_gap"],
            case["counterevidence"],
            review_feedback=case["review_feedback"],
        )
        if decision["recommendation"] != case["expected_recommendation"]:
            failures.append(
                {
                    "suite": "feedback_calibration",
                    "name": case["name"],
                    "expected": case["expected_recommendation"],
                    "actual": decision["recommendation"],
                }
            )

    for case in gold.get("scoring_calibration", []):
        counts["scoring_calibration"] += 1
        atom = ProblemAtom(
            signal_id=1,
            finding_id=1,
            cluster_key="gold-score",
            segment=case["atom"]["segment"],
            user_role=case["atom"]["user_role"],
            job_to_be_done=case["atom"]["job_to_be_done"],
            trigger_event=case["atom"].get("trigger_event", ""),
            pain_statement=case["atom"]["pain_statement"],
            failure_mode=case["atom"].get("failure_mode", ""),
            current_workaround=case["atom"].get("current_workaround", ""),
            current_tools=case["atom"].get("current_tools", ""),
            urgency_clues=case["atom"].get("urgency_clues", ""),
            frequency_clues=case["atom"].get("frequency_clues", ""),
            emotional_intensity=case["atom"].get("emotional_intensity", 0.4),
            cost_consequence_clues=case["atom"].get("cost_consequence_clues", ""),
            why_now_clues=case["atom"].get("why_now_clues", ""),
            confidence=case["atom"].get("confidence", 0.7),
            atom_json="{}",
        )
        signal = RawSignal(
            source_name="gold",
            source_type=case["signal"].get("source_type", "forum"),
            source_url="https://example.com",
            title=case["signal"]["title"],
            body_excerpt=case["signal"]["body_excerpt"],
            content_hash=f"gold-score-{case['name']}",
        )
        scorecard = score_opportunity(
            atom,
            signal,
            case["cluster_summary"],
            case["validation_evidence"],
            case["market_gap"],
            review_feedback=case.get("review_feedback"),
        )
        failed = []
        for field, minimum in case.get("min_fields", {}).items():
            if scorecard.get(field, 0.0) < minimum:
                failed.append({"field": field, "minimum": minimum, "actual": scorecard.get(field)})
        for field, maximum in case.get("max_fields", {}).items():
            if scorecard.get(field, 0.0) > maximum:
                failed.append({"field": field, "maximum": maximum, "actual": scorecard.get(field)})
        if failed:
            failures.append(
                {
                    "suite": "scoring_calibration",
                    "name": case["name"],
                    "failures": failed,
                    "scorecard": scorecard,
                }
            )

    total = sum(counts.values())
    passed = total - len(failures)
    return {
        "fixtures_path": str(Path(path).resolve()),
        "counts": counts,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": len(failures),
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "failures": failures,
    }
