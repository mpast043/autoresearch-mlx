"""Tests for orchestrator routing behavior."""

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from src.database import BuildBrief, Database, Finding, Opportunity, OpportunityCluster, Validation
from src.messaging import MessageType, create_message
from src.orchestrator import Orchestrator


@pytest.fixture
def temp_db():
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    db.init_schema()
    try:
        yield db
    finally:
        db.close()
        if os.path.exists(path):
            os.remove(path)


def test_orchestrator_routes_finding_to_evidence(temp_db):
    orchestrator = Orchestrator(temp_db)
    message = create_message(
        from_agent="discovery",
        to_agent="orchestrator",
        msg_type=MessageType.FINDING,
        payload={"finding_id": 123, "source": "reddit-problem"},
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("evidence"))
    assert queued is not None
    assert queued.msg_type == MessageType.FINDING
    assert queued.payload["finding_id"] == 123


def test_orchestrator_routes_evidence_to_validation(temp_db):
    orchestrator = Orchestrator(temp_db)
    message = create_message(
        from_agent="evidence",
        to_agent="orchestrator",
        msg_type=MessageType.EVIDENCE,
        payload={"finding_id": 456},
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("validation"))
    assert queued is not None
    assert queued.msg_type == MessageType.EVIDENCE
    assert queued.payload["finding_id"] == 456


def test_orchestrator_routes_prototype_candidate_to_solution_framing(temp_db):
    orchestrator = Orchestrator(temp_db)
    message = create_message(
        from_agent="validation",
        to_agent="orchestrator",
        msg_type=MessageType.VALIDATION,
        payload={
            "finding_id": 1,
            "decision": "park",
            "selection_status": "prototype_candidate",
            "build_brief_id": 7,
            "opportunity_id": 9,
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("solution_framing"))
    assert queued is not None
    assert queued.msg_type == MessageType.BUILD_BRIEF
    assert queued.payload["build_brief_id"] == 7


def test_orchestrator_routes_spec_draft_to_solution_framing(temp_db):
    orchestrator = Orchestrator(temp_db)
    message = create_message(
        from_agent="validation",
        to_agent="orchestrator",
        msg_type=MessageType.VALIDATION,
        payload={
            "finding_id": 1,
            "decision": "promote",
            "selection_status": "research_more",
            "build_brief_id": 8,
            "build_brief_purpose": "product_spec_draft",
            "opportunity_id": 10,
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("solution_framing"))
    assert queued is not None
    assert queued.msg_type == MessageType.BUILD_BRIEF
    assert queued.payload["build_brief_id"] == 8
    assert queued.payload["build_brief_purpose"] == "product_spec_draft"


def test_stop_on_hit_matches_defaults():
    cfg = {"enabled": True}
    assert Orchestrator.stop_on_hit_matches(
        cfg,
        {"selection_status": "prototype_candidate", "decision": "park"},
    )
    assert Orchestrator.stop_on_hit_matches(
        cfg,
        {"selection_status": "research_more", "decision": "promote"},
    )
    assert not Orchestrator.stop_on_hit_matches(
        cfg,
        {"selection_status": "research_more", "decision": "park"},
    )
    assert not Orchestrator.stop_on_hit_matches(
        {"enabled": False},
        {"selection_status": "prototype_candidate", "decision": "promote"},
    )


def test_stop_on_hit_sets_shutdown_event(temp_db):
    ev = asyncio.Event()
    orch = Orchestrator(
        temp_db,
        shutdown_event=ev,
        stop_on_hit_config={
            "enabled": True,
            "selection_status_any": ["prototype_candidate"],
            "decision_any": [],
        },
    )
    msg = create_message(
        from_agent="validation",
        to_agent="orchestrator",
        msg_type=MessageType.VALIDATION,
        payload={
            "finding_id": 1,
            "decision": "park",
            "selection_status": "prototype_candidate",
            "build_brief_id": 0,
            "opportunity_id": 9,
        },
    )
    asyncio.run(orch._handle_orchestrator_message(msg))
    assert ev.is_set()


def test_stop_on_hit_exit_on_hit_false_does_not_shutdown(temp_db):
    ev = asyncio.Event()
    orch = Orchestrator(
        temp_db,
        shutdown_event=ev,
        stop_on_hit_config={
            "enabled": True,
            "exit_on_hit": False,
            "selection_status_any": ["prototype_candidate"],
            "decision_any": [],
        },
    )
    msg = create_message(
        from_agent="validation",
        to_agent="orchestrator",
        msg_type=MessageType.VALIDATION,
        payload={
            "finding_id": 1,
            "decision": "park",
            "selection_status": "prototype_candidate",
            "build_brief_id": 0,
            "opportunity_id": 9,
        },
    )
    asyncio.run(orch._handle_orchestrator_message(msg))
    assert not ev.is_set()


def test_consume_until_quiet_drains_orchestrator_queue(temp_db):
    orchestrator = Orchestrator(temp_db)

    async def scenario():
        await orchestrator._message_queue.put(
            create_message(
                from_agent="discovery",
                to_agent="orchestrator",
                msg_type=MessageType.FINDING,
                payload={"finding_id": 321},
            )
        )
        processed = await orchestrator.consume_until_quiet(timeout=0.05)
        queued = await orchestrator._message_queue.get_for_agent("evidence")
        return processed, queued

    processed, queued = asyncio.run(scenario())
    assert processed == 1
    assert queued is not None
    assert queued.payload["finding_id"] == 321


def test_validation_decision_breakdown_prefers_canonical_policy(temp_db):
    orchestrator = Orchestrator(temp_db)
    breakdown = orchestrator._validation_decision_breakdown(
        [
            {
                "decision": "",
                "evidence": {
                    "opportunity_evaluation": {
                        "policy": {"decision": "promote"},
                    }
                },
            },
            {
                "decision": "",
                "evidence": {
                    "opportunity_evaluation": {
                        "policy": {"decision": "park"},
                    }
                },
            },
        ]
    )

    assert breakdown == {"promote": 1, "park": 1}


def test_orchestrator_routes_spec_generation_completion_to_builder_when_auto_build_enabled(temp_db):
    orchestrator = Orchestrator(temp_db, auto_build=True)
    finding_id = temp_db.insert_finding(
        Finding(source="reddit", source_url="https://example.com/brief-ready", content_hash="orch-build-ready")
    )
    validation_id = temp_db.insert_validation(
        Validation(
            finding_id=finding_id,
            passed=True,
            overall_score=0.8,
            run_id="test-run",
            evidence={"decision": "promote"},
        )
    )
    cluster_id = temp_db.upsert_cluster(
        OpportunityCluster(label="Ready brief cluster", cluster_key="ready-brief-cluster", summary={})
    )
    opportunity_id = temp_db.upsert_opportunity(
        Opportunity(
            cluster_id=cluster_id,
            title="Ready brief opportunity",
            market_gap="gap",
            recommendation="promote",
            status="promoted",
            selection_status="build_ready",
        )
    )
    build_brief_id = temp_db.upsert_build_brief(
        BuildBrief(
            run_id="test-run",
            opportunity_id=opportunity_id,
            validation_id=validation_id,
            cluster_id=cluster_id,
            status="build_ready",
            brief_json=json.dumps({"title": "Ready brief"}),
        )
    )
    message = create_message(
        from_agent="spec_generation",
        to_agent="orchestrator",
        msg_type=MessageType.BUILD_PREP,
        payload={
            "build_brief_id": build_brief_id,
            "opportunity_id": opportunity_id,
            "validation_id": validation_id,
            "prep_stage": "spec_generation",
            "next_agent": "",
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("builder"))
    assert queued is not None
    assert queued.msg_type == MessageType.BUILD_REQUEST
    assert queued.payload["build_brief_id"] == build_brief_id


def test_orchestrator_does_not_route_spec_generation_to_builder_when_brief_not_build_ready(temp_db):
    orchestrator = Orchestrator(temp_db, auto_build=True)
    finding_id = temp_db.insert_finding(
        Finding(source="reddit", source_url="https://example.com/brief-not-ready", content_hash="orch-not-build-ready")
    )
    validation_id = temp_db.insert_validation(
        Validation(
            finding_id=finding_id,
            passed=True,
            overall_score=0.8,
            run_id="test-run",
            evidence={"decision": "promote"},
        )
    )
    cluster_id = temp_db.upsert_cluster(
        OpportunityCluster(label="Not ready brief cluster", cluster_key="not-ready-brief-cluster", summary={})
    )
    opportunity_id = temp_db.upsert_opportunity(
        Opportunity(
            cluster_id=cluster_id,
            title="Not ready brief opportunity",
            market_gap="gap",
            recommendation="promote",
            status="promoted",
            selection_status="research_more",
        )
    )
    build_brief_id = temp_db.upsert_build_brief(
        BuildBrief(
            run_id="test-run",
            opportunity_id=opportunity_id,
            validation_id=validation_id,
            cluster_id=cluster_id,
            status="research_more",
            brief_json=json.dumps({"title": "Not ready brief"}),
        )
    )
    message = create_message(
        from_agent="spec_generation",
        to_agent="orchestrator",
        msg_type=MessageType.BUILD_PREP,
        payload={
            "build_brief_id": build_brief_id,
            "opportunity_id": opportunity_id,
            "validation_id": validation_id,
            "prep_stage": "spec_generation",
            "next_agent": "",
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("builder"))
    assert queued is None


def test_orchestrator_routes_spec_completion_to_registered_security_agent(temp_db):
    orchestrator = Orchestrator(temp_db)
    orchestrator.register_agent(SimpleNamespace(name="SecurityAgent"))
    spec = {"product_spec": {"name": "Narrow plugin"}}
    message = create_message(
        from_agent="spec_generation",
        to_agent="orchestrator",
        msg_type=MessageType.BUILD_PREP,
        payload={
            "build_brief_id": 42,
            "opportunity_id": 7,
            "prep_stage": "spec_generation",
            "spec_content": spec,
            "next_agent": "",
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("SecurityAgent"))
    assert queued is not None
    assert queued.msg_type == MessageType.SECURITY_SCAN
    assert queued.payload["spec"] == spec
    assert queued.payload["spec_content"] == spec


def test_orchestrator_routes_security_result_to_registered_technical_writer(temp_db):
    orchestrator = Orchestrator(temp_db)
    orchestrator.register_agent(SimpleNamespace(name="TechnicalWriterAgent"))
    spec = {"product_spec": {"name": "Narrow plugin"}}
    message = create_message(
        from_agent="SecurityAgent",
        to_agent="orchestrator",
        msg_type=MessageType.SECURITY_SCAN,
        payload={
            "build_brief_id": 42,
            "opportunity_id": 7,
            "spec": spec,
            "security_safe": True,
        },
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("TechnicalWriterAgent"))
    assert queued is not None
    assert queued.msg_type == MessageType.DOC_GENERATION
    assert queued.payload["spec"] == spec
    assert queued.payload["spec_content"] == spec
    assert queued.payload["security_safe"] is True


def test_orchestrator_routes_health_check_to_registered_sre_agent(temp_db):
    orchestrator = Orchestrator(temp_db)
    orchestrator.register_agent(SimpleNamespace(name="SREAgent"))
    message = create_message(
        from_agent="monitor",
        to_agent="orchestrator",
        msg_type=MessageType.HEALTH_CHECK,
        payload={"probe": "startup"},
    )

    asyncio.run(orchestrator._handle_orchestrator_message(message))

    queued = asyncio.run(orchestrator._message_queue.get_for_agent("SREAgent"))
    assert queued is not None
    assert queued.msg_type == MessageType.HEALTH_CHECK
    assert queued.payload["probe"] == "startup"
