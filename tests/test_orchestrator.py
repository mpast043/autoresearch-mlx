"""Tests for orchestrator routing behavior."""

import asyncio
import os
import sys
import tempfile

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.database import Database
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
