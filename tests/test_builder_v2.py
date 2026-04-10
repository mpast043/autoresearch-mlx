"""Tests for BuilderV2 runtime behavior."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.builder_v2 import BuildResult, BuilderV2Agent
from src.database import BuildBrief, BuildPrepOutput, Database, Finding, Opportunity, OpportunityCluster, Validation
from src.messaging import MessageBus, MessageType, create_message


def test_builder_v2_stop_cancels_waiting_loop():
    async def scenario():
        agent = BuilderV2Agent({"llm": {"provider": "ollama"}}, db=None)
        agent._message_queue = MessageBus()
        await agent.start()
        await asyncio.sleep(0.05)
        await asyncio.wait_for(agent.stop(), timeout=0.2)

    asyncio.run(scenario())


def test_builder_v2_build_request_from_brief_persists_product_and_advances_status(monkeypatch, tmp_path):
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)
    db.init_schema()
    db.set_active_run_id("test-run")
    try:
        finding_id = db.insert_finding(
            Finding(source="reddit", source_url="https://example.com/thread", content_hash="builder-v2-brief")
        )
        validation_id = db.insert_validation(
            Validation(
                finding_id=finding_id,
                passed=True,
                overall_score=0.8,
                run_id="test-run",
                evidence={"decision": "promote"},
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                label="Ops handoff failures",
                cluster_key="ops-handoffs",
                summary={"sample_atoms": []},
            )
        )
        opportunity_id = db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="Fix spreadsheet handoff failures",
                market_gap="handoff gap",
                recommendation="promote",
                status="promoted",
                selection_status="build_ready",
            )
        )
        build_brief_id = db.upsert_build_brief(
            BuildBrief(
                run_id="test-run",
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                cluster_id=cluster_id,
                status="build_ready",
                recommended_output_type="workflow_reliability_console",
                brief_json=json.dumps(
                    {
                        "problem_summary": "Spreadsheet handoff failures",
                        "job_to_be_done": "handoff client work",
                        "launch_artifact_plan": ["console", "readme"],
                    }
                ),
            )
        )
        db.upsert_build_prep_output(
            BuildPrepOutput(
                run_id="test-run",
                build_brief_id=build_brief_id,
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                agent_name="spec_generation",
                prep_stage="spec_generation",
                output_json=json.dumps(
                    {
                        "title": "Spreadsheet Handoff Guard",
                        "output_type": "workflow_reliability_console",
                        "traceability": {"build_brief_id": build_brief_id},
                    }
                ),
            )
        )

        async def fake_build_from_spec(spec: dict, idea_id=None):
            return BuildResult(
                success=True,
                project_path=tmp_path / "brief-build",
                files_written=["README.md", "main.py"],
                confidence=0.9,
                duration_s=0.1,
            )

        async def scenario():
            agent = BuilderV2Agent({"llm": {"provider": "ollama"}}, db=db)
            agent._message_queue = MessageBus()
            monkeypatch.setattr(agent, "build_from_spec", fake_build_from_spec)
            payload = await agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="builder",
                    msg_type=MessageType.BUILD_REQUEST,
                    payload={"build_brief_id": build_brief_id},
                )
            )
            emitted = await agent._message_queue.receive("orchestrator")
            return payload, emitted

        payload, emitted = asyncio.run(scenario())
        product = db.get_product_for_build_brief(build_brief_id)
        brief = db.get_build_brief(build_brief_id)
        opportunity = db.get_opportunity(opportunity_id)

        assert payload["success"] is True
        assert payload["product_id"] == product["id"]
        assert emitted.msg_type == MessageType.RESULT
        assert emitted.payload["product_id"] == product["id"]
        assert product["status"] == "completed"
        assert product["location"] == str(tmp_path / "brief-build")
        assert brief.status == "launched"
        assert opportunity.selection_status == "launched"
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_builder_v2_build_request_from_idea_persists_product(monkeypatch, tmp_path):
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)
    db.init_schema()
    try:
        conn = db._get_connection()
        cur = conn.execute(
            """
            INSERT INTO ideas (title, description, status, audience, spec_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "Ops Console",
                "Fix the recurring ops workflow",
                "proposed",
                "ops teams",
                json.dumps({"title": "Ops Console", "output_type": "workflow_reliability_console"}),
            ),
        )
        conn.commit()
        idea_id = int(cur.lastrowid)

        async def fake_build_from_idea(idea_id_value: int):
            return BuildResult(
                success=True,
                project_path=tmp_path / f"idea-{idea_id_value}",
                files_written=["README.md"],
                confidence=0.82,
                duration_s=0.1,
            )

        async def scenario():
            agent = BuilderV2Agent({"llm": {"provider": "ollama"}}, db=db)
            agent._message_queue = MessageBus()
            monkeypatch.setattr(agent, "build_from_idea", fake_build_from_idea)
            payload = await agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="builder",
                    msg_type=MessageType.BUILD_REQUEST,
                    payload={"idea_id": idea_id},
                )
            )
            emitted = await agent._message_queue.receive("orchestrator")
            return payload, emitted

        payload, emitted = asyncio.run(scenario())
        product = db.get_product_for_idea(idea_id)
        idea = db.get_idea(idea_id)

        assert payload["success"] is True
        assert payload["product_id"] == product["id"]
        assert emitted.payload["product_id"] == product["id"]
        assert product["status"] == "completed"
        assert idea.status == "built"
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_builder_v2_build_request_from_brief_rejects_non_build_ready(monkeypatch, tmp_path):
    db_path = tempfile.mktemp(suffix=".db")
    db = Database(db_path)
    db.init_schema()
    db.set_active_run_id("test-run")
    try:
        finding_id = db.insert_finding(
            Finding(source="reddit", source_url="https://example.com/thread", content_hash="builder-v2-not-ready")
        )
        validation_id = db.insert_validation(
            Validation(
                finding_id=finding_id,
                passed=True,
                overall_score=0.8,
                run_id="test-run",
                evidence={"decision": "promote"},
            )
        )
        cluster_id = db.upsert_cluster(
            OpportunityCluster(
                label="Ops handoff failures",
                cluster_key="ops-handoffs-not-ready",
                summary={"sample_atoms": []},
            )
        )
        opportunity_id = db.upsert_opportunity(
            Opportunity(
                cluster_id=cluster_id,
                title="Fix spreadsheet handoff failures",
                market_gap="handoff gap",
                recommendation="promote",
                status="promoted",
                selection_status="research_more",
            )
        )
        build_brief_id = db.upsert_build_brief(
            BuildBrief(
                run_id="test-run",
                opportunity_id=opportunity_id,
                validation_id=validation_id,
                cluster_id=cluster_id,
                status="research_more",
                recommended_output_type="workflow_reliability_console",
                brief_json=json.dumps({"problem_summary": "Spreadsheet handoff failures"}),
            )
        )

        async def scenario():
            agent = BuilderV2Agent({"llm": {"provider": "ollama"}}, db=db)
            agent._message_queue = MessageBus()
            payload = await agent.process(
                create_message(
                    from_agent="orchestrator",
                    to_agent="builder",
                    msg_type=MessageType.BUILD_REQUEST,
                    payload={"build_brief_id": build_brief_id},
                )
            )
            emitted = await agent._message_queue.receive("orchestrator")
            return payload, emitted

        payload, emitted = asyncio.run(scenario())

        assert payload["success"] is False
        assert payload["status"] == "research_more"
        assert emitted.payload["success"] is False
        assert db.get_product_for_build_brief(build_brief_id) is None
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)
