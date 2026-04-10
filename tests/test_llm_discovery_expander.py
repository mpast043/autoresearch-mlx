"""Tests for LLMClient and LLMDiscoveryExpander."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.database import Database
from src.problem_space import ProblemSpace, EXPLORING, VALIDATED
from src.llm_discovery_expander import (
    LLMClient,
    LLMDiscoveryExpander,
    _extract_json,
    SPACE_PROPOSAL_SYSTEM,
    SPACE_PROPOSAL_USER,
    QUERY_DERIVATION_SYSTEM,
    QUERY_DERIVATION_USER,
)


class TestLLMClient(unittest.TestCase):
    """Test LLMClient provider dispatch and error handling."""

    def test_ollama_config_defaults(self) -> None:
        config = {"llm": {}}
        client = LLMClient(config)
        assert client.provider == "ollama"
        assert client.model == "gemma4:latest"
        assert client.base_url == "http://localhost:11434"

    def test_anthropic_config(self) -> None:
        config = {"llm": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "api_key": "test-key"}}
        client = LLMClient(config)
        assert client.provider == "anthropic"
        assert client.model == "claude-sonnet-4-20250514"
        assert client.api_key == "test-key"

    def test_generate_returns_none_on_ollama_failure(self) -> None:
        config = {"llm": {"provider": "ollama"}}
        client = LLMClient(config)
        with patch.object(client, "_ollama_generate", return_value=None):
            result = client.generate("system", "user")
        assert result is None

    def test_generate_returns_none_on_anthropic_no_key(self) -> None:
        config = {"llm": {"provider": "anthropic"}}
        client = LLMClient(config)
        # No ANTHROPIC_API_KEY or CLAUDE_API_KEY in env
        with patch.dict("os.environ", {}, clear=True):
            result = client.generate("system", "user")
        assert result is None

    def test_generate_ollama_success(self) -> None:
        config = {"llm": {"provider": "ollama"}}
        client = LLMClient(config)
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"message": {"content": "test output"}}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = client.generate("system", "user")
        assert result == "test output"

    def test_auto_fallback_to_anthropic(self) -> None:
        config = {"llm": {"provider": "auto", "api_key": "test-key"}}
        client = LLMClient(config)
        # Ollama fails, Anthropic succeeds
        with patch.object(client, "_ollama_generate", return_value=None):
            with patch.object(client, "_anthropic_generate", return_value="anthropic output"):
                result = client.generate("system", "user")
        assert result == "anthropic output"

    def test_agenerate_anthropic_uses_to_thread(self) -> None:
        config = {"llm": {"provider": "anthropic", "api_key": "test-key"}}
        client = LLMClient(config)
        with patch.object(client, "_anthropic_generate", return_value="anthropic output") as mock_generate:
            result = asyncio.run(client.agenerate("system", "user"))
        assert result == "anthropic output"
        mock_generate.assert_called_once_with("system", "user")


class TestExtractJson(unittest.TestCase):
    """Test JSON extraction from LLM responses."""

    def test_direct_json(self) -> None:
        raw = '{"proposed_spaces": [{"space_key": "test", "label": "Test"}]}'
        result = _extract_json(raw)
        assert result is not None
        assert "proposed_spaces" in result

    def test_markdown_json_block(self) -> None:
        raw = '```json\n{"proposed_spaces": []}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert result["proposed_spaces"] == []

    def test_markdown_code_block(self) -> None:
        raw = '```\n{"proposed_spaces": []}\n```'
        result = _extract_json(raw)
        assert result is not None

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here is the response:\n{"proposed_spaces": [{"space_key": "reconciliation"}]}\nThat is all.'
        result = _extract_json(raw)
        assert result is not None
        # The greedy regex should match the outer object
        assert "proposed_spaces" in result

    def test_empty_input(self) -> None:
        assert _extract_json("") is None
        assert _extract_json(None) is None

    def test_garbage_input(self) -> None:
        assert _extract_json("this is not json at all") is None

    def test_nested_json(self) -> None:
        raw = '{"proposed_spaces": [{"space_key": "test", "keywords": ["a", "b"]}]}'
        result = _extract_json(raw)
        assert result is not None
        assert result["proposed_spaces"][0]["keywords"] == ["a", "b"]


class TestLLMDiscoveryExpander(unittest.TestCase):
    """Test LLMDiscoveryExpander logic."""

    def setUp(self) -> None:
        import tempfile
        import os
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_schema()
        self.config = {
            "llm": {"provider": "ollama", "model": "test-model"},
            "discovery": {
                "llm_expansion": {
                    "enabled": True,
                    "max_proposed_spaces": 4,
                    "max_keywords_per_space": 10,
                    "max_subreddits_per_space": 5,
                    "max_web_queries_per_space": 5,
                    "max_github_queries_per_space": 3,
                },
            },
        }
        self.expander = LLMDiscoveryExpander(self.db, self.config)

    def tearDown(self) -> None:
        self.db.close()

    def test_gather_context_with_opportunities(self) -> None:
        # Insert a test opportunity
        from src.database import Finding, Opportunity, OpportunityCluster
        cluster = OpportunityCluster(label="Test Cluster", cluster_key="test_1")
        self.db.upsert_cluster(cluster)
        opp = Opportunity(
            cluster_id=1,
            title="Test Opportunity",
            market_gap="gap",
            recommendation="promote",
            status="active",
            composite_score=0.75,
            selection_status="prototype_candidate",
        )
        self.db.upsert_opportunity(opp)

        context = self.expander.gather_context()
        assert len(context["opportunities"]) >= 1
        assert context["opportunities"][0]["title"] == "Test Opportunity"

    def test_build_proposal_prompt(self) -> None:
        context = {
            "opportunities": [
                {"id": 1, "title": "Bank reconciliation pain", "composite_score": 0.8,
                 "selection_status": "prototype_candidate", "cluster_id": 1},
            ],
            "active_spaces": [
                {"space_key": "financial_recon", "label": "Financial Reconciliation",
                 "status": "exploring", "yield_score": 0.5, "findings": 10, "validations": 3},
            ],
            "exhausted_space_keys": ["old_space"],
            "search_coverage": {
                "keywords_sample": ["bank reconciliation", "invoice mismatch"],
                "subreddits_sample": ["accounting"],
            },
        }
        system, user = self.expander.build_proposal_prompt(context)
        assert "Bank reconciliation pain" in user
        assert "financial_recon" in user
        assert "old_space" in user

    def test_parse_proposals_valid(self) -> None:
        raw = json.dumps({
            "proposed_spaces": [
                {
                    "space_key": "tax_compliance",
                    "label": "Tax Compliance Automation",
                    "description": "Pain around tax filing and compliance",
                    "semantic_summary": "Small business owners struggle with tax compliance",
                    "keywords": ["tax filing error", "sales tax nexus"],
                    "subreddits": ["smallbusiness", "tax"],
                    "web_queries": ["sales tax compliance automation"],
                    "github_queries": ["tax calculation bug"],
                    "adjacent_spaces": ["financial_reconciliation"],
                    "rationale": "Adjacent to financial reconciliation",
                },
            ],
        })

        spaces = self.expander.parse_proposals(raw)
        assert len(spaces) == 1
        assert spaces[0].space_key == "tax_compliance"
        assert spaces[0].label == "Tax Compliance Automation"
        assert "tax filing error" in spaces[0].keywords

    def test_parse_proposals_normalizes_string_subreddit_payloads(self) -> None:
        raw = json.dumps(
            {
                "proposed_spaces": [
                    {
                        "space_key": "spreadsheet_handoff_drift",
                        "label": "Spreadsheet handoff drift",
                        "description": "Workflow drift across spreadsheet handoffs",
                        "semantic_summary": "Operators lose track of the latest sheet",
                        "keywords": "latest spreadsheet version confusion, manual handoff workflow",
                        "subreddits": "r/automation, notion, n",
                        "web_queries": "latest spreadsheet version confusion",
                        "github_queries": [],
                        "adjacent_spaces": [],
                    }
                ]
            }
        )

        spaces = self.expander.parse_proposals(raw)

        assert spaces[0].keywords == ["latest spreadsheet version confusion", "manual handoff workflow"]
        assert spaces[0].subreddits == ["automation", "notion"]

    def test_parse_proposals_with_markdown(self) -> None:
        raw = '```json\n{"proposed_spaces": [{"space_key": "test", "label": "Test", "description": "A test", "keywords": ["kw1"], "subreddits": ["sub1"]}]}\n```'
        spaces = self.expander.parse_proposals(raw)
        assert len(spaces) == 1

    def test_parse_proposals_empty(self) -> None:
        assert self.expander.parse_proposals("") == []
        assert self.expander.parse_proposals("not json") == []

    def test_parse_proposals_normalizes_space_key(self) -> None:
        raw = json.dumps({
            "proposed_spaces": [{
                "space_key": "Tax Compliance!!!",
                "label": "Tax Compliance",
                "keywords": ["kw1"],
                "subreddits": ["sub1"],
            }],
        })
        spaces = self.expander.parse_proposals(raw)
        assert spaces[0].space_key == "tax_compliance"  # special chars stripped

    def test_register_space_and_terms(self) -> None:
        space = ProblemSpace(
            space_key="test_space",
            label="Test Space",
            description="A test space",
            keywords_json='["kw1", "kw2"]',
            subreddits_json='["sub1"]',
            web_queries_json='["query1"]',
            github_queries_json='["gh1"]',
        )

        result = self.expander.register_space_and_terms(space)

        # Verify space was persisted
        fetched = self.db.get_problem_space("test_space")
        assert fetched is not None
        assert fetched.label == "Test Space"

        # Verify terms were persisted
        terms = self.db.get_problem_space_terms("test_space")
        assert len(terms) == 5  # 2 keywords + 1 subreddit + 1 web_query + 1 github_query

    def test_register_space_skips_duplicate_hash(self) -> None:
        space = ProblemSpace(
            space_key="test_space",
            label="Test Space",
            generation_prompt_hash="abc123",
        )
        self.expander.register_space_and_terms(space)

        # Same hash should return existing
        space2 = ProblemSpace(
            space_key="test_space",
            label="Test Space Updated",
            generation_prompt_hash="abc123",
        )
        result = self.expander.register_space_and_terms(space2)
        # Should return the existing one, not update
        assert result.label == "Test Space"

    def test_fallback_derive_queries(self) -> None:
        space = ProblemSpace(
            space_key="test",
            label="Bank Reconciliation",
            description="Pain around reconciling bank statements with accounting software",
        )
        result = self.expander._fallback_derive_queries(space)
        assert len(result.keywords) > 0
        assert len(result.web_queries) > 0

    def test_compute_prompt_hash(self) -> None:
        context = {"opportunities": [{"id": 1}], "active_spaces": [{"space_key": "test"}]}
        hash1 = self.expander._compute_prompt_hash(context)
        hash2 = self.expander._compute_prompt_hash(context)
        assert hash1 == hash2  # same input = same hash

        context2 = {"opportunities": [{"id": 2}], "active_spaces": [{"space_key": "test"}]}
        hash3 = self.expander._compute_prompt_hash(context2)
        assert hash1 != hash3  # different input = different hash


# --- Standalone async tests (pytest-asyncio) ---


@pytest.fixture
def expander_with_db():
    """Create an LLMDiscoveryExpander with a temp database."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    db = Database(db_path)
    db.init_schema()
    config = {
        "llm": {"provider": "ollama", "model": "test-model"},
        "discovery": {
            "llm_expansion": {
                "enabled": True,
                "max_proposed_spaces": 4,
                "max_keywords_per_space": 10,
                "max_subreddits_per_space": 5,
                "max_web_queries_per_space": 5,
                "max_github_queries_per_space": 3,
            },
        },
    }
    expander = LLMDiscoveryExpander(db, config)
    yield db, expander
    db.close()


@pytest.mark.asyncio
async def test_expand_after_validation_with_llm(expander_with_db):
    """Test the full expansion flow with a mocked LLM response."""
    db, expander = expander_with_db
    with patch.object(LLMClient, "agenerate", new_callable=AsyncMock) as mock_agenerate:
        mock_agenerate.return_value = json.dumps({
            "proposed_spaces": [{
                "space_key": "inventory_sync",
                "label": "Inventory Sync Errors",
                "description": "Pain around multi-channel inventory sync",
                "semantic_summary": "E-commerce sellers struggle with inventory sync",
                "keywords": ["inventory sync error", "stock mismatch"],
                "subreddits": ["ecommerce", "shopify"],
                "web_queries": ["inventory sync automation"],
                "github_queries": ["inventory sync bug"],
                "adjacent_spaces": [],
                "rationale": "Adjacent to validated opportunities",
            }],
        })

        from src.database import Opportunity, OpportunityCluster
        cluster = OpportunityCluster(label="Test Cluster", cluster_key="test_1")
        db.upsert_cluster(cluster)
        opp = Opportunity(cluster_id=1, title="Test", composite_score=0.8, selection_status="prototype_candidate", market_gap="gap", recommendation="promote", status="active")
        db.upsert_opportunity(opp)

        new_spaces = await expander.expand_after_validation()
        assert len(new_spaces) >= 1
        assert new_spaces[0].space_key == "inventory_sync"


@pytest.mark.asyncio
async def test_expand_after_validation_skips_existing_space_with_same_prompt_hash(expander_with_db):
    db, expander = expander_with_db
    with patch.object(LLMClient, "agenerate", new_callable=AsyncMock) as mock_agenerate:
        mock_agenerate.return_value = json.dumps(
            {
                "proposed_spaces": [
                    {
                        "space_key": "inventory_sync",
                        "label": "Inventory Sync Errors",
                        "description": "Pain around multi-channel inventory sync",
                        "semantic_summary": "E-commerce sellers struggle with inventory sync",
                        "keywords": ["inventory sync error"],
                        "subreddits": ["ecommerce"],
                        "web_queries": ["inventory sync automation"],
                        "github_queries": ["inventory sync bug"],
                        "adjacent_spaces": [],
                    }
                ]
            }
        )

        from src.database import Opportunity, OpportunityCluster
        cluster = OpportunityCluster(label="Test Cluster", cluster_key="test_1")
        db.upsert_cluster(cluster)
        opp = Opportunity(cluster_id=1, title="Test", composite_score=0.8, selection_status="prototype_candidate", market_gap="gap", recommendation="promote", status="active")
        db.upsert_opportunity(opp)

        db.upsert_problem_space(
            ProblemSpace(
                space_key="inventory_sync",
                label="Inventory Sync Errors",
                generation_prompt_hash="placeholder",
                keywords_json=json.dumps(["inventory sync error"]),
                subreddits_json=json.dumps(["ecommerce"]),
            )
        )
        prompt_hash = expander._compute_prompt_hash(expander.gather_context())
        db.upsert_problem_space(
            ProblemSpace(
                space_key="inventory_sync",
                label="Inventory Sync Errors",
                generation_prompt_hash=prompt_hash,
                keywords_json=json.dumps(["inventory sync error"]),
                subreddits_json=json.dumps(["ecommerce"]),
            )
        )

        new_spaces = await expander.expand_after_validation()

        assert new_spaces == []


@pytest.mark.asyncio
async def test_expand_after_validation_no_opportunities(expander_with_db):
    """Test that expansion is skipped when there are no opportunities."""
    db, expander = expander_with_db
    with patch.object(LLMClient, "agenerate", new_callable=AsyncMock) as mock_agenerate:
        new_spaces = await expander.expand_after_validation()
        assert new_spaces == []
        mock_agenerate.assert_not_called()


@pytest.mark.asyncio
async def test_expand_after_validation_llm_failure(expander_with_db):
    """Test that expansion returns empty list when LLM fails."""
    db, expander = expander_with_db
    with patch.object(LLMClient, "agenerate", new_callable=AsyncMock) as mock_agenerate:
        mock_agenerate.return_value = None  # LLM unavailable

        from src.database import Opportunity, OpportunityCluster
        cluster = OpportunityCluster(label="Test Cluster", cluster_key="test_1")
        db.upsert_cluster(cluster)
        opp = Opportunity(cluster_id=1, title="Test", composite_score=0.8, market_gap="gap", recommendation="promote", status="active")
        db.upsert_opportunity(opp)

        new_spaces = await expander.expand_after_validation()
        assert new_spaces == []


if __name__ == "__main__":
    unittest.main()
