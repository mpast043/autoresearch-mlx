"""Tests for ProblemSpace dataclass, lifecycle, and DB operations."""

import json
import sqlite3
import unittest

from src.database import Database
from src.problem_space import (
    EXPLORING,
    VALIDATED,
    EXHAUSTED,
    ARCHIVED,
    ProblemSpace,
    ProblemSpaceTerm,
    SOURCE_LLM,
    SOURCE_MANUAL,
)
from src.problem_space_lifecycle import ProblemSpaceLifecycleManager


class TestProblemSpaceDataclass(unittest.TestCase):
    """Test ProblemSpace dataclass construction and properties."""

    def test_basic_construction(self) -> None:
        space = ProblemSpace(
            space_key="financial_reconciliation",
            label="Financial reconciliation",
            description="Pain points around reconciling bank feeds with accounting software",
        )
        assert space.space_key == "financial_reconciliation"
        assert space.status == EXPLORING
        assert space.source == SOURCE_LLM
        assert space.total_findings == 0

    def test_invalid_status_raises(self) -> None:
        with self.assertRaises(ValueError):
            ProblemSpace(space_key="test", label="Test", status="invalid")

    def test_keywords_property(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        space.keywords = ["bank reconciliation", "invoice mismatch"]
        assert space.keywords == ["bank reconciliation", "invoice mismatch"]
        assert json.loads(space.keywords_json) == ["bank reconciliation", "invoice mismatch"]

    def test_subreddits_property(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        space.subreddits = ["accounting", "bookkeeping"]
        assert space.subreddits == ["accounting", "bookkeeping"]

    def test_web_queries_property(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        space.web_queries = ["how to reconcile bank statements"]
        assert space.web_queries == ["how to reconcile bank statements"]

    def test_github_queries_property(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        space.github_queries = ["reconciliation bug"]
        assert space.github_queries == ["reconciliation bug"]

    def test_adjacent_spaces_property(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        space.adjacent_spaces = ["tax_compliance", "payment_fraud"]
        assert space.adjacent_spaces == ["tax_compliance", "payment_fraud"]

    def test_empty_json_defaults(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        assert space.keywords == []
        assert space.subreddits == []
        assert space.web_queries == []
        assert space.github_queries == []
        assert space.adjacent_spaces == []


class TestProblemSpaceTerm(unittest.TestCase):
    """Test ProblemSpaceTerm dataclass."""

    def test_basic_construction(self) -> None:
        term = ProblemSpaceTerm(
            space_key="financial_reconciliation",
            term_type="keyword",
            term_value="bank reconciliation",
        )
        assert term.space_key == "financial_reconciliation"
        assert term.term_type == "keyword"
        assert term.term_value == "bank reconciliation"
        assert term.source == "derived"


class TestProblemSpaceLifecycle(unittest.TestCase):
    """Test ProblemSpace lifecycle state transitions."""

    def test_exploring_to_validated_via_prototype_candidates(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(
            space_key="test",
            label="Test",
            status=EXPLORING,
            total_prototype_candidates=1,
        )
        new_state = manager.compute_next_state(space)
        assert new_state == VALIDATED

    def test_exploring_to_validated_via_validations(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(
            space_key="test",
            label="Test",
            status=EXPLORING,
            total_validations=2,
        )
        new_state = manager.compute_next_state(space)
        assert new_state == VALIDATED

    def test_exploring_stays_when_no_signals(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=EXPLORING)
        new_state = manager.compute_next_state(space, idle_cycles=0)
        assert new_state == EXPLORING

    def test_exploring_to_exhausted(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=EXPLORING)
        new_state = manager.compute_next_state(space, idle_cycles=3)
        assert new_state == EXHAUSTED

    def test_validated_to_exhausted(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=VALIDATED)
        new_state = manager.compute_next_state(space, idle_cycles=3)
        assert new_state == EXHAUSTED

    def test_validated_stays_when_active(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=VALIDATED)
        new_state = manager.compute_next_state(space, idle_cycles=0)
        assert new_state == VALIDATED

    def test_exhausted_to_archived(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=EXHAUSTED)
        new_state = manager.compute_next_state(space, idle_cycles=10)
        assert new_state == ARCHIVED

    def test_archived_stays(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=ARCHIVED)
        new_state = manager.compute_next_state(space)
        assert new_state == ARCHIVED

    def test_transition_valid(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=EXPLORING)
        manager.transition_space(space, VALIDATED)
        assert space.status == VALIDATED

    def test_transition_invalid_raises(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test", status=ARCHIVED)
        with self.assertRaises(ValueError):
            manager.transition_space(space, VALIDATED)

    def test_yield_score_zero_with_no_findings(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(space_key="test", label="Test")
        assert manager.compute_yield_score(space) == 0.0

    def test_yield_score_with_signals(self) -> None:
        manager = ProblemSpaceLifecycleManager(None)
        space = ProblemSpace(
            space_key="test",
            label="Test",
            total_findings=10,
            total_validations=5,
            total_prototype_candidates=2,
            total_build_briefs=1,
        )
        score = manager.compute_yield_score(space)
        assert 0.0 < score <= 1.0


class TestProblemSpaceDB(unittest.TestCase):
    """Test ProblemSpace database operations."""

    def setUp(self) -> None:
        import tempfile
        import os
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_schema()

    def tearDown(self) -> None:
        self.db.close()

    def test_upsert_and_get_problem_space(self) -> None:
        space = ProblemSpace(
            space_key="financial_reconciliation",
            label="Financial reconciliation",
            description="Pain points around reconciling bank feeds",
            keywords_json='["bank reconciliation", "invoice mismatch"]',
            subreddits_json='["accounting", "bookkeeping"]',
        )
        self.db.upsert_problem_space(space)

        fetched = self.db.get_problem_space("financial_reconciliation")
        assert fetched is not None
        assert fetched.space_key == "financial_reconciliation"
        assert fetched.label == "Financial reconciliation"
        assert fetched.keywords == ["bank reconciliation", "invoice mismatch"]
        assert fetched.subreddits == ["accounting", "bookkeeping"]
        assert fetched.status == EXPLORING

    def test_upsert_updates_existing(self) -> None:
        space = ProblemSpace(
            space_key="test_space",
            label="Test",
            description="Original",
        )
        self.db.upsert_problem_space(space)

        updated = ProblemSpace(
            space_key="test_space",
            label="Test Updated",
            description="Updated description",
            total_validations=5,
        )
        self.db.upsert_problem_space(updated)

        fetched = self.db.get_problem_space("test_space")
        assert fetched is not None
        assert fetched.label == "Test Updated"
        assert fetched.description == "Updated description"
        assert fetched.total_validations == 5

    def test_list_problem_spaces_by_status(self) -> None:
        for key, status in [("a", EXPLORING), ("b", EXPLORING), ("c", VALIDATED)]:
            space = ProblemSpace(space_key=key, label=key, status=status)
            self.db.upsert_problem_space(space)

        exploring = self.db.list_problem_spaces(status=EXPLORING)
        assert len(exploring) == 2

        validated = self.db.list_problem_spaces(status=VALIDATED)
        assert len(validated) == 1

        all_spaces = self.db.list_problem_spaces()
        assert len(all_spaces) == 3

    def test_update_problem_space_status(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        self.db.upsert_problem_space(space)

        self.db.update_problem_space_status("test", VALIDATED)

        fetched = self.db.get_problem_space("test")
        assert fetched is not None
        assert fetched.status == VALIDATED

    def test_update_problem_space_metrics(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        self.db.upsert_problem_space(space)

        self.db.update_problem_space_metrics(
            space_key="test",
            total_findings=10,
            total_validations=5,
            total_prototype_candidates=2,
            total_build_briefs=1,
            yield_score=0.75,
        )

        fetched = self.db.get_problem_space("test")
        assert fetched is not None
        assert fetched.total_findings == 10
        assert fetched.total_validations == 5
        assert fetched.total_prototype_candidates == 2
        assert fetched.total_build_briefs == 1
        assert fetched.yield_score == 0.75

    def test_add_and_get_problem_space_terms(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        self.db.upsert_problem_space(space)

        self.db.add_problem_space_term("test", "keyword", "bank reconciliation")
        self.db.add_problem_space_term("test", "keyword", "invoice mismatch")
        self.db.add_problem_space_term("test", "subreddit", "accounting")

        all_terms = self.db.get_problem_space_terms("test")
        assert len(all_terms) == 3

        kw_terms = self.db.get_problem_space_terms("test", term_type="keyword")
        assert len(kw_terms) == 2
        assert all(t.term_type == "keyword" for t in kw_terms)

        sub_terms = self.db.get_problem_space_terms("test", term_type="subreddit")
        assert len(sub_terms) == 1

    def test_add_duplicate_term_ignored(self) -> None:
        space = ProblemSpace(space_key="test", label="Test")
        self.db.upsert_problem_space(space)

        self.db.add_problem_space_term("test", "keyword", "same_term")
        self.db.add_problem_space_term("test", "keyword", "same_term")  # duplicate

        terms = self.db.get_problem_space_terms("test")
        assert len(terms) == 1

    def test_get_nonexistent_space(self) -> None:
        result = self.db.get_problem_space("does_not_exist")
        assert result is None


if __name__ == "__main__":
    unittest.main()