"""Tests for the signal classification module."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research.classification import (
    AI_TOOL_KEYWORDS,
    PAIN_KEYWORDS,
    VALUE_KEYWORDS,
    RECURRENCE_KEYWORDS,
    classify_signal,
    contains_ai_keyword,
    contains_pain_keyword,
    contains_value_keyword,
    contains_recurrence_keyword,
)


class TestContainsPainKeyword:
    """Tests for the contains_pain_keyword function."""

    def test_known_pain_keyword(self):
        assert contains_pain_keyword("this is frustrating") is True

    def test_multiple_pain_keywords(self):
        assert contains_pain_keyword("manual and error-prone work") is True

    def test_no_pain_keyword(self):
        assert contains_pain_keyword("everything is great") is False

    def test_case_insensitive(self):
        assert contains_pain_keyword("FRUSTRATING") is True
        assert contains_pain_keyword("Annoying") is True

    def test_empty_string(self):
        assert contains_pain_keyword("") is False

    def test_partial_match_within_word(self):
        # "overkill" is itself in PAIN_KEYWORDS
        assert contains_pain_keyword("this solution is overkill") is True

    def test_pain_keyword_with_surrounding_text(self):
        assert contains_pain_keyword("I wish there was a tool for this") is True


class TestContainsAiKeyword:
    """Tests for the contains_ai_keyword function."""

    def test_known_ai_keyword(self):
        assert contains_ai_keyword("using chatgpt for work") is True

    def test_multiple_ai_keywords(self):
        assert contains_ai_keyword("openai and anthropic models") is True

    def test_no_ai_keyword(self):
        assert contains_ai_keyword("using traditional databases") is False

    def test_case_insensitive(self):
        assert contains_ai_keyword("CHATGPT") is True
        assert contains_ai_keyword("Llama3 is an LLM") is True

    def test_empty_string(self):
        assert contains_ai_keyword("") is False

    def test_ai_short_keyword(self):
        # "ai" is a very short keyword, substring match
        assert contains_ai_keyword("AI is changing things") is True


class TestContainsValueKeyword:
    """Tests for the contains_value_keyword function."""

    def test_known_value_keyword(self):
        # "daily" is a VALUE_KEYWORD (exact substring match)
        assert contains_value_keyword("used every day and daily") is True

    def test_multiple_value_keywords(self):
        assert contains_value_keyword("revenue and customers") is True

    def test_no_value_keyword(self):
        assert contains_value_keyword("random text without keywords") is False

    def test_case_insensitive(self):
        assert contains_value_keyword("PRODUCTIVITY") is True

    def test_exact_phrase_match(self):
        # "save time" is a VALUE_KEYWORD; it must appear as exact substring
        assert contains_value_keyword("how to save time on tasks") is True
        # "saves time" does NOT contain "save time" as exact substring
        assert contains_value_keyword("this saves time") is False


class TestContainsRecurrenceKeyword:
    """Tests for the contains_recurrence_keyword function."""

    def test_known_recurrence_keyword(self):
        assert contains_recurrence_keyword("keep having this issue") is True

    def test_no_recurrence_keyword(self):
        assert contains_recurrence_keyword("happened once") is False

    def test_case_insensitive(self):
        assert contains_recurrence_keyword("ALWAYS breaks") is True

    def test_recurrence_with_spaces(self):
        # " repeatedly " has spaces around it in the keyword list
        assert contains_recurrence_keyword("it repeatedly fails") is True


class TestClassifySignal:
    """Tests for the classify_signal function.

    Classification priority: pain (exclusive) -> value (exclusive) -> ai -> competition -> low_signal
    - pain_signal: has_pain=True AND has_value=False
    - success_signal: has_value=True AND has_pain=False
    - demand_signal: has_ai=True (and neither pure pain nor pure value)
    - competition_signal: has "competitor" or "alternative" (and no ai)
    - low_signal_summary: fallback
    """

    def test_pain_only_classified_as_pain_signal(self):
        # "frustrating" and "annoying" are PAIN_KEYWORDS, no VALUE_KEYWORDS
        result = classify_signal("this is frustrating and annoying")
        assert result == "pain_signal"

    def test_value_only_classified_as_success_signal(self):
        # "productivity" is a VALUE_KEYWORD; no PAIN_KEYWORDS present
        result = classify_signal("save time and improve productivity")
        assert result == "success_signal"

    def test_pain_and_value_together_not_pure_signal(self):
        # When both pain and value keywords are present, neither exclusive condition holds.
        # "frustrating" (PAIN) + "workflow" (VALUE) => has_pain=True, has_value=True
        # Falls through to ai/competition/low_signal checks.
        result = classify_signal("frustrating workflow")
        assert result == "low_signal_summary"

    def test_ai_keyword_classified_as_demand_signal(self):
        # "chatgpt" is an AI keyword; no pain or value keywords present
        result = classify_signal("using chatgpt for everything")
        assert result == "demand_signal"

    def test_competitor_keyword_classified_as_competition_signal(self):
        result = classify_signal("looking for a competitor tool")
        assert result == "competition_signal"

    def test_alternative_keyword_classified_as_competition_signal(self):
        result = classify_signal("need an alternative solution")
        assert result == "competition_signal"

    def test_unknown_text_classified_as_low_signal(self):
        result = classify_signal("random text without keywords")
        assert result == "low_signal_summary"

    def test_empty_title_and_body_classified_as_low_signal(self):
        result = classify_signal("", "")
        assert result == "low_signal_summary"

    def test_body_contributes_keywords(self):
        result = classify_signal("something", "this is frustrating manual work")
        assert result == "pain_signal"

    def test_title_contributes_keywords(self):
        # Use only PAIN_KEYWORDS, no VALUE_KEYWORDS
        result = classify_signal("frustrating issue", "")
        assert result == "pain_signal"

    def test_ai_keyword_takes_precedence_over_competition(self):
        # "chatgpt" (AI) + "alternative" (competition trigger)
        # Logic order: pain -> value -> ai -> competition
        # No pain or value keywords, has_ai=True => demand_signal
        # Note: must avoid VALUE_KEYWORDS like "workflow"
        result = classify_signal("chatgpt alternative approach")
        assert result == "demand_signal"

    def test_pain_with_ai_keyword(self):
        # "frustrating" (PAIN) + "chatgpt" (AI), but no VALUE_KEYWORDS
        # has_pain=True, has_value=False => pain_signal (checked before ai)
        result = classify_signal("frustrating chatgpt experience")
        assert result == "pain_signal"

    def test_value_with_ai_keyword(self):
        # "productivity" (VALUE) + "chatgpt" (AI), but no PAIN_KEYWORDS
        # has_value=True, has_pain=False => success_signal (checked before ai)
        result = classify_signal("chatgpt productivity gains")
        assert result == "success_signal"

    def test_pain_value_and_ai_all_present(self):
        # "frustrating" (PAIN) + "workflow" (VALUE) + "chatgpt" (AI)
        # has_pain=True, has_value=True => neither pure condition
        # has_ai=True => demand_signal
        result = classify_signal("frustrating workflow with chatgpt")
        assert result == "demand_signal"

    def test_case_insensitive_classification(self):
        result = classify_signal("FRUSTRATING EXPERIENCE")
        assert result == "pain_signal"

    def test_keyword_lists_are_non_empty(self):
        assert len(AI_TOOL_KEYWORDS) > 0
        assert len(PAIN_KEYWORDS) > 0
        assert len(VALUE_KEYWORDS) > 0
        assert len(RECURRENCE_KEYWORDS) > 0

    def test_each_pain_keyword_matches(self):
        for kw in PAIN_KEYWORDS:
            assert contains_pain_keyword(kw) is True, f"Pain keyword '{kw}' did not match"

    def test_each_ai_keyword_matches(self):
        for kw in AI_TOOL_KEYWORDS:
            assert contains_ai_keyword(kw) is True, f"AI keyword '{kw}' did not match"

    def test_each_value_keyword_matches(self):
        for kw in VALUE_KEYWORDS:
            assert contains_value_keyword(kw) is True, f"Value keyword '{kw}' did not match"

    def test_each_recurrence_keyword_matches(self):
        for kw in RECURRENCE_KEYWORDS:
            assert contains_recurrence_keyword(kw) is True, f"Recurrence keyword '{kw}' did not match"
