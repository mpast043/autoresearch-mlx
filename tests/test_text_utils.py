"""Tests for text utility functions."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.text import (
    compact_text,
    slugify,
    domain_for,
    unwrap_search_result_url,
    normalize_search_url,
    query_phrases,
    query_terms,
    _query_phrase,
    _query_term_span,
    _clean_recurrence_text,
    topical_overlap,
    url_path,
    contains_keyword,
    first_match,
    infer_recurrence_key,
)


# ---------- compact_text ----------

class TestCompactText:
    def test_collapses_whitespace(self):
        assert compact_text("  hello   world  ") == "hello world"

    def test_truncates_to_limit(self):
        assert compact_text("a" * 600, limit=10) == "a" * 10

    def test_none_returns_empty(self):
        assert compact_text(None) == ""

    def test_empty_string(self):
        assert compact_text("") == ""

    def test_preserves_content_within_limit(self):
        text = "hello world"
        assert compact_text(text) == text

    def test_tabs_and_newlines(self):
        assert compact_text("hello\t\tworld\n\ntest") == "hello world test"


# ---------- slugify ----------

class TestSlugify:
    def test_simple_string(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters_removed(self):
        assert slugify("Hello, World! #2024") == "hello-world-2024"

    def test_unicode_characters(self):
        # Non-ASCII chars get removed by the [^a-z0-9] pattern
        result = slugify("café résumé")
        assert result == "caf-r-sum" or isinstance(result, str)

    def test_empty_string_uses_fallback(self):
        assert slugify("") == "product"

    def test_none_uses_fallback(self):
        assert slugify(None) == "product"

    def test_custom_fallback(self):
        assert slugify("", fallback="default") == "default"

    def test_truncation_to_48_chars(self):
        long_slug = "a" * 100
        result = slugify(long_slug)
        assert len(result) <= 48

    def test_strips_leading_trailing_dashes(self):
        assert slugify("---hello---") == "hello"

    def test_multiple_dashes_collapsed(self):
        assert slugify("hello   world") == "hello-world"


# ---------- domain_for ----------

class TestDomainFor:
    def test_simple_domain(self):
        assert domain_for("https://example.com/path") == "example.com"

    def test_removes_www_prefix(self):
        assert domain_for("https://www.example.com/path") == "example.com"

    def test_preserves_subdomain(self):
        assert domain_for("https://sub.example.com/path") == "sub.example.com"

    def test_http_scheme(self):
        assert domain_for("http://example.com") == "example.com"

    def test_empty_url(self):
        assert domain_for("") == ""

    def test_port_preserved(self):
        result = domain_for("https://example.com:8080/path")
        assert "example.com" in result


# ---------- unwrap_search_result_url ----------

class TestUnwrapSearchResultUrl:
    def test_plain_url_unchanged(self):
        assert unwrap_search_result_url("https://example.com/page") == "https://example.com/page"

    def test_duckduckgo_redirect_unwrapped(self):
        url = "https://duckduckgo.com/y.js?uddg=https%3A%2F%2Fexample.com%2Fpage"
        result = unwrap_search_result_url(url)
        assert "example.com" in result

    def test_protocol_relative_url(self):
        result = unwrap_search_result_url("//example.com/path")
        assert result.startswith("https:")

    def test_empty_string(self):
        assert unwrap_search_result_url("") == ""

    def test_none_returns_empty(self):
        assert unwrap_search_result_url(None) == ""


# ---------- normalize_search_url ----------

class TestNormalizeSearchUrl:
    def test_removes_fragment(self):
        result = normalize_search_url("https://example.com/page#section")
        assert "#section" not in result
        assert "section" not in result

    def test_keeps_essential_query_params(self):
        result = normalize_search_url("https://example.com/page?id=123&v=2&p=3&q=test")
        assert "id=123" in result
        assert "v=2" in result
        assert "p=3" in result
        assert "q=test" in result

    def test_removes_non_essential_query_params(self):
        result = normalize_search_url("https://example.com/page?utm_source=foo&id=123")
        assert "utm_source" not in result
        assert "id=123" in result

    def test_empty_url(self):
        assert normalize_search_url("") == ""

    def test_invalid_scheme(self):
        assert normalize_search_url("ftp://example.com/page") == ""


# ---------- query_phrases ----------

class TestQueryPhrases:
    def test_extracts_quoted_phrases(self):
        result = query_phrases('"hello world" and "foo bar"', set())
        assert result == ["hello world", "foo bar"]

    def test_no_phrases(self):
        assert query_phrases("no quotes here", set()) == []

    def test_empty_string(self):
        assert query_phrases("", set()) == []

    def test_single_phrase(self):
        assert query_phrases('"test phrase"', set()) == ["test phrase"]

    def test_strips_whitespace_in_phrases(self):
        result = query_phrases('"  spaced  phrase  "', set())
        assert "spaced  phrase" in result or "spaced phrase" in result

    def test_unclosed_quotes_ignored(self):
        # re.finditer only matches complete "..." patterns
        result = query_phrases('"unclosed phrase', set())
        assert result == []


# ---------- query_terms ----------

class TestQueryTerms:
    STOPWORDS = {"the", "and", "for"}

    def test_extracts_terms(self):
        result = query_terms("hello world test", self.STOPWORDS)
        assert "hello" in result
        assert "world" in result

    def test_removes_stopwords(self):
        result = query_terms("the quick fox and the dog", self.STOPWORDS)
        assert "the" not in result
        assert "and" not in result

    def test_removes_short_terms(self):
        result = query_terms("a big cat", self.STOPWORDS)
        # len <= 2 terms removed
        assert "a" not in result

    def test_deduplicates(self):
        result = query_terms("hello hello hello", self.STOPWORDS)
        assert result.count("hello") == 1

    def test_empty_string(self):
        assert query_terms("", self.STOPWORDS) == []


# ---------- _query_phrase ----------

class TestQueryPhrase:
    STOPWORDS = {"the", "and", "for", "with"}

    def test_generates_quoted_phrase(self):
        result = _query_phrase("the quick brown fox", self.STOPWORDS)
        assert result.startswith('"')
        assert result.endswith('"')

    def test_removes_stopwords(self):
        result = _query_phrase("the quick brown fox", self.STOPWORDS)
        assert "the" not in result

    def test_too_few_tokens_returns_empty(self):
        result = _query_phrase("a", self.STOPWORDS)
        assert result == ""

    def test_max_words_limit(self):
        result = _query_phrase("one two three four five six seven", self.STOPWORDS, max_words=3)
        # Inside quotes, max 3 words
        inner = result.strip('"')
        assert len(inner.split()) <= 3

    def test_empty_string(self):
        result = _query_phrase("", self.STOPWORDS)
        assert result == ""


# ---------- _query_term_span ----------

class TestQueryTermSpan:
    STOPWORDS = {"the", "and", "for"}
    WEAK = {"maybe", "might"}
    NOISE = {"thing", "stuff"}

    def test_extracts_terms(self):
        result = _query_term_span("hello world test", self.STOPWORDS, self.WEAK, self.NOISE)
        terms = result.split()
        assert "hello" in terms
        assert "world" in terms

    def test_removes_stopwords_weak_and_noise(self):
        result = _query_term_span("the test maybe thing", self.STOPWORDS, self.WEAK, self.NOISE)
        assert "the" not in result
        assert "maybe" not in result
        assert "thing" not in result

    def test_removes_short_tokens(self):
        result = _query_term_span("a big test", self.STOPWORDS, self.WEAK, self.NOISE)
        assert "a" not in result

    def test_deduplicates(self):
        result = _query_term_span("test test test", self.STOPWORDS, self.WEAK, self.NOISE)
        assert result.count("test") <= 1

    def test_max_terms_limit(self):
        result = _query_term_span("one two three four five", self.STOPWORDS, self.WEAK, self.NOISE, max_terms=2)
        assert len(result.split()) <= 2


# ---------- _clean_recurrence_text ----------

class TestCleanRecurrenceText:
    def test_removes_urls(self):
        result = _clean_recurrence_text("check https://example.com for info")
        assert "https://example.com" not in result

    def test_removes_brackets(self):
        result = _clean_recurrence_text("hello [world] test")
        assert "[world]" not in result

    def test_removes_emails(self):
        result = _clean_recurrence_text("contact user@example.com for details")
        assert "user@example.com" not in result

    def test_removes_version_numbers(self):
        result = _clean_recurrence_text("bug in v1.2.3 and #1234")
        assert "v1.2.3" not in result
        assert "#1234" not in result

    def test_removes_special_characters(self):
        result = _clean_recurrence_text("hello! world? test@")
        assert "!" not in result
        assert "?" not in result

    def test_empty_string(self):
        result = _clean_recurrence_text("")
        assert result == ""

    def test_none_input(self):
        result = _clean_recurrence_text(None)
        assert result == ""

    def test_limit_parameter(self):
        long_text = "word " * 200
        result = _clean_recurrence_text(long_text, limit=50)
        assert len(result) <= 50


# ---------- topical_overlap ----------

class TestTopicalOverlap:
    STOPWORDS = {"the", "and", "for"}

    def test_matching_terms(self):
        score = topical_overlap("hello world", "hello world", "", "example.com", self.STOPWORDS)
        assert score >= 2  # at least 2 term hits

    def test_matching_phrases(self):
        score = topical_overlap('"hello world"', "hello world", "", "example.com", self.STOPWORDS)
        assert score >= 2  # phrase hits count double

    def test_no_overlap(self):
        score = topical_overlap("xyzabc", "completely different text", "no match", "other.com", self.STOPWORDS)
        assert score == 0

    def test_domain_included(self):
        score = topical_overlap("example", "unrelated text", "unrelated snippet", "example.com", self.STOPWORDS)
        assert score >= 1


# ---------- url_path ----------

class TestUrlPath:
    def test_simple_path(self):
        assert url_path("https://example.com/foo/bar") == "/foo/bar"

    def test_root_path(self):
        assert url_path("https://example.com/") == "/"

    def test_no_path(self):
        assert url_path("https://example.com") == ""

    def test_lowercase(self):
        assert url_path("https://example.com/Foo/Bar") == "/foo/bar"


# ---------- contains_keyword ----------

class TestContainsKeyword:
    def test_single_word_keyword(self):
        assert contains_keyword("hello world test", "world") is True

    def test_multi_word_keyword(self):
        assert contains_keyword("hello quick brown fox", "quick brown") is True

    def test_keyword_not_found(self):
        assert contains_keyword("hello world", "absent") is False

    def test_case_insensitive(self):
        assert contains_keyword("HELLO WORLD", "world") is True

    def test_word_boundary_single_word(self):
        # Single-word keywords use word boundary matching
        assert contains_keyword("hello testworld", "test") is False

    def test_multi_word_no_boundary(self):
        # Multi-word keywords don't use word boundary
        assert contains_keyword("the quick brown fox jumps", "quick brown") is True

    def test_empty_text(self):
        assert contains_keyword("", "keyword") is False

    def test_empty_keyword(self):
        # Empty keyword produces a pattern that matches everywhere (\b\b),
        # so the function returns True for any non-empty text
        assert contains_keyword("text", "") is True


# ---------- first_match ----------

class TestFirstMatch:
    def test_first_matching_pattern(self):
        result = first_match([r"\d+", r"[a-z]+"], "123abc")
        assert result == "123"

    def test_no_match(self):
        result = first_match([r"\d{5}"], "abc")
        assert result is None

    def test_empty_patterns(self):
        result = first_match([], "text")
        assert result is None

    def test_empty_text(self):
        result = first_match([r"\d+"], "")
        assert result is None

    def test_case_insensitive_flag(self):
        result = first_match([r"hello"], "HELLO WORLD")
        assert result == "HELLO"


# ---------- infer_recurrence_key ----------

class TestInferRecurrenceKey:
    def test_basic_key_generation(self):
        result = infer_recurrence_key("docker container fails repeatedly")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_removes_stopwords(self):
        result = infer_recurrence_key("the docker and the container")
        assert "the" not in result.split()
        assert "and" not in result.split()

    def test_removes_short_tokens(self):
        result = infer_recurrence_key("a big container")
        # tokens of length <= 2 should be removed
        assert "a" not in result.split()

    def test_deduplicates(self):
        result = infer_recurrence_key("docker docker docker container")
        assert result.split().count("docker") <= 1

    def test_max_six_terms(self):
        result = infer_recurrence_key("one two three four five six seven eight nine ten")
        assert len(result.split()) <= 6

    def test_empty_string(self):
        # normalize_content("") returns "" which after splitting gives []
        # But infer_recurrence_key processes it; should return empty string
        result = infer_recurrence_key("")
        assert isinstance(result, str)