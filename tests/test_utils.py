"""Tests for utility modules.

This module contains tests for the hashing utilities used by the Discovery agent.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.hashing import (
    find_duplicate_finding,
    generate_content_hash,
    is_similar_content,
    normalize_content,
)
from utils.retry import (
    PermanentError,
    TransientError,
    retry_decorator,
    retry_with_backoff,
)


class TestNormalizeContent:
    """Tests for the normalize_content function."""

    def test_lowercase_conversion(self):
        assert normalize_content("Hello World") == "hello world"
        assert normalize_content("HELLO") == "hello"
        assert normalize_content("Test123") == "test123"

    def test_remove_urls(self):
        assert normalize_content("Check out https://example.com") == "check out"
        assert normalize_content("Visit http://test.org/page") == "visit"
        assert normalize_content("Multiple https://a.com and http://b.com") == "multiple and"

    def test_remove_mentions(self):
        assert normalize_content("Hello @username") == "hello"
        assert normalize_content("@admin check this") == "check this"
        assert normalize_content("@user1 and @user2") == "and"

    def test_remove_extra_whitespace(self):
        assert normalize_content("  hello   world  ") == "hello world"
        assert normalize_content("a\t\tb\n\nc") == "a b c"
        assert normalize_content("multiple   spaces") == "multiple spaces"

    def test_combined_normalization(self):
        text = "  Hello @user, check https://example.com  "
        assert normalize_content(text) == "hello , check"

    def test_empty_string(self):
        assert normalize_content("") == ""
        assert normalize_content("   ") == ""

    def test_only_urls_and_mentions(self):
        assert normalize_content("https://example.com @user") == ""


class TestGenerateContentHash:
    """Tests for the generate_content_hash function."""

    def test_consistent_hash(self):
        text = "Hello World"
        hash1 = generate_content_hash(text)
        hash2 = generate_content_hash(text)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_different_texts_different_hashes(self):
        hash1 = generate_content_hash("Hello")
        hash2 = generate_content_hash("World")
        assert hash1 != hash2

    def test_normalization_affects_hash(self):
        hash1 = generate_content_hash("Hello World")
        hash2 = generate_content_hash("  hello   WORLD  ")
        assert hash1 == hash2

    def test_case_insensitive_hashing(self):
        hash1 = generate_content_hash("HELLO")
        hash2 = generate_content_hash("hello")
        assert hash1 == hash2

    def test_url_removal_in_hash(self):
        hash1 = generate_content_hash("Hello World")
        hash2 = generate_content_hash("Hello https://example.com World")
        assert hash1 == hash2


class TestIsSimilarContent:
    """Tests for the is_similar_content function."""

    def test_identical_texts(self):
        assert is_similar_content("Hello World", "Hello World") is True

    def test_similar_texts(self):
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy dog."
        assert is_similar_content(text1, text2, threshold=0.9) is True

    def test_different_texts(self):
        text1 = "Hello World"
        text2 = "Completely different content here"
        assert is_similar_content(text1, text2) is False

    def test_threshold_boundary(self):
        text1 = "Hello World"
        text2 = "Hello Universe"
        assert is_similar_content(text1, text2, threshold=0.8) is False
        assert is_similar_content(text1, text2, threshold=0.5) is True

    def test_empty_strings(self):
        assert is_similar_content("", "") is True
        assert is_similar_content("hello", "") is False
        assert is_similar_content("", "hello") is False

    def test_normalization_before_comparison(self):
        text1 = "Hello @user World"
        text2 = "hello world"
        assert is_similar_content(text1, text2, threshold=0.9) is True


class TestFindDuplicateFinding:
    """Tests for the find_duplicate_finding function."""

    def test_no_duplicates_empty_list(self):
        assert find_duplicate_finding("Hello", []) is False

    def test_exact_duplicate(self):
        existing = ["Hello World", "Test Content"]
        assert find_duplicate_finding("Hello World", existing) is True

    def test_no_duplicate(self):
        existing = ["Hello World", "Test Content"]
        assert find_duplicate_finding("Different Content", existing) is False

    def test_similar_duplicate(self):
        existing = ["The quick brown fox jumps over the lazy dog"]
        similar = "The quick brown fox jumps over the lazy dog."
        assert find_duplicate_finding(similar, existing) is True

    def test_normalized_duplicate(self):
        existing = ["hello world"]
        new_text = "Hello https://example.com World"
        assert find_duplicate_finding(new_text, existing, threshold=0.9) is True

    def test_empty_text(self):
        existing = ["Hello World"]
        assert find_duplicate_finding("", existing) is False

    def test_multiple_existing_items(self):
        existing = [
            "First finding about machine learning",
            "Second finding about deep learning",
            "Third finding about neural networks",
        ]
        assert find_duplicate_finding("Second finding about deep learning", existing) is True
        assert find_duplicate_finding("Completely unrelated topic here", existing) is False

    def test_custom_threshold(self):
        existing = ["Hello World"]
        new_text = "Hello Universe"
        assert find_duplicate_finding(new_text, existing, threshold=0.9) is False
        assert find_duplicate_finding(new_text, existing, threshold=0.5) is True


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff function."""

    def test_success_on_first_try(self):
        mock_func = Mock(return_value="success")
        result = retry_with_backoff(mock_func, max_retries=3)
        assert result == "success"
        assert mock_func.call_count == 1

    @patch("utils.retry.time.sleep")
    @patch("utils.retry.random.uniform", return_value=0)
    def test_retry_on_transient_error_then_success(self, _mock_jitter, mock_sleep):
        mock_func = Mock(side_effect=[TransientError("temporary"), "success"])
        result = retry_with_backoff(mock_func, max_retries=3, base_delay=1.0)
        assert result == "success"
        assert mock_func.call_count == 2
        # Full jitter: random.uniform(0, ceiling) returns 0 when mocked, so sleep(0)
        mock_sleep.assert_called_once_with(0)

    @patch("utils.retry.time.sleep")
    @patch("utils.retry.random.uniform", return_value=0)
    def test_retry_exhaustion_raises(self, _mock_jitter, mock_sleep):
        mock_func = Mock(side_effect=TransientError("still broken"))
        with pytest.raises(TransientError):
            retry_with_backoff(mock_func, max_retries=2, base_delay=1.0)
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_permanent_error_not_retried(self):
        mock_func = Mock(side_effect=PermanentError("bad auth"))
        with pytest.raises(PermanentError):
            retry_with_backoff(mock_func, max_retries=3)
        assert mock_func.call_count == 1


class TestRetryDecorator:
    """Tests for the retry_decorator function."""

    @patch("utils.retry.time.sleep")
    @patch("utils.retry.random.uniform", return_value=0)
    def test_decorator_retries_wrapped_function(self, _mock_jitter, _mock_sleep):
        call_count = {"count": 0}

        @retry_decorator(max_retries=2, base_delay=1.0)
        def flaky() -> str:
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise TransientError("try again")
            return "ok"

        assert flaky() == "ok"
        assert call_count["count"] == 2
