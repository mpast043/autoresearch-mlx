"""Tests for structured logging utilities."""

import json
import logging
from io import StringIO

import pytest

from src.utils.logging_utils import (
    StructuredJsonFormatter,
    StructuredTextFormatter,
    get_correlation_id,
    get_logger,
    get_run_id,
    new_correlation_id,
    set_correlation_id,
    set_run_id,
    setup_logging,
)


class TestContextVars:
    """Tests for run_id and correlation_id context variables."""

    def test_default_run_id_is_empty(self):
        assert get_run_id() == ""

    def test_default_correlation_id_is_empty(self):
        assert get_correlation_id() == ""

    def test_set_and_get_run_id(self):
        set_run_id("run_001")
        assert get_run_id() == "run_001"

    def test_set_and_get_correlation_id(self):
        set_correlation_id("finding_42")
        assert get_correlation_id() == "finding_42"

    def test_new_correlation_id_generates_uuid(self):
        cid = new_correlation_id()
        assert len(cid) == 12
        assert get_correlation_id() == cid

    def test_new_correlation_id_changes_each_call(self):
        cid1 = new_correlation_id()
        cid2 = new_correlation_id()
        assert cid1 != cid2


class TestStructuredJsonFormatter:
    """Tests for JSON log formatter."""

    def test_format_basic_record(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert "time" in parsed

    def test_format_includes_context_vars(self):
        set_run_id("run_abc")
        set_correlation_id("corr_123")
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["run_id"] == "run_abc"
        assert parsed["correlation_id"] == "corr_123"

    def test_format_includes_extra_fields(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        record.finding_id = "f42"
        record.score = 0.95
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["finding_id"] == "f42"
        assert parsed["score"] == 0.95

    def test_format_exception_info(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="",
            lineno=0, msg="error", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestStructuredTextFormatter:
    """Tests for human-readable text formatter."""

    def test_format_basic_record(self):
        formatter = StructuredTextFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "test.logger" in output
        assert "Hello world" in output

    def test_format_includes_run_id_prefix(self):
        set_run_id("run_001")
        formatter = StructuredTextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "[run_001]" in output

    def test_format_includes_correlation_id_prefix(self):
        set_correlation_id("corr_42")
        formatter = StructuredTextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "[corr_42]" in output

    def test_format_no_prefix_without_context(self):
        set_run_id("")
        set_correlation_id("")
        formatter = StructuredTextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "[" not in output.split("WARNING")[0]  # No bracketed prefixes


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_json_formatter(self):
        setup_logging(logging.INFO, json_output=True)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, StructuredJsonFormatter)

    def test_setup_text_formatter(self):
        setup_logging(logging.INFO, json_output=False)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, StructuredTextFormatter)

    def test_setup_string_level(self):
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_get_logger(self):
        logger = get_logger("test.module")
        assert logger.name == "test.module"