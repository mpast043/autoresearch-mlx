"""Structured logging with correlation IDs.

Provides a custom log formatter that emits JSON-structured log lines with
run_id and correlation_id fields for tracing requests through the pipeline.

Usage::

    from src.utils.logging_utils import get_logger, set_run_id, set_correlation_id

    # At pipeline start:
    set_run_id("run_20250407_001")

    # Per-agent or per-finding:
    set_correlation_id("finding_42")

    logger = get_logger(__name__)
    logger.info("Processing finding", extra={"finding_id": "42"})

Output (with json formatter)::

    {"time": "2025-04-07T12:00:00", "level": "INFO", "logger": "src.agents.discovery",
     "run_id": "run_20250407_001", "correlation_id": "finding_42",
     "message": "Processing finding", "finding_id": "42"}
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


# ─── Context variables for async-safe correlation ────────────────────────────

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def set_run_id(run_id: str) -> None:
    """Set the current run_id for all log records in this context."""
    _run_id.set(run_id)


def set_correlation_id(cid: str) -> None:
    """Set the current correlation_id for all log records in this context."""
    _correlation_id.set(cid)


def get_run_id() -> str:
    """Get the current run_id."""
    return _run_id.get()


def get_correlation_id() -> str:
    """Get the current correlation_id."""
    return _correlation_id.get()


def new_correlation_id() -> str:
    """Generate and set a new correlation_id, returning it."""
    import uuid
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


# ─── Structured JSON formatter ────────────────────────────────────────────────


class StructuredJsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Includes run_id and correlation_id from context vars, plus any
    extra fields passed via ``logger.info("msg", extra={...})``.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": _run_id.get(),
            "correlation_id": _correlation_id.get(),
        }

        # Include any extra fields
        if record.exc_info and record.exc_text is None:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_entry["exception"] = record.exc_text

        # Merge extra fields (skip standard LogRecord attributes)
        standard_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in log_entry:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StructuredTextFormatter(logging.Formatter):
    """Human-readable formatter with run_id and correlation_id prefix.

    Format: ``[run_id] [correlation_id] LEVEL logger: message``
    """

    def format(self, record: logging.LogRecord) -> str:
        run_id = _run_id.get()
        correlation_id = _correlation_id.get()
        prefix = ""
        if run_id:
            prefix += f"[{run_id}] "
        if correlation_id:
            prefix += f"[{correlation_id}] "

        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
        level = record.levelname
        logger = record.name
        message = record.getMessage()

        result = f"{timestamp} {prefix}{level} {logger}: {message}"

        if record.exc_info and record.exc_text is None:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result += "\n" + record.exc_text

        return result


# ─── Convenience ──────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name (thin wrapper around logging.getLogger)."""
    return logging.getLogger(name)


def setup_logging(
    level: int | str = logging.WARNING,
    *,
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure root logger with structured formatting.

    Args:
        level: Log level (default WARNING).
        json_output: If True, use JSON formatter; otherwise human-readable.
        log_file: Optional file path for log output (in addition to stderr).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    formatter = StructuredJsonFormatter() if json_output else StructuredTextFormatter()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)