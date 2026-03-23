"""Project-rooted runtime path helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SOURCES_DB_PATH = PROJECT_ROOT / "data" / "sources_db.json"


def ensure_src_root_on_path() -> None:
    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def resolve_project_path(path: str | Path | None, *, default: str | Path | None = None) -> Path:
    target = path if path is not None else default
    if target is None:
        raise ValueError("A path or default must be provided")
    resolved = Path(target)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def build_runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = resolve_project_path(config.get("output_dir"), default="output")
    db_path = resolve_project_path(config.get("database", {}).get("path"), default="data/autoresearch.db")
    sources_db_path = resolve_project_path(config.get("sources_db_path"), default=DEFAULT_SOURCES_DB_PATH)
    log_path = output_dir / "autoresearcher.log"
    status_path = output_dir / "pipeline_status.json"
    return {
        "output_dir": output_dir,
        "db_path": db_path,
        "sources_db_path": sources_db_path,
        "log_path": log_path,
        "status_path": status_path,
    }
