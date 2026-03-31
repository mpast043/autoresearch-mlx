"""Environment loading helpers for runtime entrypoints."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.runtime.paths import DEFAULT_ENV_PATH, resolve_project_path


def load_local_env(env_path: Path | str | None = None) -> None:
    env_file = resolve_project_path(env_path, default=DEFAULT_ENV_PATH)
    if env_file.exists():
        load_dotenv(env_file)
