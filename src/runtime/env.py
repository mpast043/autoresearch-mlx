"""Environment loading helpers for runtime entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from runtime.paths import DEFAULT_ENV_PATH, resolve_project_path


def load_local_env(env_path: Path | str | None = None) -> None:
    env_file = resolve_project_path(env_path, default=DEFAULT_ENV_PATH)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
