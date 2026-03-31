"""Tooling utilities extracted from research_tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.utils.search_plan import SkillAudit


class ToolingManager:
    """Creates reusable skill packs, MCP config, and helper scripts."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}

    def default_mcp_config(self, allowed_root: str = ".") -> dict[str, Any]:
        return {
            "mcpServers": {
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
                },
                "fetch": {"command": "npx", "args": ["-y", "mcp-fetch-server"]},
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
                },
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_root],
                },
                "browser": {"command": "npx", "args": ["-y", "@playwright/mcp"]},
            }
        }

    def write_mcp_config(self, path: Path) -> None:
        path.write_text(json.dumps(self.default_mcp_config(str(path.parent)), indent=2))