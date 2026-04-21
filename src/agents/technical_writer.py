"""Technical writer agent for auto-generating documentation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent
from src.messaging import MessageType


@dataclass
class DocumentedEndpoint:
    """API endpoint documentation."""

    path: str
    method: str
    summary: str
    description: str = ""
    request_body: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)
    auth_required: bool = True
    errors: list[dict] = field(default_factory=list)


@dataclass
class DocumentationBundle:
    """Complete documentation package for a solution."""

    solution_id: str
    wedge_id: int
    generated_at: float
    readme: str = ""
    api_docs: str = ""
    usage_guide: str = ""
    changelog: str = ""
    endpoints: list[DocumentedEndpoint] = field(default_factory=list)


class TechnicalWriterAgent(BaseAgent):
    """Technical writer agent for auto-generating documentation.

    Generates README, API docs, usage guides, and changelogs
    from build specifications and solution metadata.
    """

    def __init__(self, db=None, message_queue=None, config: dict[str, Any] | None = None):
        super().__init__("TechnicalWriterAgent")
        self.db = db
        self.config = config or {}
        self._template_dir = self.config.get("template_dir")

    async def process(self, message) -> dict[str, Any]:
        """Process a documentation generation request."""
        payload = message.payload if hasattr(message, "payload") else message
        spec = payload.get("spec")
        output_dir = payload.get("output_dir") or self.config.get("output_dir", "output/docs")

        if spec:
            bundle = self.generate_docs(spec)
            if output_dir:
                files = self.save_docs(bundle, Path(output_dir))
                result = {"bundle": bundle, "files": files, "summary": self.format_summary(bundle)}
            else:
                result = {"bundle": bundle, "summary": self.format_summary(bundle)}
        else:
            result = {"error": "No spec provided"}

        logger.info(
            "doc generation complete: solution=%s endpoints=%d",
            bundle.solution_id if spec else "none",
            len(bundle.endpoints) if spec else 0,
        )
        # Forward results to orchestrator
        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.DOC_GENERATION,
            payload={
                **payload,
                "doc_result": {k: str(v) for k, v in result.items()},
                "from_agent": self.name,
            },
            priority=4,
        )
        return result

    def generate_docs(self, spec: dict[str, Any]) -> DocumentationBundle:
        """Generate complete documentation package from spec.

        Args:
            spec: Solution specification dict

        Returns:
            DocumentationBundle with all docs
        """
        bundle = DocumentationBundle(
            solution_id=spec.get("id", "unknown"),
            wedge_id=spec.get("wedge_id", 0),
            generated_at=time.time(),
        )

        # Generate each component
        bundle.readme = self._generate_readme(spec)
        bundle.api_docs = self._generate_api_docs(spec)
        bundle.usage_guide = self._generate_usage_guide(spec)
        bundle.changelog = self._generate_changelog(spec)
        bundle.endpoints = self._extract_endpoints(spec)

        return bundle

    def _generate_readme(self, spec: dict[str, Any]) -> str:
        """Generate README.md from spec."""
        title = spec.get("title", spec.get("name", "Solution"))
        description = spec.get("description", spec.get("summary", ""))
        tagline = spec.get("tagline", "")

        sections = [
            f"# {title}",
            "",
        ]

        if tagline:
            sections.append(f"> {tagline}")
            sections.append("")

        if description:
            sections.append(f"{description}")
            sections.append("")

        sections.extend([
            "## Quick Start",
            "",
            "```bash",
            f"# Install",
            spec.get("install_command", "pip install <package>"),
            "",
            f"# Usage",
            spec.get("usage_example", "<example>"),
            "```",
            "",
        ])

        # Features
        features = spec.get("features", [])
        if features:
            sections.extend([
                "## Features",
                "",
            ])
            for feature in features:
                sections.append(f"- {feature}")
            sections.append("")

        # Requirements
        requirements = spec.get("requirements", [])
        if requirements:
            sections.extend([
                "## Requirements",
                "",
            ])
            for req in requirements:
                sections.append(f"- {req}")
            sections.append("")

        # Configuration
        config_schema = spec.get("config_schema", {})
        if config_schema:
            sections.extend([
                "## Configuration",
                "",
                "```yaml",
                self._format_yaml(config_schema),
                "```",
                "",
            ])

        sections.extend([
            "## License",
            "",
            spec.get("license", "MIT"),
        ])

        return "\n".join(sections)

    def _generate_api_docs(self, spec: dict[str, Any]) -> str:
        """Generate API documentation from spec."""
        endpoints = self._extract_endpoints(spec)

        if not endpoints:
            return "# API Documentation\n\nNo API endpoints defined."

        lines = [
            "# API Reference",
            "",
        ]

        for ep in endpoints:
            lines.extend([
                f"## {ep.method.upper()} {ep.path}",
                "",
                f"**Summary**: {ep.summary}",
                "",
            ])

            if ep.description:
                lines.append(ep.description)
                lines.append("")

            if ep.auth_required:
                lines.append("🔒 **Authentication Required**")
                lines.append("")

            if ep.request_body:
                lines.append("### Request Body")
                lines.append("```json")
                lines.append(self._format_json(ep.request_body))
                lines.append("```")
                lines.append("")

            if ep.response:
                lines.append("### Response")
                lines.append("```json")
                lines.append(self._format_json(ep.response))
                lines.append("```")
                lines.append("")

            if ep.errors:
                lines.append("### Error Responses")
                for err in ep.errors:
                    lines.append(f"- **{err.get('code', '')}**: {err.get('message', '')}")
                lines.append("")

        return "\n".join(lines)

    def _generate_usage_guide(self, spec: dict[str, Any]) -> str:
        """Generate usage guide from spec."""
        title = spec.get("title", "Solution")

        sections = [
            "# Usage Guide",
            "",
            f"Welcome to **{title}**! This guide will help you get started.",
            "",
            "## Installation",
            "",
            "```bash",
            spec.get("install_command", "pip install <package>"),
            "```",
            "",
        ]

        # Basic usage
        basic_usage = spec.get("basic_usage", [])
        if basic_usage:
            sections.extend([
                "## Basic Usage",
                "",
            ])
            for usage in basic_usage:
                sections.append(f"1. {usage}")
            sections.append("")

        # Advanced usage
        advanced_usage = spec.get("advanced_usage", [])
        if advanced_usage:
            sections.extend([
                "## Advanced Usage",
                "",
            ])
            for usage in advanced_usage:
                sections.append(f"- {usage}")
            sections.append("")

        # Examples
        examples = spec.get("examples", [])
        if examples:
            sections.extend([
                "## Examples",
                "",
            ])
            for ex in examples:
                sections.append(f"### {ex.get('title', 'Example')}")
                if ex.get("description"):
                    sections.append(ex["description"])
                if ex.get("code"):
                    sections.append("```python")
                    sections.append(ex["code"])
                    sections.append("```")
                sections.append("")

        # Troubleshooting
        troubleshooting = spec.get("troubleshooting", [])
        if troubleshooting:
            sections.extend([
                "## Troubleshooting",
                "",
            ])
            for issue in troubleshooting:
                sections.append(f"### {issue.get('problem', 'Issue')}")
                sections.append(f"**Solution**: {issue.get('solution', '')}")
                sections.append("")

        return "\n".join(sections)

    def _generate_changelog(self, spec: dict[str, Any]) -> str:
        """Generate changelog from spec."""
        sections = [
            "# Changelog",
            "",
            f"All notable changes to this project will be documented in this file.",
            "",
            "## [Unreleased]",
            "",
        ]

        changes = spec.get("changes", [])
        if changes:
            for change in changes:
                change_type = change.get("type", "Changed")
                sections.append(f"### {change_type}")
                sections.append(f"- {change.get('description', '')}")
        else:
            sections.append("### Added")
            sections.append("- Initial release")

        return "\n".join(sections)

    def _extract_endpoints(self, spec: dict[str, Any]) -> list[DocumentedEndpoint]:
        """Extract API endpoints from spec."""
        endpoints = []
        api_spec = spec.get("api", {})

        for path, methods in api_spec.items():
            for method, details in methods.items():
                endpoint = DocumentedEndpoint(
                    path=path,
                    method=method.upper(),
                    summary=details.get("summary", ""),
                    description=details.get("description", ""),
                    request_body=details.get("requestBody", {}),
                    response=details.get("response", {}),
                    auth_required=details.get("auth", True),
                    errors=details.get("errors", []),
                )
                endpoints.append(endpoint)

        return endpoints

    def _format_endpoints(self, endpoints: list[DocumentedEndpoint]) -> str:
        """Format endpoints as markdown."""
        lines = ["# API Endpoints", ""]

        for ep in endpoints:
            lines.extend([
                f"### `{ep.method.upper()} {ep.path}`",
                "",
                ep.summary,
                "",
            ])

        return "\n".join(lines)

    def _format_yaml(self, obj: dict, indent: int = 0) -> str:
        """Format dict as YAML-like string."""
        lines = []
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(self._format_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{'  ' * indent}{key}:")
                for item in value:
                    lines.append(f"{'  ' * (indent + 1)}- {item}")
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return "\n".join(lines)

    def _format_json(self, obj: Any) -> str:
        """Format object as JSON-like string."""
        import json

        return json.dumps(obj, indent=2)

    def save_docs(self, bundle: DocumentationBundle, output_dir: Path) -> dict[str, Path]:
        """Save documentation bundle to files.

        Args:
            bundle: Documentation bundle to save
            output_dir: Output directory

        Returns:
            Dict mapping doc type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        if bundle.readme:
            readme_path = output_dir / "README.md"
            readme_path.write_text(bundle.readme)
            files["readme"] = readme_path

        if bundle.api_docs:
            api_path = output_dir / "API.md"
            api_path.write_text(bundle.api_docs)
            files["api"] = api_path

        if bundle.usage_guide:
            guide_path = output_dir / "USAGE.md"
            guide_path.write_text(bundle.usage_guide)
            files["usage"] = guide_path

        if bundle.changelog:
            changelog_path = output_dir / "CHANGELOG.md"
            changelog_path.write_text(bundle.changelog)
            files["changelog"] = changelog_path

        return files

    def format_summary(self, bundle: DocumentationBundle) -> str:
        """Format documentation summary."""
        lines = [
            f"# Documentation Generated",
            "",
            f"Solution: {bundle.solution_id}",
            f"Wedge ID: {bundle.wedge_id}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(bundle.generated_at))}",
            "",
            "## Output Files",
            "",
        ]

        if bundle.readme:
            lines.append("✅ README.md")
        if bundle.api_docs:
            lines.append("✅ API.md")
        if bundle.usage_guide:
            lines.append("✅ USAGE.md")
        if bundle.changelog:
            lines.append("✅ CHANGELOG.md")

        lines.append("")
        lines.append(f"Total endpoints documented: {len(bundle.endpoints)}")

        return "\n".join(lines)