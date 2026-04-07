"""Tests for TechnicalWriterAgent."""

import pytest
import tempfile
from pathlib import Path

from src.agents.technical_writer import (
    TechnicalWriterAgent,
    DocumentationBundle,
    DocumentedEndpoint,
)


class TestTechnicalWriterAgent:
    """Test suite for TechnicalWriterAgent."""

    def test_generate_docs_basic(self):
        """Test basic documentation generation."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "solution-1",
            "wedge_id": 1,
            "title": "My SaaS Solution",
            "description": "A solution for solving problems",
            "tagline": "Solve problems faster",
            "features": ["Feature 1", "Feature 2"],
            "install_command": "pip install my-saas",
            "usage_example": "my-saas run",
        }

        bundle = writer.generate_docs(spec)

        assert bundle.solution_id == "solution-1"
        assert bundle.wedge_id == 1
        assert "My SaaS Solution" in bundle.readme
        assert "Solve problems faster" in bundle.readme
        assert "Feature 1" in bundle.readme

    def test_generate_readme_complete(self):
        """Test README with all sections."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "solution-1",
            "title": "Test App",
            "description": "A test application",
            "features": ["Auth", "API"],
            "requirements": ["Python 3.8+", "PostgreSQL"],
            "install_command": "pip install test-app",
            "usage_example": "test-app serve",
            "config_schema": {
                "database": {"host": "localhost", "port": 5432},
                "debug": False,
            },
            "license": "MIT",
        }

        bundle = writer.generate_docs(spec)

        assert "## Quick Start" in bundle.readme
        assert "## Features" in bundle.readme
        assert "## Requirements" in bundle.readme
        assert "## Configuration" in bundle.readme
        assert "## License" in bundle.readme

    def test_generate_api_docs_with_endpoints(self):
        """Test API documentation with endpoints."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "api-solution",
            "title": "API Solution",
            "api": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "description": "Returns all users",
                        "auth": True,
                        "response": {"users": []},
                    },
                    "post": {
                        "summary": "Create user",
                        "requestBody": {"username": "string"},
                        "auth": False,
                    },
                }
            },
        }

        bundle = writer.generate_docs(spec)

        assert len(bundle.endpoints) == 2
        assert bundle.api_docs
        assert "/users" in bundle.api_docs
        assert "List users" in bundle.api_docs

    def test_generate_usage_guide(self):
        """Test usage guide generation."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "solution-1",
            "title": "Test App",
            "basic_usage": ["Install the package", "Configure settings", "Run the app"],
            "advanced_usage": ["Custom plugins", "Webhooks"],
            "examples": [
                {
                    "title": "Basic Example",
                    "description": "How to do X",
                    "code": "from app import run\nrun()",
                }
            ],
            "troubleshooting": [
                {"problem": "Connection error", "solution": "Check your network"}
            ],
        }

        bundle = writer.generate_docs(spec)

        assert "## Basic Usage" in bundle.usage_guide
        assert "## Advanced Usage" in bundle.usage_guide
        assert "## Examples" in bundle.usage_guide
        assert "## Troubleshooting" in bundle.usage_guide

    def test_generate_changelog(self):
        """Test changelog generation."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "solution-1",
            "title": "Test App",
            "changes": [
                {"type": "Added", "description": "New feature X"},
                {"type": "Changed", "description": "Updated Y"},
            ],
        }

        bundle = writer.generate_docs(spec)

        assert "## [Unreleased]" in bundle.changelog
        assert "New feature X" in bundle.changelog
        assert "Updated Y" in bundle.changelog

    def test_save_docs(self):
        """Test saving documentation to files."""
        writer = TechnicalWriterAgent()
        spec = {
            "id": "solution-1",
            "title": "Test App",
            "description": "A test app",
        }

        bundle = writer.generate_docs(spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            files = writer.save_docs(bundle, output_dir)

            assert "readme" in files
            assert files["readme"].exists()
            assert "Test App" in files["readme"].read_text()

    def test_save_docs_no_readme(self):
        """Test saving docs when readme is empty."""
        writer = TechnicalWriterAgent()
        bundle = DocumentationBundle(
            solution_id="test",
            wedge_id=1,
            generated_at=0,
            readme="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            files = writer.save_docs(bundle, output_dir)

            # Should create no files when all empty
            assert len(files) == 0

    def test_format_summary(self):
        """Test documentation summary formatting."""
        writer = TechnicalWriterAgent()
        bundle = DocumentationBundle(
            solution_id="test-1",
            wedge_id=5,
            generated_at=1000,
            readme="# Test",
            api_docs="# API",
            endpoints=[
                DocumentedEndpoint("/users", "GET", "List users"),
            ],
        )

        summary = writer.format_summary(bundle)

        assert "test-1" in summary
        assert "wedge id: 5" in summary.lower()
        assert "README.md" in summary
        assert "total endpoints documented: 1" in summary.lower()


class TestDocumentedEndpoint:
    """Test DocumentedEndpoint dataclass."""

    def test_endpoint_creation(self):
        """Test creating an API endpoint."""
        endpoint = DocumentedEndpoint(
            path="/api/users",
            method="GET",
            summary="Get all users",
            description="Returns a list of users",
            auth_required=True,
        )

        assert endpoint.path == "/api/users"
        assert endpoint.method == "GET"
        assert endpoint.auth_required is True