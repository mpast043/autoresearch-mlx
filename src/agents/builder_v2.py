"""
BuilderAgentV2 — Real code-generating builder that takes a spec and produces
working project code via LLM.

Unlike BuilderAgent (v1) which only wrote static HTML from templates,
v2 generates actual project structures: package.json, API routes, components,
database schemas, and README — from a structured spec.
"""

import asyncio
import importlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.build_prep import is_allowed_selection_transition
from src.messaging import MessageType, create_message
from src.runtime.paths import DEFAULT_CONFIG_PATH, resolve_project_path

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    success: bool
    project_path: Optional[Path] = None
    confidence: float = 0.0
    files_written: list[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_s: float = 0.0


class PromptLibrary:
    """Type-specific prompt templates for different output_type builds."""

    SYSTEM_PROMPTS = {
        "workflow_reliability_console": """You are a senior full-stack engineer building a Python CLI/console tool for diagnosing and fixing workflow reliability issues.
Your job is to produce a complete, working Python project.
Output ONLY the code files listed in your plan. No markdown explanations.
Use asyncio for I/O. Include type hints. Write production-quality code.
Install dependencies: standard library only where possible, else add to requirements.txt.""",
        "workflow_diagnostic_prototype": """You are a senior full-stack engineer building a prototype web dashboard for diagnosing workflow reliability issues.
Your job is to produce a complete, working web app.
Use React + Vite (frontend) and FastAPI (backend). Output ONLY the code files.
Include realistic mock data. Style with Tailwind CSS.
Write production-quality code. No placeholders.""",
        "operator_evidence_workspace": """You are a senior full-stack engineer building an evidence-collection workspace for operators.
Your job is to produce a complete, working project for capturing and organizing evidence.
Use React + Vite frontend, FastAPI backend, SQLite for local storage.
Output ONLY the code files. Use Tailwind for styling.
Write production-quality code. No TODO comments or placeholders.""",
    }

    USER_TEMPLATE = """Generate a {output_type} project from this spec:

## Spec
{json_spec}

## Product Requirements
Generate a complete, working product that solves the user's problem. This should be a deployable web application (SaaS) unless the problem specifically calls for a different format.

### Product Type Decision
Based on the problem_statement and job_to_be_done in the spec:
- If it involves workflow management, automation, or team processes → build a Flask/FastAPI web app
- If it involves data entry, forms, or databases → build a web app with SQLite
- If it involves monitoring or alerts → build a web dashboard
- Default to a Flask web application with SQLite database

### What the product must include:
1. **A working Flask web application** with real functionality
2. **Database models** (using SQLAlchemy) that match the problem domain
3. **Real API endpoints** that do actual things (not just return mock data)
4. **A frontend** with HTML/CSS/JS templates that show real data
5. **Actual business logic** that solves the stated problem

### DO NOT:
- Do NOT generate placeholder code or stub functions
- Do NOT just return mock data in API responses
- Do NOT create a project that won't run

### File Structure to generate:
- app.py (Flask application - use port from env var PORT or default to 5001 to avoid common port conflicts)
- models.py (SQLAlchemy database models)
- requirements.txt (Python dependencies - use >= versions, not exact pins, to avoid conflicts. Do NOT include built-in modules like datetime, os, sys, json, etc.)
- templates/base.html, templates/index.html, templates/*.html (frontend)
- static/style.css (styling)
- .env.example
- run.sh (bash script that creates venv, installs deps, and runs the app - use relative paths)
- README.md (complete documentation - see below)

### README format (must be proper Markdown):
```markdown
# Product Name

## What is this?
[A clear one-line description of what this product is, e.g., "A web-based tool that automates client onboarding workflows for operations teams"]

## Product Type
[Is this a SaaS web app, CLI tool, browser extension, mobile app, etc.]

## What problem does it solve?
[Explain the specific pain point from the spec and how this product addresses it]

## Target Users
[Who is this for? Be specific]

## Key Features
- [Feature 1]
- [Feature 2]
- [Feature 3]

## Tech Stack
- Backend: [Flask/FastAPI + SQLAlchemy]
- Database: [SQLite/PostgreSQL]
- Frontend: [HTML/CSS/JS]

## How to Run
1. Install: `pip install -r requirements.txt`
2. Configure: Copy `.env.example` to `.env` and fill in values
3. Run: `python app.py` or `bash run.sh`
4. Access: Open http://localhost:5000 in browser

## API Endpoints
- GET / - Main page
- GET/POST /api/... - [describe what each does]

## License
MIT
```

IMPORTANT: Return ONLY valid JSON. The content field must be properly escaped JSON (double quotes escaped as \\").
For example: {{"path": "package.json", "content": "[JSON STRING WITH ESCAPED QUOTES]"}}

Return ONLY the final JSON:
{{"files": [{{"path": "relative/path/file.ext", "content": "FILE_CONTENT_GOES_HERE"}}]}}
"""

    @classmethod
    def for_output_type(cls, output_type: str) -> tuple[str, str]:
        system = cls.SYSTEM_PROMPTS.get(output_type, cls.SYSTEM_PROMPTS["workflow_reliability_console"])
        return system, cls.USER_TEMPLATE


def _fix_nested_json_content(text: str) -> str:
    """Fix LLM output where content field contains unescaped JSON.

    The LLM often outputs nested JSON like:
    "content": "{"name": "value"}"
    or
    "content": '{"name": "value"}'

    Which is invalid JSON because the inner quotes are not escaped.
    This function attempts to fix such cases.
    """
    def _escape_content_quotes(value: str) -> str:
        """Escape unescaped double quotes in content value."""
        result = []
        i = 0
        while i < len(value):
            if value[i] == '\\' and i + 1 < len(value):
                # Keep escaped character as-is
                result.append(value[i])
                result.append(value[i + 1])
                i += 2
            elif value[i] == '"':
                # Escape unescaped quote
                result.append('\\"')
                i += 1
            else:
                result.append(value[i])
                i += 1
        return ''.join(result)

    # Fix pattern: "content": '{...}'
    # Extract content value, escape quotes, rebuild
    def fix_single_quoted_content(match):
        prefix = match.group(1)  # "content": "
        content = match.group(2)  # The content value between single quotes

        # If wrapped in single quotes, remove them
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]

        # Escape inner quotes
        fixed = _escape_content_quotes(content)
        return f'{prefix}"{fixed}"'

    text = re.sub(r'("content":\s*)\'([^\']+)\'', fix_single_quoted_content, text)

    # Fix pattern: "content": "{...}" where inner quotes aren't escaped
    def fix_double_quoted_content(match):
        prefix = match.group(1)  # "content": "
        content = match.group(2)  # The content value

        # Escape inner quotes
        fixed = _escape_content_quotes(content)
        return f'{prefix}"{fixed}"'

    # Match content that's already in double quotes but inner quotes need escaping
    # This is tricky because we need to find where the content ends
    # Use a pattern that matches from "content": " to the closing "
    text = re.sub(r'("content":\s*)"((?:[^{}"]|\{[^{}]*\})*)"', fix_double_quoted_content, text)

    return text


def _extract_json_from_response(text: str) -> list[dict]:
    """Find JSON array of {path, content} objects in LLM response."""
    # Pre-process to fix common LLM issues
    original_text = text

    # First, try with the original text
    # Try the original regex first
    match = re.search(r'\{[^{]*"files"\s*:\s*(\[[\s\S]*\])\s*\}', text)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text as JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            return data["files"]
    except json.JSONDecodeError:
        pass

    # Try to find and parse JSON in code blocks
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}\s*,\s*\{[\s\S]*\}|\[[\s\S]*\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"```\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array at the end of the text
    match = re.search(r"(\[[\s\S]+\])\s*$", text, re.MULTILINE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any bracket-balanced JSON array
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Try to fix common LLM JSON issues and re-parse
    # Fix single quotes used as string delimiters
    text = text.replace("'", '"')
    # Remove any trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            return data["files"]
    except json.JSONDecodeError:
        pass

    # Fix single-escaped quotes (e.g., \"name\" -> \\"name\\")
    text_fixed = text.replace('\\"', '\\\\"')

    try:
        data = json.loads(text_fixed.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            return data["files"]
    except json.JSONDecodeError:
        pass

    # Try to fix nested JSON content issues
    text = _fix_nested_json_content(original_text)
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            return data["files"]
    except json.JSONDecodeError:
        pass

    # Try to extract files from malformed JSON with nested content
    # Look for the "files" array and extract it with proper balance
    files_match = re.search(r'"files"\s*:\s*\[', text)
    if files_match:
        start = files_match.end() - 1  # Position at [
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        # Try to parse just the files array
                        files_str = text[start:i + 1]
                        # Fix common issues in nested content
                        files_str = re.sub(r'"content":\s*"({[^}]+})"', r'"content": "\1"', files_str)
                        files_str = re.sub(r'"content":\s*"(\[.+\])"', r'"content": "\1"', files_str)
                        result = json.loads(files_str)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return []


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename component."""
    return re.sub(r"[^\w\-_.]", "_", name)[:80]


def _resolve_cli_paths(
    config_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_config = resolve_project_path(config_path, default=DEFAULT_CONFIG_PATH)
    resolved_db = resolve_project_path(db_path, default="data/autoresearch.db")
    return resolved_config, resolved_db


class BuilderV2Agent:
    """Real code-generating builder that produces working projects from specs."""

    name = "builder"

    def __init__(self, config: dict, db=None):
        self.status = "ready"
        self._message_queue = None
        self._shutdown_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.config = config or {}
        self.db = db
        self.model = ""
        self.max_tokens = 8000
        self.base_url = ""
        self.client = None
        self._aiohttp = None
        self._provider = ""
        self.output_root = resolve_project_path(
            self.config.get("paths", {}).get("generated_projects"),
            default="data/generated_projects",
        )
        self._configure_provider()

    def _configure_provider(self) -> None:
        provider = self.config.get("llm", {}).get("provider", "anthropic").lower()
        self.model = self.config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
        self.max_tokens = self.config.get("llm", {}).get("max_tokens", 8000)
        self.base_url = self.config.get("llm", {}).get("base_url", "")

        if provider == "ollama":
            import aiohttp

            self.client = None
            self._aiohttp = aiohttp
            self._provider = "ollama"
            self.model = self.model or "codellama"
            if not self.base_url:
                self.base_url = "http://localhost:11434"
            return

        if provider == "anthropic":
            api_key = self.config.get("llm", {}).get("api_key") or self.config.get("anthropic_api_key")
            if not api_key:
                raise ValueError("No LLM API key found (llm.api_key or ANTHROPIC_API_KEY)")
            try:
                anthropic = importlib.import_module("anthropic")
            except ImportError as exc:
                raise RuntimeError(
                    "Anthropic support requires the 'anthropic' package. Install requirements-optional.txt to enable this feature."
                ) from exc
            self.client = anthropic.Anthropic(api_key=api_key)
            self._provider = "anthropic"
            return

        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'ollama' or 'anthropic'")

    async def start(self) -> None:
        """Start the agent's message loop."""
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the agent."""
        self._shutdown_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _handle_task_result(self, task: asyncio.Task) -> None:
        """Handle task result for better debugging."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Expected on shutdown
        except Exception as e:
            print(f"[BuilderV2Agent] crashed: {e}")

    async def _run_loop(self) -> None:
        """Main message processing loop."""
        while not self._shutdown_event.is_set():
            if self._message_queue is None:
                await asyncio.sleep(0.1)
                continue
            message = await self._message_queue.receive(self.name)
            await self.process(message)

    async def build_from_spec(self, spec: dict, idea_id: Optional[int] = None) -> BuildResult:
        """Generate a complete project from a spec dict."""
        t0 = time.time()
        slug = _sanitize_filename(spec.get("title", "build").lower().replace(" ", "_"))
        output_dir = self.output_root / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        output_type = spec.get("output_type", "workflow_reliability_console")
        system_prompt, user_template = PromptLibrary.for_output_type(output_type)

        spec_json = json.dumps(spec, indent=2)
        user_prompt = user_template.format(
            output_type=output_type,
            json_spec=spec_json,
            slug=slug,
        )

        try:
            if self._provider == "ollama":
                async with self._aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                    }
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self._aiohttp.ClientTimeout(total=180),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            return BuildResult(
                                success=False,
                                error=f"Ollama error {resp.status}: {error_text}",
                                duration_s=time.time() - t0,
                            )
                        result = await resp.json()
                        raw = result.get("response", "")
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw = response.content[0].text

            files = _extract_json_from_response(raw)
            logger.info(
                "Parsed %s files from LLM response, type: %s, first: %s",
                len(files),
                type(files),
                files[0] if files else "empty",
            )
            logger.info("Raw response (first 300 chars): %s", raw[:300])

            if not files:
                return BuildResult(
                    success=False,
                    error=f"Could not parse file list from LLM response (first 200 chars: {raw[:200]})",
                    duration_s=time.time() - t0,
                )

            written: list[str] = []
            try:
                for item in files:
                    fpath = output_dir / item["path"]
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(item["content"])
                    written.append(str(item["path"]))
            except Exception as exc:
                logger.error("Error writing files: %s", exc)
                return BuildResult(success=False, error=str(exc), duration_s=time.time() - t0)

            # Validate generated code - check for basic issues
            validation_errors: list[str] = []

            # Check Python files can be parsed/imported
            app_py = output_dir / "app.py"
            if app_py.exists():
                content = app_py.read_text()
                try:
                    import ast
                    ast.parse(content)
                except SyntaxError as e:
                    validation_errors.append(f"app.py has syntax error: {e}")

                # Check for deprecated Flask APIs
                if ".before_first_request" in content:
                    validation_errors.append("app.py uses deprecated Flask API 'before_first_request' (removed in Flask 2.3)")

            # Check requirements.txt doesn't have fake packages
            req_txt = output_dir / "requirements.txt"
            if req_txt.exists():
                content = req_txt.read_text()
                # Check for common fake packages (built-in modules)
                fake_packages = ["datetime", "os", "sys", "json", "re", "logging", "collections"]
                for pkg in fake_packages:
                    # Match lines like "datetime==1.0.0" but not imports
                    if re.search(rf"^{pkg}==", content, re.MULTILINE):
                        validation_errors.append(f"requirements.txt contains fake package '{pkg}' (it's built-in)")

                # Check for exact pins that might conflict
                if re.search(r"Flask==2\.1\.", content):
                    validation_errors.append("requirements.txt has Flask==2.1.x which conflicts with flask_sqlalchemy 3.x")

            if validation_errors:
                logger.warning("Validation errors found: %s", validation_errors)
                # Return partial success with warning instead of full failure
                return BuildResult(
                    success=False,
                    error=f"Validation errors: {'; '.join(validation_errors)}",
                    project_path=output_dir,
                    files_written=written,
                    confidence=0.1,
                    duration_s=time.time() - t0,
                )

            spec_path = output_dir / "SPEC.md"
            spec_path.write_text(spec_json)
            written.append("SPEC.md")

            confidence = min(1.0, len(written) / 5 * 0.6 + 0.3)
            logger.info("BuilderV2: wrote %s files to %s", len(written), output_dir)
            return BuildResult(
                success=True,
                project_path=output_dir,
                files_written=written,
                confidence=confidence,
                duration_s=time.time() - t0,
            )

        except Exception as exc:
            logger.error("BuilderV2 build failed: %s", exc)
            return BuildResult(success=False, error=str(exc), duration_s=time.time() - t0)

    async def build_from_idea(self, idea_id: int) -> BuildResult:
        """Build a project from an idea_id by fetching the idea from the database."""
        if not self.db:
            return BuildResult(success=False, error="No database connection")

        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT id, title, description, slug, spec_json, audience FROM ideas WHERE id=?",
            (idea_id,),
        ).fetchone()
        if not row:
            return BuildResult(success=False, error=f"Idea {idea_id} not found")

        spec = {
            "title": row["title"] or "",
            "output_type": "workflow_reliability_console",
            "problem_statement": row["description"] or "",
            "value_hypothesis": row["description"] or "",
            "audience": row["audience"] or "",
            "core_features": [],
        }

        if row["spec_json"]:
            try:
                idea_spec = json.loads(row["spec_json"])
                spec.update(idea_spec)
            except json.JSONDecodeError:
                pass

        return await self.build_from_spec(spec, idea_id)

    async def build_from_build_brief(self, build_brief_id: int) -> BuildResult:
        """Build a project from a build brief plus the spec-generation output."""
        if not self.db:
            return BuildResult(success=False, error="No database connection")

        brief = self.db.get_build_brief(build_brief_id)
        if brief is None:
            return BuildResult(success=False, error=f"Build brief {build_brief_id} not found")

        prep_outputs = self.db.list_build_prep_outputs(build_brief_id=build_brief_id, run_id=brief.run_id, limit=20)
        spec_output = next((item for item in prep_outputs if item.prep_stage == "spec_generation"), None)
        brief_payload = brief.brief

        spec = dict(spec_output.output if spec_output else {})
        problem_summary = brief_payload.get("problem_summary", "")
        spec.setdefault("title", problem_summary or f"build_brief_{build_brief_id}")
        spec.setdefault("output_type", brief.recommended_output_type or "workflow_reliability_console")
        spec.setdefault("problem_statement", problem_summary)
        spec.setdefault(
            "value_hypothesis",
            brief_payload.get("value_hypothesis")
            or brief_payload.get("pain_workaround", {}).get("pain_statement", "")
            or problem_summary,
        )
        spec.setdefault("audience", brief_payload.get("job_to_be_done", ""))
        spec.setdefault("core_features", brief_payload.get("launch_artifact_plan", []))
        traceability = dict(spec.get("traceability") or {})
        traceability.setdefault("build_brief_id", build_brief_id)
        traceability.setdefault("opportunity_id", brief.opportunity_id)
        traceability.setdefault("validation_id", brief.validation_id)
        spec["traceability"] = traceability

        return await self.build_from_spec(spec)

    async def build_from_brief(self, brief_path: Path, idea_id: Optional[int] = None) -> BuildResult:
        """Build from a build brief markdown file (output of SpecGenerationAgent)."""
        if not brief_path.exists():
            return BuildResult(success=False, error=f"Brief not found: {brief_path}")

        brief_content = brief_path.read_text()
        spec = self._parse_brief(brief_content)
        return await self.build_from_spec(spec, idea_id)

    def _record_build_artifact(
        self,
        *,
        build_brief_id: int | None,
        idea_id: int | None,
        result: BuildResult,
        spec: dict[str, Any],
    ) -> int:
        if not self.db:
            return 0
        product_status = "completed" if result.success else "failed"
        metadata: dict[str, Any] = {
            "output_type": spec.get("output_type", "workflow_reliability_console"),
            "files_written": result.files_written,
            "confidence": result.confidence,
            "duration_s": result.duration_s,
            "error": result.error,
            "traceability": spec.get("traceability", {}),
        }
        if build_brief_id:
            brief = self.db.get_build_brief(build_brief_id)
            if brief is None:
                return 0
            title = str(spec.get("title") or brief.brief.get("problem_summary") or f"Build {build_brief_id}")
            return self.db.upsert_product_for_build(
                build_brief_id=build_brief_id,
                opportunity_id=brief.opportunity_id,
                validation_id=brief.validation_id,
                name=title,
                location=str(result.project_path) if result.project_path else "",
                status=product_status,
                metadata=metadata,
            )
        if idea_id:
            idea = self.db.get_idea(int(idea_id))
            title = str(spec.get("title") or (idea.title if idea else "") or f"Idea {idea_id}")
            return self.db.upsert_product_for_idea(
                idea_id=int(idea_id),
                name=title,
                location=str(result.project_path) if result.project_path else "",
                status=product_status,
                metadata=metadata,
            )
        return 0

    def _advance_build_lifecycle(self, *, build_brief_id: int, success: bool) -> None:
        if not self.db:
            return
        brief = self.db.get_build_brief(build_brief_id)
        if brief is None:
            return
        target_status = "launched" if success else "iterate"
        if is_allowed_selection_transition(brief.status, target_status):
            self.db.update_build_brief_status(build_brief_id, target_status)
        opportunity = self.db.get_opportunity(brief.opportunity_id)
        if opportunity and is_allowed_selection_transition(opportunity.selection_status, target_status):
            self.db.update_opportunity_selection(
                brief.opportunity_id,
                selection_status=target_status,
                selection_reason="builder_v2_completed" if success else "builder_v2_failed",
            )

    def _advance_idea_lifecycle(self, *, idea_id: int, success: bool) -> None:
        if not self.db:
            return
        self.db.update_idea_status(idea_id, "built" if success else "failed")

    def _parse_brief(self, brief: str) -> dict:
        """Extract a spec dict from a build brief markdown file."""
        spec = {"raw_brief": brief, "output_type": "workflow_reliability_console"}

        title_match = re.search(r"^#\s+(.+)$", brief, re.MULTILINE)
        if title_match:
            spec["title"] = title_match.group(1).strip()

        pain_match = re.search(r"## Pain[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if pain_match:
            spec["pain"] = pain_match.group(1).strip()[:500]

        sol_match = re.search(r"## (?:Proposed |Solution)[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if sol_match:
            spec["proposed_solution"] = sol_match.group(1).strip()[:500]

        brief_lower = brief.lower()
        if any(k in brief_lower for k in ["dashboard", "monitor", "ui", "prototype"]):
            spec["output_type"] = "workflow_diagnostic_prototype"
        elif any(k in brief_lower for k in ["workspace", "evidence", "capture"]):
            spec["output_type"] = "operator_evidence_workspace"
        else:
            spec["output_type"] = "workflow_reliability_console"

        return spec

    async def process(self, message) -> Dict[str, Any]:
        """Handle incoming messages from the orchestrator."""
        if message.msg_type == MessageType.BUILD_REQUEST:
            idea_id = message.payload.get("idea_id")
            build_brief_id = message.payload.get("build_brief_id")
            if idea_id:
                idea = self.db.get_idea(int(idea_id)) if self.db else None
                spec = dict(idea.spec if idea else {})
                if idea:
                    spec.setdefault("title", idea.title or "")
                    spec.setdefault("output_type", "workflow_reliability_console")
                    spec.setdefault("problem_statement", idea.description or "")
                    spec.setdefault("value_hypothesis", idea.description or "")
                    spec.setdefault("audience", idea.audience or "")
                    spec.setdefault("core_features", [])
                result = await self.build_from_idea(idea_id)
            elif build_brief_id:
                build_brief_id = int(build_brief_id)
                brief = self.db.get_build_brief(build_brief_id) if self.db else None
                prep_outputs = self.db.list_build_prep_outputs(
                    build_brief_id=build_brief_id,
                    run_id=brief.run_id if brief else None,
                    limit=20,
                ) if self.db and brief else []
                spec_output = next((item for item in prep_outputs if item.prep_stage == "spec_generation"), None)
                spec = dict(spec_output.output if spec_output else {})
                if brief:
                    brief_payload = brief.brief
                    problem_summary = brief_payload.get("problem_summary", "")
                    spec.setdefault("title", problem_summary or f"build_brief_{build_brief_id}")
                    spec.setdefault("output_type", brief.recommended_output_type or "workflow_reliability_console")
                    spec.setdefault("problem_statement", problem_summary)
                    spec.setdefault(
                        "value_hypothesis",
                        brief_payload.get("value_hypothesis")
                        or brief_payload.get("pain_workaround", {}).get("pain_statement", "")
                        or problem_summary,
                    )
                    spec.setdefault("audience", brief_payload.get("job_to_be_done", ""))
                    spec.setdefault("core_features", brief_payload.get("launch_artifact_plan", []))
                    traceability = dict(spec.get("traceability") or {})
                    traceability.setdefault("build_brief_id", build_brief_id)
                    traceability.setdefault("opportunity_id", brief.opportunity_id)
                    traceability.setdefault("validation_id", brief.validation_id)
                    spec["traceability"] = traceability
                result = await self.build_from_spec(spec)
            else:
                return {"success": False, "error": "No idea_id or build_brief_id in BUILD_REQUEST"}

            product_id = 0
            if build_brief_id or idea_id:
                product_id = self._record_build_artifact(
                    build_brief_id=build_brief_id,
                    idea_id=idea_id,
                    result=result,
                    spec=spec,
                )
            if build_brief_id:
                self._advance_build_lifecycle(build_brief_id=build_brief_id, success=result.success)
            if idea_id:
                self._advance_idea_lifecycle(idea_id=int(idea_id), success=result.success)

            payload = {
                "success": result.success,
                "idea_id": idea_id,
                "build_brief_id": build_brief_id,
                "product_id": product_id or None,
                "project_path": str(result.project_path) if result.project_path else None,
                "files_written": result.files_written,
                "confidence": result.confidence,
                "error": result.error,
                "duration_s": result.duration_s,
            }
            if self._message_queue is not None:
                await self._message_queue.put(
                    create_message(
                        from_agent=self.name,
                        to_agent="orchestrator",
                        msg_type=MessageType.RESULT,
                        payload=payload,
                        priority=2,
                    )
                )
            return payload

        return {"processed": True}


async def main():
    import yaml
    from src.database import Database

    if len(sys.argv) < 2:
        print("Usage: python builder_v2.py <idea_id> [brief_path]")
        sys.exit(1)

    idea_id = int(sys.argv[1])
    brief_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    config_path, db_path = _resolve_cli_paths()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db = Database(str(db_path))
    agent = BuilderV2Agent(config, db=db)

    if brief_path:
        result = await agent.build_from_brief(brief_path, idea_id)
    else:
        conn = db._get_connection()
        row = conn.execute(
            "SELECT id, slug, spec_json FROM build_briefs WHERE idea_id=? LIMIT 1",
            (idea_id,),
        ).fetchone()
        if not row:
            print(f"No build_brief found for idea_id={idea_id}")
            sys.exit(1)

        spec = json.loads(row["spec_json"])
        result = await agent.build_from_spec(spec, idea_id)

    print(f"\n{'=' * 50}")
    print("BuilderV2 Results")
    print(f"{'=' * 50}")
    print(f"  success:   {result.success}")
    print(f"  project:   {result.project_path}")
    print(f"  confidence:{result.confidence:.2f}")
    print(f"  files:     {len(result.files_written)}")
    if result.error:
        print(f"  ERROR:     {result.error}")
    else:
        for file_name in result.files_written:
            print(f"    {file_name}")


if __name__ == "__main__":
    asyncio.run(main())
