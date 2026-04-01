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

from src.messaging import MessageType
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
        "saas": """You are a senior full-stack engineer building a production-ready Flask web application (SaaS).
Your job is to produce a complete, working web app that is demo-ready and solves the user's problem.
Output ONLY the code files. No markdown explanations.
Use Flask, SQLAlchemy, SQLite. Write production-quality code.
The app must have a landing page with hero section, pricing page, not just a form.""",
        "microsaas": """You are a senior full-stack engineer building a focused MicroSaaS web application.
Your job is to produce a complete, working single-purpose web app that solves ONE specific problem well.
Output ONLY the code files. No markdown explanations.
Use Flask, SQLAlchemy, SQLite. Keep it simple - one core feature, done well.
Include a landing page, real functionality, working database.""",
        "browser_extension": """You are a senior engineer building a browser extension (Chrome/Firefox).
Your job is to produce a complete, working extension that solves the user's problem.
Output ONLY the code files. No markdown explanations.
Include manifest.json, background.js, popup.html, content.js, options.html.
The extension must have a real popup UI and actual functionality.""",
        "plugin": """You are a senior engineer building a plugin/integration for an existing platform.
Your job is to produce a complete, working plugin that can be deployed.
Output ONLY the code files. No markdown explanations.
Identify the target platform (Notion, Slack, Jira, WordPress, etc.) and build accordingly.
Include all necessary files, configuration, and setup instructions.""",
        "slack_bot": """You are a senior engineer building a Slack bot/integration.
Your job is to produce a complete, working Slack app that solves the user's problem.
Output ONLY the code files. No markdown explanations.
Use Python with Slack SDK. Include app.py, requirements.txt, and setup instructions.
The bot must respond to commands and have real functionality.""",
        "api_service": """You are a senior backend engineer building a standalone API service.
Your job is to produce a complete, working API that solves the user's problem.
Output ONLY the code files. No markdown explanations.
Use Flask or FastAPI with SQLAlchemy. Include proper endpoints, models, error handling.
The API must have real business logic, not just placeholder endpoints.""",
        "desktop_app": """You are a senior full-stack engineer building a desktop application.
Your job is to produce a complete, working desktop app that solves the user's problem.
Output ONLY the code files. No markdown explanations.
Use Electron (Node.js + React) or Tauri (Rust + web). Include proper build config.
The app must have a real UI with working functionality.""",
        "cli_tool": """You are a senior full-stack engineer building a Python CLI/console tool.
Your job is to produce a complete, working Python project.
Output ONLY the code files listed in your plan. No markdown explanations.
Use asyncio for I/O. Include type hints. Write production-quality code.
Install dependencies: standard library only where possible, else add to requirements.txt.""",
    }

    USER_TEMPLATE = """Generate a {output_type} from this spec:

## Spec
{json_spec}

## Product Requirements
Generate a complete, working product that solves the user's problem. This should be a deployable web application (SaaS).

### Product Type Decision
Based on the problem_statement in the spec:
- If it involves workflow management, automation, or team processes → build a Flask/FastAPI web app
- If it involves data entry, forms, or databases → build a web app with SQLite
- If it involves monitoring or alerts → build a web dashboard
- Default to a Flask web application with SQLite database

### What the product must include:
1. **A landing page with hero section** - first impression matters, make it demo-worthy
2. **Navigation and branding** - header with logo, nav links
3. **A working Flask web application** with real functionality
4. **Database models** (using SQLAlchemy) that match the problem domain
5. **Real API endpoints** that do actual things (not just return mock data)
6. **A frontend** with HTML/CSS/JS templates that show real data
7. **Actual business logic** that solves the stated problem

### DO NOT:
- Do NOT generate placeholder code or stub functions
- Do NOT just return mock data in API responses
- Do NOT create a project that won't run
- Do NOT create just a basic contact form - build a REAL application

### File Structure to generate:
- app.py (Flask application - use port from env var PORT or default to 5001)
- models.py (SQLAlchemy database models)
- requirements.txt (Python dependencies - use >= versions)
- templates/base.html, templates/index.html, templates/*.html (frontend with hero section)
- static/style.css (styling - make it look professional)
- .env.example
- run.sh (bash script that creates venv, installs deps, runs app)
- README.md (complete documentation)

### README format:
```markdown
# Product Name

## What is this?
[A clear one-line description]

## What problem does it solve?
[Explain the pain point]

## Key Features
- [Feature 1]
- [Feature 2]

## How to Run
1. Install: `pip install -r requirements.txt`
2. Run: `python app.py`
3. Access: http://localhost:5000
```
"""

    @classmethod
    def for_output_type(cls, output_type: str) -> tuple[str, str]:
        system = cls.SYSTEM_PROMPTS.get(output_type, cls.SYSTEM_PROMPTS["saas"])
        return system, cls.USER_TEMPLATE


def _extract_json_from_response(text: str) -> list[dict]:
    """Find JSON array of {path, content} objects in LLM response."""
    match = re.search(r'\{[^{]*"files"\s*:\s*(\[[\s\S]*\])\s*\}', text)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("files"), list):
            return data["files"]
    except json.JSONDecodeError:
        pass

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

    match = re.search(r"(\[[\s\S]+\])\s*$", text, re.MULTILINE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

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
        if self._task:
            await self._task

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

        output_type = spec.get("output_type", "saas")
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
            "output_type": "saas",
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

    async def build_from_brief(self, brief_path: Path, idea_id: Optional[int] = None) -> BuildResult:
        """Build from a build brief markdown file (output of SpecGenerationAgent)."""
        if not brief_path.exists():
            return BuildResult(success=False, error=f"Brief not found: {brief_path}")

        brief_content = brief_path.read_text()
        spec = self._parse_brief(brief_content)
        return await self.build_from_spec(spec, idea_id)

    def _parse_brief(self, brief: str) -> dict:
        """Extract a spec dict from a build brief markdown file."""
        spec = {"raw_brief": brief, "output_type": "saas"}

        title_match = re.search(r"^#\s+(.+)$", brief, re.MULTILINE)
        if title_match:
            spec["title"] = title_match.group(1).strip()

        pain_match = re.search(r"## Pain[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if pain_match:
            spec["pain"] = pain_match.group(1).strip()[:500]

        sol_match = re.search(r"## (?:Proposed |Solution)[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if sol_match:
            spec["proposed_solution"] = sol_match.group(1).strip()[:500]

        # Detect output type from brief content
        brief_lower = brief.lower()
        if any(k in brief_lower for k in ["chrome extension", "firefox extension", "browser extension"]):
            spec["output_type"] = "browser_extension"
        elif any(k in brief_lower for k in ["notion", "slack", "jira", "wordpress", "plugin", "integration"]):
            spec["output_type"] = "plugin"
        elif any(k in brief_lower for k in ["slack bot", "slack app", "slack integration"]):
            spec["output_type"] = "slack_bot"
        elif any(k in brief_lower for k in ["api", "rest", "endpoint", "microservice"]):
            spec["output_type"] = "api_service"
        elif any(k in brief_lower for k in ["desktop", "electron", "tauri", "app"]):
            spec["output_type"] = "desktop_app"
        elif any(k in brief_lower for k in ["cli", "command line", "terminal", "console tool"]):
            spec["output_type"] = "cli_tool"
        elif any(k in brief_lower for k in ["microsaas", "micro-saas", "single feature", "one purpose", "focused"]):
            spec["output_type"] = "microsaas"
        elif any(k in brief_lower for k in ["dashboard", "monitor", "prototype", "ui"]):
            spec["output_type"] = "saas"
        else:
            spec["output_type"] = "saas"

        return spec

    async def process(self, message) -> Dict[str, Any]:
        """Handle incoming messages from the orchestrator."""
        if message.msg_type == MessageType.BUILD_REQUEST:
            idea_id = message.payload.get("idea_id")
            if not idea_id:
                return {"success": False, "error": "No idea_id in BUILD_REQUEST"}

            result = await self.build_from_idea(idea_id)
            return {
                "success": result.success,
                "idea_id": idea_id,
                "project_path": str(result.project_path) if result.project_path else None,
                "files_written": result.files_written,
                "confidence": result.confidence,
                "error": result.error,
                "duration_s": result.duration_s,
            }

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
