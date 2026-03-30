"""
BuilderAgentV2 — Real code-generating builder that takes a spec and produces
working project code via LLM.

Unlike BuilderAgent (v1) which only wrote static HTML from templates,
v2 generates actual project structures: package.json, API routes, components,
database schemas, and README — from a structured spec.
"""

import asyncio
import json
import logging
import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Import MessageType for process() method
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from messaging import MessageType

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

## Requirements
1. Create ALL files under /tmp/autoresearch_build/{slug}/
2. Each file must be complete, runnable code (no stubs, no TODOs)
3. Include package.json with all dependencies (frontend only)
4. Include requirements.txt (backend only)
5. Include a README with setup instructions
6. Include a Makefile or run.sh for one-command startup
7. Use environment variables for any secrets (put placeholder in .env.example)
8. Generate realistic mock data so it runs out of the box

## Output structure to generate:
- package.json (frontend)
- requirements.txt (backend)
- src/ (frontend components or backend routes)
- server.py or main.py
- README.md
- .env.example
- run.sh or Makefile

Return ONLY the final JSON:
{{"files": [{{"path": "relative/path/file.ext", "content": "..."}}]}}
"""

    @classmethod
    def for_output_type(cls, output_type: str) -> tuple[str, str]:
        system = cls.SYSTEM_PROMPTS.get(output_type, cls.SYSTEM_PROMPTS["workflow_reliability_console"])
        user_template = cls.USER_TEMPLATE
        return system, user_template


def _extract_json_from_response(text: str) -> list[dict]:
    """Find JSON array of {path, content} objects in LLM response."""
    # First try: look for {"files": [...]} wrapper - more lenient
    match = re.search(r'\{[^{]*"files"\s*:\s*(\[[\s\S]*\])\s*\}', text)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try direct full JSON parse if text starts with {
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "files" in data:
            files = data.get("files")
            if isinstance(files, list):
                return files
    except json.JSONDecodeError:
        pass

    # Try markdown code block first - be more permissive
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}\s*,\s*\{[\s\S]*\}|\[[\s\S]*\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a json code block with just ```
    match = re.search(r"```\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare array anywhere in text
    match = re.search(r"(\[[\s\S]+\])\s*$", text, re.MULTILINE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: look for any JSON array-like structure
    # Find the first [ and matching ]
    start = text.find("[")
    if start >= 0:
        # Try to find matching ]
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return []


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename component."""
    return re.sub(r"[^\w\-_.]", "_", name)[:80]


class BuilderV2Agent:
    """Real code-generating builder that produces working projects from specs."""

    name = "builder"

    def __init__(self, config: dict, db=None):
        # Minimal interface for orchestrator - builder is message-driven
        self.status = "ready"
        self._message_queue = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the agent's message loop."""
        self._shutdown_event.clear()
        asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the agent."""
        self._shutdown_event.set()

    async def _run_loop(self) -> None:
        """Main message processing loop."""
        while not self._shutdown_event.is_set():
            if self._message_queue is None:
                await asyncio.sleep(0.1)
                continue
            message = await self._message_queue.get_for_agent(self.name)
            if message is None:
                await asyncio.sleep(0.05)
                continue
            await self.process(message)
        self.config = config
        self.db = db
        provider = config.get("llm", {}).get("provider", "anthropic").lower()
        self.model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
        self.max_tokens = config.get("llm", {}).get("max_tokens", 8000)
        self.base_url = config.get("llm", {}).get("base_url", "")

        if provider == "ollama":
            # Ollama runs locally - no API key needed
            import aiohttp
            self.client = None
            self._aiohttp = aiohttp
            self._provider = "ollama"
            self.model = self.model or "codellama"
            if not self.base_url:
                self.base_url = "http://localhost:11434"
        elif provider == "anthropic":
            import anthropic
            api_key = config.get("llm", {}).get("api_key") or config.get("anthropic_api_key")
            if not api_key:
                raise ValueError("No LLM API key found (llm.api_key or ANTHROPIC_API_KEY)")
            self.client = anthropic.Anthropic(api_key=api_key)
            self._provider = "anthropic"
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'ollama' or 'anthropic'")

        self.output_root = Path(config.get("paths", {}).get("generated_projects",
                                                          "data/generated_projects"))

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
                # Use Ollama API
                async with self._aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                    }
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self._aiohttp.ClientTimeout(total=180)
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
                # Use Anthropic API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw = response.content[0].text
            files = _extract_json_from_response(raw)
            logger.info(f"Parsed {len(files)} files from LLM response, type: {type(files)}, first: {files[0] if files else 'empty'}")
            logger.info(f"Raw response (first 300 chars): {raw[:300]}")

            if not files:
                return BuildResult(
                    success=False,
                    error=f"Could not parse file list from LLM response (first 200 chars: {raw[:200]})",
                    duration_s=time.time() - t0,
                )

            written = []
            try:
                for f in files:
                    fpath = output_dir / f["path"]
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_text(f["content"])
                    written.append(str(f["path"]))
            except Exception as e:
                logger.error(f"Error writing files: {e}")
                return BuildResult(success=False, error=str(e), duration_s=time.time() - t0)

            # Write SPEC.md
            spec_path = output_dir / "SPEC.md"
            spec_path.write_text(spec_json)
            written.append("SPEC.md")

            confidence = min(1.0, len(written) / 5 * 0.6 + 0.3)

            logger.info(f"BuilderV2: wrote {len(written)} files to {output_dir}")
            return BuildResult(
                success=True,
                project_path=output_dir,
                files_written=written,
                confidence=confidence,
                duration_s=time.time() - t0,
            )

        except Exception as e:
            logger.error(f"BuilderV2 build failed: {e}")
            return BuildResult(success=False, error=str(e), duration_s=time.time() - t0)

    async def build_from_idea(self, idea_id: int) -> BuildResult:
        """Build a project from an idea_id by fetching the idea from the database."""
        if not self.db:
            return BuildResult(success=False, error="No database connection")

        # Fetch idea via direct SQL
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT id, title, description, slug, spec_json, audience FROM ideas WHERE id=?",
            (idea_id,)
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

        # Try to get more details from spec_json if available
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

        # Extract key fields from brief for the spec
        spec = self._parse_brief(brief_content)
        return await self.build_from_spec(spec, idea_id)

    def _parse_brief(self, brief: str) -> dict:
        """Extract a spec dict from a build brief markdown file."""
        # Simple field extraction — this is heuristic and can be improved
        spec = {"raw_brief": brief, "output_type": "workflow_reliability_console"}

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", brief, re.MULTILINE)
        if title_match:
            spec["title"] = title_match.group(1).strip()

        # Extract pain
        pain_match = re.search(r"## Pain[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if pain_match:
            spec["pain"] = pain_match.group(1).strip()[:500]

        # Extract proposed solution
        sol_match = re.search(r"## (?:Proposed |Solution)[:\s]+(.+?)(?=##|\Z)", brief, re.DOTALL | re.IGNORECASE)
        if sol_match:
            spec["proposed_solution"] = sol_match.group(1).strip()[:500]

        # Detect output_type from keywords
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


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

async def main():
    import sys, yaml
    from src.database import Database
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python builder_v2.py <idea_id> [brief_path]")
        sys.exit(1)

    idea_id = int(sys.argv[1])
    brief_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db = Database(str(Path(__file__).parent.parent / "data" / "autoresearch.db"))

    agent = BuilderV2Agent(config)

    if brief_path:
        result = await agent.build_from_brief(brief_path, idea_id)
    else:
        # Try to find the build_brief for this idea
        conn = db._get_connection()
        row = conn.execute(
            "SELECT id, slug, spec_json FROM build_briefs WHERE idea_id=? LIMIT 1",
            (idea_id,)
        ).fetchone()
        if not row:
            print(f"No build_brief found for idea_id={idea_id}")
            sys.exit(1)

        spec = json.loads(row["spec_json"])
        result = await agent.build_from_spec(spec, idea_id)

    print(f"\n{'='*50}")
    print(f"BuilderV2 Results")
    print(f"{'='*50}")
    print(f"  success:   {result.success}")
    print(f"  project:   {result.project_path}")
    print(f"  confidence:{result.confidence:.2f}")
    print(f"  files:     {len(result.files_written)}")
    if result.error:
        print(f"  ERROR:     {result.error}")
    else:
        for f in result.files_written:
            print(f"    {f}")


if __name__ == "__main__":
    asyncio.run(main())
