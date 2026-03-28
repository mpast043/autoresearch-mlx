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
from typing import Optional
import anthropic

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
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\[[\s\S]+?\])\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare array
    match = re.search(r"(\[[\s\S]+?\])\s*$", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return []


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename component."""
    return re.sub(r"[^\w\-_.]", "_", name)[:80]


class BuilderV2Agent:
    """Real code-generating builder that produces working projects from specs."""

    def __init__(self, config: dict):
        self.config = config
        api_key = config.get("llm", {}).get("api_key") or config.get("anthropic_api_key") or config.get("openai_api_key")
        provider = config.get("llm", {}).get("provider", "anthropic")
        self.model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
        self.max_tokens = config.get("llm", {}).get("max_tokens", 8000)
        if not api_key:
            raise ValueError("No LLM API key found in config (llm.api_key or ANTHROPIC_API_KEY)")
        self.client = anthropic.Anthropic(api_key=api_key)
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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            files = _extract_json_from_response(raw)

            if not files:
                return BuildResult(
                    success=False,
                    error=f"Could not parse file list from LLM response (first 200 chars: {raw[:200]})",
                    duration_s=time.time() - t0,
                )

            written = []
            for f in files:
                fpath = output_dir / f["path"]
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(f["content"])
                written.append(str(f["path"]))

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
