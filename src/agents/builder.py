"""Builder agent that produces local runnable MVP artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.database import Database, Product
from src.messaging import MessageQueue, MessageType
from src.research_tools import ToolingManager, slugify


class BuilderAgent(BaseAgent):
    """Builds deterministic local product artifacts from approved ideas."""

    def __init__(
        self,
        db: Database,
        message_queue: Optional[MessageQueue] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("builder", message_queue)
        self.db = db
        self.config = config or {}
        self.output_dir = Path(self.config.get("builder", {}).get("output_dir", "output/builds"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tooling = ToolingManager(self.config)

    async def process(self, message) -> Dict[str, Any]:
        if message.msg_type == MessageType.BUILD_REQUEST:
            return await self._build(message.payload)
        return {"processed": True}

    async def _build(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        idea_id = payload.get("idea_id")
        idea = self.db.get_idea(idea_id)
        if idea is None:
            return {"success": False, "error": f"idea {idea_id} not found"}

        spec = idea.spec
        slug = spec.get("slug") or slugify(idea.title)
        build_dir = self.output_dir / slug
        build_dir.mkdir(parents=True, exist_ok=True)

        existing_product = self.db.get_product_for_idea(idea_id)
        if existing_product is None:
            product = Product(
                idea_id=idea_id,
                name=idea.title,
                location=str(build_dir),
                status="in_progress",
                metadata={"test_results": {}, "tooling_manifest": {}},
            )
            product_id = self.db.insert_product(product)
        else:
            product_id = existing_product["id"]
            self.db.update_product_status(product_id, "in_progress", json.dumps({"rebuild": True}))

        self._write_app(build_dir, idea.title, spec)
        self.tooling.write_mcp_config(build_dir / "mcp_config.json")
        skill_paths = self.tooling.create_skill_pack(
            build_dir,
            self.tooling.skill_and_tooling_manifest(
                title=idea.title,
                audience=spec.get("audience", ""),
                problem_statement=spec.get("problem_statement", idea.description),
            ),
        )

        test_results = {
            "files_exist": all(
                (build_dir / file_name).exists()
                for file_name in ["index.html", "styles.css", "app.js", "mcp_config.json"]
            ),
            "skill_paths": skill_paths,
        }
        self.db.update_product_status(
            product_id,
            "completed",
            json.dumps(
                {
                    "test_results": test_results,
                    "tooling_manifest": {"skill_paths": skill_paths},
                }
            ),
        )
        self.db.update_idea_status(idea_id, "built")

        await self.send_message(
            to_agent="orchestrator",
            msg_type=MessageType.RESULT,
            payload={
                "product_id": product_id,
                "idea_id": idea_id,
                "location": str(build_dir),
                "status": "completed",
                "rebuilt": existing_product is not None,
            },
            priority=2,
        )
        return {
            "success": True,
            "product_id": product_id,
            "location": str(build_dir),
            "rebuilt": existing_product is not None,
        }

    def _write_app(self, build_dir: Path, title: str, spec: Dict[str, Any]) -> None:
        product_type = spec.get("product_type", "solution")
        if product_type == "workflow-automation":
            self._write_workflow_app(build_dir, title, spec)
        else:
            self._write_pain_solver_app(build_dir, title, spec)

    def _write_common_files(self, build_dir: Path, title: str, description: str) -> None:
        (build_dir / "README.md").write_text(
            f"# {title}\n\n{description}\n\nRun `./scripts/serve.sh` and open http://localhost:4173.\n"
        )

    def _write_pain_solver_app(self, build_dir: Path, title: str, spec: Dict[str, Any]) -> None:
        description = spec.get("value_hypothesis", "")
        features = spec.get("core_features", [])
        audience = spec.get("audience", "")
        problem = spec.get("problem_statement", description)
        monetization = spec.get("monetization_strategy", "")

        index_html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">Validated by recurring customer pain</p>
        <h1>{title}</h1>
        <p class="lede">{description}</p>
        <div class="hero-grid">
          <article class="card">
            <h2>Problem</h2>
            <p>{problem}</p>
          </article>
          <article class="card">
            <h2>Audience</h2>
            <p>{audience}</p>
          </article>
          <article class="card">
            <h2>Business model</h2>
            <p>{monetization}</p>
          </article>
        </div>
      </section>

      <section class="layout">
        <article class="card">
          <h2>ROI estimator</h2>
          <form id="roi-form" class="stack">
            <label>Hours wasted per week <input type="number" name="hours" value="6" min="1" /></label>
            <label>Hourly value ($) <input type="number" name="rate" value="45" min="1" /></label>
            <label>Team members affected <input type="number" name="people" value="1" min="1" /></label>
            <button type="submit">Estimate savings</button>
          </form>
          <div id="roi-result" class="result"></div>
        </article>

        <article class="card">
          <h2>Core workflow</h2>
          <ul class="feature-list">
            {''.join(f'<li>{feature}</li>' for feature in features)}
          </ul>
          <div class="note">
            <strong>What it does:</strong> turns scattered complaints into a concrete workflow and value case.
          </div>
        </article>

        <article class="card">
          <h2>Get the pilot build</h2>
          <form id="lead-form" class="stack">
            <label>Email <input type="email" name="email" placeholder="you@example.com" required /></label>
            <label>Biggest pain right now <textarea name="pain" rows="4" placeholder="What keeps breaking?"></textarea></label>
            <button type="submit">Save interest</button>
          </form>
          <div id="lead-result" class="result"></div>
        </article>
      </section>
    </main>
    <script src="app.js"></script>
  </body>
</html>
"""

        styles = self._base_styles()
        app_js = """
const roiForm = document.getElementById("roi-form");
const roiResult = document.getElementById("roi-result");
const leadForm = document.getElementById("lead-form");
const leadResult = document.getElementById("lead-result");

roiForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(roiForm);
  const hours = Number(form.get("hours") || 0);
  const rate = Number(form.get("rate") || 0);
  const people = Number(form.get("people") || 1);
  const monthly = hours * rate * people * 4;
  roiResult.textContent = `Estimated monthly savings: $${monthly.toFixed(0)}`;
});

leadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(leadForm);
  const payload = {
    email: String(form.get("email") || ""),
    pain: String(form.get("pain") || ""),
    savedAt: new Date().toISOString(),
  };
  const current = JSON.parse(localStorage.getItem("pilot-leads") || "[]");
  current.push(payload);
  localStorage.setItem("pilot-leads", JSON.stringify(current));
  leadResult.textContent = `Saved locally for ${payload.email}.`;
  leadForm.reset();
});
"""

        (build_dir / "index.html").write_text(index_html)
        (build_dir / "styles.css").write_text(styles)
        (build_dir / "app.js").write_text(app_js)
        self._write_common_files(build_dir, title, description)

    def _write_workflow_app(self, build_dir: Path, title: str, spec: Dict[str, Any]) -> None:
        description = spec.get("value_hypothesis", "")
        features = spec.get("core_features", [])
        monetization = spec.get("monetization_strategy", "")

        index_html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">Validated workflow opportunity</p>
        <h1>{title}</h1>
        <p class="lede">{description}</p>
      </section>

      <section class="layout">
        <article class="card">
          <h2>Workflow builder</h2>
          <form id="step-form" class="stack">
            <label>Step name <input name="step" placeholder="Collect inputs" required /></label>
            <label>Automation hint <input name="hint" placeholder="Use AI to classify requests" /></label>
            <button type="submit">Add step</button>
          </form>
          <ol id="steps" class="feature-list"></ol>
        </article>

        <article class="card">
          <h2>What ships in the MVP</h2>
          <ul class="feature-list">
            {''.join(f'<li>{feature}</li>' for feature in features)}
          </ul>
        </article>

        <article class="card">
          <h2>Monetization</h2>
          <p>{monetization}</p>
          <div class="note">The builder stores workflow drafts in local storage, so the prototype is immediately usable.</div>
        </article>
      </section>
    </main>
    <script src="app.js"></script>
  </body>
</html>
"""

        styles = self._base_styles()
        app_js = """
const form = document.getElementById("step-form");
const stepsList = document.getElementById("steps");

const render = () => {
  const steps = JSON.parse(localStorage.getItem("workflow-steps") || "[]");
  stepsList.innerHTML = "";
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${index + 1}. ${step.step}</strong><span>${step.hint}</span>`;
    stepsList.appendChild(item);
  });
};

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const values = new FormData(form);
  const steps = JSON.parse(localStorage.getItem("workflow-steps") || "[]");
  steps.push({
    step: String(values.get("step") || ""),
    hint: String(values.get("hint") || ""),
  });
  localStorage.setItem("workflow-steps", JSON.stringify(steps));
  render();
  form.reset();
});

render();
"""

        (build_dir / "index.html").write_text(index_html)
        (build_dir / "styles.css").write_text(styles)
        (build_dir / "app.js").write_text(app_js)
        self._write_common_files(build_dir, title, description)

    def _base_styles(self) -> str:
        return """
:root {
  --bg: #f5f0e7;
  --panel: #fffaf2;
  --ink: #1f2937;
  --muted: #5b6472;
  --accent: #b45309;
  --accent-strong: #7c2d12;
  --line: rgba(31, 41, 55, 0.12);
  font-family: "Georgia", "Iowan Old Style", serif;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(180, 83, 9, 0.15), transparent 35%),
    linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
}

.shell {
  width: min(1100px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 40px 0 56px;
}

.hero {
  padding: 16px 0 32px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: var(--accent-strong);
}

h1, h2 { margin: 0 0 12px; line-height: 1.05; }
h1 { font-size: clamp(2.4rem, 6vw, 4.8rem); max-width: 10ch; }
h2 { font-size: 1.35rem; }

.lede, .card p, .card li, label, .note, .result {
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.lede {
  max-width: 64ch;
  color: var(--muted);
  font-size: 1.1rem;
}

.hero-grid, .layout {
  display: grid;
  gap: 16px;
}

.hero-grid { grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-top: 24px; }
.layout { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }

.card {
  background: color-mix(in srgb, var(--panel) 90%, white 10%);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 20px;
  box-shadow: 0 18px 40px rgba(124, 45, 18, 0.08);
}

.stack {
  display: grid;
  gap: 12px;
}

label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 0.95rem;
}

input, textarea, button {
  border-radius: 14px;
  border: 1px solid rgba(31, 41, 55, 0.18);
  padding: 12px 14px;
  font: inherit;
}

button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
  color: white;
  border: none;
  cursor: pointer;
}

.feature-list {
  margin: 0;
  padding-left: 20px;
  display: grid;
  gap: 10px;
}

.note, .result {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(180, 83, 9, 0.08);
  color: var(--ink);
}

@media (max-width: 720px) {
  .shell { padding-top: 24px; }
}
"""
