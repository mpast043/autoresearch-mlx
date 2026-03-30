#!/usr/bin/env uv run python
import json
import sqlite3
from pathlib import Path

conn = sqlite3.connect('data/autoresearch.db')
cur = conn.cursor()

def base_styles():
    return '''
:root {
  --bg: #0f0f0f;
  --fg: #fafafa;
  --muted: #737373;
  --border: #2e2e2e;
  --card: #1a1a1a;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --success: #22c55e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.6; }
.shell { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
.hero { text-align: center; margin-bottom: 60px; }
.eyebrow { color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
h1 { font-size: 36px; margin-bottom: 16px; }
.lede { color: var(--muted); font-size: 18px; max-width: 600px; margin: 0 auto 40px; }
.hero-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }
.layout { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
.card h2 { font-size: 18px; margin-bottom: 16px; color: var(--fg); }
.stack { display: flex; flex-direction: column; gap: 12px; }
label { display: flex; flex-direction: column; gap: 6px; font-size: 14px; color: var(--muted); }
input, textarea { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; color: var(--fg); font-size: 14px; }
input:focus, textarea:focus { outline: none; border-color: var(--accent); }
button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 12px 20px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
button:hover { background: var(--accent-hover); }
.result { margin-top: 12px; padding: 12px; background: rgba(34, 197, 94, 0.1); border-radius: 8px; color: var(--success); font-weight: 500; }
.feature-list { list-style: none; }
.feature-list li { padding: 8px 0; border-bottom: 1px solid var(--border); }
.feature-list li:last-child { border-bottom: none; }
.note { margin-top: 16px; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; font-size: 13px; color: var(--muted); }
.note strong { color: var(--accent); }
'''

for idea_id in [1, 2]:
    cur.execute('SELECT title, spec_json FROM ideas WHERE id = ?', (idea_id,))
    row = cur.fetchone()
    if not row:
        continue

    title, spec_json = row
    spec = json.loads(spec_json) if spec_json else {}

    slug = spec.get('slug', f'idea-{idea_id}')
    build_dir = Path(f'output/builds/{slug}')

    description = spec.get('value_hypothesis', 'Solution to reduce workflow failures')
    audience = spec.get('audience', '')
    problem = spec.get('problem_statement', description)
    features = spec.get('core_features', ['Automated workflow detection', 'Problem-solution matching', 'Value calculation'])

    index_html = f'''<!doctype html>
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
            <h2>Target Audience</h2>
            <p>{audience}</p>
          </article>
          <article class="card">
            <h2>Value Proposition</h2>
            <p>Reduce recurring workflow failures before replacing entire systems</p>
          </article>
        </div>
      </section>

      <section class="layout">
        <article class="card">
          <h2>ROI Calculator</h2>
          <form id="roi-form" class="stack">
            <label>Hours wasted per week <input type="number" name="hours" value="6" min="1" /></label>
            <label>Hourly value ($) <input type="number" name="rate" value="45" min="1" /></label>
            <label>Team members affected <input type="number" name="people" value="1" min="1" /></label>
            <button type="submit">Calculate Savings</button>
          </form>
          <div id="roi-result" class="result"></div>
        </article>

        <article class="card">
          <h2>Core Features</h2>
          <ul class="feature-list">
            {''.join(f'<li>{feature}</li>' for feature in features)}
          </ul>
          <div class="note">
            <strong>What it does:</strong> transforms scattered complaints into concrete workflow solutions.
          </div>
        </article>

        <article class="card">
          <h2>Get Early Access</h2>
          <form id="lead-form" class="stack">
            <label>Email <input type="email" name="email" placeholder="you@example.com" required /></label>
            <label>Your biggest pain point <textarea name="pain" rows="4" placeholder="What keeps breaking?"></textarea></label>
            <button type="submit">Request Access</button>
          </form>
          <div id="lead-result" class="result"></div>
        </article>
      </section>
    </main>
    <script src="app.js"></script>
  </body>
</html>'''

    app_js = '''
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
  const yearly = monthly * 12;
  roiResult.textContent = "Monthly: $" + monthly.toLocaleString() + " | Yearly: $" + yearly.toLocaleString();
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
  leadResult.textContent = "Thanks! You've been added to the early access list.";
  leadForm.reset();
});
'''

    (build_dir / 'index.html').write_text(index_html)
    (build_dir / 'styles.css').write_text(base_styles())
    (build_dir / 'app.js').write_text(app_js)

    print(f'Built {slug}')
    print(f'  Files: {list(build_dir.glob("*"))}')
    print()