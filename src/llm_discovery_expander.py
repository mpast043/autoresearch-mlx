"""LLM-driven autonomous discovery expansion.

After each validation cycle, analyses validated opportunities and proposes
new adjacent problem spaces to explore.  Each problem space generates its
own derived search queries (keywords, subreddits, web queries).

Falls back to keyword-based expansion when the LLM is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import asyncio
from collections import Counter
from typing import Any

from src.problem_space import (
    EXPLORING,
    ProblemSpace,
    ProblemSpaceTerm,
    TERM_KEYWORD,
    TERM_SUBREDDIT,
    TERM_WEB_QUERY,
    TERM_GITHUB_QUERY,
)
from src.problem_space_lifecycle import ProblemSpaceLifecycleManager

logger = logging.getLogger(__name__)


_SUBREDDIT_NAME_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")


def _normalize_string_list(value: Any, *, kind: str = "term", limit: int | None = None) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        items = [piece for piece in re.split(r"[\n,]+", value) if piece]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if kind == "subreddit":
            text = re.sub(r"^/?r/", "", text, flags=re.IGNORECASE).strip()
            if not _SUBREDDIT_NAME_RE.fullmatch(text):
                continue
        if text not in normalized:
            normalized.append(text)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


# ---------------------------------------------------------------------------
# System and user prompts
# ---------------------------------------------------------------------------

SPACE_PROPOSAL_SYSTEM = """\
You are a problem-space discovery engine for a product research pipeline.
Your job is to analyse validated SaaS/plugin/microSaaS opportunities and
propose NEW adjacent problem spaces to explore.

Rules:
- Each problem space must be a NARROW, SPECIFIC problem domain that could
  sustain a plugin, add-on, or microSaaS product.
- DEEPEN WINNING LANES instead of proposing adjacent markets.
- Stay in the SAME buyer/workflow lane as the supplied opportunities and
  strong findings. Preserve buyer type, workflow phase, data artifacts, and
  incumbent systems where possible.
- Prefer tighter variants such as: same workflow but different channel,
  same reconciliation but different document type, same close process but a
  sharper exception case.
- AVOID vague spaces like "productivity", "workflow automation", "data sync".
- DO NOT jump to different industries, compliance domains, construction,
  facilities, warranty tracking, or other neighboring admin categories unless
  the supplied lane anchors explicitly point there.
- PREFER spaces with: concrete failure modes, specific user roles, clear
  trigger events, measurable consequences (costs, errors, delays).
- Each space MUST include 5-10 specific search keywords, 3-5 relevant
  subreddits, and 3-5 web search queries.
- Spaces MUST be same-lane narrowings of current validated opportunities or
  strong findings, NOT adjacent/divergent market brainstorms.
- Output ONLY valid JSON. No markdown. No prose."""

SPACE_PROPOSAL_USER = """\
## Current Validated Opportunities
{opportunities_table}

## Strong Recent Pain Findings
{strong_findings_table}

## Winning Lane Anchors
{lane_anchor_table}

## Currently Exploring Problem Spaces
{current_spaces_table}

## Exhausted/Archived Spaces (DO NOT re-propose these)
{exhausted_spaces_table}

## Search Coverage So Far
{search_coverage}

Based on the above, propose 2-4 NEW problem spaces that DEEPEN the same lane.
Focus on:
1. Tighter failure modes inside the same buyer/workflow lane
2. Different channels, edge cases, or exception paths within the same workflow
3. Underserved same-lane variants where current keywords have low coverage

Output ONLY the JSON object with this exact structure:
{{
  "proposed_spaces": [
    {{
      "space_key": "slug_identifier",
      "label": "Human-Readable Label",
      "description": "1-2 sentence semantic description",
      "parent_space_key": null,
      "semantic_summary": "What this space is about and why it matters",
      "keywords": ["keyword1", "keyword2", "..."],
      "subreddits": ["subreddit1", "..."],
      "web_queries": ["query1", "..."],
      "github_queries": ["query1", "..."],
      "adjacent_spaces": ["adjacent_space_key1", "..."],
      "rationale": "Why this space is worth exploring now"
    }}
  ]
}}"""

QUERY_DERIVATION_SYSTEM = """\
You are generating search queries for problem-space discovery.
Given a problem space, produce specific search keywords, relevant subreddits,
web search queries, and GitHub issue queries that would find people
experiencing this problem.

Rules:
- Keywords should be specific pain expressions, not generic terms
- Subreddits should be communities where affected users congregate
- Web queries should target forum posts, blog complaints, support threads
- GitHub queries should find real issues describing the problem

Output ONLY valid JSON:
{{
  "keywords": ["..."],
  "subreddits": ["..."],
  "web_queries": ["..."],
  "github_queries": ["..."]
}}"""

QUERY_DERIVATION_USER = """\
Problem Space: {label}
Description: {description}
Semantic Summary: {semantic_summary}

Generate search queries for this problem space."""

STRONG_FINDING_OPERATIONAL_TERMS = [
    "reconcile",
    "reconciliation",
    "bank deposit",
    "bank deposits",
    "payout",
    "payouts",
    "ledger",
    "month end",
    "month-end",
    "import",
    "imports",
    "export",
    "exports",
    "invoice",
    "invoices",
    "payment",
    "payments",
    "order",
    "orders",
    "inventory",
    "approval",
    "approvals",
    "handoff",
    "handoffs",
    "label",
    "labels",
    "returns",
    "supplier data",
    "csv",
    "spreadsheet",
]

STRONG_FINDING_FAILURE_TERMS = [
    "duplicate",
    "duplicates",
    "duplicated",
    "mismatch",
    "not matching",
    "wrong",
    "error",
    "errors",
    "broken",
    "fails",
    "failed",
    "manual",
    "manually",
    "copy paste",
    "copy/paste",
    "workaround",
    "cleanup",
    "staring at spreadsheets",
    "rebuild",
]

STRONG_FINDING_BUSINESS_RISK_TERMS = [
    "biggest client",
    "major client",
    "client concentration",
    "revenue concentration",
    "owes me",
    "wholesale account",
    "good fit for me",
    "review my resume",
    "resume review",
    "resume roast",
    "career advice",
    "hiring first sales",
    "first sales",
    "structuring msp business",
    "se fue un cliente",
]

LANE_TOKEN_STOPWORDS = {
    "about",
    "across",
    "after",
    "almost",
    "around",
    "automation",
    "automations",
    "business",
    "businesses",
    "cleanup",
    "close",
    "company",
    "customer",
    "customers",
    "daily",
    "data",
    "doing",
    "every",
    "error",
    "errors",
    "finance",
    "founder",
    "founders",
    "issue",
    "issues",
    "manual",
    "manually",
    "monthly",
    "operator",
    "operators",
    "operations",
    "owner",
    "owners",
    "pain",
    "problem",
    "problems",
    "process",
    "processes",
    "software",
    "spreadsheet",
    "spreadsheets",
    "still",
    "system",
    "systems",
    "team",
    "teams",
    "their",
    "tool",
    "tools",
    "using",
    "workflow",
    "workflows",
}


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified LLM client supporting Ollama and Anthropic providers.

    Follows the same patterns as builder_v2.py (async) and
    build_prep.py (sync) for provider dispatch.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        llm_config = config.get("llm", {})
        self.provider = llm_config.get("provider", "ollama")
        self.model = llm_config.get("model", "gemma4:latest")
        self.base_url = llm_config.get("base_url", "http://localhost:11434")
        self.api_key = llm_config.get("api_key", "") or ""
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.temperature = llm_config.get("temperature", 0.3)
        self.timeout = llm_config.get("timeout", 120)

    def generate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Synchronous generation. Returns raw text or None on failure."""
        if self.provider == "ollama":
            return self._ollama_generate(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._anthropic_generate(system_prompt, user_prompt)
        else:
            # auto: try Ollama first, fall back to Anthropic
            result = self._ollama_generate(system_prompt, user_prompt)
            if result is not None:
                return result
            return self._anthropic_generate(system_prompt, user_prompt)

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Async generation. Returns raw text or None on failure."""
        if self.provider == "ollama":
            return await self._ollama_agenerate(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return await asyncio.to_thread(self._anthropic_generate, system_prompt, user_prompt)
        else:
            result = await self._ollama_agenerate(system_prompt, user_prompt)
            if result is not None:
                return result
            return await asyncio.to_thread(self._anthropic_generate, system_prompt, user_prompt)

    def _ollama_generate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Synchronous Ollama call via /api/chat endpoint."""
        import urllib.request
        import urllib.error

        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(request_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"Ollama generation failed: {e}")
            return None

    async def _ollama_agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Async Ollama call via /api/chat endpoint."""
        import aiohttp

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"Ollama async error {resp.status}: {error_text[:200]}")
                        return None
                    result = await resp.json()
                    return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Ollama async generation failed: {e}")
            return None

    def _anthropic_generate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Synchronous Anthropic call via SDK (same pattern as build_prep.py)."""
        if not self.api_key:
            # Try env vars as fallback
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY", "")
        if not self.api_key:
            logger.warning("No Anthropic API key configured")
            return None

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic generation failed: {e}")
            return None


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract JSON from an LLM response, handling markdown and partial output.

    Follows the same multi-strategy pattern as build_prep.py's
    _parse_platform_fit_response.
    """
    if not raw:
        return None

    cleaned = raw.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code blocks
    if "```json" in cleaned:
        try:
            block = cleaned.split("```json")[1].split("```")[0]
            return json.loads(block.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    if "```" in cleaned:
        try:
            block = cleaned.split("```")[1].split("```")[0]
            return json.loads(block.strip())
        except (json.JSONDecodeError, IndexError):
            pass

    # Strategy 3: Find largest { ... } block (greedy, handles nested objects)
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find smaller { ... } block (non-greedy, for flat objects)
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _is_operational_strong_finding(finding: Any) -> bool:
    title = str(getattr(finding, "product_built", "") or "")
    summary = str(getattr(finding, "outcome_summary", "") or "")
    text = f"{title} {summary}".lower()
    if any(term in text for term in STRONG_FINDING_BUSINESS_RISK_TERMS):
        has_operational_terms = any(term in text for term in STRONG_FINDING_OPERATIONAL_TERMS)
        has_failure_terms = any(term in text for term in STRONG_FINDING_FAILURE_TERMS)
        if not (has_operational_terms and has_failure_terms):
            return False
    return True


def _lane_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]{4,}", (text or "").lower())
    return [token for token in tokens if token not in LANE_TOKEN_STOPWORDS]


def _build_lane_signature(context: dict[str, Any]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    anchor_sources: list[str] = []

    for opp in context.get("opportunities", []):
        title = str(opp.get("title", "") or "")
        counter.update(_lane_tokens(title))
        if title:
            anchor_sources.append(title[:120])

    for finding in context.get("strong_findings", []):
        title = str(finding.get("title", "") or "")
        summary = str(finding.get("summary", "") or "")
        counter.update(_lane_tokens(title))
        counter.update(_lane_tokens(summary))
        if title:
            anchor_sources.append(title[:120])

    anchor_terms = [term for term, _count in counter.most_common(12)]
    return {
        "anchor_terms": anchor_terms,
        "anchor_titles": anchor_sources[:6],
    }


def _proposal_matches_lane(proposal: dict[str, Any], lane_signature: dict[str, Any] | None) -> bool:
    anchor_terms = list((lane_signature or {}).get("anchor_terms") or [])
    if len(anchor_terms) < 2:
        return True

    proposal_parts = [
        str(proposal.get("space_key", "") or ""),
        str(proposal.get("label", "") or ""),
        str(proposal.get("description", "") or ""),
        str(proposal.get("semantic_summary", "") or ""),
        " ".join(_normalize_string_list(proposal.get("keywords", []))),
        " ".join(_normalize_string_list(proposal.get("web_queries", []))),
        " ".join(_normalize_string_list(proposal.get("github_queries", []))),
    ]
    proposal_text = " ".join(part for part in proposal_parts if part).lower()
    overlap = [term for term in anchor_terms if term in proposal_text]
    return len(set(overlap)) >= 2


# ---------------------------------------------------------------------------
# LLMDiscoveryExpander
# ---------------------------------------------------------------------------

class LLMDiscoveryExpander:
    """LLM-driven problem-space discovery expansion.

    After each validation cycle, analyses validated opportunities and
    proposes new adjacent problem spaces to explore.  Each problem space
    generates its own derived search queries.
    """

    def __init__(self, db: Any, config: dict[str, Any]) -> None:
        self.db = db
        self.config = config
        self.llm_config = config.get("llm", {})
        self.expansion_config = config.get("discovery", {}).get("llm_expansion", {})
        self.client = LLMClient(config)
        self.lifecycle = ProblemSpaceLifecycleManager(db, config)

    def gather_context(self) -> dict[str, Any]:
        """Gather structured context for the LLM prompt.

        Returns dict with validated opportunities, current spaces,
        exhausted spaces, and search coverage.
        """
        # Validated opportunities (top N by score)
        opportunities = self.db.get_opportunities(limit=20)
        opp_rows = []
        prototype_candidate_count = 0
        for opp in opportunities:
            selection_status = getattr(opp, "selection_status", "") or ""
            if selection_status == "prototype_candidate":
                prototype_candidate_count += 1
            opp_rows.append({
                "id": opp.id,
                "title": opp.title or "",
                "composite_score": getattr(opp, "composite_score", 0) or 0,
                "selection_status": selection_status,
                "cluster_id": getattr(opp, "cluster_id", 0),
            })

        min_validated = int(self.expansion_config.get("min_validated_for_promoted", 1) or 1)
        min_prototypes = int(self.expansion_config.get("min_prototype_candidates", 1) or 1)

        strong_finding_rows: list[dict[str, Any]] = []
        should_include_strong_findings = len(opp_rows) < min_validated or prototype_candidate_count < min_prototypes
        if should_include_strong_findings:
            findings = self.db.get_findings(limit=80)
            for finding in findings:
                if finding.source_class != "pain_signal":
                    continue
                if finding.status not in {"promoted", "qualified"}:
                    continue
                if not _is_operational_strong_finding(finding):
                    continue
                evidence = dict(finding.evidence or {})
                high_leverage = evidence.get("high_leverage", {}) if isinstance(evidence.get("high_leverage", {}), dict) else {}
                hl_score = float(high_leverage.get("score", 0.0) or 0.0)
                hl_status = str(high_leverage.get("status", "") or "")
                if finding.status != "promoted" and hl_score < 0.6 and hl_status not in {"candidate", "confirmed"}:
                    continue
                strong_finding_rows.append(
                    {
                        "id": finding.id,
                        "title": finding.product_built or finding.outcome_summary[:120],
                        "summary": finding.outcome_summary[:220],
                        "status": finding.status,
                        "hl_score": hl_score,
                        "hl_status": hl_status or ("candidate" if hl_score >= 0.62 else "thin"),
                    }
                )
                if len(strong_finding_rows) >= 8:
                    break

        # Current problem spaces (exploring + validated)
        active_spaces = self.db.list_problem_spaces(limit=25)
        space_rows = []
        for space in active_spaces:
            space_rows.append({
                "space_key": space.space_key,
                "label": space.label,
                "status": space.status,
                "yield_score": space.yield_score,
                "findings": space.total_findings,
                "validations": space.total_validations,
            })

        # Exhausted/archived spaces (to avoid re-proposing)
        exhausted_keys = self.lifecycle.get_exhausted_space_keys()

        # Search coverage: keywords and subreddits already in use
        terms = self.db.list_search_terms(limit=200) if hasattr(self.db, "list_search_terms") else []
        keywords = [t.get("term_value", "") for t in terms if t.get("term_type") == "keyword"][:30]
        subreddits = [t.get("term_value", "") for t in terms if t.get("term_type") == "subreddit"][:15]

        return {
            "opportunities": opp_rows,
            "strong_findings": strong_finding_rows,
            "active_spaces": space_rows,
            "exhausted_space_keys": exhausted_keys,
            "lane_signature": _build_lane_signature(
                {
                    "opportunities": opp_rows,
                    "strong_findings": strong_finding_rows,
                }
            ),
            "search_coverage": {
                "keywords_sample": keywords,
                "subreddits_sample": subreddits,
            },
        }

    def build_proposal_prompt(self, context: dict[str, Any]) -> tuple[str, str]:
        """Build system and user prompts for space proposal."""
        # Format opportunities table
        opp_lines = []
        for opp in context["opportunities"]:
            opp_lines.append(
                f"  #{opp['id']} [{opp.get('selection_status', '?')}] "
                f"score={opp['composite_score']:.2f}: {opp['title'][:80]}"
            )
        opportunities_table = "\n".join(opp_lines) if opp_lines else "No validated opportunities yet."

        finding_lines = []
        for finding in context.get("strong_findings", []):
            finding_lines.append(
                f"  #{finding['id']} [{finding.get('status', '?')}/{finding.get('hl_status', '?')}] "
                f"hl={finding.get('hl_score', 0.0):.2f}: {finding['title'][:80]} :: {finding.get('summary', '')[:120]}"
            )
        strong_findings_table = "\n".join(finding_lines) if finding_lines else "No strong recent pain findings."

        lane_signature = context.get("lane_signature", {}) or {}
        lane_terms = ", ".join(lane_signature.get("anchor_terms", [])[:10]) or "No lane anchors yet."
        lane_titles = "\n".join(f"  - {title}" for title in lane_signature.get("anchor_titles", [])[:4])
        lane_anchor_table = f"Anchor terms: {lane_terms}\nExample source titles:\n{lane_titles or '  - None'}"

        # Format current spaces table
        space_lines = []
        for space in context["active_spaces"]:
            space_lines.append(
                f"  {space['space_key']} [{space['status']}] "
                f"yield={space['yield_score']:.2f} "
                f"findings={space['findings']} validations={space['validations']}: "
                f"{space['label']}"
            )
        current_spaces_table = "\n".join(space_lines) if space_lines else "No active problem spaces yet."

        # Format exhausted spaces
        exhausted = context["exhausted_space_keys"]
        exhausted_table = ", ".join(exhausted) if exhausted else "None"

        # Search coverage
        coverage = context.get("search_coverage", {})
        kw_sample = ", ".join(coverage.get("keywords_sample", [])[:15])
        sub_sample = ", ".join(coverage.get("subreddits_sample", [])[:10])
        search_coverage = f"Keywords: {kw_sample}\nSubreddits: {sub_sample}"

        user_prompt = SPACE_PROPOSAL_USER.format(
            opportunities_table=opportunities_table,
            strong_findings_table=strong_findings_table,
            lane_anchor_table=lane_anchor_table,
            current_spaces_table=current_spaces_table,
            exhausted_spaces_table=exhausted_table,
            search_coverage=search_coverage,
        )

        return SPACE_PROPOSAL_SYSTEM, user_prompt

    def parse_proposals(self, raw: str, lane_signature: dict[str, Any] | None = None) -> list[ProblemSpace]:
        """Parse LLM response into ProblemSpace objects."""
        data = _extract_json(raw)
        if not data:
            logger.warning("Could not parse LLM proposal response")
            return []

        proposals = data.get("proposed_spaces", [])
        if not proposals and isinstance(data, dict):
            # Maybe the response is a single space
            if "space_key" in data:
                proposals = [data]

        spaces = []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            if not _proposal_matches_lane(proposal, lane_signature):
                logger.info(
                    "Rejected out-of-lane LLM problem space proposal: %s",
                    proposal.get("space_key") or proposal.get("label") or "<unknown>",
                )
                continue
            space_key = proposal.get("space_key", "")
            if not space_key:
                continue
            # Normalize space_key
            space_key = re.sub(r'[^a-z0-9]+', '_', space_key.lower().strip())[:64].strip('_')
            if not space_key:
                continue

            space = ProblemSpace(
                space_key=space_key,
                label=proposal.get("label", space_key),
                description=proposal.get("description", ""),
                semantic_summary=proposal.get("semantic_summary", ""),
                adjacent_spaces_json=json.dumps(_normalize_string_list(proposal.get("adjacent_spaces", []))),
                keywords_json=json.dumps(_normalize_string_list(proposal.get("keywords", []))),
                subreddits_json=json.dumps(_normalize_string_list(proposal.get("subreddits", []), kind="subreddit")),
                web_queries_json=json.dumps(_normalize_string_list(proposal.get("web_queries", []))),
                github_queries_json=json.dumps(_normalize_string_list(proposal.get("github_queries", []))),
                source="llm",
                llm_provider=self.client.provider,
                llm_model=self.client.model,
            )
            spaces.append(space)

        return spaces

    async def generate_derived_queries(self, space: ProblemSpace) -> ProblemSpace:
        """Generate derived search queries for a problem space via LLM.

        If the space already has keywords, skip. Otherwise, call the LLM
        to produce keywords, subreddits, web queries, and GitHub queries.
        """
        # Only generate if the space has no keywords yet
        if space.keywords and space.subreddits:
            return space

        max_kw = self.expansion_config.get("max_keywords_per_space", 10)
        max_sub = self.expansion_config.get("max_subreddits_per_space", 5)
        max_web = self.expansion_config.get("max_web_queries_per_space", 5)
        max_gh = self.expansion_config.get("max_github_queries_per_space", 3)

        user_prompt = QUERY_DERIVATION_USER.format(
            label=space.label,
            description=space.description,
            semantic_summary=space.semantic_summary,
        )

        # Add limits to the user prompt
        user_prompt += f"\n\nLimits: max {max_kw} keywords, {max_sub} subreddits, {max_web} web queries, {max_gh} github queries."

        raw = await self.client.agenerate(QUERY_DERIVATION_SYSTEM, user_prompt)
        if not raw:
            logger.info("Async query derivation failed, falling back to sync")
            raw = self.client.generate(QUERY_DERIVATION_SYSTEM, user_prompt)
        if raw:
            data = _extract_json(raw)
            if data:
                space.keywords = _normalize_string_list(data.get("keywords", []), limit=max_kw)
                space.subreddits = _normalize_string_list(data.get("subreddits", []), kind="subreddit", limit=max_sub)
                space.web_queries = _normalize_string_list(data.get("web_queries", []), limit=max_web)
                space.github_queries = _normalize_string_list(data.get("github_queries", []), limit=max_gh)
                return space

        # Fallback: derive minimal queries from the label/description
        logger.info(f"LLM query derivation failed for '{space.space_key}', using fallback")
        return self._fallback_derive_queries(space)

    def _fallback_derive_queries(self, space: ProblemSpace) -> ProblemSpace:
        """Derive minimal queries from the space label when LLM is unavailable."""
        label_words = [w.lower() for w in re.findall(r'[a-zA-Z]{3,}', space.label)]
        desc_words = [w.lower() for w in re.findall(r'[a-zA-Z]{3,}', space.description)]

        # Combine label and description words as keywords
        all_words = list(dict.fromkeys(label_words + desc_words))[:10]
        if all_words and not space.keywords:
            space.keywords = all_words

        # Use label words as web queries
        if label_words and not space.web_queries:
            space.web_queries = [" ".join(label_words[:4]) + " pain point problem"]

        return space

    def register_space_and_terms(self, space: ProblemSpace) -> ProblemSpace:
        """Persist a ProblemSpace and its derived terms to the database.

        Also registers derived keywords/subreddits in the existing
        discovery_search_terms table for backward compatibility.
        """
        # Check for duplicate
        existing = self.db.get_problem_space(space.space_key)
        if existing and existing.generation_prompt_hash == space.generation_prompt_hash:
            logger.info(f"ProblemSpace '{space.space_key}' already exists with same prompt hash, skipping")
            return existing

        # Persist the space
        self.db.upsert_problem_space(space)

        # Persist derived terms
        for kw in space.keywords:
            self.db.add_problem_space_term(space.space_key, TERM_KEYWORD, kw, source="derived")
        for sub in space.subreddits:
            self.db.add_problem_space_term(space.space_key, TERM_SUBREDDIT, sub, source="derived")
        for wq in space.web_queries:
            self.db.add_problem_space_term(space.space_key, TERM_WEB_QUERY, wq, source="derived")
        for gq in space.github_queries:
            self.db.add_problem_space_term(space.space_key, TERM_GITHUB_QUERY, gq, source="derived")

        # Also register in discovery_search_terms for backward compat
        lifecycle_mgr = self.lifecycle
        for kw in space.keywords:
            try:
                lifecycle_mgr.db.list_search_terms(term_type="keyword", limit=1)  # ensure table exists
            except Exception:
                pass
            # Use ensure_term_exists if available, otherwise skip silently
            if hasattr(lifecycle_mgr, "ensure_term_exists"):
                lifecycle_mgr.ensure_term_exists("keyword", kw, source="problem_space")

        for sub in space.subreddits:
            if hasattr(lifecycle_mgr, "ensure_term_exists"):
                lifecycle_mgr.ensure_term_exists("subreddit", sub, source="problem_space")

        return space

    def _compute_prompt_hash(self, context: dict[str, Any]) -> str:
        """Compute a hash of the core prompt content for deduplication."""
        opp_ids = sorted(o["id"] for o in context.get("opportunities", []))
        finding_ids = sorted(f["id"] for f in context.get("strong_findings", []))
        space_keys = sorted(s["space_key"] for s in context.get("active_spaces", []))
        content = json.dumps({"opp_ids": opp_ids, "finding_ids": finding_ids, "space_keys": space_keys}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def expand_after_validation(self) -> list[ProblemSpace]:
        """Main entry point. Called after validation cycle completes.

        1. Gather current validated opportunities and their clusters
        2. Call LLM with structured prompt
        3. Parse LLM response into ProblemSpace proposals
        4. For each proposed space, generate derived queries
        5. Persist new problem spaces with derived queries
        6. Return new spaces for the next discovery wave
        """
        max_proposed = self.expansion_config.get("max_proposed_spaces", 4)

        # Step 1: Gather context
        context = self.gather_context()

        # If neither validated opportunities nor strong recent findings exist, skip expansion
        if not context["opportunities"] and not context.get("strong_findings"):
            logger.info("No validated opportunities or strong recent findings yet, skipping LLM expansion")
            return []

        # Step 2: Build prompt and call LLM
        system_prompt, user_prompt = self.build_proposal_prompt(context)
        prompt_hash = self._compute_prompt_hash(context)

        raw = await self.client.agenerate(system_prompt, user_prompt)
        if not raw:
            # Fallback to sync generation if async failed (e.g. event loop conflict)
            logger.info("Async LLM call failed, falling back to sync")
            raw = self.client.generate(system_prompt, user_prompt)
        if not raw:
            logger.warning("LLM expansion failed, no response received")
            return []

        # Step 3: Parse proposals
        proposed_spaces = self.parse_proposals(raw, context.get("lane_signature"))
        if not proposed_spaces:
            logger.info("LLM proposed no new problem spaces")
            return []

        # Limit to max_proposed_spaces
        proposed_spaces = proposed_spaces[:max_proposed]

        # Step 4: For each proposed space, generate derived queries
        new_spaces = []
        for space in proposed_spaces:
            # Skip if space already exists with same hash
            space.generation_prompt_hash = prompt_hash
            existing = self.db.get_problem_space(space.space_key)
            if existing and existing.generation_prompt_hash == prompt_hash:
                logger.info(f"ProblemSpace '{space.space_key}' already exists with same prompt hash, skipping")
                continue

            # Generate derived queries via LLM
            space = await self.generate_derived_queries(space)

            # Step 5: Persist
            space = self.register_space_and_terms(space)
            new_spaces.append(space)
            logger.info(
                f"LLM expansion created problem space '{space.space_key}' "
                f"with {len(space.keywords)} keywords, {len(space.subreddits)} subreddits"
            )

        return new_spaces


# ---------------------------------------------------------------------------
# LLM-enhanced atom extraction
# ---------------------------------------------------------------------------

ATOM_EXTRACT_SYSTEM = """\
You are a structured problem-atom extraction engine. Given raw text from a user
complaint or discussion, extract the following fields as JSON:

- user_role: WHO is experiencing this problem (e.g., "ad operations manager",
  "small business owner", "bookkeeper"). Be specific, not generic like "user".
- job_to_be_done: What they are TRYING to accomplish (e.g., "reconcile monthly
  bank statements with QuickBooks"). Be specific about the workflow.
- trigger_event: WHEN this problem occurs (e.g., "at month-end close",
  "when importing CSV files"). Must be a moment or situation, not generic.
- pain_statement: The core problem in their words, paraphrased to be clear
  (e.g., "Spending 4+ hours manually adjusting bids across 40+ campaigns
  after every budget overrun"). NOT the raw title.
- failure_mode: What specifically breaks or goes wrong (e.g., "single bad CSV
  row silently corrupts inventory counts"). Must be concrete, not "it doesn't work".
- current_workaround: What they do instead (e.g., "manual copy-paste between
  spreadsheets", "checking each row by hand"). Empty string if not mentioned.
- consequence: The cost or impact (e.g., "3 hours lost per week",
  "client-facing errors in invoices"). Empty string if not mentioned.

Rules:
- Extract ONLY what is clearly stated or strongly implied. Do NOT invent details.
- If a field cannot be determined, use an empty string "".
- Be SPECIFIC. "data sync issue" is bad. "QuickBooks invoices don't match
  Stripe payouts during reconciliation" is good.
- Output ONLY valid JSON. No markdown. No prose."""

ATOM_EXTRACT_USER = """\
Raw text:
{raw_text}

Current heuristic extraction (may have empty or poor fields):
- user_role: {user_role}
- job_to_be_done: {job_to_be_done}
- trigger_event: {trigger_event}
- pain_statement: {pain_statement}
- failure_mode: {failure_mode}
- current_workaround: {current_workaround}

Extract the structured problem atom. Fill in empty fields and improve vague ones."""


class LLMAtomExtractor:
    """LLM-enhanced problem atom extraction.

    Tries LLM first, falls back to heuristic extraction unchanged.
    Only replaces fields that the LLM improves (non-empty and more specific).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.client = LLMClient(config)

    def extract(self, raw_text: str, heuristic_atom: dict[str, Any]) -> dict[str, Any]:
        """Try LLM extraction, falling back to heuristic on failure.

        Returns the atom dict with fields improved where the LLM provided
        better values. Heuristic fields are preserved if the LLM fails
        or provides empty/worse values.
        """
        # Truncate raw text to avoid excessive token usage
        text = raw_text[:2000] if len(raw_text) > 2000 else raw_text

        user_prompt = ATOM_EXTRACT_USER.format(
            raw_text=text,
            user_role=heuristic_atom.get("user_role", ""),
            job_to_be_done=heuristic_atom.get("job_to_be_done", ""),
            trigger_event=heuristic_atom.get("trigger_event", ""),
            pain_statement=heuristic_atom.get("pain_statement", ""),
            failure_mode=heuristic_atom.get("failure_mode", ""),
            current_workaround=heuristic_atom.get("current_workaround", ""),
        )

        raw = self.client.generate(ATOM_EXTRACT_SYSTEM, user_prompt)
        if not raw:
            logger.info("LLM atom extraction failed, keeping heuristic values")
            return heuristic_atom

        data = _extract_json(raw)
        if not data:
            logger.info("LLM atom extraction returned unparseable JSON, keeping heuristic values")
            return heuristic_atom

        # Merge: only replace heuristic values with LLM values that are
        # non-empty and more specific (longer or contains more detail)
        result = dict(heuristic_atom)  # start with heuristic
        field_map = {
            "user_role": "user_role",
            "job_to_be_done": "job_to_be_done",
            "trigger_event": "trigger_event",
            "pain_statement": "pain_statement",
            "failure_mode": "failure_mode",
            "current_workaround": "current_workaround",
            "consequence": "cost_consequence_clues",  # maps to existing field
        }

        # Generic phrases that indicate heuristic extraction failed
        _GENERIC_PHRASES = {"welcome to", "use this", "forum", "find customizable",
                            "this is an", "here is", "this spreadsheet"}

        for llm_key, atom_key in field_map.items():
            llm_val = str(data.get(llm_key, "")).strip()
            if not llm_val:
                continue

            heuristic_val = result.get(atom_key, "")
            # Replace if heuristic is empty, generic, or LLM is more specific
            heuristic_is_generic = any(
                phrase in heuristic_val.lower() for phrase in _GENERIC_PHRASES
            )
            if not heuristic_val or heuristic_is_generic or len(llm_val) > len(heuristic_val) * 1.1:
                result[atom_key] = llm_val

        # Handle consequence separately (it maps to cost_consequence_clues)
        consequence = str(data.get("consequence", "")).strip()
        if consequence:
            existing = result.get("cost_consequence_clues", "")
            if not existing or existing == consequence:
                result["cost_consequence_clues"] = consequence

        result["atom_extraction_method"] = "llm_enhanced"
        return result

    async def aextract(self, raw_text: str, heuristic_atom: dict[str, Any]) -> dict[str, Any]:
        """Async version of extract."""
        text = raw_text[:2000] if len(raw_text) > 2000 else raw_text

        user_prompt = ATOM_EXTRACT_USER.format(
            raw_text=text,
            user_role=heuristic_atom.get("user_role", ""),
            job_to_be_done=heuristic_atom.get("job_to_be_done", ""),
            trigger_event=heuristic_atom.get("trigger_event", ""),
            pain_statement=heuristic_atom.get("pain_statement", ""),
            failure_mode=heuristic_atom.get("failure_mode", ""),
            current_workaround=heuristic_atom.get("current_workaround", ""),
        )

        raw = await self.client.agenerate(ATOM_EXTRACT_SYSTEM, user_prompt)
        if not raw:
            logger.info("LLM async atom extraction failed, trying sync")
            return self.extract(raw_text, heuristic_atom)

        data = _extract_json(raw)
        if not data:
            logger.info("LLM atom extraction returned unparseable JSON, keeping heuristic values")
            return heuristic_atom

        # Same merge logic as sync version
        result = dict(heuristic_atom)
        field_map = {
            "user_role": "user_role",
            "job_to_be_done": "job_to_be_done",
            "trigger_event": "trigger_event",
            "pain_statement": "pain_statement",
            "failure_mode": "failure_mode",
            "current_workaround": "current_workaround",
        }

        _GENERIC_PHRASES = {"welcome to", "use this", "forum", "find customizable",
                            "this is an", "here is", "this spreadsheet"}

        for llm_key, atom_key in field_map.items():
            llm_val = str(data.get(llm_key, "")).strip()
            if not llm_val:
                continue
            heuristic_val = result.get(atom_key, "")
            heuristic_is_generic = any(
                phrase in heuristic_val.lower() for phrase in _GENERIC_PHRASES
            )
            if not heuristic_val or heuristic_is_generic or len(llm_val) > len(heuristic_val) * 1.1:
                result[atom_key] = llm_val

        consequence = str(data.get("consequence", "")).strip()
        if consequence:
            result["cost_consequence_clues"] = consequence

        result["atom_extraction_method"] = "llm_enhanced"
        return result
