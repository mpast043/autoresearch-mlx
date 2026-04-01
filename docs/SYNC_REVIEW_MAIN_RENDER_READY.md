# Sync Review: `main` vs `codex/render-ready`

This branch treats `codex/render-ready` as the source of truth and prepares a clean content-sync candidate for `main`.

## Branch Topology

- `main` and `codex/render-ready` currently have no merge base in this clone.
- A normal merge or rebase is intentionally avoided.
- The sync strategy is content reconciliation from `codex/render-ready` onto a clean branch.

## Main-Only Commits

| Commit | Summary | Disposition | Notes |
| --- | --- | --- | --- |
| `989794a` | Add `discovery_schema.py` and numerous dependencies | Exclude from merge | Non-mergeable because it vendors `dashboard/node_modules`; sampled source intent already exists on `codex/render-ready`. |
| `f13f001` | source-restored, pyc-fallback removed | Mined, not merged | Sampled source/docs already exist on `codex/render-ready`; tracked recovery backup is removed in this sync cleanup. |

## `codex/render-ready` Commit Matrix

### Runtime / Deploy Surface

| Commit | Summary | Disposition |
| --- | --- | --- |
| `2de0d26` | prepare render-ready runtime surface | Keep as-is |
| `8547012` | remove unsupported render worker shutdown delay | Keep as-is |
| `142271a` | switch render blueprint to relay only | Keep as-is |
| `60a8832` | harden relay seeding task cleanup | Keep as-is |
| `de45dd9` | wire render relay auth token from config | Keep as-is |
| `69b11fa` | implement reddit-devvit bridge service with relay mirroring | Keep as-is |
| `42c2876` | timing-safe auth comparison + `.env.example` | Keep as-is |
| `1efe33f` | add codespaces runtime setup | Keep as-is |

### Pipeline Architecture

| Commit | Summary | Disposition |
| --- | --- | --- |
| `6a1e183` | extract reddit transport layer | Keep as-is |
| `b60abfc` | extract database read models | Keep as-is |
| `79d7481` | add `MessageBus` per-agent queue messaging | Keep as-is |
| `256205d` | fix MessageBus priority handling | Keep as-is |
| `a89756f` | fix orchestrator routing and `consume_until_quiet` | Keep as-is |
| `5bc84d1` | await BuilderV2 task lifecycle | Keep as-is |
| `9546de5` | normalize imports to `src.` prefix | Squash into cleanup |
| `a30b4de` | normalize test imports to `src.` prefix | Squash into cleanup |
| `f92e5a2` | normalize `cli.py` imports | Squash into cleanup |
| `4cf00f9` | normalize monkeypatch paths/import order in reddit transport | Squash into cleanup |

### Discovery / Validation Expansion

| Commit | Summary | Disposition |
| --- | --- | --- |
| `0792d35` | unseeded discovery loop + validation gate fix + routing | Keep as-is |
| `146bb94` | integrate 5 skills as native pipeline stages | Keep as-is |
| `b50271d` | fix deep research / DB schema / dispatch | Keep as-is |
| `d0dcfbb` | configurable DuckDuckGo settings | Keep as-is |
| `91f3d1a` | use DuckDuckGo backend for DDGS | Keep as-is |
| `27f2463` | use `source_family_diversity` in gate | Keep as-is |
| `1e68022` | configurable DuckDuckGo settings | Superseded by `d0dcfbb` |
| `9545b05` | update timeouts, year filter, docs | Keep as-is |
| `2be535f` | expand review sources and enable YouTube | Keep as-is |
| `208e1d1` | remove invalid Shopify app handle | Keep as-is |
| `a568966` | respect `max_apps` for Shopify discovery | Keep as-is |
| `3d5a8c7` | Shopify sitemap auto-discovery | Keep as-is |
| `c2c50f1` | handle empty `app_handles` for sitemap discovery | Keep as-is |
| `95cea6c` | add YouTube comments for discovery | Keep as-is |
| `76f931b` | fix discovery expander, RAG pathing, full-table scans | Keep as-is |
| `509d7d4` | cleanup discovery expander iterator bug | Squash into cleanup |
| `dfaba16` | add Jina Reader extraction | Keep as-is |
| `79265a8` | integrate Jina Reader extraction | Squash into `dfaba16` |
| `4c2439b` | fetch live from Reddit on cache miss | Superseded by `e3ffc3e` |
| `3dd5189` | revert live Reddit cache-miss fetch | Superseded / exclude |
| `e3ffc3e` | retry live Reddit cache-miss fetch | Keep as-is |
| `726d719` | replace naive env parser with `python-dotenv` | Keep as-is |

### Build / Product Loop

| Commit | Summary | Disposition |
| --- | --- | --- |
| `6e52b4b` | trigger ideation on `prototype_candidate` + build brief | Keep as-is |
| `af40d9f` | self-expanding discovery and BuilderV2 generation | Keep as-is |
| `d04d79e` | builder_v2 / reddit_relay pathing and provider config | Keep as-is |
| `47d7401` | add `apify-client` to optional deps | Keep as-is |

### Repo Hygiene / Tooling / Docs

| Commit | Summary | Disposition |
| --- | --- | --- |
| `f0a20a2` | add 10 antigravity skills | Keep as-is |
| `e8c8af0` | add additional antigravity skills | Squash into skills/docs cleanup |
| `befc8e4` | `ok` | Squash into adjacent cleanup |
| `4adc7d8` | packaging cleanup and dependency honesty | Keep as-is |
| `a9fe961` | extract text utilities | Keep as-is |
| `31dee17` | extract search planning/tooling utils | Keep as-is |
| `46aa15a` | extract opportunity helpers | Keep as-is |
| `761ff16` | add async retry functions | Keep as-is |
| `19cf506` | add clean repo export script and update gitignore | Keep as-is |
| `efc3819` | replace absolute paths in README | Squash into docs cleanup |
| `e8d3cad` | split `research_tools.py` into modular package | Keep as-is |

## Sync Cleanup Applied On This Branch

- Removed tracked vendor/runtime/generated content:
  - `dashboard/node_modules/**`
  - `data/generated_projects/**`
  - `outputs/deep_research/**`
  - `data/unseeded_results/**`
  - `data/discovery_expansion.json`
  - `recovery_artifacts/_pyc_loader.py.bak`
- Updated `.gitignore` so these artifacts do not re-enter future sync branches.

## Acceptance Notes

- Public behavior should match `codex/render-ready`, not `main`.
- `main` history is accounted for, but not merged wholesale.
- This branch is intended to become the clean sync candidate for updating or replacing `main`.
