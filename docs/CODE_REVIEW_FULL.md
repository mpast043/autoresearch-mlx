# Full codebase review (including database alignment)

**Date:** 2026-03-24  
**Scope:** Repository-wide architectural and data-layer review. **Not** a line-by-line audit of `opportunity_engine.py` / `research_tools.py` (very large); those are sampled and risk-rated.

---

## 1. Executive summary

| Area | Assessment |
|------|------------|
| **SQLite schema ↔ Python models** | Generally aligned; evolution handled via `_ensure_column` and dynamic `_table_columns` inserts. Some **intentional** duplication (mirror tables). |
| **Run-scoped correctness** | Core tables use `(run_id, finding_id)` or similar with **UNIQUE** indexes and `ON CONFLICT` upserts where it matters (`validations`, `corroborations`, `market_enrichments`, `build_briefs`, `build_prep_outputs`). |
| **Integrity risks** | Medium: dual-write experiment path; `evidence_ledger` dedupe in app only; async + thread-local DB usage. |
| **Documentation** | `docs/state-model.md` is strong on semantics; **absolute `/Users/...` links** break on other machines. |
| **Security** | Secrets belong in `.env`; no full secrets audit performed. |

---

## 2. Database alignment

### 2.1 Authoritative schema

- **Single source of truth:** `Database.init_schema()` in `src/database.py` plus incremental `_ensure_column` migrations.
- **Threading:** Connections are **thread-local** (`threading.local`). Safe for threaded use; **async** code should stay on one thread or risk subtle bugs if multiple threads share `Database` (typical asyncio single-thread is OK).

### 2.2 Run identity

- **`active_run_id`** is set at `initialize(start_new_run=True)` to a timestamp string.
- **Run-scoped rows:** `validations`, `corroborations`, `market_enrichments`, `evidence_ledger`, `review_feedback`, `build_briefs`, `build_prep_outputs`, `experiments`, `validation_experiments`, `discovery_themes` (optional).
- **`get_latest_run_id()`** unions several tables’ `run_id` by time. **Gap:** `experiments.run_id` is **not** included in that union (only validations, corroborations, market, ledger, review_feedback, build_briefs, build_prep). A hypothetical run that **only** touched `experiments` might not surface as “latest” — **low severity**, edge case.

### 2.3 Upserts and uniqueness

| Table | Mechanism | Notes |
|-------|-----------|--------|
| `validations` | `UNIQUE(run_id, finding_id)` + `ON CONFLICT DO UPDATE` | Aligns with “one validation row per finding per run”. |
| `corroborations` | Same pattern | OK. |
| `market_enrichments` | Same pattern | OK. |
| `build_briefs` | `UNIQUE(run_id, opportunity_id)` + upsert | OK. |
| `build_prep_outputs` | `UNIQUE(run_id, build_brief_id, agent_name)` + upsert | OK. |
| `evidence_ledger` | **No UNIQUE constraint** | Dedupe is **application-level** (`SELECT` then `UPDATE` or `INSERT` in `insert_ledger_entry`). Concurrent writers could duplicate — **low** risk in single-client CLI. |

### 2.4 Mirror / dual tables (documented in `docs/state-model.md`)

| Pair | Purpose |
|------|---------|
| `clusters` vs `opportunity_clusters` | Projection / compatibility; `upsert_cluster` maintains both. |
| `experiments` vs `validation_experiments` | `insert_experiment` writes **validation_experiments first**, then mirrors into `experiments` with **same `id`**. |

**Risk (medium):** `insert_experiment` is **not** wrapped in an explicit transaction. If the second `INSERT` into `experiments` fails after the first succeeds, you could have a row only in `validation_experiments`. `commit()` is once at the end — SQLite default transaction still wraps both in one transaction **if** they share the same connection without intermediate commit (they do). **Verify:** both executes before `commit` — **yes**. Rollback on exception is not explicit; failed mid-flight could leave uncommitted work — typically OK.

### 2.5 `problem_atoms` physical columns vs JSON

- Base `CREATE TABLE` is **minimal**; many atom fields are stored in **`atom_json` / `score_json` / `metadata_json`**.
- Code paths (`insert_problem_atom`) merge dataclass fields into JSON and use `_table_columns` so inserts never reference nonexistent columns — **aligned**.
- **Implication:** SQL-level filtering on `cluster_key` / `segment` is weak unless you query JSON — code uses Python filters (`get_problem_atoms_by_cluster_key` loads all atoms in one path) — **performance** concern at scale, not correctness.

### 2.6 `findings.evidence` vs column name

- SQLite column is **`evidence_json`**; dataclass maps via `Finding.__post_init__` — **aligned**.

### 2.7 Foreign keys

- SQLite **does not enforce** foreign keys unless `PRAGMA foreign_keys=ON` — not set in `Database._get_connection()`. Referential integrity is **logical only** — **medium** for corrupted manual DB edits; normal app path is consistent.

---

## 3. Pipeline and runtime (`run.py`, `orchestrator.py`, agents)

- **Message flow** matches documented stages: discovery → evidence → validation → optional build_prep.
- **`run_once`** waits on **`completion_state()["drained"]`** so validation/evidence don’t get cut off early — good.
- **`stop_on_hit`** (orchestrator + `shutdown_event`) is consistent with validation payload fields (`selection_status`, `decision`).
- **Config resolution** for promotion/park thresholds is centralized in `validation_thresholds.py` — reduces drift vs raw `config.yaml`.

---

## 4. Large modules (risk-rated, not fully read)

| File | ~Size | Risk notes |
|------|-------|------------|
| `src/opportunity_engine.py` | Large | Core scoring / `stage_decision`; changes need regression tests. |
| `src/research_tools.py` | Large | External I/O, rate limits, recurrence budgets; primary source of “timeout” / thin evidence. |
| `src/source_policy.py` | Smaller | Gates what becomes `pain_signal` — critical for intake. |

---

## 5. Documentation drift

- **`docs/state-model.md`**: Accurate on mirror tables and JSON payloads; **fix** file links from absolute Mac paths to **repo-relative** paths.
- **`CLAUDE.md`**: Updated over time; keep in sync with `config.yaml` keys (`stop_on_hit`, `pipeline-health`, etc.).

---

## 6. Security & operations

- **API keys:** `${ENV_VAR}` in config — ensure `.env` not committed.
- **Reddit bridge token:** treat as secret; rotate if leaked.
- **SQLite file:** `data/autoresearch.db` — backup for production; no encryption at rest in repo.

---

## 7. Testing gaps (suggested)

- **DB:** Migration test: fresh `init_schema` + smoke CRUD for each major entity.
- **Integration:** Already have `test_build_prep_integration.py`; add optional test for `insert_experiment` dual-table row counts.
- **Ledger:** Optional test that duplicate `(run_id, entity_type, entity_id, entry_kind)` updates in place.

---

## 8. Prioritized recommendations

1. ~~**P1 — Foreign keys:**~~ **Done:** `PRAGMA foreign_keys=ON` in `Database._get_connection()` (see `tests/test_database.py::test_foreign_keys_pragma_enabled`).
2. ~~**P2 — `get_latest_run_id`:**~~ **Done:** `experiments` and `validation_experiments` included in `get_latest_run_id` / `get_recent_run_ids` unions.
3. ~~**P2 — Docs:**~~ **Done:** Relative links in `docs/*.md` (removed absolute `/Users/...` paths).
4. ~~**P3 — `evidence_ledger`:**~~ **Done:** `idx_evidence_ledger_run_entity_kind` + dedupe on `init_schema`; `insert_ledger_entry` uses `ON CONFLICT` upsert (`test_evidence_ledger_upsert_idempotent`).
5. **P3 — Performance:** Avoid full-table `problem_atoms` scans for cluster_key as data grows (index JSON with generated columns or denormalize hot fields). — *Not implemented; future work.*

---

## 9. Conclusion

The **database layer and application code are largely aligned**: run-scoped keys, upsert patterns, and mirror tables are **documented and mostly consistent**. Remaining issues are **incremental** (FK pragma, ledger uniqueness, latest-run union, doc links), not a fundamental schema mismatch. For **full** assurance on scoring behavior, add targeted tests around `stage_decision` and representative `validate_problem` outputs rather than reading every line of the largest files.
