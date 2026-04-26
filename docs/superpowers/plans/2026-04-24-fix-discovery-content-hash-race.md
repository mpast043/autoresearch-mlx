# Fix Discovery Content-Hash Race Crash — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `sqlite3.IntegrityError: UNIQUE constraint failed: findings.content_hash` crash that halts discovery after 4-5 waves.

**Architecture:** The crash is a race condition in concurrent Reddit discovery: two coroutines discover the same URL, both pass the `seen_urls` / `_seen_hashes` dedup (which is not thread-safe across coroutines), and both attempt `insert_finding` with the same `content_hash`. The first insert succeeds; the second crashes. The fix is a defense-in-depth guard at the insert boundary: re-check for existing hash right before insert, and wrap insert in a try/catch that gracefully skips duplicates.

**Tech Stack:** Python 3.14, SQLite, asyncio, pytest.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/agents/discovery.py` | `_process_finding` — dedup logic, insert call |
| `src/database.py` | `insert_finding` — raw INSERT into `findings` |
| `tests/test_agents/test_discovery.py` | Discovery agent tests |
| `tests/test_database.py` | Database layer tests |

---

## Task 1: Reproduce the Crash in a Test

**Files:**
- Modify: `tests/test_agents/test_discovery.py`

- [ ] **Step 1: Add a test that simulates the race**

```python
import pytest
from unittest.mock import MagicMock, patch

from src.agents.discovery import DiscoveryAgent


class TestDiscoveryRaceCondition:
    def test_process_finding_skips_duplicate_content_hash(self, temp_db):
        """Simulate two concurrent discoveries of the same URL — second must not crash."""
        agent = DiscoveryAgent(
            config={"discovery": {"sources": ["reddit"]}, "llm": {"enabled": False}},
            db=temp_db,
            message_bus=MagicMock(),
        )
        agent._seen_hashes.clear()
        agent.sources = ["reddit"]

        finding_data = {
            "source": "reddit",
            "source_url": "https://reddit.com/r/test/comments/abc123/same_post/",
            "product_built": "Same post title",
            "outcome_summary": "Same post body content",
            "finding_kind": "problem_signal",
            "evidence": {"discovery_query": "test query", "source_plan": "reddit-problem"},
        }

        # First call succeeds
        id1 = agent._process_finding_sync(finding_data)
        assert id1 is not None
        assert id1 > 0

        # Simulate race: same content_hash bypasses _seen_hashes guard
        agent._seen_hashes.clear()

        # Second call must NOT crash — should return None or existing id
        id2 = agent._process_finding_sync(finding_data)
        # Should skip gracefully (either None or the existing id)
        assert id2 is None or id2 == id1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_agents/test_discovery.py::TestDiscoveryRaceCondition::test_process_finding_skips_duplicate_content_hash -v
```

Expected: FAIL with `sqlite3.IntegrityError: UNIQUE constraint failed: findings.content_hash`.

---

## Task 2: Add DB-Level Duplicate Guard Before Insert

**Files:**
- Modify: `src/agents/discovery.py:1148-1151`

- [ ] **Step 1: Insert a re-check right before `insert_finding`**

```python
# After line 1148 (finding.evidence_json = json.dumps(evidence))
# and before line 1150 (finding_id = self.db.insert_finding(finding)):

# Race-condition guard: another coroutine may have inserted the same
# content_hash between our initial get_finding_by_hash check and now.
existing_now = self.db.get_finding_by_hash(content_hash)
if existing_now:
    logger.debug("content_hash %s inserted by concurrent coroutine; skipping", content_hash[:16])
    self._seen_hashes.add(content_hash)
    return None

finding_id = self.db.insert_finding(finding)
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_agents/test_discovery.py::TestDiscoveryRaceCondition::test_process_finding_skips_duplicate_content_hash -v
```

Expected: PASS (the re-check catches the duplicate before insert).

---

## Task 3: Wrap `insert_finding` in Try/Catch as Final Defense

**Files:**
- Modify: `src/agents/discovery.py:1150`

- [ ] **Step 1: Wrap the insert in a try/catch**

Replace:
```python
finding_id = self.db.insert_finding(finding)
```

With:
```python
try:
    finding_id = self.db.insert_finding(finding)
except sqlite3.IntegrityError as exc:
    if "content_hash" in str(exc):
        logger.warning("content_hash collision on insert for %s: %s", content_hash[:16], exc)
        self._seen_hashes.add(content_hash)
        return None
    raise
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_agents/test_discovery.py::TestDiscoveryRaceCondition::test_process_finding_skips_duplicate_content_hash -v
```

Expected: PASS.

---

## Task 4: Apply Same Guard to Pre-Atom Filter Insert Path

**Files:**
- Modify: `src/agents/discovery.py:1061`

- [ ] **Step 1: Wrap the pre-atom-filter insert**

Replace:
```python
finding_id = self.db.insert_finding(finding)
```

With:
```python
try:
    finding_id = self.db.insert_finding(finding)
except sqlite3.IntegrityError as exc:
    if "content_hash" in str(exc):
        logger.warning("content_hash collision on pre-atom insert for %s: %s", content_hash[:16], exc)
        self._seen_hashes.add(content_hash)
        return None
    raise
```

- [ ] **Step 2: Run all discovery tests**

```bash
pytest tests/test_agents/test_discovery.py -v
```

Expected: All PASS.

---

## Task 5: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add tests/test_agents/test_discovery.py src/agents/discovery.py
git commit -m "fix: guard discovery insert against content-hash race condition

Fixes wave-5 crash where concurrent Reddit discovery coroutines
discovered the same URL, bypassed seen_urls dedup, and both
tried to insert identical content_hash. Adds:

1. Re-check get_finding_by_hash right before insert
2. try/catch on insert_finding that skips duplicate gracefully
3. Same guard on the pre-atom-filter insert path
4. Test simulating the race condition

Wedge impact: unblocks continuous discovery waves → more findings →
more atoms → more opportunities → faster speed to pilot."
```

---

## Self-Review

**Spec coverage:**
- Race condition simulation? Task 1 test covers it.
- Re-check before insert? Task 2 covers it.
- Try/catch on insert? Task 3 covers it.
- Pre-atom path? Task 4 covers it.
- Test? Task 1 and follow-on runs cover it.

**Placeholder scan:** None.

**Type consistency:** All types match existing codebase (sqlite3.IntegrityError, logger.warning, return Optional[int]).
