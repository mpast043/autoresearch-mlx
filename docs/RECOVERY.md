# If you deleted the database or “nothing works” anymore

## What Git cannot restore

`*.db` files are **gitignored** (see `.gitignore`). **`data/autoresearch.db` is not in version control.** If you deleted it, Git cannot bring it back unless you have a **copy elsewhere** (download, Time Machine, another machine, Render volume snapshot, etc.).

## Backup from now on

```bash
python cli.py backup-db
```

Copies the configured DB to **`data/backups/autoresearch-<timestamp>.db`**. Run this before risky changes or periodically.

## “Working” again when the DB is exhausted

`python cli.py pipeline-health` may show **`actionable_qualified_for_pipeline: 0`** because every finding is already **parked / killed / screened_out** and discovery **dedupes** on `content_hash`. That feels broken but is **data state**, not necessarily a crashed process.

**Option A — keep history, force new discovery surface**

- Widen `discovery.reddit` subs/keywords, add `web`, tune Shopify/WordPress handles.
- Wait for **new URLs** so new rows insert.

**Option B — clean slate (dev)**

1. `python cli.py backup-db` (save what you have if you care).
2. Move or delete `data/autoresearch.db` (and optionally `data/reddit_relay_cache.db`).
3. Run `python cli.py run-once --verbose` or `python cli.py run`.

You get a **fresh pipeline** with **qualified backlog** again; scores will differ until you accumulate clusters.

## Point config at a new DB file

In `config.yaml`:

```yaml
database:
  path: data/autoresearch_fresh.db
```

## GitHub discovery “skipped after repeated zero-yield feedback”

The discovery agent **stops calling GitHub** if `discovery_feedback` shows **many runs with zero findings**. That’s adaptive — not a crash.

To let GitHub run again (SQLite):

```bash
sqlite3 data/autoresearch.db "DELETE FROM discovery_feedback WHERE source_name LIKE 'github%';"
```

Then restart `python cli.py run`.

## Shopify reviews 404 / “listing skipped”

`discovery.shopify_reviews.app_handles` must match a **real** app URL `https://apps.shopify.com/<handle>`. A bad handle → **404** and **no reviews**. Fix handles in `config.yaml`, then rerun.

## Still stuck?

1. `python cli.py check-bridge` — relay OK if using Render.
2. `python cli.py pipeline-health` — read `likely_blockers` / `hints`.
3. `python cli.py gate-diagnostics --limit 10` — see promote/park floors.
