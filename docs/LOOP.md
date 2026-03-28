# Getting the discovery → evidence → validation loop working (hosted relay on Render)

The Reddit relay on Render is **not** the usual failure point. The loop stalls when there is **no qualified backlog** (nothing passes screening), or when **gates** never produce `prototype_candidate` / `parked` opportunities.

## 1. Verify the bridge (URL + token)

```bash
# .env must set REDDIT_BRIDGE_BASE_URL and REDDIT_BRIDGE_AUTH_TOKEN (same token as Render)
python cli.py check-bridge
```

Expect `"ok": true` and a `health` object from `/api/health`. If `auth_failed`, the token in `.env` does not match the relay.

## 2. Pipeline health (why the loop looks “stuck”)

```bash
python cli.py pipeline-health
```

- **0 qualified findings** → discovery is not accepting new items (dedup, subs empty, or all screened out). Widen `discovery.reddit` subs/keywords or use a fresh DB for dev.
- **Qualified but no clusters** → need `cluster_min_atoms` atoms; single-shot pain may never cluster until more runs accumulate.

## 3. One full run with visibility

```bash
python cli.py run-once --verbose
```

Then inspect:

```bash
python cli.py gate-diagnostics
python cli.py report
```

## 4. Long-running discovery (`run`)

In `config.yaml`, set **`orchestration.continuous_waves: true`**. Set **`stop_on_hit.retry_interval_seconds: 0`** for back-to-back waves (no idle gap between passes; only the pipeline drain wait runs). **`stop_on_hit.enabled`** remains optional: turn it on only if you want the process to **exit** after a `prototype_candidate` / `promote`; use **`stop_on_hit.exit_on_hit: false`** to log hits but keep looping.

## 5. Continuous watch (status only)

```bash
python cli.py watch
```

`output/pipeline_status.json` updates as stages complete.

## 6. If validation never promotes/parks

See `python cli.py gate-diagnostics` — thresholds are in `config.yaml` (`validation.promotion_threshold`, `validation.park_threshold`, etc.). Lower **park** slightly to see `parked` opportunities while tuning.
