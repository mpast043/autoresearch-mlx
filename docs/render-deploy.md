# Render Deploy

This repo now includes a Render Blueprint at [render.yaml](/Users/meganpastore/Projects/autoresearch-mlx/render.yaml).

## Services

- `autoresearch-relay`
  - Render `web` service
  - Docker runtime
  - starts: `python cli.py reddit-relay --host 0.0.0.0 --port 10000`
  - health check: `/api/health`

- `autoresearch-pipeline`
  - Render `worker` service
  - Docker runtime
  - starts: `python cli.py run`
  - attaches a persistent disk at `/app/data` so the SQLite runtime database survives restarts and deploys

## Environment

The Blueprint wires:

- `REDDIT_BRIDGE_BASE_URL`
  - sourced from the relay service's `RENDER_EXTERNAL_URL`
- `REDDIT_RELAY_AUTH_TOKEN`
  - required for a hosted deployment
- `REDDIT_BRIDGE_AUTH_TOKEN`
  - required, and must exactly match the relay token

## Deploy

1. Push this branch to GitHub.
2. In Render, create a new Blueprint from this repository.
3. Select [render.yaml](/Users/meganpastore/Projects/autoresearch-mlx/render.yaml).
4. Review the two services and create the Blueprint instance.
5. Set the same token value for:
   - `REDDIT_RELAY_AUTH_TOKEN`
   - `REDDIT_BRIDGE_AUTH_TOKEN`

## Notes

- This is a first Render-ready surface, not a full hosted operations stack.
- The dashboard frontend is not included in this Blueprint.
- The worker keeps state in SQLite on its own persistent disk, so this is suitable for a single-worker deployment shape.
- The relay cache is currently ephemeral on Render; that is acceptable for this first deploy surface because the critical durable runtime state is the pipeline database.
