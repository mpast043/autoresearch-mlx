# Render Deploy

This repo now includes a Render Blueprint at [render.yaml](/Users/meganpastore/Projects/autoresearch-mlx/render.yaml).

## Service

- `autoresearch-relay`
  - Render `web` service
  - Docker runtime
  - starts: `python cli.py reddit-relay --host 0.0.0.0 --port 10000`
  - health check: `/api/health`

The pipeline stays local in this lower-cost deployment shape.

## Environment

On Render:

- `REDDIT_RELAY_AUTH_TOKEN`
  - required for a hosted deployment

Locally, point the pipeline at the hosted relay:

- `REDDIT_BRIDGE_BASE_URL`
  - set this to your Render relay URL
- `REDDIT_BRIDGE_AUTH_TOKEN`
  - must exactly match the relay token

## Deploy

1. Push this branch to GitHub.
2. In Render, create a new Blueprint from this repository.
3. Select [render.yaml](/Users/meganpastore/Projects/autoresearch-mlx/render.yaml).
4. Review the single relay service and create the Blueprint instance.
5. Set `REDDIT_RELAY_AUTH_TOKEN` on the Render service.
6. Locally, export:
   - `REDDIT_BRIDGE_BASE_URL=https://<your-render-relay-url>`
   - `REDDIT_BRIDGE_AUTH_TOKEN=<same token>`

## Notes

- This is the recommended low-cost hosted shape for now.
- The dashboard frontend is not included in this Blueprint.
- The pipeline remains local, which avoids paying for an always-on worker while the research engine is still evolving.
