# Production Runtime

This repo now has a minimal production-style package surface for the Python pipeline and the local Reddit relay.

If you want a cloud dev environment instead of a local laptop setup, use:

- [docs/codespaces.md](/Users/meganpastore/Projects/autoresearch-mlx/docs/codespaces.md)

## What Runs

- `pipeline`: long-running discovery/evidence/validation runtime
- `relay`: local Reddit relay HTTP service on port `8787`

Both services use the same image and the same checked-in [config.yaml](/Users/meganpastore/Projects/autoresearch-mlx/config.yaml).

## Environment

Environment variables:

- `REDDIT_BRIDGE_BASE_URL`
  Used by the pipeline bridge client. In Docker Compose this is set to `http://relay:8787`.
- `REDDIT_BRIDGE_AUTH_TOKEN`
  Bearer token for the bridge client in protected environments.
- `REDDIT_RELAY_AUTH_TOKEN`
  Bearer token for the relay server in hosted environments.

## Local Virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
REDDIT_RELAY_AUTH_TOKEN=dev-relay-token python3 cli.py reddit-relay --host 0.0.0.0 --port 8787
REDDIT_BRIDGE_AUTH_TOKEN=dev-relay-token REDDIT_BRIDGE_BASE_URL=http://127.0.0.1:8787 python3 cli.py run
```

If you intentionally want a no-auth local relay for ad hoc development, use a local config override instead of changing the checked-in default:

```yaml
reddit_relay:
  allow_no_auth: true
```

## Docker Compose

```bash
docker compose up --build
```

This starts:

- relay on `http://127.0.0.1:8787`
- the long-running pipeline container

Runtime state is persisted through mounted host directories:

- `./data`
- `./output`

## Notes

- The repo expects `config.yaml` at project root unless `--config` is passed.
- Runtime paths are project-rooted, so absolute-path invocation from another cwd is supported.
- The dashboard is not packaged in this first production runtime pass.
