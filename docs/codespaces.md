# Codespaces Runtime

This repo is now set up to run cleanly in GitHub Codespaces with the checked-in devcontainer.

## Recommended Shape

Use the hosted Render relay and run the Python pipeline inside Codespaces.

Why this is the best default:

- no extra always-on worker bill
- no need to warm a second local relay cache unless you want to
- the Codespace only needs the pipeline runtime and your local review workflow

## First Open

When the Codespace opens:

1. Let the devcontainer build finish.
2. The post-create step installs [requirements-dev.txt](../requirements-dev.txt).
3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Edit `.env` and set:

- `REDDIT_BRIDGE_BASE_URL`
- `REDDIT_BRIDGE_AUTH_TOKEN`

The recommended values are your hosted Render relay URL and the matching relay token.

## Run The Pipeline

```bash
python3 cli.py run
```

## Run The Eval Suite

```bash
python3 cli.py eval
pytest tests -q
```

## Optional: Run A Local Relay In Codespaces

If you want a relay inside the Codespace instead of the hosted Render relay:

1. Set `REDDIT_RELAY_AUTH_TOKEN` in `.env`
2. Start the relay:

```bash
python3 cli.py reddit-relay --host 0.0.0.0 --port 8787
```

3. Point the bridge client at that local relay:

```bash
REDDIT_BRIDGE_BASE_URL=http://127.0.0.1:8787 \
REDDIT_BRIDGE_AUTH_TOKEN=$REDDIT_RELAY_AUTH_TOKEN \
python3 cli.py run
```

Port `8787` is forwarded by the devcontainer for this case.

## Notes

- runtime paths are project-rooted, so the pipeline does not depend on the shell cwd
- `.env` stays local and should not be committed
- `data/` and `output/` are created automatically by the devcontainer post-create step
