#!/bin/bash

# Export a clean copy of the repository for safe sharing
# Excludes: .git, .venv, __pycache__, .pytest_cache, .env, outputs, recovery_artifacts, *.db

set -e

rm -rf dist_clean
mkdir dist_clean

rsync -av \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.env' \
  --exclude='outputs' \
  --exclude='recovery_artifacts' \
  . dist_clean/

echo "Exported to dist_clean/"