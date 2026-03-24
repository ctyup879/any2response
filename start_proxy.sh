#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

set -a
source "$ROOT_DIR/.env"
set +a

exec "$ROOT_DIR/.venv/bin/uvicorn" app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
