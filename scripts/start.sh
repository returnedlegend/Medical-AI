#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETRACTION_VENV_DIR="${RETRACTION_VENV_DIR:-$PROJECT_DIR/.venv}"
if [[ ! -x "$RETRACTION_VENV_DIR/bin/python" || ! -f "$PROJECT_DIR/frontend/dist/index.html" ]]; then
  printf 'Run scripts/setup.sh first.\n' >&2
  exit 1
fi
export RETRACTION_DATA_DIR="${RETRACTION_DATA_DIR:-$PROJECT_DIR/.data}"
cd "$PROJECT_DIR/backend"
exec "$RETRACTION_VENV_DIR/bin/python" -m uvicorn retraction.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
