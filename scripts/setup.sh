#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETRACTION_VENV_DIR="${RETRACTION_VENV_DIR:-$PROJECT_DIR/.venv}"
python3 -m venv "$RETRACTION_VENV_DIR"
"$RETRACTION_VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/backend/requirements.lock"
npm --prefix "$PROJECT_DIR/frontend" ci --no-audit --no-fund
npm --prefix "$PROJECT_DIR/frontend" run build
printf '\nSetup complete. Run scripts/start.sh to launch Retraction Lab.\n'
