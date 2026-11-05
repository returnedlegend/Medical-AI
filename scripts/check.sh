#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETRACTION_VENV_DIR="${RETRACTION_VENV_DIR:-$PROJECT_DIR/.venv}"
cd "$PROJECT_DIR/backend"
"$RETRACTION_VENV_DIR/bin/python" -m pytest tests -q
npm --prefix "$PROJECT_DIR/frontend" run build
