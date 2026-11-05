#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETRACTION_VENV_DIR="${RETRACTION_VENV_DIR:-$PROJECT_DIR/.venv}"
test_report="${RETRACTION_TEST_REPORT:-$PROJECT_DIR/test-results/backend.xml}"
mkdir -p "$(dirname -- "$test_report")"
cd "$PROJECT_DIR/backend"
"$RETRACTION_VENV_DIR/bin/python" -m pytest tests -q --junitxml="$test_report"
npm --prefix "$PROJECT_DIR/frontend" run build
