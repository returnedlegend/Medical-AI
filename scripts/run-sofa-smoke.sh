#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${RETRACTION_SOFA_DIR:-/workspace/.sofa-runtime}"
export SOFA_ROOT="$runtime_root/SOFA_v24.06.00_Linux"
export SOFAPYTHON3_ROOT="$SOFA_ROOT/plugins/SofaPython3"
export SOFA_ENV_SOURCE="$runtime_root/source"
export PYTHONPATH="$SOFAPYTHON3_ROOT/lib/python3/site-packages:$SOFA_ENV_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
toolchain_lib="$("$runtime_root/venv/bin/python" -c 'import sys; print(sys.base_prefix + "/lib")')"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFAPYTHON3_ROOT/lib:$toolchain_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec timeout 300 "$runtime_root/venv/bin/python" "$project_root/scripts/sofa_smoke.py" "$@"
