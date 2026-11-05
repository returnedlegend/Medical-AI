#!/usr/bin/env bash
# Optional isolated upstream-physics runtime. Does not change the MVP backend.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${RETRACTION_SOFA_DIR:-/workspace/.sofa-runtime}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.toolchains}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_SYSTEM_CERTS=true
upstream_commit=85bf7e05dd088b824794dda0046679df13b13e6e
archive_name=SOFA_v24.06.00_Linux.zip
archive_sha=9d515e2f25f657c744821be8a5361e22803c18947b33af7a0b357c259202236a
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This optional setup supports Linux x86_64 only.' >&2
  exit 1
fi
for prerequisite in uv git curl tar timeout; do
  command -v "$prerequisite" >/dev/null || { echo "Missing prerequisite: $prerequisite" >&2; exit 1; }
done
mkdir -p "$runtime_root"
uv python install 3.10.22
python_runtime="$(uv python find --managed-python 3.10.22)"
if [[ ! -x "$runtime_root/venv/bin/python" ]]; then
  "$python_runtime" -m venv "$runtime_root/venv"
fi
"$runtime_root/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing optional runtime must use Python 3.10"'
if [[ ! -f "$runtime_root/$archive_name" ]]; then
  curl --fail --location --show-error \
    "https://github.com/sofa-framework/sofa/releases/download/v24.06.00/$archive_name" \
    -o "$runtime_root/$archive_name.partial"
  mv "$runtime_root/$archive_name.partial" "$runtime_root/$archive_name"
fi
# Expected digest is published on the official SOFA v24.06.00 release page.
# Verify before extraction or loading any native code. Preserve archive symlinks.
"$python_runtime" - "$runtime_root" "$archive_sha" <<'PY'
import hashlib
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

root = Path(sys.argv[1]).resolve()
archive = root / 'SOFA_v24.06.00_Linux.zip'
with archive.open('rb') as handle:
    actual = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
if actual is None:  # Python 3.10 compatibility
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
if actual != sys.argv[2]:
    raise SystemExit('SOFA archive checksum mismatch; refusing to extract or use it')
destination = root / 'SOFA_v24.06.00_Linux'
if destination.is_dir():
    print('Official SOFA archive verified; preserving existing extracted runtime')
else:
    with tempfile.TemporaryDirectory(prefix='extract-', dir=root) as temporary:
        temporary = Path(temporary)
        links = []
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                path = temporary / item.filename
                if not path.resolve().is_relative_to(temporary):
                    raise SystemExit('Unsafe archive path')
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    links.append((path, z.read(item).decode()))
                elif item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(item) as source, path.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    if mode:
                        path.chmod(stat.S_IMODE(mode))
        for path, target in links:
            if not (path.parent / target).resolve().is_relative_to(temporary):
                # The verified official archive contains one Ubuntu-specific
                # absolute symlink in an unused optional CImg plugin. It points
                # to an absent zlib build dependency on Debian 13. Do not create
                # an external link or invent an ABI replacement. This plugin is
                # not loaded by the checked GraspLiftTouch scene.
                if path.relative_to(temporary).as_posix() == 'SOFA_v24.06.00_Linux/plugins/CImgPlugin/lib/libz.so' and target == '/lib/x86_64-linux-gnu/libz.so.1.2.11':
                    print('Omitting unused CImgPlugin absolute zlib symlink from official archive')
                    continue
                raise SystemExit('Unsafe archive symlink: ' + str(path.relative_to(temporary)))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        shutil.move(str(temporary / destination.name), destination)
    print('Official SOFA archive verified and extracted')
PY
if [[ ! -f "$runtime_root/source/.upstream_commit" ]]; then
  if [[ -d "$runtime_root/source" && -n "$(ls -A "$runtime_root/source")" ]]; then
    echo 'Existing source has no provenance marker; refusing to overwrite it.' >&2
    exit 1
  fi
  source_checkout="$(mktemp -d /tmp/retraction-sofa-source.XXXXXX)"
  trap 'rm -rf "$source_checkout"' EXIT
  git -C "$source_checkout" init -q
  git -C "$source_checkout" remote add origin https://github.com/ScheiklP/sofa_env.git
  git -C "$source_checkout" fetch --depth 1 origin "$upstream_commit"
  test "$(git -C "$source_checkout" rev-parse FETCH_HEAD)" = "$upstream_commit"
  mkdir -p "$runtime_root/source"
  git -C "$source_checkout" archive FETCH_HEAD sofa_env assets LICENSE README.md setup.py | tar -x -C "$runtime_root/source"
  printf '%s\n' "$upstream_commit" > "$runtime_root/source/.upstream_commit"
elif [[ "$(cat "$runtime_root/source/.upstream_commit")" != "$upstream_commit" ]]; then
  echo 'Existing upstream source differs from the pinned commit; use a fresh RETRACTION_SOFA_DIR.' >&2
  exit 1
fi
"$runtime_root/venv/bin/python" -m pip install -r "$project_root/scripts/requirements-sofa.lock"
RETRACTION_SOFA_DIR="$runtime_root" bash "$project_root/scripts/run-sofa-smoke.sh" --output "$runtime_root/smoke.json"
"$runtime_root/venv/bin/python" "$project_root/scripts/audit_sofa_meshes.py" \
  --runtime "$runtime_root" --scene-json "$runtime_root/smoke.json" --strict-scene \
  --output "$runtime_root/setup-mesh-audit.json" > "$runtime_root/setup-mesh-audit.log"
