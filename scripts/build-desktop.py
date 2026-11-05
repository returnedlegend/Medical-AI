#!/usr/bin/env python3
"""Build a native platform desktop bundle; run on Windows to produce .exe."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import stat
import ssl
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-frontend", action="store_true", help="Use a frontend build already produced in this checkout")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--installer", action="store_true", help="Also create a Windows Setup.exe using NSIS 3.11")
    parser.add_argument("--makensis", type=Path, help="Path to the NSIS compiler")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = "windows" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else "macos"
    output = (args.output_dir or root / "artifacts" / "desktop" / target).resolve()
    work = root / ".build" / "desktop" / target
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if not args.skip_frontend:
        npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
        if not npm:
            raise SystemExit("Node/npm is required to build the interface; or pass --skip-frontend for an existing build")
        subprocess.run([npm, "ci"], cwd=root / "frontend", check=True)
        subprocess.run([npm, "run", "build"], cwd=root / "frontend", check=True)
    if not (root / "frontend/dist/index.html").is_file():
        raise SystemExit("frontend/dist/index.html is missing; build the frontend first")

    # Small local vector mark: no external assets, fonts, or image downloads.
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    assets = root / "desktop" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    image = QImage(256, 256, QImage.Format.Format_ARGB32)
    image.fill(QColor("#162d32"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#cbdcd7"), 9))
    painter.drawLine(68, 176, 130, 65)
    painter.drawLine(130, 65, 194, 176)
    painter.drawLine(68, 176, 194, 176)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#d8ad63"))
    painter.drawEllipse(109, 44, 42, 42)
    painter.setBrush(QColor("#84b6a4"))
    painter.drawEllipse(47, 155, 42, 42)
    painter.drawEllipse(173, 155, 42, 42)
    painter.end()
    image.save(str(assets / "icon.png"))
    png = (assets / "icon.png").read_bytes()
    (assets / "icon.ico").write_bytes(struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22) + png)

    # Freeze an explicit source snapshot before analysis: concurrently edited
    # worktrees must not silently change the code partway through packaging.
    snapshot = work / "source-snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir()
    for relative in ("backend/retraction", "desktop", "frontend/dist", "docs"):
        shutil.copytree(root / relative, snapshot / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"))
    for relative in ("README.md", "backend/requirements.txt", "backend/requirements-desktop.txt", "scripts/build-desktop.py"):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    # Managed Python distributions may ship an adjacent explicit distrust
    # certificate used by their ssl.py. Preserve that exact trust behavior.
    distrust_certificate = Path(ssl.__file__).with_name("codex-rejected-cas.pem")
    if distrust_certificate.is_file():
        destination = snapshot / "desktop" / "runtime-ca" / distrust_certificate.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(distrust_certificate, destination)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        git_commit = git.stdout.strip() if git.returncode == 0 else None
        git_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except OSError:
        git_commit, git_dirty = os.environ.get("RETRACTION_BUILD_GIT_COMMIT"), None
    sys.path.insert(0, str(snapshot / "backend"))
    from retraction.simulation import MODEL_VERSION
    from retraction.environment import ENV_VERSION, PHYSICS_FINGERPRINT, PHYSICS_FINGERPRINT_ALGORITHM
    from retraction import __version__
    source_hashes = {path.relative_to(snapshot).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(snapshot.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}
    provenance = {
        "version": __version__, "target": target, "architecture": platform.machine(),
        "python": platform.python_version(), "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "git_dirty": git_dirty,
        "model_versions": {"cohort": MODEL_VERSION, "procedure": ENV_VERSION},
        "physics_fingerprint": PHYSICS_FINGERPRINT,
        "physics_fingerprint_algorithm": PHYSICS_FINGERPRINT_ALGORITHM,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("PySide6", "pyinstaller", "numpy", "scipy", "scikit-learn", "gymnasium", "fastapi")},
        "source_sha256": source_hashes,
    }
    (snapshot / "build-info.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
               "--distpath", str(output), "--workpath", str(work / "pyinstaller"), str(snapshot / "desktop" / "RetractionLab.spec")]
    subprocess.run(command, cwd=snapshot, check=True)
    bundle = output / "RetractionLab"
    executable = bundle / ("RetractionLab.exe" if target == "windows" else "RetractionLab")
    if not executable.is_file():
        raise SystemExit(f"Build completed without expected executable: {executable}")
    archive = output / f"RetractionLab-{target}-x64.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(bundle.rglob("*")):
            if path.is_symlink():
                entry = zipfile.ZipInfo(str(path.relative_to(output)).replace('\\', '/'))
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(entry, str(path.readlink()))
            elif path.is_file():
                z.write(path, path.relative_to(output))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{sha}  {archive.name}\n")
    manifest = {**provenance,
                "archive": archive.name, "archive_sha256": sha, "executable": str(executable.relative_to(output)),
                "packaging": "PyInstaller onedir; extract the complete folder before launching", "signed": False}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.installer:
        if target != "windows":
            raise SystemExit("--installer requires an existing Windows-target build")
        sys.path.insert(0, str(root))
        from desktop.build_installer import build_installer
        build_installer(bundle, output, work / "installer", __version__, args.makensis)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
