"""Preserve installed dependency license files in the immutable build snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys


def verify_pinned_notices(directory: Path) -> dict[str, str]:
    """Check verbatim source notices before they enter a distributed bundle."""
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    verified = {}
    for name, metadata in sources["files"].items():
        path = directory / name
        if path.parent.resolve() != directory.resolve():
            raise RuntimeError(f"Pinned license notice must be a direct file: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["sha256"]:
            raise RuntimeError(f"Pinned license notice checksum mismatch: {name}")
        verified[name] = digest
    return verified


def collect_interpreter_notice(directory: Path, source: Path, version: str) -> dict | None:
    """Keep installed-runtime notices separate from byte-pinned upstream text."""
    if not source.is_file():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"CPython-{version}-installed-LICENSE.txt"
    shutil.copy2(source, target)
    provenance = {
        "schema": "retraction-installed-interpreter-notice-v1",
        "python_version": version,
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "source_distribution": "LICENSE.txt from the installed build interpreter's base directory",
        "file": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (directory / "installed-interpreter-notice.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def collect_notices(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    packages = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
        name = distribution.metadata.get("Name", "unknown")
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        files = []
        for relative in distribution.files or ():
            if ".." in relative.parts:
                continue
            if not relative.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE")):
                continue
            if Path(relative).suffix.lower() not in ("", ".txt", ".md", ".rst"):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            target = destination / slug / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({"file": target.relative_to(destination).as_posix(),
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        packages.append({"name": name, "version": distribution.version,
                         "license_expression": distribution.metadata.get("License-Expression"), "files": files})
    manifest = {"scope": "License files available in the installed build environment; includes build tools.",
                "packages": packages}
    (destination / "build-environment-notices.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
