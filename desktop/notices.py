"""Preserve installed dependency license files in the immutable build snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil


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
