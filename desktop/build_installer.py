"""Assemble an NSIS installer from an already frozen Windows application."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys


def nsis_literal(value: str) -> str:
    return value.replace("$", "$$").replace('"', '$\\"')


def build_installer(bundle: Path, output: Path, work: Path, version: str, makensis: Path | None = None) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Installer version must contain three numeric components")
    if not (bundle / "RetractionLab.exe").is_file():
        raise ValueError("The Windows application bundle is missing RetractionLab.exe")
    compiler = str(makensis) if makensis else shutil.which("makensis")
    if not compiler:
        raise RuntimeError("NSIS 3.11 is required for --installer; install it or provide --makensis")
    switch = "/" if sys.platform == "win32" else "-"
    compiler_version = subprocess.check_output([compiler, f"{switch}VERSION"], text=True).strip()
    if not compiler_version.startswith("v3.11"):
        raise RuntimeError(f"NSIS 3.11 is required, found {compiler_version}")
    work.mkdir(parents=True, exist_ok=True)
    # The compiler may compress for several minutes. Keep its payload immutable
    # even if another build later replaces the portable application directory.
    payload = work / "payload"
    if payload.exists():
        shutil.rmtree(payload)
    shutil.copytree(bundle, payload)
    bundle = payload
    uninstall = work / "uninstall-files.nsh"
    paths = sorted(bundle.rglob("*"))
    commands = [f'Delete "$INSTDIR\\{nsis_literal(str(path.relative_to(bundle)).replace("/", chr(92)))}"'
                for path in paths if path.is_file()]
    directories = sorted((path for path in paths if path.is_dir()), key=lambda path: len(path.parts), reverse=True)
    commands.extend(f'RMDir "$INSTDIR\\{nsis_literal(str(path.relative_to(bundle)).replace("/", chr(92)))}"' for path in directories)
    uninstall.write_text("\n".join(commands) + "\n", encoding="utf-8")
    installer = output / f"RetractionLab-{version}-windows-x64-Setup.exe"
    partial = installer.with_name(f".{installer.stem}.partial.exe")
    definitions = {"APP_BUNDLE": bundle.resolve(), "INSTALLER_OUT": partial.resolve(),
                   "UNINSTALL_INCLUDE": uninstall.resolve(), "VERSION": version}
    script = Path(__file__).with_name("installer.nsi")
    try:
        result = subprocess.run([compiler, f"{switch}V2", f"{switch}WX", *[f"{switch}D{key}={value}" for key, value in definitions.items()], str(script)], capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode and os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
                handle.write("\n### Installer compilation failed\n\n```text\n" + (result.stdout + result.stderr)[-12000:] + "\n```\n")
        result.check_returncode()
        partial.replace(installer)
    finally:
        partial.unlink(missing_ok=True)
    checksum = hashlib.sha256(installer.read_bytes()).hexdigest()
    installer.with_suffix(".exe.sha256").write_text(f"{checksum}  {installer.name}\n", encoding="utf-8")
    manifest = output / "build-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data.update(installer=installer.name, installer_sha256=checksum,
                    installer_type="NSIS amd64 Unicode per-user; no administrator elevation", installer_signed=False,
                    installer_compiler_version=compiler_version,
                    installer_compiler_sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest())
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return installer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--makensis", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = bundle.parent
    manifest = json.loads((output / "build-manifest.json").read_text(encoding="utf-8"))
    print(build_installer(bundle, output, args.work_dir, manifest["version"], args.makensis))


if __name__ == "__main__":
    main()
