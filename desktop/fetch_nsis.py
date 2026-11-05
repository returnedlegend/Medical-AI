"""Fetch the exact Debian NSIS compiler used by the installer assembly job.

Package hashes were copied from Debian trixie's signature-verified Packages
index. Extraction requires dpkg-deb; Windows applications are still frozen
with Windows Python before this independent installer-assembly stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen


PACKAGES = {
    "nsis_3.11-1_amd64.deb": "e6eb6ffb70f0df09e95dedb2d2cf55ad3c2679f7c2ae357d0c00cc4c83178690",
    "nsis-common_3.11-1_all.deb": "103a3284c1a5356efa0aba90fdfc391c3cde8e7df8726377cef0f98471584047",
}
BASE_URL = "https://deb.debian.org/debian/pool/main/n/nsis/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for name, expected in PACKAGES.items():
        url = BASE_URL + name
        # urllib retains the configured proxy and normal certificate validation.
        with urlopen(url, timeout=60) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise RuntimeError(f"Unexpected compiler package size: {name}")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Compiler package checksum mismatch: {name}")
        archive = archives / name
        archive.write_bytes(data)
        subprocess.run(["dpkg-deb", "--extract", str(archive), str(root)], check=True)
    compiler = root / "usr/bin/makensis"
    environment = dict(os.environ, NSISDIR=str(root / "usr/share/nsis"))
    version = subprocess.check_output([str(compiler), "-VERSION"], env=environment, text=True).strip()
    if version != "v3.11-1":
        raise RuntimeError(f"Unexpected extracted compiler version: {version}")
    provenance = {
        "schema": "retraction-installer-compiler-v1",
        "repository": "https://deb.debian.org/debian",
        "distribution": "trixie",
        "package_version": "3.11-1",
        "hash_source": "Debian archive signature-verified Packages index",
        "packages": [{"url": BASE_URL + name, "sha256": digest} for name, digest in PACKAGES.items()],
        "compiler_version": version,
        "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
    }
    (root / "compiler-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
