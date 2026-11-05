"""Runtime-specific notices must never replace byte-pinned upstream notices."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from desktop.notices import collect_interpreter_notice, verify_pinned_notices


def test_installed_crlf_notice_preserves_pinned_upstream_bytes(tmp_path):
    directory = tmp_path / "licenses"
    directory.mkdir()
    pinned = directory / "CPython-3.12.10-LICENSE.txt"
    upstream = b"Verbatim upstream notice\nCopyright\n"
    pinned.write_bytes(upstream)
    digest = hashlib.sha256(upstream).hexdigest()
    (directory / "sources.json").write_text(json.dumps({"files": {pinned.name: {"sha256": digest}}}))
    installed = tmp_path / "LICENSE.txt"
    installed.write_bytes(upstream.replace(b"\n", b"\r\n"))

    provenance = collect_interpreter_notice(directory, installed, "3.12.10")
    assert pinned.read_bytes() == upstream
    assert verify_pinned_notices(directory) == {pinned.name: digest}
    copied = directory / provenance["file"]
    assert copied != pinned and copied.read_bytes() == installed.read_bytes()
    assert provenance["sha256"] == hashlib.sha256(installed.read_bytes()).hexdigest()

    pinned.write_bytes(b"Modified upstream notice")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_pinned_notices(directory)
