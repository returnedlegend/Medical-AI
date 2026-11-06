"""Offline security checks for the pinned Windows release archive.

Small synthetic PE/ZIP fixtures exercise validation, never the network or a real
release. Only unit fixtures override the pinned outer archive digest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import warnings
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "publish_desktop_release.py"
SPEC = importlib.util.spec_from_file_location("publish_desktop_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


SOURCE_COMMIT = "e62f26c44d964a360e057338ad8ce7c6a4e039b1"
PHYSICS_FINGERPRINT = (
    "99bcc146975fff6221c65ee927d4bc38373a18069e56ce8cd45bf7a75c984898"
)
SETUP_NAME = "RetractionLab-0.2.0-windows-x64-Setup.exe"
COMPILER_SHA256 = (
    "b945886428f881b04a504e95a5e008188e8de59ecc25fc89636710fdd98bd1bc"
)


def synthetic_pe(machine: int = 0x8664) -> bytes:
    """A bounded DOS/PE header, deliberately never executed by these tests."""
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, machine, 1, 0, 0, 0, 240, 0x22)
    struct.pack_into("<H", data, 0x98, 0x20B)
    return bytes(data)


class ValidateReleaseArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0

    def fixture(self, *, metadata=None, executable=None, sidecar=None, members=None):
        self.counter += 1
        executable = synthetic_pe() if executable is None else executable
        executable_hash = hashlib.sha256(executable).hexdigest()
        manifest = {
            "version": "0.2.0",
            "target": "windows",
            "architecture": "AMD64",
            "python": "3.12.10",
            "git_commit": SOURCE_COMMIT,
            "git_dirty": False,
            "physics_fingerprint": PHYSICS_FINGERPRINT,
            "model_versions": {"procedure": "tissue-retraction-1.1"},
            "installer": SETUP_NAME,
            "installer_sha256": executable_hash,
            "installer_compiler_sha256": COMPILER_SHA256,
        }
        if metadata:
            manifest.update(metadata)
        entries = [
            (SETUP_NAME, executable),
            (
                SETUP_NAME + ".sha256",
                sidecar if sidecar is not None else f"{executable_hash}  {SETUP_NAME}\n",
            ),
            ("build-manifest.json", json.dumps(manifest)),
        ]
        if members:
            entries = members(entries)
        archive = self.root / f"fixture-{self.counter}.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, content in entries:
                    bundle.writestr(name, content)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest(), manifest

    def validate(self, archive, digest):
        return publisher.validate_archive(
            archive, self.root / f"validated-{self.counter}", expected_digest=digest
        )

    def assert_rejected(self, **kwargs):
        archive, digest, _ = self.fixture(**kwargs)
        with self.assertRaises(publisher.ReleaseError):
            self.validate(archive, digest)

    def test_accepts_exact_three_file_pinned_release_fixture(self):
        archive, digest, manifest = self.fixture()
        result = self.validate(archive, digest)
        self.assertEqual(result["manifest"], manifest)
        self.assertEqual(result["installer_sha256"], manifest["installer_sha256"])
        self.assertEqual(
            {p.name for p in result["assets"]},
            {SETUP_NAME, SETUP_NAME + ".sha256", "build-manifest.json"},
        )
        self.assertTrue(all(isinstance(p, Path) and p.is_file() for p in result["assets"]))

    def test_rejects_outer_archive_hash_mismatch(self):
        archive, _, _ = self.fixture()
        with self.assertRaises(publisher.ReleaseError):
            self.validate(archive, "0" * 64)

    def test_rejects_untrusted_or_inconsistent_provenance(self):
        alterations = {
            "version": "0.2.1",
            "target": "linux",
            "architecture": "x86",
            "python": "3.12.15",
            "git_commit": "0" * 40,
            "git_dirty": True,
            "physics_fingerprint": "0" * 64,
            "model_versions": {"procedure": "tissue-retraction-1.0"},
            "installer": "other.exe",
            "installer_sha256": "0" * 64,
        }
        for key, value in alterations.items():
            with self.subTest(field=key):
                self.assert_rejected(metadata={key: value})

    def test_rejects_mismatched_or_wrongly_named_checksum_sidecar(self):
        expected_hash = hashlib.sha256(synthetic_pe()).hexdigest()
        for sidecar in (
            "0" * 64 + f"  {SETUP_NAME}\n",
            "not a checksum\n",
            expected_hash + "  other.exe\n",
        ):
            with self.subTest(sidecar=sidecar[:20]):
                self.assert_rejected(sidecar=sidecar)

    def test_rejects_non_pe_and_non_amd64_executables(self):
        for executable in (b"not a PE executable", synthetic_pe(0x14C), synthetic_pe(0xAA64)):
            with self.subTest(size=len(executable), machine=executable[0x84:0x86]):
                self.assert_rejected(executable=executable)

    def test_rejects_extra_traversal_and_duplicate_archive_members(self):
        for name in ("extra.txt", "../escape.txt", "/absolute.txt", "folder/../../escape.txt", "..\\escape.txt"):
            with self.subTest(member=name):
                self.assert_rejected(members=lambda rows, name=name: rows + [(name, b"extra")])
        self.assert_rejected(members=lambda rows: rows + [rows[0]])

    def test_rejects_missing_required_archive_members(self):
        for index in range(3):
            with self.subTest(missing=index):
                self.assert_rejected(members=lambda rows, index=index: rows[:index] + rows[index + 1:])


if __name__ == "__main__":
    unittest.main()
