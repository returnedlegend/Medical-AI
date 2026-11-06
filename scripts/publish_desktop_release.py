#!/usr/bin/env python3
"""Publish the already-tested, byte-pinned native Windows build; never rebuild it.

Run only in this repository's GitHub Actions publisher with GH_TOKEN set to its
ordinary job token. --verify-only --artifact-zip ZIP is an offline, read-only
archive check; it deliberately retains the production artifact checksum pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import tempfile
import zipfile

REPOSITORY = "returnedlegend/Medical-AI"
SOURCE_COMMIT = "e62f26c44d964a360e057338ad8ce7c6a4e039b1"
RUN_ID = 42085408337
ARTIFACT_ID = 13443587116
OUTER_SHA256 = "1abf17f67b52f030627bf0b0ffa0b8e4bda24f8144ff41acaac4210beae466a2"
PHYSICS_FINGERPRINT = "99bcc146975fff6221c65ee927d4bc38373a18069e56ce8cd45bf7a75c984898"
VERSION = "0.2.0"
TAG = "desktop-v0.2.0"
TITLE = "Retraction Lab 0.2.0"
SETUP_NAME = "RetractionLab-0.2.0-windows-x64-Setup.exe"
ASSET_NAMES = (SETUP_NAME, SETUP_NAME + ".sha256", "build-manifest.json")
MAX_ARCHIVE_BYTES = 400 * 1024 * 1024


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_message(message: str) -> str:
    """Keep HTTP/error context while removing credentials and terminal controls."""
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(variable)
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)authorization\s*:[^\r\n]+", "Authorization: [REDACTED]", message)
    message = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", message)
    message = re.sub(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]+", "[REDACTED]", message)
    message = re.sub(r"https://[^\s/@]+:[^\s/@]+@", "https://[REDACTED]@", message)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", message).strip()[:1800]


def validate_archive(archive: Path, output: Path, expected_digest: str = OUTER_SHA256) -> dict:
    """Validate before extraction. The optional digest supports isolated unit fixtures.

    The CLI never accepts a replacement digest; production always uses its pin.
    """
    require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "Artifact archive exceeds size bound")
    require(sha256(archive) == expected_digest, "Native artifact outer ZIP checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [item.filename for item in entries]
        require(len(names) == len(set(names)), "Duplicate artifact ZIP members")
        for item in entries:
            path = PurePosixPath(item.filename)
            require(not item.is_dir() and not path.is_absolute() and ".." not in path.parts
                    and "\\" not in item.filename and len(path.parts) == 1,
                    "Unsafe artifact ZIP member")
            require((item.external_attr >> 16) & 0o170000 != 0o120000,
                    "Artifact ZIP symlinks are forbidden")
        require(set(names) == set(ASSET_NAMES), "Artifact must contain exactly Setup, checksum, and manifest")
        require(sum(item.file_size for item in entries) <= MAX_ARCHIVE_BYTES,
                "Uncompressed artifact exceeds size bound")
        output.mkdir(parents=True, exist_ok=True)
        for name in ASSET_NAMES:
            destination = output / name
            require(not destination.exists(), "Extraction destination already exists")
            with bundle.open(name) as source, destination.open("xb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
    manifest = json.loads((output / "build-manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("git_commit") == SOURCE_COMMIT and manifest.get("git_dirty") is False,
            "Manifest is not the clean frozen release source")
    require(manifest.get("version") == VERSION and manifest.get("target") == "windows"
            and manifest.get("architecture") in ("AMD64", "x86_64")
            and manifest.get("python") == "3.12.10", "Manifest is not the native Windows CI build")
    require(manifest.get("physics_fingerprint") == PHYSICS_FINGERPRINT
            and manifest.get("model_versions", {}).get("procedure") == "tissue-retraction-1.1",
            "Manifest model provenance mismatch")
    require(manifest.get("installer") == SETUP_NAME, "Manifest installer name mismatch")
    setup = output / SETUP_NAME
    inner_digest = sha256(setup)
    require(manifest.get("installer_sha256") == inner_digest, "Installer manifest checksum mismatch")
    checksum = (output / (SETUP_NAME + ".sha256")).read_text(encoding="utf-8").strip()
    require(re.fullmatch(re.escape(inner_digest) + r"[ \t]+\*?" + re.escape(SETUP_NAME), checksum)
            is not None, "Installer checksum sidecar mismatch")
    with setup.open("rb") as handle:
        header = handle.read(64)
        require(len(header) == 64 and header[:2] == b"MZ", "Installer is not a Windows PE executable")
        offset = struct.unpack_from("<I", header, 0x3C)[0]
        require(64 <= offset <= setup.stat().st_size - 6, "Invalid installer PE header offset")
        handle.seek(offset)
        pe = handle.read(6)
        require(pe[:4] == b"PE\0\0" and pe[4:] == b"\x64\x86", "Installer is not an AMD64 PE executable")
    return {"manifest": manifest, "assets": [output / name for name in ASSET_NAMES],
            "installer_sha256": inner_digest}


def gh(*arguments: str, destination: Path | None = None) -> str:
    """Use gh's supported authentication/redirect handling; never log token values."""
    if destination is not None:
        with destination.open("xb") as handle:
            result = subprocess.run(["gh", *arguments], stdout=handle, stderr=subprocess.PIPE, timeout=600)
        require(result.returncode == 0, "GitHub CLI download failed: "
                + safe_message(result.stderr.decode("utf-8", "replace")))
        return ""
    result = subprocess.run(["gh", *arguments], capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, "GitHub CLI operation failed: " + " ".join(arguments[:3])
            + ": " + safe_message(result.stderr))
    return result.stdout


def api(endpoint: str) -> dict | list:
    return json.loads(gh("api", endpoint))


def get_release() -> dict | None:
    pages = json.loads(gh("api", f"repos/{REPOSITORY}/releases?per_page=100", "--paginate", "--slurp"))
    matches = [release for page in pages for release in page if release.get("tag_name") == TAG]
    require(len(matches) <= 1, "Multiple releases use the frozen tag")
    return matches[0] if matches else None


def check_tag(*, required: bool = False) -> None:
    refs = api(f"repos/{REPOSITORY}/git/matching-refs/tags/{TAG}")
    matches = [ref for ref in refs if ref.get("ref") == f"refs/tags/{TAG}"]
    require(len(matches) <= 1 and (matches or not required), "Release tag missing or ambiguous")
    if not matches:
        return
    obj = matches[0]["object"]
    for _ in range(10):
        if obj.get("type") == "commit":
            require(obj.get("sha") == SOURCE_COMMIT, "Existing release tag targets different source")
            return
        require(obj.get("type") == "tag", "Unexpected release tag object type")
        obj = api(f"repos/{REPOSITORY}/git/tags/{obj['sha']}")["object"]
    raise ReleaseError("Excessive annotated release tag nesting")


def check_release(release: dict, notes: str) -> None:
    require(release.get("tag_name") == TAG and release.get("target_commitish") == SOURCE_COMMIT
            and release.get("name") == TITLE, "Existing release metadata differs from frozen release")
    require((release.get("body") or "").strip() == notes.strip(), "Existing release notes differ; refusing to overwrite")
    require(not release.get("prerelease"), "Existing release prerelease state differs")


def check_assets(release: dict, assets: list[Path], work: Path, *, complete: bool) -> set[str]:
    expected = {path.name: path for path in assets}
    seen: set[str] = set()
    for asset in release.get("assets", []):
        name = asset.get("name")
        require(name in expected and name not in seen, "Existing release contains unexpected or duplicate assets")
        require(asset.get("state") == "uploaded" and asset.get("size") == expected[name].stat().st_size,
                "Existing release asset is incomplete or has different size")
        digest = "sha256:" + sha256(expected[name])
        if asset.get("digest") is not None:
            require(asset["digest"] == digest, "Existing release asset digest mismatch; refusing to clobber")
        else:
            remote = work / ("remote-" + str(asset["id"]))
            gh("api", f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
               "-H", "Accept: application/octet-stream", destination=remote)
            require(sha256(remote) == digest.removeprefix("sha256:"), "Downloaded release asset checksum mismatch")
            remote.unlink()
        # GitHub may use an untagged preview URL until the draft is published.
        # The final public URL is checked after publication, never guessed for a draft.
        if not release.get("draft"):
            require(asset.get("browser_download_url") == f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{name}",
                    "Release asset public URL mismatch")
        seen.add(name)
    require(not complete or seen == set(expected), "Release assets are incomplete")
    return seen


def download_native_artifact(destination: Path) -> dict:
    repository = api(f"repos/{REPOSITORY}")
    require(repository.get("full_name") == REPOSITORY and repository.get("private") is False,
            "Publisher target must be the expected public repository")
    run = api(f"repos/{REPOSITORY}/actions/runs/{RUN_ID}")
    require(run.get("head_sha") == SOURCE_COMMIT and run.get("status") == "completed"
            and run.get("conclusion") == "success"
            and run.get("repository", {}).get("full_name") == REPOSITORY
            and run.get("path") == ".github/workflows/desktop.yml", "Pinned native Windows run provenance mismatch")
    artifact = api(f"repos/{REPOSITORY}/actions/artifacts/{ARTIFACT_ID}")
    workflow = artifact.get("workflow_run", {})
    require(artifact.get("id") == ARTIFACT_ID and artifact.get("name") == "RetractionLab-windows-x64-Setup"
            and artifact.get("expired") is False and artifact.get("digest") == "sha256:" + OUTER_SHA256
            and workflow.get("id") == RUN_ID and workflow.get("head_sha") == SOURCE_COMMIT
            and workflow.get("repository_id") == repository.get("id"), "Pinned Setup artifact metadata mismatch")
    require(0 < artifact.get("size_in_bytes", 0) <= MAX_ARCHIVE_BYTES, "Pinned artifact size is invalid")
    gh("api", f"repos/{REPOSITORY}/actions/artifacts/{ARTIFACT_ID}/zip", destination=destination)
    require(destination.stat().st_size == artifact["size_in_bytes"], "Downloaded artifact size mismatch")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--artifact-zip", type=Path, help="Offline verification only; retains the native artifact hash pin")
    args = parser.parse_args(argv)
    require(not args.artifact_zip or args.verify_only, "Local archives are allowed only with --verify-only")
    if not args.verify_only:
        require(os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY,
                "Publishing is restricted to this repository's GitHub Actions job")
        require(bool(os.environ.get("GH_TOKEN")), "GH_TOKEN must use the standard GitHub Actions job token")
    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else (
        "Windows 10/11 x64 per-user installer for Retraction Lab 0.2.0.\n\n"
        f"Frozen source: {SOURCE_COMMIT}. Native Windows build and installation checks: "
        f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}.\n\n"
        "Download Setup.exe and its SHA256 sidecar below. No Python or Node installation is needed. "
        "This unsigned research MVP uses synthetic, uncalibrated tissue models and is not a clinical tool.\n")
    require(0 < len(notes.encode()) <= 65536, "Release notes must contain at most 64KiB")
    with tempfile.TemporaryDirectory(prefix="retraction-release-") as temporary:
        work = Path(temporary)
        archive = args.artifact_zip or work / "native-artifact.zip"
        if not args.artifact_zip:
            download_native_artifact(archive)
        verified = validate_archive(archive, work / "payload")
        assets = verified["assets"]
        result = {"status": "verified", "source_commit": SOURCE_COMMIT, "native_run_id": RUN_ID,
                  "artifact_id": ARTIFACT_ID, "outer_zip_sha256": OUTER_SHA256,
                  "validation_scope": "pinned archive only" if args.artifact_zip else "native run, artifact metadata, and pinned archive",
                  "installer_sha256": verified["installer_sha256"]}
        if not args.verify_only:
            check_tag()
            release = get_release()
            if release:
                check_release(release, notes)
                check_assets(release, assets, work, complete=not release.get("draft"))
                if not release.get("draft"):
                    check_tag(required=True)
            else:
                notes_path = work / "release-notes.md"
                notes_path.write_text(notes, encoding="utf-8")
                gh("release", "create", TAG, "--repo", REPOSITORY, "--draft", "--title", TITLE,
                   "--target", SOURCE_COMMIT, "--notes-file", str(notes_path))
                release = get_release()
                require(release is not None and release.get("draft") is True, "Draft release creation did not complete")
                check_release(release, notes)
            present = check_assets(release, assets, work, complete=not release.get("draft"))
            for path in assets:
                if path.name not in present:
                    gh("release", "upload", TAG, str(path), "--repo", REPOSITORY)
            release = api(f"repos/{REPOSITORY}/releases/{release['id']}")
            check_release(release, notes)
            check_assets(release, assets, work, complete=True)
            check_tag(required=not release.get("draft"))
            if release.get("draft"):
                gh("release", "edit", TAG, "--repo", REPOSITORY, "--draft=false", "--latest")
            release = api(f"repos/{REPOSITORY}/releases/{release['id']}")
            require(release.get("draft") is False, "Release is still a draft")
            check_tag(required=True)
            check_release(release, notes)
            check_assets(release, assets, work, complete=True)
            result.update(status="published", release_url=release["html_url"], tag=TAG,
                          assets=[{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                                   "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{path.name}"}
                                  for path in assets])
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseError, OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        message = safe_message(str(error))
        print(f"Release publication failed: {message}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            annotation = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error title=Verified desktop publication failed::{annotation}")
        raise SystemExit(1)
