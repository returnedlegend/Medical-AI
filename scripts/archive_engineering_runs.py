#!/usr/bin/env python3
"""Preserve all saved early synthetic RL engineering runs, including failures.

Only recognized complete research-result JSON and adjacent protocol/Python
source snapshots are included. No database, upload, environment dump, or model
execution is involved. Frozen multi-seed studies have their separate archives.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def archive(project):
    data = project / ".data"
    candidates = list(data.glob("rl-*.json"))
    candidates += [p / "results.json" for p in data.glob("rl-*")
                   if p.is_dir() and "frozen" not in p.name and (p / "results.json").is_file()]
    payload, runs = {}, []
    for path in sorted(candidates):
        raw = path.read_bytes()
        result = json.loads(raw)
        if (not isinstance(result, dict) or not isinstance(result.get("checkpoint"), dict)
                or not isinstance(result.get("metadata"), dict) or not isinstance(result.get("evaluation"), dict)):
            continue
        name = path.relative_to(project).as_posix()
        payload[name] = raw
        adjacent = []
        if path.name == "results.json":
            protocol = path.parent / "protocol.json"
            if protocol.is_file():
                adjacent.append(protocol)
            source = path.parent / "source"
            if source.is_dir():
                adjacent.extend(source.glob("*.py"))
        for item in adjacent:
            payload[item.relative_to(project).as_posix()] = item.read_bytes()
        metrics = result["evaluation"]["learned"]
        runs.append({"source": name, "result_sha256": hashlib.sha256(raw).hexdigest(),
                     "algorithm": result.get("algorithm"), "model_version": result.get("model_version"),
                     "config": result.get("config"),
                     "curve_rows": len(result.get("learning_curve", [])),
                     "learned_validation": {key: metrics[key] for key in
                                             ("n", "success_rate", "severe_harm_rate", "mean_return", "mean_max_damage")},
                     "adjacent_records": [p.relative_to(project).as_posix() for p in adjacent],
                     "historical_source_provenance": "Preserve recorded protocol/hash fields where present. Exact historical source snapshots are incomplete; do not claim an exact reproducible rerun."})
    if not runs:
        raise ValueError("No recognized saved engineering runs were found")
    manifest = {"format": "retraction-engineering-development-archive-v1",
                "purpose": "Preserve early synthetic development outcomes, including negative/harmful runs; these inspected validation streams are development data.",
                "limitations": ["Older engine checkpoints may be incompatible with the current app and must not be relabeled or promoted.",
                                "Early source capture was incomplete. JSON/version/hash fields do not replace missing exact source snapshots.",
                                "No independent final patients or clinical evidence are included.",
                                "Frozen studies001–003 are separate archives; this archive does not silently replace their protocols."],
                "run_count": len(runs), "runs": runs,
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                          for name, raw in sorted(payload.items())}}
    payload["MANIFEST.json"] = json_bytes(manifest)
    payload["README.md"] = ("# Early synthetic RL engineering outputs\n\n"
                             "All recognized saved early RL result JSON files are preserved, including handoff-only, zero-success and harmful variants. "
                             "Read MANIFEST.json for per-run outcomes, original paths and exact byte hashes. "
                             "These inspected runs are development data, and historical source capture is incomplete. "
                             "Older model versions are not current playable references. No weights, reward or physiology are changed by this archive. "
                             "The frozen multi-seed studies and independent final test have separate artifacts.\n").encode()
    payload["scripts/archive_engineering_runs.py"] = Path(__file__).read_bytes()

    def compress():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for name, raw in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                output.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        return stream.getvalue()
    compressed = compress()
    if compressed != compress():
        raise AssertionError("Archive construction is not deterministic")
    with zipfile.ZipFile(io.BytesIO(compressed)) as check:
        if check.testzip() is not None or set(check.namelist()) != set(payload):
            raise AssertionError("Archive integrity validation failed")
        for name, raw in payload.items():
            if check.read(name) != raw:
                raise AssertionError("Archived payload differs from exact source bytes")
    return compressed, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, manifest = archive(args.project)
    with args.output.open("xb") as output:
        output.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest + "  " + args.output.name + "\n")
    print(json.dumps({"runs": manifest["run_count"], "source_files": len(manifest["files"]),
                      "archive_bytes": len(raw), "sha256": digest}))


if __name__ == "__main__":
    main()
