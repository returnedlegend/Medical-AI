#!/usr/bin/env python3
"""Inventory recorded and conservatively excluded synthetic patient seeds.

This script reads artifacts only. It does not import the simulator, instantiate an
environment, train, or evaluate a policy. Planned RNG streams are excluded even
when their training job has not yet finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def stream_seeds(config):
    seed = int(config["training_seed"])
    episodes = int(config["episodes"])
    count_demo = int(config.get("demonstration_episodes", 0))
    count_test = int(config.get("eval_episodes", 20))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 72]))
    train = [int(n) for n in rng.integers(0, 2**32, episodes, dtype=np.uint64)]
    used = set(train)
    generated = {}
    for label, stream, count in (("demonstration", 74, count_demo), ("validation", 73, count_test)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        values = []
        while len(values) < count:
            value = int(rng.integers(0, 2**32, dtype=np.uint64))
            if value not in used:
                used.add(value)
                values.append(value)
        generated[label] = values
    return {"training": train, **generated}


def collect(project, additional):
    groups = {}
    records = []

    def record(kind, source, values, **details):
        values = set(values)
        if any(type(n) is not int or not 0 <= n < 2**32 for n in values):
            raise ValueError(f"Invalid patient seed at {source}")
        groups.setdefault(kind, set()).update(values)
        digest = hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()
        records.append({"kind": kind, "source": source, "unique_count": len(values),
                        "seed_set_sha256": digest, **details})

    def result_seeds(value, source, **details):
        if not isinstance(value, dict):
            return
        if value.get("format") == "retraction-bundled-reference-v1":
            value = value["result"]
        metadata = value.get("metadata", {})
        keys = ("training_seeds", "training_episode_seeds", "demonstration_seeds", "test_seeds")
        if any(key in metadata for key in keys):
            seeds = [n for key in keys for n in metadata.get(key, [])]
            for summary in value.get("evaluation", {}).values():
                if isinstance(summary, dict):
                    seeds.extend(ep["seed"] for ep in summary.get("episodes", []) if "seed" in ep)
            record("recorded_learning_histories", source, seeds, **details)

    data = project / ".data"
    for path in sorted(data.rglob("*.json")):
        if path.name.startswith("rl-all-prior-patient-seeds") or path.name == "reserved-final-patient-seeds.json":
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        source = str(path.relative_to(project))
        if not isinstance(value, dict):
            continue
        result_seeds(value, source, source_sha256=hashlib.sha256(raw).hexdigest(),
                     source_hash_meaning="Exact JSON file bytes")
        if path.name == "protocol.json":
            configs = value.get("configs", [])
            configs += [item["config"] for item in value.get("entries", [])]
            for config in configs:
                if "training_seed" in config and "episodes" in config:
                    streams = stream_seeds(config)
                    record("planned_protocol_histories", source,
                           [n for seq in streams.values() for n in seq], config=config,
                           source_sha256=hashlib.sha256(raw).hexdigest(),
                           stream_counts={key: len(vals) for key, vals in streams.items()})

    for path in sorted(data.rglob("*.sqlite3")):
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            source = str(path.relative_to(project))
            if "sessions" in tables:
                query = "SELECT seed,engine_version FROM sessions ORDER BY seed,engine_version"
                rows = db.execute(query).fetchall()
                record("recorded_stateful_sessions", source,
                       [seed for seed, version in rows if version.startswith("tissue-retraction-")],
                       row_count=len(rows), query=query,
                       query_projection_sha256=hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
            if "runs" in tables:
                query = "SELECT id,config,result FROM runs ORDER BY id"
                for identity, config_raw, result_raw in db.execute(query):
                    projection = {"id": identity, "config": config_raw, "result": result_raw}
                    details = {"query": query, "query_row_id": identity,
                               "query_projection_sha256": hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
                    if result_raw:
                        result_seeds(json.loads(result_raw), source + "#" + identity, **details)
                    config = json.loads(config_raw)
                    if "training_seed" in config and "episodes" in config:
                        streams = stream_seeds(config)
                        record("planned_database_training_histories", source + "#" + identity,
                               [n for vals in streams.values() for n in vals], config=config, **details)
        finally:
            db.close()

    # These are real training fixtures/manual integration runs. Some test outcomes
    # are ephemeral; excluding their full deterministic streams is conservative.
    fixtures = [(19, 20, 40, 5), (7, 20, 0, 5), (29, 20, 0, 5),
                (42, 20, 40, 5), (79, 20, 5, 5), (91, 20, 5, 5),
                (923, 20, 0, 5), (932, 20, 0, 5), (998, 20, 5, 5), (999, 20, 0, 5)]
    for seed, episodes, demos, evaluations in fixtures:
        config = {"training_seed": seed, "episodes": episodes,
                  "demonstration_episodes": demos, "eval_episodes": evaluations}
        streams = stream_seeds(config)
        record("conservative_fixture_and_inline_streams", "documented training fixtures/inline integration", 
               [n for vals in streams.values() for n in vals], config=config)

    record("conservative_direct_development_range", "manual/controller/test seeds and numerical validation",
           range(10020), meaning="Exclude every integer 0–10019 as a conservative superset; this is not a claim that all were executed.")
    record("documented_direct_development_patients", "backend/tests/test_policy_evaluation.py",
           range(4_000_000_000, 4_000_000_005),
           meaning="Explicit development-only independent-evaluator test patients.")

    for path in additional:
        raw = path.read_bytes()
        audit_manifest = path.with_name(path.stem + "-manifest" + path.suffix)
        audit_details = ({"audit_manifest": str(audit_manifest),
                          "audit_manifest_sha256": hashlib.sha256(audit_manifest.read_bytes()).hexdigest()}
                         if audit_manifest.is_file() else {})
        record("additional_read_only_audit", str(path), json.loads(raw),
               source_sha256=hashlib.sha256(raw).hexdigest(), **audit_details)
    union = sorted(set().union(*groups.values()))
    return union, {"created_at": datetime.now(timezone.utc).isoformat(),
                   "format": "retraction-prior-patient-exclusions-v1",
                   "method": "Read-only artifact/SQLite inventory plus deterministic planned streams and explicitly conservative exclusions; no simulator resets.",
                   "rng_streams": {"training": 72, "demonstration": 74, "validation": 73},
                   "unique_excluded_count": len(union),
                   "group_unique_counts": {key: len(vals) for key, vals in groups.items()},
                   "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--additional-seeds", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    args = parser.parse_args()
    seeds, manifest = collect(args.project, args.additional_seeds)
    reserved = json.loads(args.reserved.read_text())
    overlap = sorted(set(seeds) & set(reserved))
    manifest["reserved_seed_count"] = len(reserved)
    manifest["reserved_overlap"] = overlap
    manifest["seed_list_sha256"] = hashlib.sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()
    manifest["collector_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, value in ((args.output, seeds), (args.manifest, manifest)):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ("unique_excluded_count", "group_unique_counts", "reserved_seed_count", "reserved_overlap", "seed_list_sha256")}))
    return 2 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
