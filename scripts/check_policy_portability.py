#!/usr/bin/env python3
"""Collect or compare fixed development trajectories on Linux and Windows.

This check never trains and never accepts a final-cohort seed. Run the same
checkout with each platform's Python, then compare the two saved JSON files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "retraction-development-portability-v1"
SEEDS = [2, 17, 42, 74, 100]
POLICIES = {
    "v1": "stateful-policy-1.1.json",
    "v2": "stateful-policy-1.1-study-002.json",
    "v3": "stateful-policy-1.1-study-003-v3.json",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def collect():
    import numpy as np
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(ROOT / "backend"))
    from retraction import learning
    from retraction.environment import TissueRetractionEnv, PHYSICS_FINGERPRINT

    rows, inputs = [], {}
    with threadpool_limits(limits=1):
        for label, name in POLICIES.items():
            raw = (ROOT / "examples" / name).read_bytes()
            run = json.loads(raw)["result"]
            checkpoint = run["checkpoint"]
            inputs[label] = {"file": name, "sha256": sha(raw)}
            model, low, high, actions = learning.load_checkpoint(checkpoint)
            for seed in SEEDS:
                env = TissueRetractionEnv(checkpoint["environment_config"])
                try:
                    observation, info = env.reset(seed=seed)
                    total, digest = 0.0, hashlib.sha256()
                    for step in range(checkpoint["max_steps"]):
                        encoded = learning._encode(observation, low, high, checkpoint.get("feature_transform"))
                        values, _ = model.forward(encoded)
                        index = int(np.argmax(values))
                        observation, reward, terminated, truncated, info = env.step(actions[index])
                        total += reward
                        digest.update(canonical({"index": index, "reward": reward, "state": info["state"]}))
                        if terminated or truncated:
                            break
                    rows.append({"policy": label, "seed": seed, "steps": step + 1,
                                 "outcome": info["termination_reason"], "return": total,
                                 "trajectory_sha256": digest.hexdigest(), "state": info["state"]})
                finally:
                    env.close()
    return {"format": FORMAT, "development_only": True, "platform": sys.platform,
            "python": platform.python_version(), "numpy": np.__version__,
            "physics_fingerprint": PHYSICS_FINGERPRINT, "policies": inputs, "seeds": SEEDS,
            "source_sha256": {name: sha((ROOT / "backend/retraction" / name).read_bytes())
                              for name in ("environment.py", "tissue.py", "learning.py", "tabular.py")},
            "rows": rows}


def compare(paths):
    raw = [path.read_bytes() for path in paths]
    reports = [json.loads(value) for value in raw]
    first, second = reports
    for value in reports:
        if value.get("format") != FORMAT or value.get("seeds") != SEEDS or len(value.get("rows", [])) != 15:
            raise ValueError("Expected a complete fixed development-portability report")
    if {first["platform"], second["platform"]} != {"linux", "win32"}:
        raise ValueError("Comparison requires one actual Linux-Python and one Windows-Python report")
    for key in ("policies", "source_sha256", "physics_fingerprint", "numpy"):
        if first[key] != second[key]:
            raise ValueError(f"The two runs differ in their frozen {key}")
    expected = [(label, seed) for label in POLICIES for seed in SEEDS]
    for value in reports:
        if [(row["policy"], row["seed"]) for row in value["rows"]] != expected:
            raise ValueError("Development cases or their order differ")
    differences = [{"policy": a["policy"], "seed": a["seed"]}
                   for a, b in zip(first["rows"], second["rows"]) if a != b]
    return {"format": FORMAT, "development_only": True, "cases": 15,
            "exact_matches": 15 - len(differences), "all_equal": not differences,
            "differences": differences, "report_sha256": [sha(value) for value in raw],
            "platforms": [{key: value[key] for key in ("platform", "python", "numpy")} for value in reports],
            "physics_fingerprint": first["physics_fingerprint"],
            "limitations": ["Fifteen development cases do not establish equivalence for every patient or platform.",
                            "Windows Python under Wine is distinct from native Windows GUI and installer validation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; never overwrites")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("LINUX_REPORT", "WINDOWS_REPORT"))
    args = parser.parse_args()
    result = compare(args.compare) if args.compare else collect()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "development_only": True,
                      "cases": result.get("cases", len(result.get("rows", []))),
                      "all_equal": result.get("all_equal")}))
    return 0 if result.get("all_equal", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
