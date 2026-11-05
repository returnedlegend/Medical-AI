#!/usr/bin/env python3
"""Evaluate fixed declarative policies on paired patients; never train or select.

The caller must freeze policies and supply the union of all prior fitting,
demonstration and validation patient seeds. Do not execute a reserved final
cohort before its owner authorizes that one-time evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))

from retraction import environment, learning, tabular, tissue  # noqa: E402
from retraction.environment import TissueRetractionEnv, normalize_env_config  # noqa: E402

MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 256 * 1024 * 1024
MAX_BOOTSTRAP_INDEX_BYTES = 4 * 1024 * 1024
FORMAT = "retraction-fixed-policy-evaluation-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json_bounded(path: Path, *, remaining_bytes: int | None = None):
    limit = MAX_INPUT_BYTES if remaining_bytes is None else min(MAX_INPUT_BYTES, remaining_bytes)
    if limit < 0:
        raise ValueError('Combined policy input size exceeds 256 MiB')
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        if remaining_bytes is not None and remaining_bytes < MAX_INPUT_BYTES:
            raise ValueError('Combined policy input size exceeds 256 MiB')
        raise ValueError(f"Input exceeds the 32 MiB limit: {path}")
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key}")
            result[key] = value
        return result
    return json.loads(content, parse_constant=reject_constant, object_pairs_hook=object_pairs), content


def validate_seeds(values, name, *, nonempty=True):
    if (not isinstance(values, list) or (nonempty and not values) or len(values) > 1_000_000
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)):
        raise ValueError(f"{name} must be an integer seed list in [0, 2**32)")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct seeds")
    return list(values)


def write_json_exclusive(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def wilson_interval(successes: int, count: int) -> list[float]:
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("Wilson interval requires positive n and 0 <= successes <= n")
    z = 1.959963984540054
    p, z2 = successes / count, z * z
    denominator = 1 + z2 / count
    center = (p + z2 / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _prepare_policy(label: str, path: Path, *, remaining_bytes: int | None = None):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", label):
        raise ValueError("Policy labels must be lowercase letters/digits/underscore/hyphen and start with a letter")
    payload, raw = read_json_bounded(path, remaining_bytes=remaining_bytes)
    run = payload.get("result") if isinstance(payload, dict) and "result" in payload else payload
    if not isinstance(run, dict) or not isinstance(run.get("checkpoint"), dict):
        raise ValueError("Provide a complete run or bundled reference JSON, not a bare checkpoint")
    checkpoint, config, metadata = run["checkpoint"], run.get("config"), run.get("metadata")
    learning.normalize_training_config(config)
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("Policy run lacks complete configuration or seed provenance")
    if checkpoint.get("training_config") != config:
        raise ValueError("Policy training configuration differs from its run")
    expected_environment = normalize_env_config({**config["environment"], "max_steps": config["max_steps"]})
    if checkpoint.get("environment_config") != expected_environment:
        raise ValueError("Policy environment configuration differs from its run")
    if run.get("model_version") != checkpoint.get("model_version"):
        raise ValueError("Run and checkpoint model versions disagree")
    if run.get("algorithm") != checkpoint.get("algorithm"):
        raise ValueError("Run and checkpoint algorithms disagree")
    training = metadata.get("training_episode_seeds")
    demos = metadata.get("demonstration_seeds")
    validation = validate_seeds(metadata.get("test_seeds"), "Recorded validation patients")
    for values in (training, demos):
        if not isinstance(values, list) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
            raise ValueError("Policy run has invalid fitting/demonstration seed provenance")
    curve = run.get("learning_curve")
    if (not isinstance(curve, list) or any(not isinstance(row, dict) for row in curve)
            or len(training) != config["episodes"] or len(demos) != config["demonstration_episodes"]
            or len(validation) != config["eval_episodes"] or metadata.get("training_seeds") != training + demos
            or len(curve) != config["episodes"] or [row.get("seed") for row in curve] != training):
        raise ValueError("Policy run is incomplete or its fitting/validation counts disagree")
    used = set(training) | set(demos) | set(validation)
    if (set(training) | set(demos)) & set(validation):
        raise ValueError("Policy run reused fitting patients for validation")
    validation_results = run.get("evaluation")
    if not isinstance(validation_results, dict):
        raise ValueError("Policy run lacks its completed validation evaluation")
    for mode in ("learned", "random", "random_no_handoff", "scripted"):
        mode_result = validation_results.get(mode)
        rows = mode_result.get("episodes") if isinstance(mode_result, dict) else None
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or [row.get("seed") for row in rows] != validation):
            raise ValueError("Policy run does not contain a complete paired validation evaluation")
    return {"label": label, "path": str(Path(path).resolve()), "raw": raw, "file_sha256": sha256(raw),
            "checkpoint": checkpoint, "environment_config": expected_environment, "used_seeds": used,
            "algorithm": checkpoint["algorithm"], "training_config": config}


def _episode_summary(seed, total, steps, terminated, truncated, info, maxima, reward_totals, patient):
    state, physiology = info["state"], info["state"]["physiology"]
    return {"seed": seed, "return": float(total), "steps": steps, "terminated": bool(terminated),
            "truncated": bool(truncated), "outcome": info["termination_reason"],
            "success": bool(info["success"]), "severe_harm": bool(info["severe_harm"]),
            "handoff": info["termination_reason"] == "handoff", "patient": patient,
            "max_damage": float(state["max_damage"]), "critical_damage": float(state["critical_damage"]),
            "blood_loss_ml": float(physiology["blood_loss_ml"]), "bile_leak_ml": float(physiology["bile_leak_ml"]),
            "final_exposure": float(state["exposure"]), "peak_exposure": float(state["peak_exposure"]),
            "simulation_time_s": float(state["time_s"]), **maxima,
            "reward_components_total": reward_totals, "final_state_sha256": sha256(canonical_bytes(state))}


def _rollout(controller, prepared, config, seed, rng_seed, canonical_primitives):
    env = TissueRetractionEnv(config)
    try:
        observation, info = env.reset(seed=seed)
        patient = dict(info["state"]["patient"])
        # Common baseline streams are independent of policy label/training seed.
        mode_key = 1 if controller == "baseline:random" else 2
        rng = np.random.default_rng(np.random.SeedSequence([rng_seed, seed, mode_key]))
        available = canonical_primitives
        if controller == "baseline:random_no_handoff":
            available = canonical_primitives[canonical_primitives[:, 6] <= 0.5]
            if not len(available):
                raise ValueError("The canonical baseline library has no non-handoff action")
        total, reward_totals = 0.0, {}
        maxima = {"peak_contact_force_n": 0.0, "peak_temperature_c": 37.0}
        terminated = truncated = False
        for step in range(config["max_steps"]):
            if prepared is not None:
                model, low, high, primitives = prepared["compiled"]
                encoded = learning._encode(observation, low, high, prepared["checkpoint"].get("feature_transform"))
                values, _ = model.forward(encoded)
                action = primitives[int(np.argmax(values))]
            elif controller in ("baseline:random", "baseline:random_no_handoff"):
                action = available[int(rng.integers(len(available)))]
            else:
                action = env.scripted_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            for name, value in info["reward_components"].items():
                reward_totals[name] = reward_totals.get(name, 0.0) + float(value)
            maxima["peak_contact_force_n"] = max(maxima["peak_contact_force_n"], info["state"]["tools"]["grasper"]["force_n"])
            maxima["peak_temperature_c"] = max(maxima["peak_temperature_c"], max(node["temperature_c"] for node in info["state"]["tissue"]["nodes"]))
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise RuntimeError("Episode exhausted its horizon without a terminal/truncation outcome")
        return _episode_summary(seed, total, step + 1, terminated, truncated, info, maxima, reward_totals, patient)
    finally:
        env.close()


def aggregate(episodes):
    if not episodes:
        raise ValueError("Cannot summarize an empty evaluation")
    count = len(episodes)
    successes = sum(row["success"] for row in episodes)
    harmed = sum(row["severe_harm"] for row in episodes)
    return {"n": count, "successes": successes, "severe_harm_episodes": harmed,
            "handoffs": sum(row["handoff"] for row in episodes),
            "success_rate": successes / count, "success_rate_wilson_ci": wilson_interval(successes, count),
            "severe_harm_rate": harmed / count, "severe_harm_rate_wilson_ci": wilson_interval(harmed, count),
            "mean_return": math.fsum(row["return"] for row in episodes) / count,
            "mean_max_damage": math.fsum(row["max_damage"] for row in episodes) / count,
            "mean_blood_loss_ml": math.fsum(row["blood_loss_ml"] for row in episodes) / count}


def paired_return_contrasts(result_rows, *, bootstrap_seed, bootstrap_replicates):
    """Share each bounded index batch across every paired controller contrast.

    The default 128-patient/2,000-replicate call fits in one batch and retains
    the original RNG call and arithmetic exactly. Larger calls retain the same
    flat RNG stream while bounding both indices and indexed-return temporaries.
    Only one scalar bootstrap mean per contrast/replicate is retained.
    """
    pairs = list(combinations(result_rows, 2))
    if not pairs:
        return []
    returns = {name: np.asarray([row['return'] for row in value['episodes']], dtype=float)
               for name, value in result_rows.items()}
    count = len(next(iter(returns.values())))
    if not count or any(len(values) != count for values in returns.values()):
        raise ValueError('Paired bootstrap requires equal nonempty patient cohorts')
    rows_per_batch = max(1, min(bootstrap_replicates,
                               MAX_BOOTSTRAP_INDEX_BYTES // (count * np.dtype(np.int64).itemsize)))
    samples = np.empty((len(pairs), bootstrap_replicates), dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    for start in range(0, bootstrap_replicates, rows_per_batch):
        stop = min(start + rows_per_batch, bootstrap_replicates)
        indices = rng.integers(0, count, size=(stop - start, count))
        for position, (negative, positive) in enumerate(pairs):
            differences = returns[positive] - returns[negative]
            samples[position, start:stop] = differences[indices].mean(axis=1)
    return [{'positive_controller': positive, 'negative_controller': negative,
             'mean_return_difference': float((returns[positive] - returns[negative]).mean()),
             'paired_patient_bootstrap_ci': np.quantile(samples[position], [.025, .975]).tolist()}
            for position, (negative, positive) in enumerate(pairs)]


def evaluate_fixed_policies(policy_paths: dict[str, Path], seeds: list[int], exclusion_seeds: list[int],
                            output_directory: Path, *, phase="development", baseline_rng_seed=99173,
                            bootstrap_seed=99174, bootstrap_replicates=2000, progress_callback=None):
    if not isinstance(policy_paths, dict) or not 1 <= len(policy_paths) <= 32:
        raise ValueError("Provide 1–32 fixed labeled policy files")
    seeds = validate_seeds(seeds, "Evaluation patients")
    if len(seeds) > 10000:
        raise ValueError("At most 10000 evaluation patients are supported")
    exclusion_seeds = validate_seeds(exclusion_seeds, "Prior-used patient exclusions", nonempty=False)
    if phase not in ("development", "final"):
        raise ValueError("phase must be development or final")
    if type(baseline_rng_seed) is not int or not 0 <= baseline_rng_seed < 2**32 or type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError("Evaluation RNG seeds must be integers in [0, 2**32)")
    if type(bootstrap_replicates) is not int or not 100 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates must be 100..10000")
    prepared = []
    remaining_bytes = MAX_TOTAL_INPUT_BYTES
    for label, path in sorted(policy_paths.items()):
        policy = _prepare_policy(label, Path(path), remaining_bytes=remaining_bytes)
        prepared.append(policy)
        remaining_bytes -= len(policy['raw'])
    used = set().union(*(policy["used_seeds"] for policy in prepared))
    if not used <= set(exclusion_seeds):
        raise ValueError("The exclusion manifest does not cover every policy's prior fitting/demo/validation patients")
    if set(seeds) & (used | set(exclusion_seeds)):
        raise ValueError("Evaluation patient overlap with prior fitting/demo/validation or exclusion seeds")
    config = prepared[0]["environment_config"]
    if any(policy["environment_config"] != config for policy in prepared):
        raise ValueError("All fixed policies must use the same environment configuration")
    env = TissueRetractionEnv(config)
    try:
        description = env.describe()
        for policy in prepared:
            learning.validate_checkpoint_compatibility(policy["checkpoint"], description)
            if (policy["checkpoint"].get("observation_low") != description["observation_low"] or
                    policy["checkpoint"].get("observation_high") != description["observation_high"]):
                raise ValueError("Policy observation bounds differ from the current environment")
            policy["compiled"] = learning.load_checkpoint(policy["checkpoint"])
    finally:
        env.close()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "inputs").mkdir()
    for policy in prepared:
        with (output_directory / "inputs" / f'policy-{policy["label"]}.json').open("xb") as stream:
            stream.write(policy["raw"])
            stream.flush()
            os.fsync(stream.fileno())
    names, primitives = learning.action_primitives()
    source_paths = {"environment.py": Path(environment.__file__), "tissue.py": Path(tissue.__file__),
                    "learning.py": Path(learning.__file__), "tabular.py": Path(tabular.__file__),
                    "evaluate_frozen_policies.py": Path(__file__)}
    protocol = {"format": FORMAT, "created_at": timestamp(), "phase": phase,
                "purpose": "Fixed-policy paired evaluation only; no training, ranking or model selection.",
                "environment_version": description["version"], "physics_fingerprint": description["physics_fingerprint"],
                "environment_config": config, "python_version": platform.python_version(), "numpy_version": np.__version__,
                "source_sha256": {name: sha256(path.read_bytes()) for name, path in source_paths.items()},
                "policies": [{"label": p["label"], "source_path": p["path"], "file_sha256": p["file_sha256"],
                              "checkpoint_sha256": sha256(canonical_bytes(p["checkpoint"])), "algorithm": p["algorithm"],
                              "training_config": p["training_config"]} for p in prepared],
                "patient_seeds": seeds, "patient_seed_sha256": sha256(canonical_bytes(seeds)),
                "prior_used_patient_seeds": sorted(exclusion_seeds),
                "prior_used_seed_set_sha256": sha256(canonical_bytes(sorted(exclusion_seeds))),
                "baseline_rng_seed": baseline_rng_seed,
                "baseline_rng_rule": "SeedSequence([baseline_rng_seed, patient_seed, mode]) with mode1=random,2=random_no_handoff; streams are independent of policy label and training seed.",
                "baseline_primitives": {"names": names, "actions": primitives.tolist()},
                "scripted_controller_version": description["scripted_controller_version"],
                "bootstrap": {"seed": bootstrap_seed, "replicates": bootstrap_replicates,
                              "max_index_batch_bytes": MAX_BOOTSTRAP_INDEX_BYTES,
                              "method": "Paired patient percentile bootstrap; common resampled indices for every contrast; no multiple-comparison adjustment."},
                "limitations": ["Synthetic unvalidated dynamics; no claim of clinical performance or generalization.",
                                "Exclusion completeness across all prior studies is supplied by the caller; input-policy histories are explicitly checked.",
                                "Intervals condition on these fixed trained policies and exclude training-run and model uncertainty.",
                                "Zero observed harm does not establish zero risk; Wilson intervals remain nonzero.",
                                "This evaluator never selects or exports a new reference policy."]}
    # This immutable protocol and copied input bytes exist before the first reset.
    write_json_exclusive(output_directory / "protocol.json", protocol)
    controllers = [(f'policy:{p["label"]}', p) for p in prepared] + [
        ("baseline:random", None), ("baseline:random_no_handoff", None), ("baseline:scripted", None)]
    result_rows = {}
    completed = 0
    current_controller, current_seed = None, None
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1), (output_directory / "episodes.jsonl").open("x", encoding="utf-8") as log:
            for controller, policy in controllers:
                episodes = []
                for seed in seeds:
                    current_controller, current_seed = controller, seed
                    row = _rollout(controller, policy, config, seed, baseline_rng_seed, primitives)
                    episodes.append(row)
                    log.write(json.dumps({"controller": controller, **row}, allow_nan=False) + "\n")
                    log.flush()
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / (len(controllers) * len(seeds)), controller, seed)
                result_rows[controller] = {**aggregate(episodes), "episodes": episodes}
        # A seeded patient must have identical starting anatomy for all controls.
        for index in range(len(seeds)):
            expected = result_rows[controllers[0][0]]["episodes"][index]["patient"]
            if any(value["episodes"][index]["patient"] != expected for value in result_rows.values()):
                raise RuntimeError("Paired patients have different initial anatomy")
        contrasts = paired_return_contrasts(result_rows, bootstrap_seed=bootstrap_seed,
                                            bootstrap_replicates=bootstrap_replicates)
        result = {"format": FORMAT, "completed_at": timestamp(), "phase": phase,
                  "protocol_sha256": sha256((output_directory / "protocol.json").read_bytes()),
                  "physics_fingerprint": description["physics_fingerprint"],
                  "controllers": result_rows, "contrasts": contrasts,
                  "observed_episodes": completed, "expected_episodes": len(controllers) * len(seeds),
                  "policy_selection_performed": False, "limitations": protocol["limitations"]}
        if completed != result["expected_episodes"]:
            raise RuntimeError("Evaluation episode count differs from frozen protocol")
        write_json_exclusive(output_directory / "results.json", result)
        write_json_exclusive(output_directory / "complete.marker", {"completed_at": timestamp(),
                             "protocol_sha256": result["protocol_sha256"],
                             "results_sha256": sha256((output_directory / "results.json").read_bytes()),
                             "completed_episodes": completed})
        return result
    except Exception as error:
        write_json_exclusive(output_directory / "failed.json", {"failed_at": timestamp(),
                             "controller": current_controller, "patient_seed": current_seed,
                             "completed_episodes": completed, "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="Fixed label=path to complete run/reference JSON; repeat for each policy")
    parser.add_argument("--seeds-file", type=Path, required=True, help="JSON list of evaluation patient seeds")
    parser.add_argument("--exclusion-seeds-file", type=Path, required=True, help="JSON union of all prior fitting/demo/validation patient seeds")
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing directories are never overwritten")
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--baseline-rng-seed", type=int, default=99173)
    parser.add_argument("--bootstrap-seed", type=int, default=99174)
    args = parser.parse_args()
    try:
        policies = {}
        for supplied in args.policy:
            if "=" not in supplied:
                raise ValueError("--policy requires label=path")
            label, path = supplied.split("=", 1)
            if label in policies:
                raise ValueError("Policy labels must be distinct")
            policies[label] = Path(path)
        seeds, _ = read_json_bounded(args.seeds_file)
        exclusions, _ = read_json_bounded(args.exclusion_seeds_file)
        result = evaluate_fixed_policies(policies, seeds, exclusions, args.output, phase=args.phase,
                                        baseline_rng_seed=args.baseline_rng_seed, bootstrap_seed=args.bootstrap_seed)
        print(json.dumps({"output": str(args.output), "observed_episodes": result["observed_episodes"],
                          "controllers": {name: {key: value for key, value in row.items() if key != "episodes"}
                                          for name, row in result["controllers"].items()}}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError) as error:
        print(f"Fixed-policy evaluation did not complete: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
