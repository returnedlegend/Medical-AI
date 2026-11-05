"""Fixed-policy evaluator checks using development patients, never final patients."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest

from retraction.environment import TissueRetractionEnv, normalize_env_config
from retraction.learning import train_policy

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_frozen_policies.py"
SPEC = importlib.util.spec_from_file_location("fixed_policy_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)

DEVELOPMENT_SEEDS = list(range(4_000_000_000, 4_000_000_005))


@pytest.fixture(scope="module")
def tiny_run():
    # Verify the tiny fixture's generated fitting/evaluation streams cannot
    # accidentally consume the separately reserved final cohort.
    for stream, count in [(72, 20), (73, 5)]:
        values = np.random.default_rng(np.random.SeedSequence([923, stream])).integers(0, 2**32, count, dtype=np.uint64)
        assert not set(map(int, values)) & set(range(20000, 20128))
    return train_policy({"algorithm": "tabular_q", "tabular_state_encoding": "v2",
                         "episodes": 20, "demonstration_episodes": 0,
                         "eval_episodes": 5, "max_steps": 20, "training_seed": 923})


@pytest.fixture
def policy_input(tmp_path, tiny_run):
    path = tmp_path / "fixed-policy.json"
    path.write_text(json.dumps(tiny_run, allow_nan=False))
    used = sorted(set(tiny_run["metadata"]["training_seeds"]) | set(tiny_run["metadata"]["test_seeds"]))
    assert not set(used) & set(DEVELOPMENT_SEEDS)
    return path, used


def test_real_paired_evaluation_writes_protocol_before_rollouts_and_never_trains(tmp_path, policy_input, monkeypatch):
    path, used = policy_input
    output = tmp_path / "paired"
    original_reset = TissueRetractionEnv.reset
    observed = []

    def checked_reset(env, *args, **kwargs):
        protocol_path = output / "protocol.json"
        assert protocol_path.is_file()
        protocol = json.loads(protocol_path.read_text())
        assert protocol["patient_seeds"] == DEVELOPMENT_SEEDS
        copied = output / "inputs" / "policy-fixed.json"
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == protocol["policies"][0]["file_sha256"]
        observed.append(kwargs.get("seed"))
        return original_reset(env, *args, **kwargs)

    def forbidden_train(*args, **kwargs):
        raise AssertionError("Evaluation attempted to train")

    monkeypatch.setattr(TissueRetractionEnv, "reset", checked_reset)
    monkeypatch.setattr(evaluation.learning, "train_policy", forbidden_train)
    result = evaluation.evaluate_fixed_policies({"fixed": path}, DEVELOPMENT_SEEDS, used, output,
                                               bootstrap_replicates=100)
    assert len(observed) == 20
    assert result["observed_episodes"] == result["expected_episodes"] == 20
    assert result["policy_selection_performed"] is False
    assert "selected" not in result
    assert (output / "complete.marker").is_file()
    assert not (output / "failed.json").exists()
    marker = json.loads((output / "complete.marker").read_text())
    assert marker["completed_episodes"] == 20
    assert marker["results_sha256"] == hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 20
    expected_patients = None
    for controller in result["controllers"].values():
        assert controller["n"] == 5
        assert [row["seed"] for row in controller["episodes"]] == DEVELOPMENT_SEEDS
        patients = [row["patient"] for row in controller["episodes"]]
        if expected_patients is None:
            expected_patients = patients
        assert patients == expected_patients
        assert all(row["terminated"] or row["truncated"] for row in controller["episodes"])
        assert controller["severe_harm_rate_wilson_ci"][1] > 0
    json.dumps(result, allow_nan=False)


def test_identical_fixed_policies_have_zero_paired_difference_and_common_baselines(tmp_path, policy_input):
    path, used = policy_input
    first = evaluation.evaluate_fixed_policies({"first": path}, DEVELOPMENT_SEEDS, used, tmp_path / "one", bootstrap_replicates=100)
    second = evaluation.evaluate_fixed_policies({"renamed": path, "twin": path}, DEVELOPMENT_SEEDS, used,
                                               tmp_path / "two", bootstrap_replicates=100)
    for baseline in ("baseline:random", "baseline:random_no_handoff", "baseline:scripted"):
        assert first["controllers"][baseline] == second["controllers"][baseline]
    pair = next(row for row in second["contrasts"] if row["positive_controller"] == "policy:twin"
                and row["negative_controller"] == "policy:renamed")
    assert pair["mean_return_difference"] == 0
    assert pair["paired_patient_bootstrap_ci"] == [0, 0]


def test_overlap_refused_before_output_or_any_episode(tmp_path, policy_input, monkeypatch):
    path, used = policy_input
    monkeypatch.setattr(TissueRetractionEnv, "reset", lambda *a, **k: pytest.fail("Overlap check ran an episode"))
    with pytest.raises(ValueError, match="overlap"):
        evaluation.evaluate_fixed_policies({"fixed": path}, [used[0]], used, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="overlap"):
        evaluation.evaluate_fixed_policies({"fixed": path}, DEVELOPMENT_SEEDS, sorted(used + [DEVELOPMENT_SEEDS[0]]), tmp_path / "bad2")


def test_exclusion_manifest_must_cover_input_policy_history(tmp_path, policy_input):
    path, used = policy_input
    with pytest.raises(ValueError, match="does not cover"):
        evaluation.evaluate_fixed_policies({"fixed": path}, DEVELOPMENT_SEEDS, used[:-1], tmp_path / "bad")


def test_configuration_and_physics_mismatches_are_refused(tmp_path, policy_input, tiny_run):
    path, used = policy_input
    changed = deepcopy(tiny_run)
    changed["config"]["environment"] = {"force_limit_n": 3.0}
    changed["checkpoint"]["training_config"] = changed["config"]
    changed["checkpoint"]["environment_config"] = normalize_env_config({"force_limit_n": 3.0, "max_steps": 20})
    other = tmp_path / "other.json"
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="same environment"):
        evaluation.evaluate_fixed_policies({"first": path, "other": other}, DEVELOPMENT_SEEDS, used, tmp_path / "bad")
    changed = deepcopy(tiny_run)
    changed["checkpoint"]["physics_fingerprint"] = "0" * 64
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        evaluation.evaluate_fixed_policies({"other": other}, DEVELOPMENT_SEEDS, used, tmp_path / "bad2")


def test_failed_rollout_keeps_protocol_and_failure_without_complete_marker(tmp_path, policy_input, monkeypatch):
    path, used = policy_input
    output = tmp_path / "failed"
    def fail(*args, **kwargs):
        raise RuntimeError("Injected numerical failure")
    monkeypatch.setattr(evaluation, "_rollout", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        evaluation.evaluate_fixed_policies({"fixed": path}, DEVELOPMENT_SEEDS, used, output)
    assert (output / "protocol.json").is_file()
    assert (output / "failed.json").is_file()
    assert not (output / "complete.marker").exists()
    assert json.loads((output / "failed.json").read_text())["completed_episodes"] == 0


def test_output_directory_is_never_overwritten(tmp_path, policy_input):
    path, used = policy_input
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_text("retain me")
    with pytest.raises(FileExistsError):
        evaluation.evaluate_fixed_policies({"fixed": path}, DEVELOPMENT_SEEDS, used, output)
    assert sentinel.read_text() == "retain me"


def test_zero_observed_harm_has_positive_upper_confidence_bound():
    interval = evaluation.wilson_interval(0, 128)
    assert 0 <= interval[0] < 1e-12
    assert interval[1] == pytest.approx(0.02913695627350843)
    assert interval[1] > 0


def test_bounded_declarative_json_rejects_oversize_nonfinite_and_duplicate_keys(tmp_path):
    path = tmp_path / "input.json"
    with path.open("wb") as stream:
        stream.truncate(evaluation.MAX_INPUT_BYTES + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        evaluation.read_json_bounded(path)
    for value in ('{"value":NaN}', '{"value":1,"value":2}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            evaluation.read_json_bounded(path)


def test_cumulative_policy_budget_rejects_next_file_before_json_parsing(tmp_path, policy_input, monkeypatch):
    path, used = policy_input
    second = tmp_path / 'second.json'
    second.write_bytes(b'{this file must never reach the JSON parser}')
    monkeypatch.setattr(evaluation, 'MAX_TOTAL_INPUT_BYTES', path.stat().st_size + 10)
    parsed_lengths = []
    original = evaluation.json.loads
    def track_parsing(contents, *args, **kwargs):
        parsed_lengths.append(len(contents))
        return original(contents, *args, **kwargs)
    monkeypatch.setattr(evaluation.json, 'loads', track_parsing)
    with pytest.raises(ValueError, match='Combined policy input size'):
        evaluation.evaluate_fixed_policies({'first': path, 'second': second}, DEVELOPMENT_SEEDS, used, tmp_path / 'bounded')
    assert parsed_lengths == [path.stat().st_size]
    assert not (tmp_path / 'bounded').exists()


def _synthetic_return_rows(count):
    x = np.arange(count, dtype=float)
    return {name: {'episodes': [{'return': float(value)} for value in values]}
            for name, values in {'first': np.sin(x / 3), 'second': np.cos(x / 5) * 2,
                                 'shifted': np.sin(x / 3) + 7, 'third': x % 11 - 4}.items()}


def _original_unbatched_contrasts(rows, seed, replicates):
    count = len(next(iter(rows.values()))['episodes'])
    indices = np.random.default_rng(seed).integers(0, count, size=(replicates, count))
    results = []
    for negative, positive in combinations(rows, 2):
        differences = np.array([a['return'] - b['return']
                                for a, b in zip(rows[positive]['episodes'], rows[negative]['episodes'])])
        results.append({'positive_controller': positive, 'negative_controller': negative,
                        'mean_return_difference': float(differences.mean()),
                        'paired_patient_bootstrap_ci': np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()})
    return results


@pytest.mark.parametrize('index_budget', [4 * 1024 * 1024, 128 * 8 * 3])
def test_batched_bootstrap_matches_original_exactly_and_uses_common_patient_draws(monkeypatch, index_budget):
    rows = _synthetic_return_rows(128)
    expected = _original_unbatched_contrasts(rows, seed=99174, replicates=2000)
    monkeypatch.setattr(evaluation, 'MAX_BOOTSTRAP_INDEX_BYTES', index_budget)
    actual = evaluation.paired_return_contrasts(rows, bootstrap_seed=99174, bootstrap_replicates=2000)
    assert actual == expected
    constant = next(row for row in actual if row['positive_controller'] == 'shifted'
                    and row['negative_controller'] == 'first')
    assert constant['paired_patient_bootstrap_ci'] == [7, 7]


def test_largest_patient_cohort_draws_bootstrap_indices_in_bounded_batches(monkeypatch):
    rows = _synthetic_return_rows(10000)
    original = evaluation.np.random.default_rng
    requested = []
    class RecordingRng:
        def __init__(self, seed):
            self.inner = original(seed)
        def integers(self, *args, **kwargs):
            result = self.inner.integers(*args, **kwargs)
            requested.append(result.shape)
            assert result.nbytes <= evaluation.MAX_BOOTSTRAP_INDEX_BYTES
            return result
    monkeypatch.setattr(evaluation.np.random, 'default_rng', RecordingRng)
    result = evaluation.paired_return_contrasts(rows, bootstrap_seed=99174, bootstrap_replicates=100)
    assert len(result) == 6
    assert len(requested) > 1
    assert sum(shape[0] for shape in requested) == 100
    assert all(shape[1] == 10000 for shape in requested)


def test_cli_executes_development_patients_and_writes_real_completion(tmp_path, policy_input):
    import subprocess
    import sys
    path, used = policy_input
    seeds_file, used_file = tmp_path / "dev-seeds.json", tmp_path / "used-seeds.json"
    seeds_file.write_text(json.dumps(DEVELOPMENT_SEEDS))
    used_file.write_text(json.dumps(used))
    output = tmp_path / "cli-development"
    process = subprocess.run([sys.executable, str(SCRIPT), "--policy", f"fixed={path}",
                              "--seeds-file", str(seeds_file), "--exclusion-seeds-file", str(used_file),
                              "--output", str(output), "--phase", "development"], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["observed_episodes"] == 20
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["patient_seeds"] == DEVELOPMENT_SEEDS
    assert protocol["phase"] == "development"
    assert (output / "complete.marker").is_file()
