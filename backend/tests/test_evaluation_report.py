"""Reporting tests from a retained genuine development evaluation, never final rollouts."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/report_frozen_evaluation.py'
SPEC = importlib.util.spec_from_file_location('frozen_evaluation_report', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
FIXTURE = Path(__file__).parent / 'fixtures/fixed-policy-development.zip'


@pytest.fixture
def development(tmp_path):
    directory = tmp_path / 'evaluation'
    directory.mkdir()
    with zipfile.ZipFile(FIXTURE) as archive:
        assert all(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in archive.namelist())
        archive.extractall(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    assert protocol['phase'] == 'development'
    assert protocol['patient_seeds'] == list(range(4_000_000_000, 4_000_000_005))
    design = {'format': 'retraction-final-evaluation-design-v1', 'physics_fingerprint': protocol['physics_fingerprint'],
              'policies': [{'label': row['label'], 'source': 'retained genuine development fixture'} for row in protocol['policies']],
              'patient_seeds': protocol['patient_seeds'], 'baselines': ['random', 'random_no_handoff', 'scripted'],
              'expected_episodes': results['expected_episodes'], 'baseline_rng_seed': protocol['baseline_rng_seed'],
              'bootstrap_rng_seed': protocol['bootstrap']['seed'], 'bootstrap_replicates': protocol['bootstrap']['replicates']}
    design_path = tmp_path / 'development-design.json'
    design_path.write_text(json.dumps(design, indent=2) + '\n')
    return directory, design_path


def reseal_results(directory, results):
    contents = (json.dumps(results, indent=2) + '\n').encode()
    (directory / 'results.json').write_bytes(contents)
    marker = json.loads((directory / 'complete.marker').read_text())
    marker['results_sha256'] = report.sha(contents)
    (directory / 'complete.marker').write_text(json.dumps(marker))


def test_genuine_completed_development_records_generate_verified_report_without_evaluation(development, tmp_path):
    directory, design = development
    before = {str(path.relative_to(directory)): report.sha(path.read_bytes()) for path in directory.rglob('*') if path.is_file()}
    output = tmp_path / 'report'
    manifest = report.generate_report(directory, design, output)
    assert manifest['phase'] == 'development'
    assert manifest['simulator_executed'] is False and manifest['policy_selection_performed'] is False
    assert manifest['shared_patients'] == 5 and manifest['recorded_rollouts'] == 20
    assert manifest['controller_order'] == ['policy:fixed', 'baseline:random', 'baseline:random_no_handoff', 'baseline:scripted']
    rows = list(csv.DictReader((output / 'controllers.csv').open()))
    assert len(rows) == 4
    assert all(float(row['severe_harm_rate_ci_upper']) > 0 for row in rows)
    assert len(list(csv.DictReader((output / 'paired-return-contrasts.csv').open()))) == 6
    text = (output / 'report.md').read_text()
    assert 'Development fixed-policy evaluation' in text
    assert 'not 20 independent patients' in text
    assert 'exclude training-run and tissue-model uncertainty' in text
    for name, digest in manifest['outputs_sha256'].items():
        assert report.sha((output / name).read_bytes()) == digest
    after = {str(path.relative_to(directory)): report.sha(path.read_bytes()) for path in directory.rglob('*') if path.is_file()}
    assert after == before


@pytest.mark.parametrize('damage', ['marker_hash', 'copied_input', 'aggregate', 'paired_ci', 'rollout_log'])
def test_inconsistent_completion_or_outcomes_are_refused_before_report_creation(development, tmp_path, damage):
    directory, design = development
    if damage == 'marker_hash':
        marker = json.loads((directory / 'complete.marker').read_text()); marker['results_sha256'] = '0' * 64
        (directory / 'complete.marker').write_text(json.dumps(marker))
    elif damage == 'copied_input':
        path = directory / 'inputs/policy-fixed.json'; path.write_bytes(path.read_bytes() + b' ')
    elif damage == 'rollout_log':
        path = directory / 'episodes.jsonl'; path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    else:
        results = json.loads((directory / 'results.json').read_text())
        if damage == 'aggregate':
            results['controllers']['policy:fixed']['mean_return'] += 1
        else:
            results['contrasts'][0]['paired_patient_bootstrap_ci'] = [100, 101]
        reseal_results(directory, results)
    with pytest.raises(ValueError):
        report.generate_report(directory, design, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_existing_report_directory_is_never_overwritten(development, tmp_path):
    directory, design = development
    output = tmp_path / 'existing'; output.mkdir(); sentinel = output / 'retain.txt'; sentinel.write_text('retain')
    with pytest.raises(FileExistsError):
        report.generate_report(directory, design, output)
    assert sentinel.read_text() == 'retain'


def test_rollout_log_has_a_cumulative_read_bound(development, tmp_path, monkeypatch):
    directory, design = development
    first_line = (directory / 'episodes.jsonl').read_bytes().splitlines(keepends=True)[0]
    monkeypatch.setattr(report, 'MAX_FILE_BYTES', len(first_line) + 1)
    with pytest.raises(ValueError, match='total report input bound'):
        report.generate_report(directory, design, tmp_path / 'refused-large-log')
    assert not (tmp_path / 'refused-large-log').exists()


def test_pre_outcome_policy_freeze_binds_actual_input_bytes_and_aliases(development, tmp_path):
    directory, design = development
    bundle = report.validate_bundle(directory, design)
    protocol = bundle['protocol']
    frozen = {'format': 'retraction-final-policy-freeze-v1', 'design_file_sha256': bundle['hashes']['design'],
              'physics_fingerprint': protocol['physics_fingerprint'], 'environment_config': protocol['environment_config'],
              'patient_seed_sha256': protocol['patient_seed_sha256'], 'source_sha256': protocol['source_sha256'],
              'prior_seed_exclusions': {'canonical_sha256': protocol['prior_used_seed_set_sha256'],
                                       'count': len(protocol['prior_used_patient_seeds'])},
              'policies': [{'label': label, **value} for label, value in bundle['policy_inputs'].items()],
              'expected_policy_count': 1, 'expected_baseline_count': 3, 'expected_episode_count': 20,
              'validation_selected_aliases': {'test_alias': 'fixed'}}
    path = tmp_path / 'policy-freeze.json'; path.write_text(json.dumps(frozen))
    manifest = report.generate_report(directory, design, tmp_path / 'valid', policy_freeze=path)
    assert manifest['policy_freeze']['manifest']['validation_selected_aliases'] == {'test_alias': 'fixed'}
    assert len(manifest['controller_order']) == 4
    frozen['policies'][0]['file_sha256'] = '0' * 64; path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='Frozen policy bytes'):
        report.generate_report(directory, design, tmp_path / 'invalid', policy_freeze=path)
    assert not (tmp_path / 'invalid').exists()


def test_crossed_bootstrap_matches_independent_frequency_weight_calculation(tmp_path):
    # Deliberately synthetic interaction matrices, not simulated/final outcomes.
    patients = list(range(4_000_000_000, 4_000_000_004))
    seeds = [201, 202, 203, 204, 205]
    matrices = {'return': np.arange(20, dtype=float).reshape(5, 4) ** 1.2,
                'success': np.array([[(-1.) ** (i + j) for j in range(4)] for i in range(5)]),
                'severe_harm': np.zeros((5, 4))}
    controllers = {}
    for index, seed in enumerate(seeds):
        for arm, positive in [('study003_v2', False), ('study003_v3', True)]:
            episodes = [{'return': matrices['return'][index, j] if positive else 0.,
                         'success': bool(matrices['success'][index, j] > 0) if positive else bool(matrices['success'][index, j] < 0),
                         'severe_harm': False} for j in range(4)]
            controllers[f'policy:{arm}_seed{seed}'] = {'episodes': episodes, 'mean_return': np.mean([r['return'] for r in episodes]),
                                                       'success_rate': np.mean([r['success'] for r in episodes]), 'severe_harm_rate': 0.}
    bundle = {'hashes': {'design': 'd' * 64}, 'protocol': {'patient_seeds': patients},
              'patient_count': 4, 'results': {'controllers': controllers}}
    plan = {'format': 'retraction-crossed-analysis-plan-v1', 'design_sha256': 'd' * 64,
            'shared_patient_seeds': patients, 'paired_training_seeds': seeds, 'positive_arm': 'study003_v3',
            'negative_arm': 'study003_v2', 'metrics': ['return', 'success', 'severe_harm'],
            'bootstrap_seed': 99175, 'bootstrap_replicates': 2000}
    path = tmp_path / 'mock-crossed-plan.json'; path.write_text(json.dumps(plan))
    actual = report.crossed_analysis(bundle, path)
    rng = np.random.default_rng(99175)
    samples = {metric: [] for metric in matrices}
    for _ in range(2000):
        row_counts = np.bincount(rng.integers(0, 5, size=5), minlength=5)
        column_counts = np.bincount(rng.integers(0, 4, size=4), minlength=4)
        weights = row_counts[:, None] * column_counts[None, :]
        for metric, matrix in matrices.items():
            samples[metric].append(float((matrix * weights).sum() / 20))
    for row in actual['contrasts']:
        expected = np.quantile(samples[row['metric']], [.025, .975])
        assert row['mean_difference'] == pytest.approx(float(matrices[row['metric']].mean()), abs=1e-12)
        assert [row['crossed_ci_lower'], row['crossed_ci_upper']] == pytest.approx(expected, abs=1e-12)
    assert all(row['fixed_policies'] == 5 and row['shared_patients'] == 4 for row in actual['fixed_policy_group_descriptors'])
    assert len(actual['paired_seed_descriptors']) == 5


def test_cli_reports_existing_development_records_without_simulation(development, tmp_path):
    directory, design = development
    output = tmp_path / 'cli-report'
    process = subprocess.run([sys.executable, str(SCRIPT), '--evaluation', str(directory), '--design', str(design),
                              '--output', str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['phase'] == 'development'
    assert (output / 'report-complete.json').is_file()
