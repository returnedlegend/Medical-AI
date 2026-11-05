#!/usr/bin/env python3
"""Audit a frozen RL bundle, summarize all runs, and export its selected example."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'backend'))

from retraction.environment import TissueRetractionEnv  # noqa: E402
from retraction.learning import normalize_training_config, validate_checkpoint_compatibility  # noqa: E402


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def hierarchical_return_interval(matrix, seed=98911):
    """Resample training seeds, then patients within each selected seed."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError('Invalid run/patient return matrix')
    rng = np.random.default_rng(seed)
    runs, patients = matrix.shape
    selected_runs = rng.integers(runs, size=(2000, runs, 1))
    selected_patients = rng.integers(patients, size=(2000, runs, patients))
    resampled = matrix[selected_runs, selected_patients].mean(axis=(1, 2))
    return np.quantile(resampled, [.025, .975]).tolist()


def audit_and_summarize(study, source_commit=None):
    protocol_path = study / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if set(protocol['source_sha256']) != {'learning.py', 'tabular.py', 'environment.py', 'tissue.py'}:
        raise ValueError('Protocol must fingerprint the four declared algorithm/physics sources')
    for name, expected in protocol['source_sha256'].items():
        snapshot = study / 'source' / name
        source = snapshot if snapshot.is_file() else PROJECT / 'backend/retraction' / name
        current = hashlib.sha256(source.read_bytes()).hexdigest()
        if current != expected:
            raise ValueError(f'Source changed after protocol freeze: {name}')
    if source_commit:
        source_commit = subprocess.check_output(['git', 'rev-parse', '--verify', source_commit],
                                               cwd=PROJECT, text=True).strip()
        for name, expected in protocol['source_sha256'].items():
            committed = subprocess.check_output(['git', 'show', f'{source_commit}:backend/retraction/{name}'], cwd=PROJECT)
            if hashlib.sha256(committed).hexdigest() != expected:
                raise ValueError(f'Source commit does not match frozen training implementation: {name}')
    configs = protocol['configs']
    # Validate supported supplied fields while retaining the exact historical
    # configuration. Later additive defaults must not rewrite a frozen study.
    for config in configs:
        normalize_training_config(config)
    if not configs:
        raise ValueError('Protocol has no configurations')
    algorithms = sorted({config['algorithm'] for config in configs})
    for algorithm in algorithms:
        marker = json.loads((study / algorithm / 'complete.json').read_text())
        expected_runs = sum(config['algorithm'] == algorithm for config in configs)
        if marker.get('completed_training_runs') != expected_runs:
            raise ValueError(f'Incomplete {algorithm} bundle')
    env = TissueRetractionEnv()
    try:
        description = env.describe()
    finally:
        env.close()
    runs, training_union, validation_union = [], set(), set()
    for config in configs:
        path = study / config['algorithm'] / f'seed-{config["training_seed"]}-results.json'
        result = json.loads(path.read_text())
        if result['config'] != config:
            raise ValueError(f'Result configuration differs from frozen protocol: {path}')
        checkpoint = result['checkpoint']
        if checkpoint.get('physics_fingerprint') != protocol['physics_fingerprint']:
            raise ValueError(f'Result used another physics source: {path}')
        validate_checkpoint_compatibility(checkpoint, description)
        training = set(result['metadata']['training_seeds'])
        validation = set(result['metadata']['test_seeds'])
        if training & validation:
            raise ValueError(f'Training/validation patient overlap: {path}')
        training_union |= training
        validation_union |= validation
        for controller in ('learned', 'random', 'random_no_handoff', 'scripted'):
            episodes = result['evaluation'][controller]['episodes']
            if [row['seed'] for row in episodes] != result['metadata']['test_seeds']:
                raise ValueError(f'Unpaired validation patients: {path}, {controller}')
        runs.append({'algorithm': config['algorithm'], 'training_seed': config['training_seed'],
                     'result': result, 'source': str(path.relative_to(study))})
    if training_union & validation_union:
        raise ValueError('A validation patient was used for fitting another run in this study')
    summaries = {}
    metrics = ('mean_return', 'success_rate', 'severe_harm_rate', 'handoff_rate', 'mean_max_damage')
    for algorithm in algorithms:
        selected = [row for row in runs if row['algorithm'] == algorithm]
        controllers = {}
        for controller in ('learned', 'random', 'random_no_handoff', 'scripted'):
            observations = [row['result']['evaluation'][controller] for row in selected]
            returns = [[episode['return'] for episode in value['episodes']] for value in observations]
            controllers[controller] = {
                'training_runs': len(selected), 'evaluated_episodes': sum(value['n'] for value in observations),
                'mean_across_training_runs': {metric: float(np.mean([value[metric] for value in observations]))
                                              for metric in metrics},
                'range_across_training_runs': {metric: [float(min(value[metric] for value in observations)),
                                                         float(max(value[metric] for value in observations))]
                                               for metric in metrics},
                'mean_return_hierarchical_bootstrap_ci': hierarchical_return_interval(returns),
                'observed_successes': sum(episode['success'] for value in observations for episode in value['episodes']),
                'observed_severe_harm': sum(episode['severe_harm'] for value in observations for episode in value['episodes']),
            }
        summaries[algorithm] = controllers
    contrasts = {}
    if set(algorithms) == {'dqn', 'tabular_q'}:
        by_seed = {algorithm: {row['training_seed']: row['result'] for row in runs if row['algorithm'] == algorithm}
                   for algorithm in algorithms}
        if set(by_seed['dqn']) != set(by_seed['tabular_q']):
            raise ValueError('Algorithms do not have paired training seeds')
        differences = []
        for seed in sorted(by_seed['dqn']):
            first, second = by_seed['tabular_q'][seed], by_seed['dqn'][seed]
            if first['metadata']['test_seeds'] != second['metadata']['test_seeds']:
                raise ValueError('Algorithms do not have paired validation patients')
            differences.append([a['return'] - b['return'] for a, b in zip(
                first['evaluation']['learned']['episodes'], second['evaluation']['learned']['episodes'])])
        contrasts['tabular_minus_dqn_return'] = {'mean': float(np.mean(differences)),
                                                'hierarchical_bootstrap_ci': hierarchical_return_interval(differences)}
    def selection_key(row):
        value = row['result']['evaluation']['learned']
        return (-value['success_rate'], value['severe_harm_rate'], value['mean_max_damage'], -value['mean_return'])
    chosen = min(runs, key=selection_key)
    result_summary = {
        'format': 'retraction-frozen-study-summary-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'physics_fingerprint': protocol['physics_fingerprint'], 'model_version': protocol['model_version'],
        'source_sha256': protocol['source_sha256'],
        'source_commit': source_commit,
        'algorithms': summaries, 'contrasts': contrasts,
        'runs': [{'algorithm': row['algorithm'], 'training_seed': row['training_seed'], 'source': row['source'],
                  'learned': {key: value for key, value in row['result']['evaluation']['learned'].items() if key != 'episodes'}}
                 for row in runs],
        'selected': {'algorithm': chosen['algorithm'], 'training_seed': chosen['training_seed'],
                     'source': chosen['source'], 'selection_rule': protocol['selection_rule']},
        'all_training_patient_seeds': sorted(training_union),
        'all_validation_patient_seeds': sorted(validation_union),
        'integrity_checks': ['All frozen configurations/source snapshots/physics match',
                             'Every comparator uses paired validation patients',
                             'No validation patient overlaps fitting patients in any run',
                             'All checkpoint signatures match the current simulator'],
        'limitations': ['Illustrative synthetic model; no clinical generalization claim',
                        'Selected run used validation data and has optimistic selection bias',
                        'Hierarchical return intervals resample five training seeds and patients, excluding model uncertainty',
                        'Zero observed harm does not imply zero future risk; consult per-run Wilson intervals',
                        'Reserved parent-owned final cohort is not part of these validation results'],
    }
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'study_protocol_sha256': result_summary['protocol_sha256'],
                  'source_sha256': protocol['source_sha256'],
                  'source_commit': source_commit,
                  'selection_rule': protocol['selection_rule'], 'validation_only': True,
                  'policy_source_run': {'algorithm': chosen['algorithm'], 'training_seed': chosen['training_seed']},
                  'checkpoint_sha256': canonical_sha(chosen['result']['checkpoint']),
                  'result_sha256': canonical_sha(chosen['result']),
                  'scientific_note': 'Selected using exploratory synthetic validation; unvalidated research policy with no clinical competence claim. No reserved final patient cohort was used for selection.'}
    example = {'format': 'retraction-bundled-reference-v1', 'result': chosen['result'], 'provenance': provenance}
    return result_summary, example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--example-output', type=Path)
    parser.add_argument('--source-commit', help='Optional Git commit, verified against all four frozen source hashes')
    args = parser.parse_args()
    try:
        summary, example = audit_and_summarize(args.study, args.source_commit)
        write_json(args.output, summary)
        if args.example_output:
            write_json(args.example_output, example)
            print('Example SHA256:', hashlib.sha256(args.example_output.read_bytes()).hexdigest())
        print(json.dumps({'selected': summary['selected'], 'algorithms': summary['algorithms']}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'Frozen study audit could not complete: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
