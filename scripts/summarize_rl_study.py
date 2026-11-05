#!/usr/bin/env python3
"""Audit a frozen RL bundle, summarize all runs, and export its selected example."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'backend'))

from retraction.environment import TissueRetractionEnv, normalize_env_config  # noqa: E402
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


def audit_and_summarize(study, source_commit=None, reserved_seeds=None):
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
    raw_entries = protocol.get('entries')
    if raw_entries is None:
        entries = [{'arm': config['algorithm'], 'config': config} for config in protocol['configs']]
    else:
        if not isinstance(raw_entries, list) or any(not isinstance(entry, dict) or
                set(entry) != {'arm', 'config'} for entry in raw_entries):
            raise ValueError('Protocol entries must contain exactly arm and config')
        entries = raw_entries
        if 'configs' in protocol and protocol['configs'] != [entry['config'] for entry in entries]:
            raise ValueError('Protocol entries and legacy configs disagree')
    if not entries:
        raise ValueError('Protocol has no configurations')
    # Reject unsafe directory names instead of silently normalizing distinct arm
    # identities to the same path. Names remain portable to Windows checkouts.
    reserved_names = {'con', 'prn', 'aux', 'nul', *{f'com{i}' for i in range(1, 10)},
                      *{f'lpt{i}' for i in range(1, 10)}}
    for entry in entries:
        arm, config = entry['arm'], entry['config']
        if (not isinstance(arm, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', arm)
                or arm in reserved_names):
            raise ValueError('Arm identifiers must be safe lowercase directory names (letters, digits, underscore, hyphen)')
        # Validate supplied fields while retaining exact historical defaults.
        normalize_training_config(config)
    identities = [(entry['arm'], entry['config']['training_seed']) for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError('Protocol repeats an arm/training-seed identity')
    arms = sorted({entry['arm'] for entry in entries})
    for arm in arms:
        algorithm_names = {entry['config']['algorithm'] for entry in entries if entry['arm'] == arm}
        if len(algorithm_names) != 1:
            raise ValueError(f'An arm must use one algorithm: {arm}')
        marker = json.loads((study / arm / 'complete.json').read_text())
        expected_runs = sum(entry['arm'] == arm for entry in entries)
        if marker.get('completed_training_runs') != expected_runs:
            raise ValueError(f'Incomplete {arm} bundle')
    require_zero_harm = protocol.get('selection_require_zero_severe_harm', False)
    if type(require_zero_harm) is not bool:
        raise ValueError('selection_require_zero_severe_harm must be a boolean')
    env = TissueRetractionEnv()
    try:
        description = env.describe()
    finally:
        env.close()
    runs, training_union, validation_union = [], set(), set()
    for entry in entries:
        arm, config = entry['arm'], entry['config']
        path = study / arm / f'seed-{config["training_seed"]}-results.json'
        result = json.loads(path.read_text())
        if result['config'] != config:
            raise ValueError(f'Result configuration differs from frozen protocol: {path}')
        checkpoint = result['checkpoint']
        if checkpoint.get('model_version') != protocol['model_version'] or result.get('model_version') != protocol['model_version']:
            raise ValueError(f'Result model version differs from the frozen protocol: {path}')
        if checkpoint.get('physics_fingerprint') != protocol['physics_fingerprint']:
            raise ValueError(f'Result used another physics source: {path}')
        effective_environment = normalize_env_config({**config['environment'], 'max_steps': config['max_steps']})
        if checkpoint.get('environment_config') != effective_environment or checkpoint.get('training_config') != config:
            raise ValueError(f'Checkpoint configuration differs from recorded training: {path}')
        if result.get('algorithm') != checkpoint.get('algorithm'):
            raise ValueError(f'Result and checkpoint algorithms disagree: {path}')
        validate_checkpoint_compatibility(checkpoint, description)
        metadata = result['metadata']
        training_values, validation_values = metadata['training_seeds'], metadata['test_seeds']
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
               for seed in training_values + validation_values):
            raise ValueError(f'Invalid fitting/validation patient seeds: {path}')
        training, validation = set(training_values), set(validation_values)
        if len(validation_values) != config['eval_episodes'] or len(validation) != len(validation_values):
            raise ValueError(f'Validation seed count or distinctness differs from protocol: {path}')
        if len(result['learning_curve']) != config['episodes'] or [row['seed'] for row in result['learning_curve']] != metadata['training_episode_seeds']:
            raise ValueError(f'Learning curve is incomplete or has inconsistent seed provenance: {path}')
        if training & validation:
            raise ValueError(f'Training/validation patient overlap: {path}')
        training_union |= training
        validation_union |= validation
        for controller in ('learned', 'random', 'random_no_handoff', 'scripted'):
            episodes = result['evaluation'][controller]['episodes']
            recorded = result['evaluation'][controller]
            if len(episodes) != config['eval_episodes'] or [row['seed'] for row in episodes] != validation_values:
                raise ValueError(f'Unpaired validation patients: {path}, {controller}')
            calculated = {'n': len(episodes), 'mean_return': math.fsum(row['return'] for row in episodes) / len(episodes),
                          'mean_max_damage': math.fsum(row['max_damage'] for row in episodes) / len(episodes),
                          'mean_steps': math.fsum(row['steps'] for row in episodes) / len(episodes)}
            for key in ('success', 'severe_harm', 'handoff'):
                if any(not isinstance(row[key], bool) for row in episodes):
                    raise ValueError(f'Invalid episode outcome flag: {path}, {controller}')
                calculated[f'{key}_rate'] = sum(row[key] for row in episodes) / len(episodes)
            if any(not math.isclose(recorded.get(key, math.nan), value, rel_tol=1e-9, abs_tol=1e-9)
                   for key, value in calculated.items()):
                raise ValueError(f'Stored aggregate differs from actual episode outcomes: {path}, {controller}')
        runs.append({'arm': arm, 'algorithm': config['algorithm'], 'training_seed': config['training_seed'],
                     'result': result, 'source': str(path.relative_to(study))})
    if training_union & validation_union:
        raise ValueError('A validation patient was used for fitting another run in this study')
    if reserved_seeds is not None:
        if (not isinstance(reserved_seeds, list) or not reserved_seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in reserved_seeds)
                or len(set(reserved_seeds)) != len(reserved_seeds)):
            raise ValueError('Reserved patient seeds must be a nonempty distinct integer list')
        if set(reserved_seeds) & (training_union | validation_union):
            raise ValueError('Reserved final patients overlap fitting or validation patients')
    summaries = {}
    metrics = ('mean_return', 'success_rate', 'severe_harm_rate', 'handoff_rate', 'mean_max_damage')
    for arm in arms:
        selected = [row for row in runs if row['arm'] == arm]
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
        summaries[arm] = controllers
    contrasts = {}
    by_seed = {arm: {row['training_seed']: row['result'] for row in runs if row['arm'] == arm}
               for arm in arms}
    for negative_arm, positive_arm in combinations(arms, 2):
        if set(by_seed[negative_arm]) != set(by_seed[positive_arm]):
            raise ValueError(f'Arms do not have paired training seeds: {negative_arm}, {positive_arm}')
        differences = []
        for seed in sorted(by_seed[negative_arm]):
            first, second = by_seed[positive_arm][seed], by_seed[negative_arm][seed]
            if first['metadata']['test_seeds'] != second['metadata']['test_seeds']:
                raise ValueError(f'Arms do not have paired validation patients: {negative_arm}, {positive_arm}')
            if first['checkpoint']['environment_config'] != second['checkpoint']['environment_config']:
                raise ValueError(f'Paired arms use different environment configurations: {negative_arm}, {positive_arm}')
            differences.append([a['return'] - b['return'] for a, b in zip(
                first['evaluation']['learned']['episodes'], second['evaluation']['learned']['episodes'])])
        # Preserve the original frozen study's published contrast key.
        name = ('tabular_minus_dqn_return' if (negative_arm, positive_arm) == ('dqn', 'tabular_q')
                else f'{positive_arm}_minus_{negative_arm}_return')
        if name in contrasts:
            raise ValueError('Arm names produce an ambiguous return contrast key')
        contrasts[name] = {'mean': float(np.mean(differences)),
                           'hierarchical_bootstrap_ci': hierarchical_return_interval(differences),
                           'positive_arm': positive_arm, 'negative_arm': negative_arm}
    def selection_key(row):
        value = row['result']['evaluation']['learned']
        return (-value['success_rate'], value['severe_harm_rate'], value['mean_max_damage'], -value['mean_return'])
    eligible = [row for row in runs if not require_zero_harm or
                not any(episode['severe_harm'] for episode in row['result']['evaluation']['learned']['episodes'])]
    chosen = min(eligible, key=selection_key) if eligible else None
    result_summary = {
        'format': 'retraction-frozen-study-summary-v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'physics_fingerprint': protocol['physics_fingerprint'], 'model_version': protocol['model_version'],
        'source_sha256': protocol['source_sha256'],
        'source_commit': source_commit,
        'algorithms': summaries, 'contrasts': contrasts,
        'runs': [{'arm': row['arm'], 'algorithm': row['algorithm'], 'training_seed': row['training_seed'], 'source': row['source'],
                  'learned': {key: value for key, value in row['result']['evaluation']['learned'].items() if key != 'episodes'}}
                 for row in runs],
        'selected': ({'arm': chosen['arm'], 'algorithm': chosen['algorithm'], 'training_seed': chosen['training_seed'],
                      'source': chosen['source'], 'selection_rule': protocol['selection_rule']} if chosen else None),
        'selection_eligibility': {'require_zero_observed_severe_harm': require_zero_harm,
                                  'eligible_runs': len(eligible), 'total_runs': len(runs),
                                  'status': 'selected' if chosen else 'no_eligible_policy'},
        'all_training_patient_seeds': sorted(training_union),
        'all_validation_patient_seeds': sorted(validation_union),
        'reserved_patient_exclusion': {'checked': reserved_seeds is not None,
                                       'count': len(reserved_seeds) if reserved_seeds else 0,
                                       'seed_set_sha256': canonical_sha(sorted(reserved_seeds)) if reserved_seeds else None},
        'integrity_checks': ['All frozen configurations/source snapshots/physics match',
                             'Every comparator uses paired validation patients',
                             'No validation patient overlaps fitting patients in any run',
                             'All checkpoint signatures match the current simulator'],
        'limitations': ['Illustrative synthetic model; no clinical generalization claim',
                        ('Selected run used validation data and has optimistic selection bias' if chosen else
                         'No run met the declared reference-selection eligibility rule; no policy exported'),
                        'Hierarchical return intervals resample the recorded training seeds and patients, excluding model uncertainty; few training seeds give unstable intervals',
                        'Zero observed harm does not imply zero future risk; consult per-run Wilson intervals',
                        'Reserved parent-owned final cohort is not part of these validation results'],
    }
    if chosen is None:
        return result_summary, None
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'study_protocol_sha256': result_summary['protocol_sha256'],
                  'source_sha256': protocol['source_sha256'],
                  'source_commit': source_commit,
                  'selection_rule': protocol['selection_rule'], 'validation_only': True,
                  'policy_source_run': {'arm': chosen['arm'], 'algorithm': chosen['algorithm'], 'training_seed': chosen['training_seed']},
                  'selection_eligibility': result_summary['selection_eligibility'],
                  'checkpoint_sha256': canonical_sha(chosen['result']['checkpoint']),
                  'result_sha256': canonical_sha(chosen['result']),
                  'reserved_patient_exclusion': result_summary['reserved_patient_exclusion'],
                  'scientific_note': 'Selected using exploratory synthetic validation; unvalidated research policy with no clinical competence claim. No reserved final patient cohort was used for selection.'}
    example = {'format': 'retraction-bundled-reference-v1', 'result': chosen['result'], 'provenance': provenance}
    return result_summary, example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--example-output', type=Path)
    parser.add_argument('--source-commit', help='Optional Git commit, verified against all four frozen source hashes')
    parser.add_argument('--reserved-seeds-file', type=Path, help='JSON list of reserved final patients; refuses any fitting/validation overlap')
    args = parser.parse_args()
    try:
        reserved = json.loads(args.reserved_seeds_file.read_text()) if args.reserved_seeds_file else None
        summary, example = audit_and_summarize(args.study, args.source_commit, reserved)
        write_json(args.output, summary)
        if args.example_output and example is None:
            print('No eligible reference policy: summary written, no example exported (any existing example target is unchanged).', file=sys.stderr)
            return 2
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
