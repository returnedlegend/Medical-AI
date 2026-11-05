#!/usr/bin/env python3
"""Validate completed fixed-policy records and write reports; never evaluate.

No simulator, model loader, trainer, or evaluator is imported. Matplotlib is
optional and needed only for --plots. Existing output directories are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / 'examples/final-evaluation-001-design.json'
DEFAULT_CROSSED = ROOT / 'examples/final-evaluation-001-crossed-analysis.json'
DEFAULT_FREEZE = ROOT / 'examples/final-evaluation-001-policy-freeze.json'
FORMAT = 'retraction-fixed-policy-evaluation-v1'
MAX_FILE_BYTES = 128 * 1024 * 1024


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(path, limit=MAX_FILE_BYTES):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    with Path(path).open('rb') as handle:
        contents = handle.read(limit + 1)
    if len(contents) > limit:
        raise ValueError('Report input exceeds its bounded size')
    return json.loads(contents, parse_constant=reject, object_pairs_hook=pairs), contents


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def wilson(count, n):
    z = 1.959963984540054
    p, z2 = count / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0, center - half), min(1, center + half)]


def return_intervals(controllers, bootstrap):
    """Patient-only intervals conditional on the fixed checkpoints."""
    labels = list(controllers)
    count = len(controllers[labels[0]]['episodes'])
    replicates = bootstrap['replicates']
    require(type(replicates) is int and 100 <= replicates <= 10000, 'Invalid bootstrap replicate count')
    values = {name: np.array([row['return'] for row in controllers[name]['episodes']]) for name in labels}
    samples = {name: np.empty(replicates) for name in labels}
    batch = max(1, min(replicates, (4 * 1024 * 1024) // (8 * count)))
    rng = np.random.default_rng(bootstrap['seed'])
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = rng.integers(0, count, size=(stop - start, count))
        for name in labels:
            samples[name][start:stop] = values[name][indices].mean(axis=1)
    return {name: np.quantile(samples[name], [.025, .975]).tolist() for name in labels}, samples


def validate_bundle(directory, design_path):
    directory = Path(directory)
    design, design_bytes = read_json(design_path, 32 * 1024 * 1024)
    marker, marker_bytes = read_json(directory / 'complete.marker', 1024 * 1024)
    protocol, protocol_bytes = read_json(directory / 'protocol.json', 32 * 1024 * 1024)
    results, result_bytes = read_json(directory / 'results.json')
    require(not (directory / 'failed.json').exists(), 'Evaluation contains a failure record')
    require(protocol['format'] == results['format'] == FORMAT, 'Unsupported evaluation format')
    require(design.get('format') == 'retraction-final-evaluation-design-v1', 'Unsupported prespecified design')
    require(marker['protocol_sha256'] == results['protocol_sha256'] == sha(protocol_bytes), 'Protocol hash mismatch')
    require(marker['results_sha256'] == sha(result_bytes), 'Result hash mismatch')
    require(results.get('policy_selection_performed') is False, 'Evaluation must not perform policy selection')
    require(protocol['phase'] == results['phase'] and protocol['phase'] in ('development', 'final'), 'Phase mismatch')
    require(protocol['physics_fingerprint'] == results['physics_fingerprint'] == design['physics_fingerprint'], 'Physics identity mismatch')
    patients = protocol['patient_seeds']
    require(isinstance(patients, list) and 0 < len(patients) <= 10000 and len(set(patients)) == len(patients)
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in patients), 'Invalid patient cohort')
    require(patients == design['patient_seeds'] and sha(canonical(patients)) == protocol['patient_seed_sha256'], 'Patient design/hash mismatch')
    excluded = protocol['prior_used_patient_seeds']
    require(sha(canonical(sorted(excluded))) == protocol['prior_used_seed_set_sha256'], 'Prior-used exclusion hash mismatch')
    require(not set(patients).intersection(excluded), 'Evaluation overlaps prior-used patients')
    policy_design = design['policies']
    labels = [row['label'] for row in policy_design]
    require(0 < len(labels) <= 32 and len(set(labels)) == len(labels)
            and all(re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', label) for label in labels), 'Invalid design labels')
    recorded = {row['label']: row for row in protocol['policies']}
    require(len(recorded) == len(protocol['policies']) and set(recorded) == set(labels), 'Policy set differs from design')
    require(design['baselines'] == ['random', 'random_no_handoff', 'scripted'], 'Unexpected baseline design')
    order = [f'policy:{label}' for label in labels] + [f'baseline:{mode}' for mode in design['baselines']]
    controllers = results['controllers']
    require(set(controllers) == set(order), 'Result controllers differ from prespecified design')
    expected = len(order) * len(patients)
    require(marker['completed_episodes'] == results['observed_episodes'] == results['expected_episodes']
            == design['expected_episodes'] == expected, 'Completion/episode count mismatch')
    for key, actual in [('baseline_rng_seed', protocol['baseline_rng_seed']), ('bootstrap_rng_seed', protocol['bootstrap']['seed']),
                        ('bootstrap_replicates', protocol['bootstrap']['replicates'])]:
        if key in design:
            require(design[key] == actual, f'Prespecified RNG differs: {key}')
    hashes = {'complete.marker': sha(marker_bytes), 'protocol.json': sha(protocol_bytes),
              'results.json': sha(result_bytes), 'design': sha(design_bytes)}
    policy_inputs = {}
    input_budget = 256 * 1024 * 1024
    for specification in policy_design:
        label = specification['label']
        filename = f'inputs/policy-{label}.json'
        payload, contents = read_json(directory / filename, min(32 * 1024 * 1024, input_budget))
        input_budget -= len(contents)
        record = recorded[label]
        require(sha(contents) == record['file_sha256'], f'Copied policy input hash mismatch: {label}')
        run = payload.get('result', payload)
        checkpoint = run['checkpoint']
        require(sha(canonical(checkpoint)) == record['checkpoint_sha256'], f'Checkpoint hash mismatch: {label}')
        require(checkpoint['physics_fingerprint'] == protocol['physics_fingerprint'], f'Policy physics differs: {label}')
        require(checkpoint['environment_config'] == protocol['environment_config'], f'Policy environment differs: {label}')
        require(run['config'] == checkpoint['training_config'] == record['training_config'], f'Policy configuration differs: {label}')
        for field in ('file_sha256', 'checkpoint_sha256'):
            if field in specification:
                require(specification[field] == record[field], f'Frozen design policy hash differs: {label}')
        if 'training_seed' in specification:
            require(specification['training_seed'] == record['training_config']['training_seed'], f'Training seed differs: {label}')
        if 'encoder' in specification:
            require(checkpoint['state_encoding']['version'].endswith('-' + specification['encoder'][1:]), f'Encoder differs: {label}')
        histories = run['metadata']
        used = set(histories['training_seeds']) | set(histories['test_seeds'])
        require(used <= set(excluded), f'Exclusion manifest misses policy history: {label}')
        hashes[filename] = sha(contents)
        policy_inputs[label] = {'file_sha256': sha(contents), 'file_bytes': len(contents),
                                'checkpoint_sha256': sha(canonical(checkpoint)), 'result_sha256': sha(canonical(run)),
                                'algorithm': run.get('algorithm'), 'model_version': run.get('model_version'),
                                'physics_fingerprint': checkpoint['physics_fingerprint'],
                                'state_encoding_version': checkpoint.get('state_encoding', {}).get('version'),
                                'training_config': run['config']}
    rows = []
    initial_patients = None
    for name in order:
        value = controllers[name]
        episodes = value['episodes']
        require(value['n'] == len(episodes) == len(patients) and [episode['seed'] for episode in episodes] == patients,
                f'Incomplete/unpaired controller episodes: {name}')
        anatomy = [row['patient'] for row in episodes]
        require(initial_patients is None or anatomy == initial_patients, 'Controllers have different initial patient anatomy')
        initial_patients = anatomy
        for episode in episodes:
            require(all(type(episode[field]) is bool for field in ('success', 'severe_harm', 'handoff', 'terminated', 'truncated')),
                    'Episode outcomes must be boolean')
            require(episode['terminated'] != episode['truncated'], 'Episode has no unique terminal/truncation outcome')
            require(close(episode['return'], math.fsum(episode['reward_components_total'].values())), 'Reward components disagree with return')
        row = {'controller': name, 'label': name.split(':', 1)[1], 'role': name.split(':', 1)[0], 'n': len(patients)}
        for field, count_key, rate_key, ci_key in [('success', 'successes', 'success_rate', 'success_rate_wilson_ci'),
                                                 ('severe_harm', 'severe_harm_episodes', 'severe_harm_rate', 'severe_harm_rate_wilson_ci')]:
            count = sum(episode[field] for episode in episodes)
            interval = wilson(count, len(patients))
            require(count == value[count_key] and close(value[rate_key], count / len(patients))
                    and len(value[ci_key]) == 2 and all(close(a, b) for a, b in zip(value[ci_key], interval)), f'Invalid {field} aggregate: {name}')
            row.update({count_key: count, rate_key: count / len(patients), rate_key + '_ci_lower': interval[0], rate_key + '_ci_upper': interval[1]})
        for key, episode_key in [('mean_return', 'return'), ('mean_max_damage', 'max_damage'), ('mean_blood_loss_ml', 'blood_loss_ml')]:
            mean = math.fsum(episode[episode_key] for episode in episodes) / len(patients)
            require(close(value[key], mean), f'Invalid {key} aggregate: {name}')
            row[key] = mean
        require(value['handoffs'] == sum(episode['handoff'] for episode in episodes), 'Invalid handoff count')
        row['handoffs'] = value['handoffs']
        if row['role'] == 'policy':
            cfg = recorded[row['label']]['training_config']
            row.update(algorithm=recorded[row['label']]['algorithm'], training_seed=cfg['training_seed'], demonstrations=cfg['demonstration_episodes'])
        else:
            row.update(algorithm='untrained baseline', training_seed='', demonstrations='')
        rows.append(row)
    # Check the append-only rollout log against completed result records.
    expected_log = ((name, episode) for name, value in controllers.items() for episode in value['episodes'])
    log_hash = hashlib.sha256()
    log_bytes = 0
    with (directory / 'episodes.jsonl').open('rb') as handle:
        for name, episode in expected_log:
            line = handle.readline(1024 * 1024 + 1)
            require(line and len(line) <= 1024 * 1024, 'Rollout log is incomplete or oversized')
            log_bytes += len(line)
            require(log_bytes <= MAX_FILE_BYTES, 'Rollout log exceeds the total report input bound')
            log_hash.update(line)
            require(json.loads(line) == {'controller': name, **episode}, 'Rollout log differs from completed results')
        require(handle.read(1) == b'', 'Rollout log contains extra episodes')
    hashes['episodes.jsonl'] = log_hash.hexdigest()
    pairs = set()
    for contrast in results['contrasts']:
        positive, negative = contrast['positive_controller'], contrast['negative_controller']
        require(positive in controllers and negative in controllers and positive != negative, 'Invalid paired contrast labels')
        pair = frozenset([positive, negative])
        require(pair not in pairs, 'Duplicate paired contrast')
        pairs.add(pair)
        expected_mean = math.fsum(a['return'] - b['return'] for a, b in zip(controllers[positive]['episodes'], controllers[negative]['episodes'])) / len(patients)
        interval = contrast['paired_patient_bootstrap_ci']
        require(close(contrast['mean_return_difference'], expected_mean) and len(interval) == 2
                and all(math.isfinite(x) for x in interval) and interval[0] <= interval[1], 'Invalid paired return contrast')
    require(len(pairs) == len(order) * (len(order) - 1) // 2, 'Missing paired contrasts')
    intervals, samples = return_intervals(controllers, protocol['bootstrap'])
    for contrast in results['contrasts']:
        expected_interval = np.quantile(samples[contrast['positive_controller']] - samples[contrast['negative_controller']], [.025, .975])
        require(all(close(a, b) for a, b in zip(contrast['paired_patient_bootstrap_ci'], expected_interval)),
                'Paired interval disagrees with the frozen common patient bootstrap')
    for row in rows:
        row['mean_return_ci_lower'], row['mean_return_ci_upper'] = intervals[row['controller']]
    return {'design': design, 'protocol': protocol, 'results': results, 'rows': rows, 'hashes': hashes,
            'phase': protocol['phase'], 'patient_count': len(patients), 'rollouts': expected, 'policy_inputs': policy_inputs}


def validate_policy_freeze(bundle, path):
    frozen, contents = read_json(path, 32 * 1024 * 1024)
    require(frozen['format'] == 'retraction-final-policy-freeze-v1'
            and frozen['design_file_sha256'] == bundle['hashes']['design'], 'Policy freeze/design hash mismatch')
    protocol = bundle['protocol']
    require(frozen['physics_fingerprint'] == protocol['physics_fingerprint']
            and frozen['environment_config'] == protocol['environment_config'], 'Frozen environment identity differs')
    require(frozen['patient_seed_sha256'] == protocol['patient_seed_sha256'], 'Frozen patient cohort differs')
    exclusions = frozen['prior_seed_exclusions']
    require(exclusions['canonical_sha256'] == protocol['prior_used_seed_set_sha256']
            and exclusions['count'] == len(protocol['prior_used_patient_seeds']), 'Frozen exclusion manifest differs')
    require([row['label'] for row in frozen['policies']] == [row['label'] for row in bundle['design']['policies']], 'Frozen policy order differs')
    for record in frozen['policies']:
        actual = bundle['policy_inputs'][record['label']]
        require(all(record.get(key) == value for key, value in actual.items()), f"Frozen policy bytes/identity differ: {record['label']}")
    require(all(protocol['source_sha256'].get(name) == digest for name, digest in frozen['source_sha256'].items()), 'Frozen evaluation source identity differs')
    require(frozen['expected_policy_count'] == len(bundle['policy_inputs'])
            and frozen['expected_baseline_count'] == 3 and frozen['expected_episode_count'] == bundle['rollouts'], 'Frozen count mismatch')
    aliases = frozen.get('validation_selected_aliases', {})
    require(set(aliases.values()) <= set(bundle['policy_inputs']), 'Frozen alias names an extra or unknown policy')
    return {'manifest': frozen, 'sha256': sha(contents)}


def crossed_analysis(bundle, plan_path):
    plan, contents = read_json(plan_path, 32 * 1024 * 1024)
    require(plan['format'] == 'retraction-crossed-analysis-plan-v1' and plan['design_sha256'] == bundle['hashes']['design'], 'Crossed plan/design hash mismatch')
    require(plan['shared_patient_seeds'] == bundle['protocol']['patient_seeds'], 'Crossed plan patient cohort differs')
    metrics = plan['metrics']
    require(metrics == ['return', 'success', 'severe_harm'], 'Unexpected crossed outcomes')
    seeds = plan['paired_training_seeds']
    require(len(seeds) == 5 and len(set(seeds)) == 5, 'Crossed analysis requires five prespecified paired training seeds')
    controllers = bundle['results']['controllers']
    differences = {}
    for metric in metrics:
        differences[metric] = np.array([[float(a[metric]) - float(b[metric])
                                        for a, b in zip(controllers[f"policy:{plan['positive_arm']}_seed{seed}"]['episodes'],
                                                        controllers[f"policy:{plan['negative_arm']}_seed{seed}"]['episodes'])]
                                       for seed in seeds], dtype=float)
    reps = plan['bootstrap_replicates']
    require(type(reps) is int and 100 <= reps <= 10000, 'Invalid crossed bootstrap replicate count')
    rng = np.random.default_rng(plan['bootstrap_seed'])
    samples = {metric: np.empty(reps) for metric in metrics}
    for replicate in range(reps):
        selected_seeds = rng.integers(0, len(seeds), size=len(seeds))
        selected_patients = rng.integers(0, bundle['patient_count'], size=bundle['patient_count'])
        for metric in metrics:
            samples[metric][replicate] = differences[metric][selected_seeds][:, selected_patients].mean()
    rows = [{'metric': metric, 'positive_arm': plan['positive_arm'], 'negative_arm': plan['negative_arm'],
             'paired_training_seeds': len(seeds), 'shared_patients': bundle['patient_count'],
             'mean_difference': float(differences[metric].mean()),
             'crossed_ci_lower': float(np.quantile(samples[metric], .025)),
             'crossed_ci_upper': float(np.quantile(samples[metric], .975))} for metric in metrics]
    per_seed = [{'training_seed': seed, **{metric + '_difference': float(differences[metric][i].mean()) for metric in metrics}}
                for i, seed in enumerate(seeds)]
    groups = []
    for arm in (plan['negative_arm'], plan['positive_arm']):
        records = [controllers[f'policy:{arm}_seed{seed}'] for seed in seeds]
        groups.append({'arm': arm, 'fixed_policies': len(seeds), 'shared_patients': bundle['patient_count'],
                       **{key: float(np.mean([record[key] for record in records]))
                          for key in ('mean_return', 'success_rate', 'severe_harm_rate')},
                       **{key + suffix: float(function([record[key] for record in records]))
                          for key in ('mean_return', 'success_rate', 'severe_harm_rate')
                          for suffix, function in (('_min', min), ('_max', max))}})
    return {'plan': plan, 'plan_sha256': sha(contents), 'contrasts': rows, 'paired_seed_descriptors': per_seed,
            'fixed_policy_group_descriptors': groups}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(bundle, crossed, freeze):
    n, phase = bundle['patient_count'], bundle['phase']
    lines = [f"# {'Final' if phase == 'final' else 'Development'} fixed-policy evaluation", '',
             f"{len(bundle['rows'])} fixed controllers were assessed on the same **{n} synthetic patients**, giving {bundle['rollouts']} recorded rollouts. These are not {bundle['rollouts']} independent patients. Rows follow the prespecified design; no ranking or replacement-policy selection is performed.", '',
             'The individual-policy intervals condition on these fixed checkpoints and exclude training-run and tissue-model uncertainty. Binary intervals are Wilson 95% bounds. Return intervals use patient-only bootstrap resampling from recorded outcomes with the frozen evaluation RNG. All pairwise comparisons are unadjusted for multiplicity; no familywise superiority claim is supported.', '',
             '| Prespecified controller | n | Success [95% CI] | Severe injury [95% CI] | Mean return [95% CI] | Mean peak injury |',
             '| --- | ---: | --- | --- | --- | ---: |']
    for row in bundle['rows']:
        interval = lambda key: f"{row[key] * 100:.1f}% [{row[key + '_ci_lower'] * 100:.1f}, {row[key + '_ci_upper'] * 100:.1f}]"
        lines.append(f"| `{row['controller']}` | {n} | {interval('success_rate')} | {interval('severe_harm_rate')} | {row['mean_return']:.3f} [{row['mean_return_ci_lower']:.3f}, {row['mean_return_ci_upper']:.3f}] | {row['mean_max_damage']:.5f} |")
    lines += ['', 'Zero observed severe injury does not establish zero risk. Low injury with failure or handoff is not successful procedure control. These illustrative mechanics and harm thresholds have no clinical validation.', '',
              'The validation-selected study-003 V3 example, if referenced elsewhere, is an alias of one included V3 checkpoint. It is not an extra controller or independent replicate.', '',
              '[All controller metrics](controllers.csv) · [All unadjusted paired return contrasts](paired-return-contrasts.csv) · [Report provenance](report-complete.json)']
    if freeze:
        aliases = freeze['manifest'].get('validation_selected_aliases', {})
        lines += ['', f"The pre-outcome policy-freeze manifest is verified (`{freeze['sha256']}`). Prespecified validation aliases: "
                  + '; '.join(f'`{name}` → `{label}`' for name, label in aliases.items()) + '. No alias adds a result row.']
    if crossed:
        lines += ['', '## Prespecified crossed V3 minus V2 supplement', '',
                  f"The five paired training seeds are crossed with the same {n} patients. A replicate resamples five seed rows and **one common patient-index vector**, then averages their Cartesian product; all three outcomes share these indices. The {5 * n} policy-patient outcomes per arm are correlated repeated measurements, not {5 * n} independent patients.", '',
                  'These descriptive 95% percentile intervals include variation across the five observed training seeds, may be unstable, exclude tissue/model error, and are unadjusted across three outcomes and other comparisons. A zero-event difference interval does not establish zero absolute risk or equivalence; inspect individual-policy Wilson bounds.', '',
                  '| Fixed-policy set | Policies | Shared patients | Mean success (range) | Mean severe injury (range) | Mean return (range) |',
                  '| --- | ---: | ---: | --- | --- | --- |']
        for group in crossed['fixed_policy_group_descriptors']:
            cells = []
            for key in ('success_rate', 'severe_harm_rate', 'mean_return'):
                scale = 1 if key == 'mean_return' else 100
                unit = '' if key == 'mean_return' else '%'
                cells.append(f"{scale * group[key]:.3f}{unit} ({scale * group[key + '_min']:.3f}–{scale * group[key + '_max']:.3f})")
            lines.append(f"| `{group['arm']}` | {group['fixed_policies']} | {group['shared_patients']} | " + ' | '.join(cells) + ' |')
        lines += ['', '| Outcome (V3 − V2) | Mean difference | Crossed 95% interval |', '| --- | ---: | --- |']
        for row in crossed['contrasts']:
            scale, unit = (1, ' return units') if row['metric'] == 'return' else (100, ' percentage points')
            lines.append(f"| {row['metric']} | {scale * row['mean_difference']:.3f}{unit} | [{scale * row['crossed_ci_lower']:.3f}, {scale * row['crossed_ci_upper']:.3f}] |")
        lines += ['', '[Five paired-seed descriptors](study003-paired-seeds.csv) · [Fixed-policy arm descriptors](study003-fixed-arm-descriptors.csv) · [Crossed contrast values](study003-crossed-contrasts.csv)']
    lines += ['', '## Integrity and interpretation', '',
              'The generator verifies completion and raw file hashes, copied policy/checkpoint hashes, declared counts, patient alignment, aggregate outcomes, Wilson bounds, rollout-log consistency, and the complete paired-return contrast set. It imports no simulator, trainer, policy loader, or evaluator. The output directory is new and input records are unchanged.', '',
              f"Physics fingerprint: `{bundle['protocol']['physics_fingerprint']}`. Protocol SHA-256: `{bundle['hashes']['protocol.json']}`.", '']
    return '\n'.join(lines)


def plot_report(bundle, output):
    with tempfile.TemporaryDirectory(prefix='retraction-report-plot-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'fixed-policy-report-v1'})
        rows = bundle['rows']
        figure, axes = plt.subplots(1, 3, figsize=(15, max(5.8, .4 * len(rows) + 2.5)))
        figure.subplots_adjust(left=.29, right=.985, top=.82, bottom=.17, wspace=.13)
        title = 'Final fixed-policy assessment' if bundle['phase'] == 'final' else 'Development fixed-policy fixture'
        figure.suptitle(title, x=.03, y=.97, ha='left', fontsize=17, fontweight='semibold')
        figure.text(.03, .91, f"Prespecified order · {len(rows)} controllers on the same {bundle['patient_count']} synthetic patients · no ranking or clinical claim", fontsize=10)
        for col, metric in enumerate(('success_rate', 'severe_harm_rate', 'mean_return')):
            axis = axes[col]
            axis.set_yticks(range(len(rows)), [row['controller'] for row in rows] if col == 0 else [])
            axis.invert_yaxis()
            axis.grid(axis='x', alpha=.2)
            axis.set_axisbelow(True)
            for side in ('top', 'right', 'left'):
                axis.spines[side].set_visible(False)
            axis.tick_params(axis='both', labelsize=8.3)
            if col < 2:
                axis.set_xlim(-2, 102)
            else:
                axis.axvline(0, color='#7B8794', linestyle='--', linewidth=.8)
            for position, row in enumerate(rows):
                if position % 2 == 0:
                    axis.axhspan(position - .48, position + .48, color='#F1F4F7', zorder=0)
                scale = 100 if col < 2 else 1
                color = '#0072B2' if row['role'] == 'policy' else '#555555'
                low, high = row[metric + '_ci_lower'] * scale, row[metric + '_ci_upper'] * scale
                axis.plot([low, high], [position, position], color=color, linewidth=1.2)
                axis.plot(row[metric] * scale, position, 'o', color=color, markersize=4)
            axis.set_xlabel(['Success (%)', 'Modeled severe injury (%)', 'Mean return'][col], fontsize=10)
        figure.text(.03, .075, '95% intervals: Wilson for rates; patient bootstrap for return, conditional on fixed policies. Training-run and model uncertainty are excluded.', fontsize=9)
        figure.text(.03, .035, 'Every controller shares the same patient cohort; policy-patient rollouts are not independent patient replicates. Zero observed harm does not establish safety.', fontsize=9)
        figure.savefig(output / 'fixed-policies.svg', metadata={'Date': None})
        figure.savefig(output / 'fixed-policies.png', dpi=240)
        plt.close(figure)


def generate_report(directory, design_path, output_directory, *, crossed_plan=None, policy_freeze=None, plots=False):
    bundle = validate_bundle(directory, design_path)
    freeze = validate_policy_freeze(bundle, policy_freeze) if policy_freeze else None
    crossed = crossed_analysis(bundle, crossed_plan) if crossed_plan else None
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'controllers.csv', bundle['rows'])
    contrasts = [{**row, 'paired_ci_lower': row['paired_patient_bootstrap_ci'][0],
                  'paired_ci_upper': row['paired_patient_bootstrap_ci'][1]}
                 for row in bundle['results']['contrasts']]
    for row in contrasts:
        row.pop('paired_patient_bootstrap_ci')
    write_csv(output / 'paired-return-contrasts.csv', contrasts)
    if crossed:
        write_csv(output / 'study003-crossed-contrasts.csv', crossed['contrasts'])
        write_csv(output / 'study003-paired-seeds.csv', crossed['paired_seed_descriptors'])
        write_csv(output / 'study003-fixed-arm-descriptors.csv', crossed['fixed_policy_group_descriptors'])
    (output / 'report.md').write_text(markdown(bundle, crossed, freeze), encoding='utf-8')
    if plots:
        plot_report(bundle, output)
        with (output / 'report.md').open('a', encoding='utf-8') as handle:
            handle.write('\n![All fixed controllers in prespecified order with individual outcome intervals.](fixed-policies.svg)\n\n[PNG figure](fixed-policies.png)\n')
    manifest = {'format': 'retraction-fixed-policy-report-v1', 'phase': bundle['phase'],
                'simulator_executed': False, 'policy_selection_performed': False,
                'controller_order': [row['controller'] for row in bundle['rows']],
                'shared_patients': bundle['patient_count'], 'recorded_rollouts': bundle['rollouts'],
                'input_sha256': bundle['hashes'], 'report_generator_sha256': sha(Path(__file__).read_bytes()),
                'return_bootstrap': bundle['protocol']['bootstrap'],
                'outputs_sha256': {path.name: sha(path.read_bytes()) for path in sorted(output.iterdir())}}
    if crossed:
        manifest['crossed_analysis'] = {'plan_sha256': crossed['plan_sha256'], 'plan': crossed['plan'],
                                       'contrasts': crossed['contrasts']}
    if freeze:
        manifest['policy_freeze'] = freeze
    (output / 'report-complete.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--design', type=Path, default=DEFAULT_DESIGN)
    parser.add_argument('--crossed-analysis', type=Path, help='Prespecified crossed supplement; automatic with the default final design')
    parser.add_argument('--policy-freeze', type=Path, help='Prespecified policy hash manifest; automatic with the default final design')
    parser.add_argument('--output', required=True, type=Path, help='A new report directory')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    crossed = args.crossed_analysis
    freeze = args.policy_freeze
    if crossed is None and args.design.resolve() == DEFAULT_DESIGN.resolve():
        crossed = DEFAULT_CROSSED
    if freeze is None and args.design.resolve() == DEFAULT_DESIGN.resolve():
        freeze = DEFAULT_FREEZE
    try:
        manifest = generate_report(args.evaluation, args.design, args.output, crossed_plan=crossed, policy_freeze=freeze, plots=args.plots)
        print(json.dumps({'output': str(args.output), 'phase': manifest['phase'],
                          'controllers': len(manifest['controller_order']), 'shared_patients': manifest['shared_patients']}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        parser.exit(2, f'Report not completed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
