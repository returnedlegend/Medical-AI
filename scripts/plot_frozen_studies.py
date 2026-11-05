#!/usr/bin/env python3
"""Render recorded study 001/002 validation results; never run a simulator.

Requires Matplotlib (including its NumPy dependency), separate from application
runtime requirements. Inputs are tracked summaries and complete study archives.
All training runs are displayed; validation-selected references are not promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

_MPL_CACHE = tempfile.TemporaryDirectory(prefix='retraction-figure-cache-')
os.environ.setdefault('MPLCONFIGDIR', _MPL_CACHE.name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ['o', 's', '^', 'D', 'v']
COLORS = {'dqn': '#0072B2', 'tabular_q': '#009E73', 'tabular_v1': '#009E73',
          'tabular_v2': '#D55E00', 'tabular_v2_unassisted': '#AA4499',
          'scripted': '#444444', 'random': '#7B8794', 'random_no_handoff': '#7B8794'}
ARMS = {
    '001': [('dqn', 'Double-DQN\n100 demonstrations'), ('tabular_q', 'Tabular V1\n100 demonstrations')],
    '002': [('tabular_v1', 'Tabular V1\n100 demonstrations'), ('tabular_v2', 'Tabular V2\n100 demonstrations'),
            ('tabular_v2_unassisted', 'Tabular V2\n0 demonstrations')],
}
CONTROLS = [('scripted', 'Continuous scripted\nno learning'), ('random', 'Uniform random\nno learning'),
            ('random_no_handoff', 'Random, no handoff\nno learning')]


def sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_study(number):
    summary_path = ROOT / f'examples/rl-study-{number}-summary.json'
    archive_path = ROOT / f'research-artifacts/rl-study-{number}-full.zip'
    summary_bytes, archive_bytes = summary_path.read_bytes(), archive_path.read_bytes()
    expected_archive = archive_path.with_suffix('.zip.sha256').read_text().split()[0]
    if sha256(archive_bytes) != expected_archive:
        raise ValueError(f'Study {number}: archive checksum failed')
    summary = json.loads(summary_bytes)
    if not summary['reserved_patient_exclusion']['checked']:
        raise ValueError(f'Study {number}: reserved evaluation exclusion was not audited')
    seeds = sorted({run['training_seed'] for run in summary['runs']})
    if len(seeds) != 5:
        raise ValueError('These figures require the five recorded training seeds, not a selected subset')
    learned = {arm: {} for arm, _ in ARMS[number]}
    controls = {mode: {} for mode, _ in CONTROLS}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('MANIFEST-SHA256.json'))
        for run in summary['runs']:
            arm = run.get('arm', run['algorithm'])
            if arm not in learned or run['training_seed'] in learned[arm]:
                raise ValueError('Unexpected or duplicate recorded training run')
            contents = archive.read(run['source'])
            if sha256(contents) != manifest[run['source']]:
                raise ValueError('A raw result does not match its archive manifest')
            result = json.loads(contents)
            evaluation = result['evaluation']
            metrics = evaluation['learned']
            for key in ('mean_return', 'success_rate', 'severe_harm_rate'):
                if not math.isclose(metrics[key], run['learned'][key], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Summary does not match the complete recorded result')
            if metrics['n'] != 30 or len(result['learning_curve']) != 3000:
                raise ValueError('A recorded run has an unexpected episode count')
            patients = [episode['seed'] for episode in metrics['episodes']]
            if len(patients) != 30 or patients != result['metadata']['test_seeds']:
                raise ValueError('Learned episodes do not match their recorded validation patients')
            seed = run['training_seed']
            learned[arm][seed] = metrics
            for mode in controls:
                if [episode['seed'] for episode in evaluation[mode]['episodes']] != patients:
                    raise ValueError('Controller results do not use the same paired validation patients')
                if seed in controls[mode] and canonical(controls[mode][seed]) != canonical(evaluation[mode]):
                    raise ValueError('Repeated control records differ across paired arms; cannot deduplicate')
                controls[mode][seed] = evaluation[mode]
    for arm, records in learned.items():
        if sorted(records) != seeds:
            raise ValueError('A complete arm is missing a training seed')
        for metric in ('mean_return', 'success_rate', 'severe_harm_rate'):
            mean = np.mean([records[seed][metric] for seed in seeds])
            recorded = summary['algorithms'][arm]['learned']['mean_across_training_runs'][metric]
            if not math.isclose(mean, recorded, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Across-seed aggregate does not match the five plotted runs')
    first_arm = ARMS[number][0][0]
    aggregates = {arm: summary['algorithms'][arm]['learned'] for arm in learned}
    aggregates.update({mode: summary['algorithms'][first_arm][mode] for mode in controls})
    provenance = {'summary': str(summary_path.relative_to(ROOT)), 'summary_sha256': sha256(summary_bytes),
                  'archive': str(archive_path.relative_to(ROOT)), 'archive_sha256': sha256(archive_bytes),
                  'protocol_sha256': summary['protocol_sha256'], 'training_seeds': seeds,
                  'learned_runs': len(summary['runs']), 'validation_patients_per_seed': 30,
                  'unique_validation_patients': len(summary['all_validation_patient_seeds']),
                  'controls_deduplicated_across_arms': True, 'physics_fingerprint': summary['physics_fingerprint']}
    return {'number': number, 'summary': summary, 'seeds': seeds, 'records': {**learned, **controls},
            'aggregates': aggregates, 'provenance': provenance}


def errorbar(axis, point, interval, y, color, marker, *, mean=False):
    if interval is not None:
        # Stored percentile intervals need not be symmetric around the estimate.
        axis.plot(interval, [y, y], color=color, linewidth=1.8 if mean else .75,
                  alpha=1 if mean else .65, zorder=4 if mean else 2)
        axis.vlines(interval, y - (.045 if mean else .025), y + (.045 if mean else .025),
                    color=color, linewidth=.9, zorder=3)
    axis.plot(point, y, marker=marker, markersize=5.5 if mean else 4.2,
              markerfacecolor=color if mean else 'white', markeredgecolor=color,
              markeredgewidth=.9, linestyle='none', zorder=6 if mean else 5)


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis='x', color='#DEE3E8', linewidth=.6)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color('#78838D')
    axis.tick_params(axis='both', labelsize=8.5, length=3, color='#78838D')
    axis.tick_params(axis='y', length=0, pad=10)


def overview(studies):
    figure, axes = plt.subplots(2, 3, figsize=(14.6, 10.8),
                                gridspec_kw={'height_ratios': [5, 6]})
    figure.subplots_adjust(left=.175, right=.985, top=.825, bottom=.145, wspace=.10, hspace=.55)
    figure.suptitle('All training seeds in two frozen validation studies', x=.175, y=.976,
                     ha='left', fontsize=17, fontweight='semibold')
    figure.text(.175, .944, 'Exploratory synthetic results · validation, not an independent test · no clinical validity claim',
                fontsize=10.5, color='#4B5967')
    definitions = [('success_rate', 'Validation success (%)', 'Success'),
                   ('severe_harm_rate', 'Modeled severe injury (%)', 'Severe injury'),
                   ('mean_return', 'Mean episode return', 'Return')]
    offsets = np.linspace(-.24, .24, 5)
    for row, study in enumerate(studies):
        methods = ARMS[study['number']] + CONTROLS
        row_top = axes[row, 0].get_position().y1
        figure.text(.175, row_top + .060, f"Study {study['number']} · five training seeds {study['seeds'][0]}–{study['seeds'][-1]}",
                    fontsize=11, fontweight='semibold')
        handles = [Line2D([], [], marker=marker, color='#586575', markerfacecolor='white',
                          linestyle='none', markersize=4.3, label=str(seed))
                   for marker, seed in zip(MARKERS, study['seeds'])]
        figure.legend(handles=handles, loc='upper right', bbox_to_anchor=(.986, row_top + .075),
                      ncol=5, frameon=False, fontsize=8.5, handletextpad=.3, columnspacing=1.0,
                      title='Training seed / paired control cohort', title_fontsize=8.5)
        for col, (metric, xlabel, panel_title) in enumerate(definitions):
            axis = axes[row, col]
            style_axis(axis)
            axis.set_title(f'{chr(65 + row * 3 + col)}  {panel_title}', loc='left', fontsize=11, pad=12)
            axis.set_xlabel(xlabel, fontsize=9.3, labelpad=8)
            axis.set_yticks(range(len(methods)))
            axis.set_yticklabels([label for _, label in methods] if col == 0 else [])
            axis.set_ylim(len(methods) - .43, -.52)
            axis.set_xlim((-3, 103) if col < 2 else (-35, 20))
            axis.set_xticks([0, 25, 50, 75, 100] if col < 2 else [-30, -20, -10, 0, 10, 20])
            if col == 2:
                axis.axvline(0, color='#74808B', linestyle=(0, (3, 3)), linewidth=.8)
            axis.axhline(len(ARMS[study['number']]) - .49, color='#A8B1BB', linewidth=.8)
            for position, (arm, _) in enumerate(methods):
                if position % 2 == 0:
                    axis.axhspan(position - .48, position + .47, color='#F3F5F7', zorder=0)
                scale = 100 if col < 2 else 1
                for seed, marker, offset in zip(study['seeds'], MARKERS, offsets):
                    metrics = study['records'][arm][seed]
                    interval_key = 'mean_return_ci' if metric == 'mean_return' else metric + '_ci'
                    interval = [scale * value for value in metrics[interval_key]]
                    errorbar(axis, scale * metrics[metric], interval, position + offset, COLORS[arm], marker)
                aggregate = study['aggregates'][arm]
                mean = scale * aggregate['mean_across_training_runs'][metric]
                interval = aggregate['mean_return_hierarchical_bootstrap_ci'] if col == 2 else None
                errorbar(axis, mean, interval, position + .36, '#15212D', 'D', mean=True)
    figure.text(.175, .091, 'Small symbols show every training seed (or matched control cohort), n = 30 patients each. Thin bars: per-run 95% Wilson intervals for rates; patient-bootstrap intervals for return.', fontsize=8.5)
    figure.text(.175, .068, 'Black diamonds: means of five seeds. Thick return bars: recorded 95% hierarchical bootstrap intervals, resampling seed then patient (2,000 replicates).', fontsize=8.5)
    figure.text(.175, .045, 'Controls are counted once per study. Studies use different patient cohorts; do not pair or pool across studies. Zero observed severe injury does not establish safety.', fontsize=8.5, color='#4B5967')
    return figure


def contrasts(studies):
    first, second = studies
    definitions = [(first, 'tabular_minus_dqn_return', 'tabular_q', 'dqn',
                    'Study 001: Tabular V1 − Double-DQN\nboth assisted'),
                   (second, 'tabular_v2_minus_tabular_v1_return', 'tabular_v2', 'tabular_v1',
                    'Study 002: V2 assisted − V1 assisted'),
                   (second, 'tabular_v2_unassisted_minus_tabular_v1_return', 'tabular_v2_unassisted', 'tabular_v1',
                    'Study 002: V2 unassisted − V1 assisted'),
                   (second, 'tabular_v2_unassisted_minus_tabular_v2_return', 'tabular_v2_unassisted', 'tabular_v2',
                    'Study 002: V2 unassisted − V2 assisted')]
    figure, axis = plt.subplots(figsize=(12.8, 5.7))
    figure.subplots_adjust(left=.315, right=.98, top=.79, bottom=.23)
    style_axis(axis)
    axis.set_xlim(-20, 43)
    axis.set_ylim(3.52, -.55)
    axis.set_yticks(range(4), [item[4] for item in definitions])
    axis.axvline(0, color='#4B5967', linestyle=(0, (4, 3)), linewidth=1)
    axis.axhline(.5, color='#A8B1BB', linewidth=.8)
    axis.set_xlabel('Paired validation mean-return difference (positive arm minus negative arm)', fontsize=10, labelpad=10)
    for row, (study, name, positive, negative, _) in enumerate(definitions):
        if row % 2 == 0:
            axis.axhspan(row - .48, row + .48, color='#F3F5F7', zorder=0)
        estimate = study['summary']['contrasts'][name]
        differences = [study['records'][positive][seed]['mean_return'] - study['records'][negative][seed]['mean_return']
                       for seed in study['seeds']]
        if not math.isclose(float(np.mean(differences)), estimate['mean'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Recorded contrast does not match the five paired seed estimates')
        for seed, marker, offset in zip(study['seeds'], MARKERS, np.linspace(-.24, .24, 5)):
            difference = study['records'][positive][seed]['mean_return'] - study['records'][negative][seed]['mean_return']
            errorbar(axis, difference, None, row + offset, COLORS[positive], marker)
            axis.annotate(str(seed), (difference, row + offset), xytext=(5, 0), textcoords='offset points',
                          fontsize=7.5, va='center', color='#4B5967')
        errorbar(axis, estimate['mean'], estimate['hierarchical_bootstrap_ci'], row + .36, '#15212D', 'D', mean=True)
    figure.suptitle('Paired return contrasts across all training seeds', x=.03, y=.965,
                     ha='left', fontsize=16, fontweight='semibold')
    figure.text(.03, .898, 'Validation only · 100 demonstrations in assisted arms; none in the unassisted arm · no independent test or clinical claim', fontsize=10, color='#4B5967')
    figure.text(.03, .125, 'Small labeled symbols: each recorded training-seed contrast on its same 30 validation patients. Black diamonds and bars: mean and recorded 95% paired hierarchical bootstrap interval.', fontsize=8.7)
    figure.text(.03, .082, 'The Study 002 assisted V2 − V1 interval includes zero; these five seeds do not establish V2 superiority. Studies use separate cohorts and are not paired with one another.', fontsize=8.7, color='#4B5967')
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/assets')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'medical-ai-frozen-validation-v1', 'axes.titleweight': 'semibold'})
    studies = [read_study(number) for number in ('001', '002')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in [('rl-validation-all-seeds', overview(studies)), ('rl-validation-paired-contrasts', contrasts(studies))]:
        figure.savefig(args.output_dir / f'{name}.svg', metadata={'Date': None, 'Creator': 'scripts/plot_frozen_studies.py'})
        figure.savefig(args.output_dir / f'{name}.png', dpi=240, metadata={'Software': 'scripts/plot_frozen_studies.py'})
        plt.close(figure)
    provenance = {'format': 'retraction-validation-figure-provenance-v1', 'validation_only': True,
                  'reserved_evaluation_used': False, 'simulator_executed': False,
                  'studies': [study['provenance'] for study in studies],
                  'uncertainty': 'Stored per-run Wilson/patient-bootstrap intervals and recorded hierarchical return intervals; no new fits or evaluations.',
                  'selection': 'Every run is plotted; selected reference policies are not singled out.'}
    provenance['renderer'] = {'script_sha256': sha256(Path(__file__).read_bytes()),
                              'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__, 'png_dpi': 240}
    provenance['outputs'] = {path.name: sha256(path.read_bytes()) for path in sorted(args.output_dir.glob('rl-validation-*'))
                             if path.suffix in ('.svg', '.png')}
    (args.output_dir / 'rl-validation-figures-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Wrote two SVG/PNG validation figures and provenance to {args.output_dir}")


if __name__ == '__main__':
    main()
