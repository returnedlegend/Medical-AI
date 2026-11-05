"""Freeze and execute an exploratory multi-seed, multi-threshold study bundle."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from .main import RunConfig
from .simulation import MODEL_VERSION, run_experiment


def write_json(path: Path, payload: dict) -> None:
    with path.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_study(output: Path, configs: list[dict]) -> dict:
    """Never overwrite existing bundles. Save the complete protocol before any run."""
    normalized = [RunConfig.model_validate(config).model_dump(exclude_none=True) for config in configs]
    if not normalized or len(normalized) > 60:
        raise ValueError('A study requires 1–60 runs')
    if len({json.dumps(c, sort_keys=True) for c in normalized}) != len(normalized):
        raise ValueError('Duplicate configurations are not independent runs')
    protocol = {'model_version': MODEL_VERSION, 'configs': normalized,
                'purpose': 'Exploratory synthetic study; not externally preregistered or clinically validated.'}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('simulation.py', 'analysis.py', 'study.py', 'main.py')}
    # mkdir is atomic and also rejects accidental reuse of old study directories.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol': protocol, 'protocol_sha256': digest, 'source_sha256': sources,
                'created_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version()}
    write_json(output / 'protocol.json', manifest)
    rows = []
    for index, config in enumerate(normalized):
        result = run_experiment(config)
        name = f'run-{index + 1:03d}-seed-{config["seed"]}-{config["threshold_profile"]}.json'
        write_json(output / name, {'config': config, 'results': result})
        summary = result['summary']
        rows.append({'file': name, 'seed': config['seed'], 'profile': config['threshold_profile'],
                     'n_patients': config['n_patients'], 'test_size': summary['test_size'],
                     'shared_success_rate': summary['shared_success_rate'],
                     'oracle_success_rate': summary['oracle_success_rate'],
                     'personalization_gain': summary['personalization_gain'],
                     'bailout_rate': summary['bailout_rate'], 'probe_auroc': result['probe']['auroc'],
                     'probe_ece': result['probe']['ece']})
        print(f'[{index + 1}/{len(normalized)}] {name}', file=sys.stderr)
    with (output / 'summary.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Exploratory surrogate study', '', f'Protocol SHA-256: `{digest}`', '',
             'Synthetic, illustrative mechanics. No clinical inference or LapGym/SOFA validation.',
             'Each shared strategy and probe classifier is fitted independently on its training split.',
             'Seeds are repeated synthetic cohorts, not independent clinical or tissue-model validations.', '',
             '| Seed | Profile | Held-out n | Shared | Grid best | Gain (pp) | Bail-out | Probe AUROC |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        auroc = f'{row["probe_auroc"]:.3f}' if row['probe_auroc'] is not None else 'undefined'
        lines.append(f'| {row["seed"]} | {row["profile"]} | {row["test_size"]} | '
                     f'{100 * row["shared_success_rate"]:.1f}% | {100 * row["oracle_success_rate"]:.1f}% | '
                     f'{100 * row["personalization_gain"]:.1f} | {100 * row["bailout_rate"]:.1f}% | {auroc} |')
    lines += ['', 'Full configurations and code fingerprints were saved before simulations began in `protocol.json`.',
              'This local protocol capture is not a time-stamped external preregistration.',
              'An absent `complete.json` indicates an interrupted or failed bundle; partial runs must not be treated as complete.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    complete = {'protocol_sha256': digest, 'completed_runs': len(rows), 'files': [r['file'] for r in rows]}
    write_json(output / 'complete.json', complete)
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Base JSON config; study seed and profile flags override it')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--profiles', nargs='+', choices=['conservative', 'nominal', 'permissive'],
                        default=['conservative', 'nominal', 'permissive'])
    parser.add_argument('--patients', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True, help='New bundle directory; never overwritten')
    args = parser.parse_args()
    try:
        base = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(base, dict):
            raise ValueError('Configuration JSON must be an object')
        if args.patients is not None:
            base['n_patients'] = args.patients
        # Threshold overrides would obscure the profile sensitivity comparison.
        if any(base.get(key) is not None for key in ('force_limit_n', 'strain_limit', 'exposure_threshold')):
            raise ValueError('Study profiles require named thresholds; remove custom threshold overrides from the base config')
        configs = [{**base, 'seed': seed, 'threshold_profile': profile}
                   for seed in args.seeds for profile in args.profiles]
        run_study(args.output, configs)
        print(f'Completed study: {args.output}')
        return 0
    except (ValueError, OSError) as error:
        print(f'Study could not complete: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
