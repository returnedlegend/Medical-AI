#!/usr/bin/env python3
"""Train/evaluate independent NumPy DQN seeds and save a reproducible bundle."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'backend'))

from retraction.learning import normalize_training_config, train_policy  # noqa: E402


def write_json(path: Path, data: dict):
    with path.open('x') as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Base JSON training configuration')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--episodes', type=int, help='Training episodes per independent seed')
    parser.add_argument('--eval-episodes', type=int, help='Disjoint evaluation episodes per controller')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--demonstration-episodes', type=int, help='Training-only scripted replay trajectories; zero disables assistance')
    parser.add_argument('--observation-encoding', choices=['box', 'characteristic'])
    parser.add_argument('--algorithm', choices=['dqn', 'tabular_q'])
    parser.add_argument('--output', type=Path, required=True, help='New bundle directory; refuses overwrite')
    args = parser.parse_args()
    try:
        base = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(base, dict):
            raise ValueError('Training config must be a JSON object')
        if not 1 <= len(args.seeds) <= 10 or len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Use 1–10 distinct training seeds')
        for key in ('episodes', 'eval_episodes', 'max_steps', 'demonstration_episodes', 'observation_encoding', 'algorithm'):
            if getattr(args, key) is not None:
                base[key] = getattr(args, key)
        configs = [normalize_training_config({**base, 'training_seed': seed}) for seed in args.seeds]
        args.output.mkdir(parents=True, exist_ok=False)
        sources = {name: hashlib.sha256((PROJECT / 'backend/retraction' / name).read_bytes()).hexdigest()
                   for name in ('learning.py', 'tabular.py', 'environment.py', 'tissue.py')}
        protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'configs': configs,
                    'source_sha256': sources, 'purpose': 'Exploratory synthetic RL evaluation, not clinical training'}
        write_json(args.output / 'protocol.json', protocol)
        summary = []
        for index, config in enumerate(configs):
            def progress(fraction, message):
                print(f'Seed {config["training_seed"]}: {100 * fraction:.0f}% · {message}', file=sys.stderr)
            result = train_policy(config, progress_callback=progress)
            prefix = f'seed-{config["training_seed"]}'
            write_json(args.output / f'{prefix}-results.json', result)
            write_json(args.output / f'{prefix}-policy.json', result['checkpoint'])
            for controller in ('learned', 'random', 'random_no_handoff', 'scripted'):
                metrics = result['evaluation'][controller]
                summary.append({'training_seed': config['training_seed'], 'controller': controller,
                                **{key: metrics[key] for key in ('n', 'mean_return', 'success_rate',
                                                               'severe_harm_rate', 'handoff_rate', 'mean_max_damage', 'mean_steps')}})
            print(f'Completed seed {config["training_seed"]} ({index + 1}/{len(configs)})')
        with (args.output / 'summary.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        write_json(args.output / 'complete.json', {'completed_training_runs': len(configs), 'rows': len(summary)})
        print(f'Completed bundle: {args.output}')
        return 0
    except (ValueError, OSError) as error:
        print(f'Training could not complete: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
