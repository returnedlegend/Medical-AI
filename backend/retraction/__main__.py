"""Run a reproducible surrogate experiment without the web server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    from .main import RunConfig
    from .simulation import run_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON RunConfig; flags override fields in the file')
    parser.add_argument('--patients', type=int, help='Synthetic cohort size (40–2000)')
    parser.add_argument('--seed', type=int, help='Reproducible cohort and split seed')
    parser.add_argument('--profile', choices=['conservative', 'nominal', 'permissive'])
    parser.add_argument('--output', type=Path, required=True, help='New JSON result path (refuses overwrite)')
    args = parser.parse_args()
    try:
        raw = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(raw, dict):
            raise ValueError('Configuration JSON must be an object')
        if args.patients is not None:
            raw['n_patients'] = args.patients
        if args.seed is not None:
            raw['seed'] = args.seed
        if args.profile is not None:
            raw['threshold_profile'] = args.profile
        config = RunConfig.model_validate(raw).model_dump(exclude_none=True)
        if args.output.exists():
            parser.error('Output already exists; choose a new path')
        result = run_experiment(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump({'config': config, 'results': result}, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps(result['summary'], indent=2, allow_nan=False))
        print(f'Results: {args.output}', file=sys.stderr)
        return 0
    except (ValueError, OSError) as error:
        print(f'Experiment could not run: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
