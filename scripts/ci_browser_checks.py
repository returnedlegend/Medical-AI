#!/usr/bin/env python3
"""Run the real browser suites, preserving failures in public CI summaries."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=Path('/tmp/retraction-ci-browser'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    suites = [
        ('cohort', [sys.executable, str(root / 'scripts/browser_smoke.py'), '--base-url', args.base_url]),
        ('procedure', [sys.executable, str(root / 'scripts/procedure_browser_smoke.py'), '--base-url', args.base_url]),
        ('accessibility', [sys.executable, str(root / 'frontend/scripts/audit_accessibility.py'), '--base-url', args.base_url]),
    ]
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    for name, command in suites:
        print(f'Running actual {name} browser checks', flush=True)
        log_path = args.output / f'{name}.log'
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        output = log_path.read_text(errors='replace')
        print(output, flush=True)
        if summary:
            with Path(summary).open('a') as report:
                report.write(f'### {name} browser suite: {"passed" if result.returncode == 0 else "FAILED"}\n\n')
                if result.returncode:
                    # These are test assertions/tracebacks, never an environment
                    # or credential dump. Keep evidence visible without log API.
                    tail = '\n'.join(output.splitlines()[-80:]).replace('```', "'''")
                    report.write(f'```text\n{tail}\n```\n\n')
        if result.returncode:
            print(f'::error title={name} browser checks failed::See the job summary and browser artifacts for the actual test failure.', flush=True)
            return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
