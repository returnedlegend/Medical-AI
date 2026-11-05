#!/usr/bin/env python3
"""Run the real browser suites, preserving failures in public CI summaries."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare_browser(output: Path) -> str:
    """Verify an installed browser, or install managed Chromium if none exists."""
    from playwright.sync_api import sync_playwright

    package_version = version('playwright')
    if package_version != '1.62.0':
        raise RuntimeError(f'Expected pinned Playwright 1.62.0; found {package_version}')
    candidates = []
    seen = set()
    for name in ('chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable'):
        executable = shutil.which(name)
        if executable and str(Path(executable).resolve()) not in seen:
            candidates.append((name, executable))
            seen.add(str(Path(executable).resolve()))
    evidence_path = output / 'browser-probes.jsonl'

    def record(value):
        with evidence_path.open('a') as stream:
            stream.write(json.dumps(value) + '\n')

    def verify(playwright, source, executable):
        browser = None
        evidence = {'source': source, 'executable': executable, 'playwright_version': package_version}
        try:
            browser = playwright.chromium.launch(executable_path=executable, headless=True, timeout=30000)
            page = browser.new_page()
            # An in-memory data page also works on managed browsers that block
            # navigation to data: URLs. DOM assertions still exercise rendering.
            page.set_content('<title>Retraction browser readiness</title><p id="probe">browser-ready</p>', timeout=10000)
            assert page.title() == 'Retraction browser readiness'
            assert page.locator('#probe').inner_text() == 'browser-ready'
            assert browser.version
            evidence.update(status='passed', browser_version=browser.version)
            record(evidence)
            (output / 'browser-discovery.json').write_text(json.dumps(evidence, indent=2) + '\n')
            print(f'Verified {source}: {executable} (browser {browser.version}, Playwright {package_version})', flush=True)
            return executable
        except Exception as error:
            evidence.update(status='failed', error=f'{type(error).__name__}: {error}')
            record(evidence)
            print(f'Browser readiness failed for {source}: {error}', flush=True)
            return None
        finally:
            if browser is not None:
                try:
                    browser.close()
                except Exception as error:
                    record({'source': source, 'executable': executable, 'status': 'cleanup_failed',
                            'error': f'{type(error).__name__}: {error}'})

    with sync_playwright() as playwright:
        for source, executable in candidates:
            verified = verify(playwright, source, executable)
            if verified:
                return verified
        if candidates:
            raise RuntimeError('Installed browsers failed the headless data-page check; see browser-probes.jsonl. No suites were skipped.')
        managed = playwright.chromium.executable_path
        if not Path(managed).is_file():
            command = [sys.executable, '-m', 'playwright', 'install', '--with-deps', 'chromium']
            print('No installed browser found; installing pinned Playwright Chromium.', flush=True)
            with (output / 'browser-install.log').open('a') as log:
                installation = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record({'source': 'playwright-install', 'returncode': installation.returncode})
            if installation.returncode:
                raise RuntimeError('Managed Chromium installation failed; see browser-install.log.')
        verified = verify(playwright, 'playwright-managed', managed)
        if verified:
            return verified
        raise RuntimeError('Managed Chromium failed the headless data-page check; see browser-probes.jsonl.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=Path('/tmp/retraction-ci-browser'))
    parser.add_argument('--prepare-browser', action='store_true', help='Verify browser readiness without running application suites.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        chromium = prepare_browser(args.output)
    except Exception as error:
        detail = f'{type(error).__name__}: {error}'
        (args.output / 'browser-preparation-error.txt').write_text(detail + '\n')
        escaped = detail.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print(f'::error title=Browser readiness failed::{escaped}', flush=True)
        return 1
    if args.prepare_browser:
        return 0
    suites = [
        ('cohort', [sys.executable, str(root / 'scripts/browser_smoke.py'), '--base-url', args.base_url]),
        ('procedure', [sys.executable, str(root / 'scripts/procedure_browser_smoke.py'), '--base-url', args.base_url]),
        ('accessibility', [sys.executable, str(root / 'frontend/scripts/audit_accessibility.py'), '--base-url', args.base_url]),
    ]
    for _, command in suites:
        command.extend(['--chromium', chromium])
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
            # Public error annotations remain readable when GitHub requires a
            # sign-in for raw logs or custom summaries. This is captured test
            # output only; never include environment or credential values.
            detail = '\n'.join(output.splitlines()[-50:])[-6000:]
            escaped = detail.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
            print(f'::error title={name} browser checks failed::{escaped}', flush=True)
            return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
