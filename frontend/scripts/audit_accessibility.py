#!/usr/bin/env python3
"""Audit the built, running workbench with locally installed axe-core.

Run the repository's real browser smoke first so a completed cohort exists.
Requires Python Playwright and Chromium, as does scripts/browser_smoke.py.
This audit does not create runs or alter the application's saved records.
Zero automated violations does not establish full accessibility conformance;
axe's incomplete/manual-review findings are preserved in the output report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--chromium', default=shutil.which('chromium'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/retraction-accessibility/report.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'node_modules/axe-core/axe.min.js'
    if not source.is_file():
        parser.error('Run npm ci in frontend to install the pinned local axe-core dependency.')
    report = {'axe_version': '4.10.3', 'states': {}, 'limitations':
              'Automated WCAG2/2.1 A/AA checks only. Incomplete findings require manual review.'}
    expect.set_options(timeout=15_000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1050})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            response = context.request.get(f'{args.base_url.rstrip("/")}/api/runs')
            assert response.ok, 'Start scripts/start.sh before running this audit.'
            completed = next((run for run in response.json()['runs'] if run['status'] == 'completed'), None)
            assert completed, 'Complete a cohort first (for example with scripts/browser_smoke.py). No records were changed.'
            page.goto(args.base_url, wait_until='networkidle')
            page.add_script_tag(path=str(source))
            if not page.locator('.metrics-grid').count():
                page.get_by_role('button', name='Run history', exact=False).click()
                page.locator('.history-list > button').filter(has_text=completed['id'][:8]).click()
            expect(page.locator('.metrics-grid')).to_be_visible()

            def audit(name):
                page.wait_for_timeout(150)
                data = page.evaluate("""async () => await axe.run(document, {
                  runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}
                })""")
                report['states'][name] = {
                    'violations': data['violations'], 'incomplete': data['incomplete'],
                    'passed_rules': len(data['passes']),
                }
                print(f'{name}: {len(data["violations"])} violations; '
                      f'{len(data["incomplete"])} incomplete/manual-review rules', flush=True)

            audit('desktop_overview')
            page.get_by_role('button', name='Patient parameter ranges', exact=True).click()
            audit('desktop_advanced_setup')
            page.get_by_role('button', name='Patient parameter ranges', exact=True).click()
            page.get_by_role('button', name='Try a custom retraction', exact=False).click()
            expect(page.locator('.strategy-scatter circle').first).to_be_visible()
            audit('desktop_custom_strategy')
            page.get_by_role('button', name='Virtual patients', exact=True).click()
            audit('desktop_patient_ledger')
            page.get_by_role('button', name='Methods & limits', exact=True).click()
            audit('desktop_methods')
            page.get_by_role('button', name='Physics sandbox', exact=True).click()
            page.wait_for_load_state('networkidle')
            page.wait_for_function("""() => document.querySelector('.physics-unavailable') ||
              document.querySelector('.physics-frame img') ||
              document.querySelector('.physics-results .empty-state')""")
            audit('desktop_physics')
            details = page.locator('.physics-messages')
            if details.count():
                details.locator('summary').click()
                audit('desktop_physics_messages')
            page.get_by_role('button', name='Experiments', exact=False).click()
            page.get_by_role('button', name='Run history', exact=False).click()
            audit('desktop_history_modal')
            page.keyboard.press('Escape')
            page.set_viewport_size({'width': 390, 'height': 844})
            audit('mobile_overview')
            page.get_by_role('button', name='Physics sandbox', exact=True).click()
            page.wait_for_load_state('networkidle')
            page.wait_for_function("""() => document.querySelector('.physics-unavailable') ||
              document.querySelector('.physics-frame img') ||
              document.querySelector('.physics-results .empty-state')""")
            audit('mobile_physics')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.output.with_name('physics-mobile.png')), full_page=True)
            page.set_viewport_size({'width': 1440, 'height': 1050})
            page.get_by_role('button', name='Experiments', exact=False).click()
            page.screenshot(path=str(args.output.with_name('overview-desktop.png')), full_page=True)
            report['javascript_errors'] = errors
        finally:
            browser.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    failed = bool(report.get('javascript_errors')) or any(
        state['violations'] for state in report['states'].values())
    print(f'Report: {args.output}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
