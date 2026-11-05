#!/usr/bin/env python3
"""Exercise the built workbench in a real Chromium browser.

Install Playwright's Python package separately from production dependencies and
provide Chromium with --chromium (the managed environment has /usr/bin/chromium).
Start scripts/start.sh first. Run data is retained in the application's normal
store; this check creates one genuine 40-patient run and does not delete data.
The empty, rejected-request, and offline UI checks use explicitly intercepted
responses. Simulation, history, downloads, and reload checks use the real API.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--chromium', default='/usr/bin/chromium')
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/retraction-browser-smoke'))
    args = parser.parse_args()
    base_url = args.base_url.rstrip('/')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []
    javascript_errors: list[str] = []
    expect.set_options(timeout=15_000)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: javascript_errors.append(str(error)))
        try:
            health = context.request.get(f'{base_url}/api/health')
            assert health.ok, f'Start the service first: health returned {health.status}'
            assert health.json()['simulator'] == 'analytic-surrogate'
            checks.append('real API health identifies analytic surrogate')

            runs_url = re.compile(re.escape(base_url) + r'/api/runs(?:\?.*)?$')

            def empty_history(route):
                if route.request.method == 'GET':
                    route.fulfill(status=200, json={'runs': []})
                else:
                    route.continue_()

            # A controlled empty response tests this state even if previous runs
            # exist. No persistent application data is cleared or fabricated.
            page.route(runs_url, empty_history)
            page.goto(base_url, wait_until='networkidle')
            expect(page.get_by_role('heading', name='Experiment workbench')).to_be_visible()
            expect(page.get_by_role('heading', name='When does personalization matter?')).to_be_visible()
            expect(page.get_by_role('button', name='Run experiment', exact=True)).to_be_enabled()
            page.unroute(runs_url, empty_history)
            checks.append('empty state and connected launch controls (intercepted empty history)')
            page.screenshot(path=str(args.output_dir / 'empty-desktop.png'), full_page=True)

            page.locator('#cohort').focus()
            page.locator('#cohort').press('Home')
            expect(page.locator('#cohort')).to_have_value('40')
            page.locator('#seed').fill('20261105')
            page.locator('#threshold').select_option('nominal')
            with page.expect_response(lambda response: urlparse(response.url).path == '/api/runs'
                                      and response.request.method == 'POST') as submitted:
                page.get_by_role('button', name='Run experiment', exact=True).click()
            response = submitted.value
            assert response.status == 202, f'Run submission failed: {response.text()}'
            record = response.json()
            run_id = record['id']
            expect(page.get_by_role('status')).to_contain_text('Searching the strategy grid')
            expect(page.get_by_text('Complete', exact=True)).to_be_visible(timeout=60_000)
            expect(page.get_by_role('heading', name=f'Run {run_id[:8]}', exact=False)).to_be_visible()
            expect(page.get_by_role('heading', name='Patient explorer', exact=True)).to_be_visible()
            checks.append('real run creation, progress state, completion, and results rendering')

            patient_select = page.get_by_label('Select virtual patient', exact=True)
            assert patient_select.locator('option').count() == 40
            patient_select.select_option(index=1)
            selected_patient = patient_select.input_value()
            expect(page.get_by_role('heading', name=selected_patient, exact=True)).to_be_visible()
            for strategy in ('Reference', 'Shared', 'Personalized'):
                button = page.get_by_role('button', name=strategy, exact=True)
                button.click()
                expect(button).to_have_attribute('aria-pressed', 'true')
            checks.append('patient selection and reference/shared/personalized strategy switching')

            with page.expect_download() as downloaded:
                page.get_by_role('link', name='JSON', exact=True).click()
            json_download = downloaded.value
            json_path = args.output_dir / f'{run_id}.json'
            json_download.save_as(json_path)
            exported = json.loads(json_path.read_text())
            assert exported['run']['id'] == run_id
            assert exported['run']['config']['n_patients'] == 40
            assert exported['run']['config']['seed'] == 20261105
            results = exported['results']
            assert results['engine'] == 'analytic-surrogate'
            assert len(results['patients']) == 40
            assert results['summary']['test_size'] == 12
            assert results['summary']['oracle_success_rate'] >= results['summary']['shared_success_rate']
            with page.expect_download() as downloaded:
                page.get_by_role('link', name='CSV', exact=True).click()
            csv_path = args.output_dir / f'{run_id}.csv'
            downloaded.value.save_as(csv_path)
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
            assert len(rows) == 40
            assert {row['run_id'] for row in rows} == {run_id}
            assert {row['id'] for row in rows} == {p['id'] for p in results['patients']}
            checks.append('real JSON/CSV downloads match persisted run and all patient IDs')
            page.screenshot(path=str(args.output_dir / 'results-desktop.png'), full_page=True)

            page.get_by_role('button', name='Virtual patients', exact=True).click()
            expect(page.get_by_role('heading', name='Cohort ledger', exact=True)).to_be_visible()
            cohort_filter = page.get_by_label('Filter cohort', exact=True)
            ledger_rows = page.locator('.cohort-panel tbody tr')
            expect(ledger_rows).to_have_count(40)
            for value, predicate in (
                ('test', lambda p: p['split'] == 'test'),
                ('personalize', lambda p: p['category'] == 'personalize'),
                ('bail_out', lambda p: p['category'] == 'bail_out'),
            ):
                cohort_filter.select_option(value)
                expect(ledger_rows).to_have_count(sum(predicate(p) for p in results['patients']))
            cohort_filter.select_option('test')
            ledger_rows.first.get_by_role('button').click()
            expect(patient_select).to_have_value(next(p['id'] for p in results['patients'] if p['split'] == 'test'))
            checks.append('cohort ledger filtering agrees with exported held-out and outcome labels')

            page.get_by_role('button', name='Methods & limits', exact=True).click()
            expect(page.get_by_role('heading', name='Methods & limitations', exact=True)).to_be_visible()
            expect(page.get_by_text('Research use only', exact=True)).to_be_visible()
            page.reload(wait_until='networkidle')
            expect(page.get_by_role('heading', name=f'Run {run_id[:8]}', exact=False)).to_be_visible()
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            page.get_by_role('button', name=re.compile(r'^Run history')).click()
            dialog = page.get_by_role('dialog', name='Experiment history')
            expect(dialog).to_be_visible()
            history_run = dialog.get_by_role('button', name=re.compile(rf'^Run {run_id[:8]}'))
            expect(history_run).to_be_visible()
            history_run.click()
            expect(dialog).not_to_be_visible()
            expect(page.get_by_role('heading', name='Patient explorer', exact=True)).to_be_visible()
            checks.append('methods disclosure, reload persistence, and saved-history selection')

            def rejected_submission(route):
                if route.request.method == 'POST':
                    route.fulfill(status=429, json={'detail': 'Smoke test: queue is full'})
                else:
                    route.continue_()

            page.route(runs_url, rejected_submission)
            page.get_by_role('button', name='Run experiment', exact=True).click()
            expect(page.get_by_role('alert')).to_contain_text('Smoke test: queue is full')
            expect(page.get_by_role('button', name='Run experiment', exact=True)).to_be_enabled()
            page.unroute(runs_url, rejected_submission)
            page.get_by_role('button', name='Dismiss error', exact=True).click()
            checks.append('rejected request shows actionable error and restores launch controls (intercepted 429)')

            page.set_viewport_size({'width': 390, 'height': 844})
            page.reload(wait_until='networkidle')
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Mobile page overflows horizontally'
            expect(page.get_by_role('heading', name='Experiment workbench')).to_be_visible()
            page.screenshot(path=str(args.output_dir / 'results-mobile.png'), full_page=True)
            page.get_by_role('button', name='Virtual patients', exact=True).click()
            expect(page.get_by_role('heading', name='Cohort ledger', exact=True)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Mobile patient ledger overflows horizontally'
            checks.append('390px mobile results and ledger remain usable without page overflow')

            health_url = f'{base_url}/api/health'
            page.route(health_url, lambda route: route.fulfill(status=503, json={'detail': 'Smoke test: service offline'}))
            page.reload(wait_until='networkidle')
            expect(page.get_by_role('alert')).to_contain_text('research service is unavailable')
            expect(page.get_by_role('button', name='Run experiment', exact=True)).to_be_disabled()
            page.unroute(health_url)
            page.get_by_role('button', name='Service offline', exact=True).click()
            expect(page.get_by_role('button', name='Service connected', exact=True)).to_be_visible()
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            expect(page.get_by_role('alert')).to_have_count(0)
            checks.append('offline launch is disabled and connection retry recovers (intercepted 503)')
            assert not javascript_errors, f'Unhandled browser errors: {javascript_errors}'
            report = {'status': 'passed', 'run_id': run_id, 'checks': checks,
                      'javascript_errors': javascript_errors, 'artifacts': str(args.output_dir)}
            (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
            return 0
        except Exception:
            page.screenshot(path=str(args.output_dir / 'failure.png'), full_page=True)
            print(json.dumps({'status': 'failed', 'checks_completed': checks,
                              'javascript_errors': javascript_errors,
                              'screenshot': str(args.output_dir / 'failure.png')}, indent=2), file=sys.stderr)
            raise
        finally:
            context.close()
            browser.close()


if __name__ == '__main__':
    raise SystemExit(main())
