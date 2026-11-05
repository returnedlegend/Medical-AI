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
import shutil
import sys
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--chromium', default=shutil.which('chromium'), help='Chromium executable; otherwise use Playwright-managed Chromium')
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

        def workspace(name):
            page.get_by_role('navigation', name='Research workspaces').get_by_role('button', name=name, exact=True).click()

        def open_study():
            workspace('Cohort experiments')
            expect(page.get_by_role('heading', name='Cohort strategy study', exact=True)).to_be_visible()
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
            open_study()
            expect(page.get_by_role('heading', name='Cohort strategy study')).to_be_visible()
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
            with page.expect_download() as downloaded:
                page.get_by_role('link', name='Report', exact=True).click()
            report_path = args.output_dir / f'{run_id}.md'
            downloaded.value.save_as(report_path)
            report = report_path.read_text()
            assert report.startswith('# Retraction Lab · Exploratory experiment report')
            assert f'Run: `{run_id}`' in report
            assert f'Seed: {exported["run"]["config"]["seed"]}' in report
            assert f'{results["summary"]["personalization_gain"] * 100:.1f} percentage points' in report
            assert 'Analytic surrogate only' in report
            assert 'clinical safety' in report
            checks.append('Markdown report download matches selected run, primary gain, and research limitations')

            # Subgroup rows must display held-out values from the genuine export.
            subgroup_panel = page.locator('.subgroup-panel')
            expect(subgroup_panel).to_be_visible()
            for dimension, button_name in (('attachment_length', 'Attachment'), ('adhesions', 'Adhesions'), ('distension', 'Distension')):
                subgroup_panel.get_by_role('button', name=button_name, exact=True).click()
                expected_groups = [g for g in results['subgroups'] if g['dimension'] == dimension]
                subgroup_rows = subgroup_panel.locator('tbody tr')
                expect(subgroup_rows).to_have_count(len(expected_groups))
                for index, group in enumerate(expected_groups):
                    cells = subgroup_rows.nth(index).locator('td')
                    expect(cells.nth(0)).to_have_text(group['label'])
                    expect(cells.nth(1)).to_have_text(str(group['n']))
                    expect(cells.nth(2)).to_have_text(f"{group['shared_success_rate'] * 100:.1f}%" if group['n'] else '—')
                    expect(cells.nth(3)).to_have_text(f"{group['oracle_success_rate'] * 100:.1f}%" if group['n'] else '—')
                    if group['n']:
                        expect(cells.nth(4)).to_contain_text(f"+{group['personalization_gain'] * 100:.1f} pp")
            checks.append('attachment/adhesion/distension subgroup table agrees with exported held-out rates')

            # Reuse changes only the draft controls; saved data remains fixed.
            page.locator('#seed').fill('111')
            page.locator('#threshold').select_option('conservative')
            page.get_by_role('button', name='Reuse setup', exact=True).click()
            expect(page.locator('#seed')).to_have_value('20261105')
            expect(page.locator('#threshold')).to_have_value('nominal')
            expect(page.locator('#cohort')).to_have_value('40')
            checks.append('reuse setup restores selected run configuration into editable controls')

            explorer = page.locator('.strategy-explorer')
            preview_url = f'{base_url}/api/preview'
            with page.expect_response(lambda r: r.url == preview_url and r.request.method == 'POST'):
                explorer.get_by_role('button', name=re.compile(r'^Try a custom retraction')).click()
            expect(explorer.locator('.constraint-meters')).to_be_visible()
            scatter = explorer.get_by_role('group', name=re.compile(r'^All strategy outcomes'))
            expect(scatter.get_by_role('button')).to_have_count(75)
            grid_response = context.request.get(f'{base_url}/api/runs/{run_id}/patients/{selected_patient}/strategies')
            assert grid_response.ok
            grid = grid_response.json()
            assert len(grid['strategies']) == 75
            for outcome in grid['strategies']:
                name = f"{outcome['strategy']['id']}: force {outcome['force_n']:.2f} N, exposure {outcome['exposure'] * 100:.0f}%, {'success' if outcome['success'] else 'failure'}"
                expect(scatter.get_by_role('button', name=name, exact=True)).to_have_count(1)
            distance_slider = explorer.get_by_label('Custom pull distance', exact=False)
            with page.expect_response(lambda r: r.url == preview_url and r.request.post_data_json['strategy']['distance_mm'] == 0) as zero_response:
                distance_slider.focus()
                distance_slider.press('Home')
            zero_outcome = zero_response.value.json()['outcome']
            assert zero_outcome['force_n'] == 0 and zero_outcome['strain'] == 0 and zero_outcome['exposure'] == 0
            assert zero_outcome['success'] is False
            expect(explorer.locator('.preview-status')).to_contain_text('Proxy constraints do not all pass')
            meters = explorer.locator('.constraint-meters .constraint b')
            expect(meters.nth(0)).to_have_text('0.00 N')
            expect(meters.nth(1)).to_have_text('0.0%')
            expect(meters.nth(2)).to_have_text('0.0%')
            with page.expect_response(lambda r: r.url == preview_url and r.request.post_data_json['strategy']['distance_mm'] == 30) as full_response:
                distance_slider.press('End')
            full_outcome = full_response.value.json()['outcome']
            assert full_outcome['force_n'] > 0 and full_outcome['strain'] > 0
            expect(meters.nth(0)).to_have_text(f"{full_outcome['force_n']:.2f} N")
            expect(meters.nth(1)).to_have_text(f"{full_outcome['strain'] * 100:.1f}%")
            candidate = grid['strategies'][0]
            grid_point = scatter.get_by_role('button', name=re.compile('^' + re.escape(candidate['strategy']['id']) + ':'))
            with page.expect_response(lambda r: r.url == preview_url and r.request.post_data_json['strategy']['distance_mm'] == candidate['strategy']['distance_mm']):
                grid_point.focus()
                grid_point.press('Enter')
            expect(explorer.get_by_label('Custom grasp position', exact=False)).to_have_value(str(candidate['strategy']['grasp_position']))
            expect(explorer.get_by_label('Custom pull direction', exact=False)).to_have_value(str(candidate['strategy']['direction_deg']))
            expect(distance_slider).to_have_value(str(candidate['strategy']['distance_mm']))
            expect(meters.nth(0)).to_have_text(f"{candidate['force_n']:.2f} N")
            checks.append('custom preview zero/30mm mechanics, all75 real grid points, keyboard point selection and slider synchronization')

            # Hold one old preview request while switching patients. Releasing
            # its response afterwards must not overwrite the new patient view.
            pending_previews = []
            def hold_preview(route):
                payload = route.request.post_data_json
                if payload['strategy']['distance_mm'] == 30:
                    pending_previews.append(route)
                else:
                    route.continue_()
            page.route(preview_url, hold_preview)
            with page.expect_request(lambda request: request.url == preview_url and request.post_data_json['strategy']['distance_mm'] == 30):
                distance_slider.focus()
                distance_slider.press('End')
            other_patient = next(p for p in results['patients'] if p['id'] != selected_patient and p['oracle']['strategy']['distance_mm'] != 30)
            with page.expect_response(lambda r: r.url == preview_url and r.request.post_data_json['patient']['attachment_length_mm'] == other_patient['attachment_length_mm']):
                patient_select.select_option(other_patient['id'])
            expect(explorer.locator('.preview-status')).not_to_contain_text('Updating mechanics')
            assert pending_previews, 'The race check must hold an actual old request'
            for held in pending_previews:
                stale_response = context.request.post(preview_url, data=held.request.post_data_json)
                try:
                    held.fulfill(status=200, json=stale_response.json())
                except PlaywrightError:
                    if not held.request.failure:
                        raise
            page.unroute(preview_url, hold_preview)
            expect(page.get_by_role('heading', name=other_patient['id'], exact=True)).to_be_visible()
            expect(distance_slider).to_have_value(str(other_patient['oracle']['strategy']['distance_mm']))
            expect(meters.nth(0)).to_have_text(f"{other_patient['oracle']['force_n']:.2f} N")
            checks.append('patient switching cancels old preview and prevents stale metrics overwriting new patient')
            page.screenshot(path=str(args.output_dir / 'custom-preview-desktop.png'), full_page=True)

            page.screenshot(path=str(args.output_dir / 'results-desktop.png'), full_page=True)

            page.get_by_role('button', name='Patient records', exact=True).click()
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

            page.get_by_role('button', name='Study methods', exact=True).click()
            expect(page.get_by_role('heading', name='Methods & limitations', exact=True)).to_be_visible()
            expect(page.get_by_role('heading', name='Research use only', exact=True)).to_be_visible()
            page.reload(wait_until='networkidle')
            open_study()
            expect(page.get_by_role('heading', name=f'Run {run_id[:8]}', exact=False)).to_be_visible()
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            page.get_by_role('button', name=re.compile(r'^Run history')).click()
            dialog = page.get_by_role('dialog', name='Experiment history')
            expect(dialog).to_be_visible()
            close_history = dialog.get_by_role('button', name='Close history', exact=True)
            expect(close_history).to_be_focused()
            close_history.press('Shift+Tab')
            expect(dialog.get_by_role('button').last).to_be_focused()
            dialog.get_by_role('button').last.press('Tab')
            expect(close_history).to_be_focused()
            page.keyboard.press('Escape')
            expect(dialog).not_to_be_visible()
            page.get_by_role('button', name=re.compile(r'^Run history')).click()
            expect(dialog).to_be_visible()
            checks.append('history modal traps forward/reverse focus and closes on Escape')
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
            open_study()
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Mobile page overflows horizontally'
            expect(page.get_by_role('heading', name='Cohort strategy study')).to_be_visible()
            page.screenshot(path=str(args.output_dir / 'results-mobile.png'), full_page=True)
            page.get_by_role('button', name='Patient records', exact=True).click()
            expect(page.get_by_role('heading', name='Cohort ledger', exact=True)).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Mobile patient ledger overflows horizontally'
            checks.append('390px mobile results and ledger remain usable without page overflow')

            health_url = f'{base_url}/api/health'
            page.route(health_url, lambda route: route.fulfill(status=503, json={'detail': 'Smoke test: service offline'}))
            workspace('Model documentation')
            open_study()
            expect(page.get_by_role('alert')).to_contain_text('research service is unavailable')
            expect(page.get_by_role('button', name='Run experiment', exact=True)).to_be_disabled()
            page.unroute(health_url)
            workspace('Model documentation')
            open_study()
            expect(page.get_by_text('Complete', exact=True)).to_be_visible()
            expect(page.get_by_role('button', name='Run experiment', exact=True)).to_be_enabled()
            expect(page.get_by_role('alert')).to_have_count(0)
            checks.append('offline study disables launch and workspace reconnection recovers (intercepted 503)')

            # Reopening saved results is the academic view's explicit retry path.
            result_url = f'{base_url}/api/runs/{run_id}/results'
            page.route(result_url, lambda route: route.fulfill(status=503, json={'detail': 'Smoke test: result temporarily unavailable'}))
            page.get_by_role('button', name=re.compile(r'^Run history')).click()
            history_dialog = page.get_by_role('dialog', name='Experiment history')
            history_dialog.get_by_role('button', name=re.compile(rf'^Run {run_id[:8]}')).click()
            expect(page.get_by_role('alert')).to_contain_text('Smoke test: result temporarily unavailable')
            page.unroute(result_url)
            page.get_by_role('button', name=re.compile(r'^Run history')).click()
            history_dialog.get_by_role('button', name=re.compile(rf'^Run {run_id[:8]}')).click()
            expect(page.get_by_role('heading', name='Patient explorer', exact=True)).to_be_visible()
            expect(page.get_by_role('alert')).to_have_count(0)
            checks.append('reselecting persisted result clears transient fetch error without changing the run')
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
