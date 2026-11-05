#!/usr/bin/env python3
"""Real Chromium/SOFA diagnostic smoke; start the app and optional runtime first.

Creates one genuine three-step neutral scene. --exercise-grasp also runs ten
steps with the bounded scripted-grasp/gentle-command controls. No clinical or
physical validation claim is made, regardless of subprocess completion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urljoin, urlparse

from playwright.sync_api import expect, sync_playwright

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--chromium', default=shutil.which('chromium'))
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/retraction-physics-browser'))
    parser.add_argument('--exercise-grasp', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip('/')
    expect.set_options(timeout=20_000)
    checks = []
    errors = []
    created_runs = []
    results_by_id = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1100}, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            runtime_response = context.request.get(f'{base_url}/api/physics/status')
            assert runtime_response.ok, 'Start the API with the physics router first'
            runtime = runtime_response.json()
            assert runtime['available'] is True, 'Install the optional runtime with scripts/setup-sofa.sh before this real-runtime check'
            assert runtime['model_ready'] is False
            assert runtime['purpose'] == 'upstream-physics-diagnostic'
            initial_cohorts = {r['id'] for r in context.request.get(f'{base_url}/api/runs?limit=200').json()['runs']}
            page.goto(base_url, wait_until='networkidle')
            page.get_by_role('button', name='Physics sandbox', exact=True).click()
            expect(page.get_by_role('heading', name='Physics sandbox', exact=True)).to_be_visible()
            expect(page.get_by_text('Runtime installed', exact=True)).to_be_visible()
            expect(page.get_by_text('Tissue-model validation remains incomplete.', exact=True)).to_be_visible()
            steps_input = page.get_by_label('Simulation steps', exact=True)
            expect(steps_input).to_have_attribute('min', '1')
            expect(steps_input).to_have_attribute('max', '50')
            steps_input.fill('51')
            assert steps_input.evaluate('(input) => input.validity.rangeOverflow')
            steps_input.fill('3')
            checks.append('optional runtime availability, explicit model limits, and bounded form controls')

            def run_diagnostic(*, grasp: bool, steps: int):
                page.get_by_label('Physics seed', exact=True).fill('42')
                steps_input.fill(str(steps))
                page.get_by_label('Request a scripted initial grasp', exact=False).set_checked(grasp)
                page.get_by_role('button', name='Gentle command' if grasp else 'Neutral', exact=True).click()
                with page.expect_response(lambda response: urlparse(response.url).path == '/api/physics/runs'
                                          and response.request.method == 'POST') as submission:
                    page.get_by_role('button', name='Run physics diagnostic', exact=True).click()
                response = submission.value
                assert response.status == 202, response.text()
                record = response.json()
                run_id = record['id']
                created_runs.append(run_id)
                expect(page.get_by_role('status')).to_contain_text('Running the upstream scene')
                expect(page.get_by_role('heading', name='Rendered scene', exact=True)).to_be_visible(timeout=125_000)
                expect(page.get_by_role('heading', name=f'Diagnostic {run_id[:8]}', exact=True)).to_be_visible()
                result_response = context.request.get(f'{base_url}/api/physics/runs/{run_id}/results')
                assert result_response.ok
                result = result_response.json()
                assert result['model_ready'] is False
                assert result['purpose'] == 'upstream-physics-diagnostic'
                assert result['requested_steps'] == steps
                assert result['start_grasped'] is grasp
                assert 1 <= len(result['steps']) <= steps
                assert result['topology_mode'] in ('upstream', 'derived')
                assert isinstance(result['topology'], dict) and result['topology']
                for topology in result['topology'].values():
                    assert isinstance(topology['component_state'], str)
                diagnostics = result['diagnostics']
                assert isinstance(diagnostics['error_count'], int) and diagnostics['error_count'] >= 0
                assert isinstance(diagnostics['topology_error_count'], int) and diagnostics['topology_error_count'] >= 0
                if diagnostics['error_count'] or diagnostics['topology_error_count']:
                    assert result['diagnostic_state'] == 'needs_review'
                expected_state = 'Needs review' if result['diagnostic_state'] == 'needs_review' else 'Infrastructure checked'
                expect(page.get_by_text(expected_state, exact=True)).to_be_visible()
                counts = page.locator('.physics-diagnostic-counts strong')
                expect(counts.nth(0)).to_have_text(str(diagnostics['error_count']))
                expect(counts.nth(1)).to_have_text(str(diagnostics['warning_count']))
                expect(counts.nth(2)).to_have_text(str(diagnostics['topology_error_count']))
                expect(page.get_by_text('Model ready: false · Computational diagnostic only', exact=True)).to_be_visible()
                expect(page.get_by_text('Raw diagnostic · NOT newtons', exact=True)).to_be_visible()
                expect(page.locator('.physics-step-table tbody tr')).to_have_count(len(result['steps']))
                image = page.get_by_role('img', name=re.compile(r'^Actual SOFA-rendered upstream'))
                for frame in ('initial', 'final'):
                    page.get_by_role('button', name=frame.title(), exact=True).click()
                    expect(image).to_have_attribute('src', re.compile(rf'/api/physics/runs/{run_id}/image\?frame={frame}$'))
                    expect(image).to_be_visible()
                    image.evaluate("image => image.complete && image.naturalWidth > 0 ? true : new Promise((resolve,reject) => { image.onload=()=>resolve(true); image.onerror=()=>reject(new Error('Rendered image failed to load')); })")
                    assert image.evaluate('(image) => image.naturalWidth === 256 && image.naturalHeight === 256')
                    png = context.request.get(urljoin(base_url, image.get_attribute('src')))
                    assert png.ok and png.headers['content-type'].startswith('image/png')
                    assert png.body().startswith(PNG_SIGNATURE)
                    (args.output_dir / f'{run_id}-{frame}.png').write_bytes(png.body())
                with page.expect_download() as downloaded:
                    page.locator('.physics-results').get_by_role('link', name='JSON', exact=True).click()
                json_path = args.output_dir / f'{run_id}.json'
                downloaded.value.save_as(json_path)
                assert json.loads(json_path.read_text()) == result
                results_by_id[run_id] = result
                return run_id

            neutral_id = run_diagnostic(grasp=False, steps=3)
            checks.append('real three-step neutral SOFA scene completes with explicit topology/error diagnostics')
            checks.append('actual initial/final PNGs load and telemetry/JSON stay separate from clinical outcomes')
            page.screenshot(path=str(args.output_dir / 'physics-neutral-desktop.png'), full_page=True)
            selected_id = neutral_id
            if args.exercise_grasp:
                selected_id = run_diagnostic(grasp=True, steps=10)
                assert results_by_id[selected_id]['commands']['insertion'] == 0.1
                checks.append('real ten-step scripted-grasp request and bounded actuator command produce recorded telemetry')
                page.screenshot(path=str(args.output_dir / 'physics-grasp-desktop.png'), full_page=True)

            # Refresh remounts the analytic view first. Returning to the physics
            # view must restore its separate persisted history and latest run.
            page.reload(wait_until='networkidle')
            page.get_by_role('button', name='Physics sandbox', exact=True).click()
            expect(page.get_by_role('heading', name=f'Diagnostic {selected_id[:8]}', exact=True)).to_be_visible()
            expect(page.get_by_role('heading', name='Rendered scene', exact=True)).to_be_visible()
            history = page.locator('.physics-history')
            for run_id in created_runs:
                expect(history.get_by_role('button', name=re.compile('^' + run_id[:8]))).to_be_visible()
            history.get_by_role('button', name=re.compile('^' + neutral_id[:8])).click()
            expect(page.get_by_role('heading', name=f'Diagnostic {neutral_id[:8]}', exact=True)).to_be_visible()
            expect(page.get_by_role('heading', name='Rendered scene', exact=True)).to_be_visible()
            final_cohorts = {r['id'] for r in context.request.get(f'{base_url}/api/runs?limit=200').json()['runs']}
            assert not set(created_runs) & final_cohorts
            assert initial_cohorts <= final_cohorts
            checks.append('physics history survives reload and cannot contaminate analytic cohort history')

            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), 'Physics mobile page overflows horizontally'
            expect(page.get_by_role('heading', name='Rendered scene', exact=True)).to_be_visible()
            page.screenshot(path=str(args.output_dir / 'physics-mobile.png'), full_page=True)
            checks.append('390px physics controls, renders, and telemetry do not overflow the page')

            # Explicitly mocked missing runtime: the browser must show setup
            # instructions without automatically launching or installing anything.
            missing = context.new_page()
            missing.on('pageerror', lambda error: errors.append(str(error)))
            posts = []
            missing.on('request', lambda request: posts.append(request.url)
                       if request.method == 'POST' and '/api/physics/runs' in request.url else None)
            missing.route(f'{base_url}/api/physics/status', lambda route: route.fulfill(status=200, json={
                'available': False, 'model_ready': False, 'purpose': 'upstream-physics-diagnostic',
                'reason': 'Optional runtime missing. Run scripts/setup-sofa.sh.'}))
            missing.goto(base_url, wait_until='networkidle')
            missing.get_by_role('button', name='Physics sandbox', exact=True).click()
            expect(missing.get_by_role('heading', name='The optional runtime is not installed.', exact=True)).to_be_visible()
            expect(missing.locator('.physics-unavailable code')).to_have_text('bash scripts/setup-sofa.sh')
            expect(missing.get_by_role('button', name='Run physics diagnostic', exact=True)).to_have_count(0)
            assert posts == []
            missing.get_by_role('button', name=re.compile(r'^Experiments')).click()
            expect(missing.get_by_role('heading', name='Experiment workbench', exact=True)).to_be_visible()
            missing.close()
            checks.append('missing-runtime UI explains optional setup and leaves analytic workflow available (mocked status)')
            assert not errors, f'Unhandled browser errors: {errors}'
            report = {'status': 'passed', 'created_runs': created_runs, 'checks': checks,
                      'javascript_errors': errors,
                      'diagnostics': {run_id: {'topology_mode': result['topology_mode'],
                                               'diagnostic_state': result['diagnostic_state'],
                                               'error_count': result['diagnostics']['error_count'],
                                               'final_grasp_established': result['steps'][-1]['grasp_established'],
                                               'model_ready': result['model_ready']}
                                      for run_id, result in results_by_id.items()},
                      'artifacts': str(args.output_dir)}
            (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
            return 0
        except Exception:
            page.screenshot(path=str(args.output_dir / 'failure.png'), full_page=True)
            print(json.dumps({'status': 'failed', 'checks_completed': checks, 'created_runs': created_runs,
                              'javascript_errors': errors, 'screenshot': str(args.output_dir / 'failure.png')}, indent=2), file=sys.stderr)
            raise
        finally:
            context.close()
            browser.close()


if __name__ == '__main__':
    raise SystemExit(main())
