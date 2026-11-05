#!/usr/bin/env python3
"""Read-only axe audit of the academic workbench's real saved states.

Run procedure_browser_smoke.py and browser_smoke.py first so saved episodes
and a completed cohort exist. This audit never advances a procedure, trains a
policy, installs a runtime, or changes saved records. Automated WCAG2/2.1 A/AA
checks supplement manual keyboard, canvas, and scientific-interpretation review.
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
    base_url = args.base_url.rstrip('/')
    root = Path(__file__).resolve().parents[1]
    source = root / 'node_modules/axe-core/axe.min.js'
    if not source.is_file():
        parser.error('Run npm ci in frontend to install the pinned local axe-core dependency.')
    report = {'axe_version': '4.10.3', 'states': {},
              'limitations': 'Automated WCAG2/2.1 A/AA checks only. Canvas content, keyboard operation, and incomplete findings require manual review.',
              'mutations': 'None: existing saved episodes, cohorts, and physics diagnostics are inspected read-only.'}
    expect.set_options(timeout=20_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1050})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            cohorts = context.request.get(f'{base_url}/api/runs')
            assert cohorts.ok, 'Start scripts/start.sh before running this audit.'
            completed = next((run for run in cohorts.json()['runs'] if run['status'] == 'completed'), None)
            assert completed, 'Complete a cohort first with scripts/browser_smoke.py. No records were changed.'
            episodes = context.request.get(f'{base_url}/api/procedure/sessions').json()['sessions']
            assert episodes, 'Save a procedure episode first with scripts/procedure_browser_smoke.py. No records were changed.'
            episode = episodes[0]
            page.goto(base_url, wait_until='networkidle')
            page.add_script_tag(path=str(source))

            def workspace(name):
                page.get_by_role('navigation', name='Research workspaces').get_by_role('button', name=name, exact=True).click()

            def audit(name):
                # Two frames settle React's layout and locally generated SVGs;
                # axe reports canvas/contrast uncertainty as manual review.
                page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                data = page.evaluate("""async () => await axe.run(document, {
                  runOnly: {type:'tag', values:['wcag2a','wcag2aa','wcag21a','wcag21aa']}
                })""")
                report['states'][name] = {
                    'violations': data['violations'], 'incomplete': data['incomplete'],
                    'passed_rules': len(data['passes']),
                }
                print(f'{name}: {len(data["violations"])} violations; '
                      f'{len(data["incomplete"])} incomplete/manual-review rules', flush=True)
                for violation in data['violations']:
                    print(f'  {violation["id"]}: {len(violation["nodes"])} affected nodes', flush=True)

            expect(page.get_by_role('heading', name='No active episode', exact=True)).to_be_visible()
            audit('desktop_procedure_uninitialized')
            page.get_by_label('Saved episode', exact=True).select_option(episode['id'])
            expect(page.locator('.viewport-caption')).to_contain_text('revision')
            page.wait_for_function("""() => document.querySelector('.three-viewport canvas') ||
              document.querySelector('.viewport-body svg[role="img"]') ||
              document.querySelector('.viewport-body canvas')""")
            audit('desktop_procedure_spatial_state')
            page.get_by_role('button', name='XY nodes', exact=True).click()
            expect(page.get_by_role('img', name='Illustrative XY tissue graph showing actual simulated node positions, instrument locations, temperature and damage')).to_be_visible()
            audit('desktop_procedure_xy_state')
            page.get_by_role('button', name='Policy learning', exact=True).click()
            expect(page.get_by_role('heading', name='Train a discrete policy', exact=True)).to_be_visible()
            audit('desktop_procedure_policy_inspector')
            page.get_by_role('button', name='Tissue / tools', exact=True).click()
            audit('desktop_procedure_tissue_inspector')

            workspace('Cohort experiments')
            expect(page.get_by_role('heading', name='Cohort strategy study', exact=True)).to_be_visible()
            if not page.locator('.metrics-grid').count():
                page.get_by_role('button', name='Run history', exact=False).click()
                page.locator('.history-list > button').filter(has_text=completed['id'][:8]).click()
            expect(page.locator('.metrics-grid')).to_be_visible()
            audit('desktop_cohort_overview')
            page.get_by_role('button', name='Patient parameter ranges', exact=True).click()
            audit('desktop_cohort_advanced_setup')
            page.get_by_role('button', name='Patient parameter ranges', exact=True).click()
            page.get_by_role('button', name='Try a custom retraction', exact=False).click()
            expect(page.locator('.strategy-scatter circle').first).to_be_visible()
            audit('desktop_custom_strategy')
            page.get_by_role('button', name='Patient records', exact=True).click()
            audit('desktop_patient_ledger')
            page.get_by_role('button', name='Study methods', exact=True).click()
            audit('desktop_study_methods')
            workspace('Model documentation')
            expect(page.get_by_role('heading', name='Stateful model, methods, and provenance', exact=True)).to_be_visible()
            audit('desktop_model_documentation')

            workspace('SOFA diagnostics')
            page.wait_for_load_state('networkidle')
            page.wait_for_function("""() => document.querySelector('.physics-unavailable') ||
              document.querySelector('.physics-frame img') ||
              document.querySelector('.physics-results .empty-state')""")
            audit('desktop_physics')
            details = page.locator('.physics-messages')
            if details.count():
                details.locator('summary').click()
                audit('desktop_physics_messages')

            workspace('Cohort experiments')
            expect(page.locator('.metrics-grid')).to_be_visible()
            page.get_by_role('button', name='Run history', exact=False).click()
            audit('desktop_history_modal')
            page.keyboard.press('Escape')
            page.set_viewport_size({'width': 390, 'height': 844})
            audit('mobile_cohort_overview')
            workspace('SOFA diagnostics')
            page.wait_for_load_state('networkidle')
            page.wait_for_function("""() => document.querySelector('.physics-unavailable') ||
              document.querySelector('.physics-frame img') ||
              document.querySelector('.physics-results .empty-state')""")
            audit('mobile_physics')
            page.screenshot(path=str(args.output.with_name('physics-mobile.png')), full_page=True)
            workspace('Procedure laboratory')
            page.get_by_label('Saved episode', exact=True).select_option(episode['id'])
            expect(page.locator('.viewport-caption')).to_contain_text('revision')
            page.get_by_role('button', name='XY nodes', exact=True).click()
            audit('mobile_procedure_xy')
            page.screenshot(path=str(args.output.with_name('procedure-mobile.png')), full_page=True)
            page.set_viewport_size({'width': 1440, 'height': 1050})
            workspace('Cohort experiments')
            expect(page.locator('.metrics-grid')).to_be_visible()
            page.screenshot(path=str(args.output.with_name('cohort-desktop.png')), full_page=True)
            report['javascript_errors'] = errors
            report['saved_record_ids'] = {'procedure': episode['id'], 'cohort': completed['id']}
        except Exception:
            page.screenshot(path=str(args.output.with_name('audit-failure.png')), full_page=True)
            raise
        finally:
            browser.close()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    failed = bool(report.get('javascript_errors')) or any(
        state['violations'] for state in report['states'].values())
    print(f'Report: {args.output}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
