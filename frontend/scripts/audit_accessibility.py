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
import re
import shutil
from urllib.parse import urlparse

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
        mutations = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: mutations.append(request.url)
                if request.method in ('POST', 'PUT', 'PATCH', 'DELETE')
                and not urlparse(request.url).path.endswith('/preview') else None)
        try:
            cohorts = context.request.get(f'{base_url}/api/runs')
            assert cohorts.ok, 'Start scripts/start.sh before running this audit.'
            completed = next((run for run in cohorts.json()['runs'] if run['status'] == 'completed'), None)
            assert completed, 'Complete a cohort first with scripts/browser_smoke.py. No records were changed.'
            episodes = context.request.get(f'{base_url}/api/procedure/sessions?limit=200').json()['sessions']
            assert episodes, 'Save a procedure episode first with scripts/procedure_browser_smoke.py. No records were changed.'
            episode = next((item for item in episodes if item['revision'] > 1), None)
            assert episode, 'Save a multi-transition procedure first with scripts/procedure_browser_smoke.py.'
            trainings = context.request.get(f'{base_url}/api/learning/runs').json()['runs']
            reference = context.request.get(f'{base_url}/api/learning/reference').json()
            reference_id = reference.get('run_id') if reference.get('available') else None
            trained = next((run for run in trainings if run['id'] == reference_id), None)
            trained = trained or next((run for run in trainings if run['status'] == 'completed'), None)
            assert trained, 'Complete a policy first with scripts/procedure_browser_smoke.py. No records were changed.'
            training_result = context.request.get(f"{base_url}/api/learning/runs/{trained['id']}/results").json()
            protocol_episode = next((item for item in episodes if item.get('protocol') and item['revision'] > 1), None)
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
            expect(page.get_by_role('group', name='Synthetic telemetry for displayed state')).to_be_visible()
            recorded = page.get_by_label('Recorded transition', exact=True)
            expect(recorded).to_be_enabled()
            recorded.focus()
            recorded.press('Home')
            expect(page.locator('.viewport-caption')).to_contain_text('Recording inspection')
            expect(page.get_by_role('button', name='Step', exact=True)).to_be_disabled()
            first_transition = context.request.get(
                f"{base_url}/api/procedure/sessions/{episode['id']}/transitions?limit=1").json()['transitions'][0]
            page.get_by_role('button', name='Policy learning', exact=True).click()
            expect(page.locator('.reward-total strong').first).to_be_visible()
            assert abs(float(page.locator('.reward-total strong').nth(0).inner_text()) - first_transition['reward']) <= .00051
            assert abs(float(page.locator('.reward-total strong').nth(1).inner_text()) - first_transition['episode_reward']) <= .00051
            for key, value in first_transition['reward_components'].items():
                component = page.locator('.reward-components > div').filter(
                    has=page.locator('span', has_text=re.compile('^' + re.escape(key.replace('_', ' ')) + '$')))
                assert abs(float(component.locator('strong').inner_text()) - value) <= .000051
            audit('desktop_recorded_episode_inspection')
            page.get_by_role('button', name='Return to latest', exact=True).click()
            page.get_by_role('button', name='XY nodes', exact=True).click()
            expect(page.get_by_role('img', name='Illustrative XY tissue graph showing actual simulated node positions, instrument locations, temperature and damage')).to_be_visible()
            audit('desktop_procedure_xy_state')
            page.get_by_role('button', name='Policy learning', exact=True).click()
            expect(page.get_by_role('heading', name='Train a discrete policy', exact=True)).to_be_visible()
            audit('desktop_procedure_policy_inspector')
            page.get_by_label('Policy checkpoint', exact=True).select_option(trained['id'])
            expect(page.get_by_role('img', name=re.compile('^Actual training returns over '))).to_be_visible()
            audit('desktop_completed_policy_learning_curve')
            page.get_by_role('button', name='Policy evaluation', exact=True).click()
            evaluation = page.get_by_role('region', name='Held-out policy evaluation', exact=True)
            expect(evaluation.get_by_role('rowheader', name='Learned policy', exact=True)).to_be_visible()
            if trained['id'] == reference_id:
                expect(page.locator('.reference-provenance')).to_contain_text('selected on validation')
                expect(evaluation.locator('caption')).to_contain_text('Training-disjoint validation')
                learned_row = evaluation.get_by_role('row', name=re.compile('^Learned policy '))
                expected_upper = training_result['evaluation']['learned']['severe_harm_rate_ci'][1] * 100
                expect(learned_row.locator('td').nth(3)).to_contain_text(f'{expected_upper:.1f}')
            audit('desktop_held_out_policy_evaluation')
            page.get_by_role('button', name='Events', exact=True).click()
            page.get_by_role('button', name='Tissue / tools', exact=True).click()
            audit('desktop_procedure_tissue_inspector')
            page.get_by_role('button', name='Response protocols', exact=True).click()
            expect(page.get_by_role('heading', name='Controlled response protocols', exact=True)).to_be_visible()
            expect(page.get_by_label('Response protocol', exact=True)).not_to_have_value('')
            audit('desktop_response_protocol_catalog')
            page.get_by_label('Response protocol', exact=True).select_option('arterial-thermal-contact')
            page.locator('.protocol-parameters summary').click()
            expect(page.locator('.protocol-parameters')).to_contain_text('not clinical')
            audit('desktop_response_protocol_parameters')
            if protocol_episode:
                page.get_by_label('Saved episode', exact=True).select_option(protocol_episode['id'])
                expect(page.locator('.protocol-state')).to_be_visible()
                audit('desktop_saved_protocol_response')
                protocol_first = context.request.get(
                    f"{base_url}/api/procedure/sessions/{protocol_episode['id']}/transitions?limit=1").json()['transitions'][0]
                page.get_by_label('Recorded transition', exact=True).focus()
                page.get_by_label('Recorded transition', exact=True).press('Home')
                expect(page.locator('.protocol-state dd').nth(2)).to_have_text(protocol_first['source']['phase'])
                audit('desktop_saved_protocol_recorded_phase')
                page.get_by_role('button', name='Return to latest', exact=True).click()
                page.get_by_label('Saved episode', exact=True).select_option(episode['id'])
            page.get_by_role('button', name='Tissue / tools', exact=True).click()
            archived = next((item for item in episodes if item.get('engine_compatible') is False), None)
            if archived:
                page.get_by_label('Saved episode', exact=True).select_option(archived['id'])
                expect(page.locator('.archived-episode-notice')).to_be_visible()
                expect(page.get_by_role('button', name='Step', exact=True)).to_be_disabled()
                expect(page.get_by_role('button', name='Run policy', exact=True)).to_be_disabled()
                expect(page.get_by_role('button', name='New episode', exact=True)).to_be_enabled()
                expect(page.get_by_role('button', name='Reset', exact=True)).to_be_enabled()
                expect(page.get_by_role('link', name='Episode JSON', exact=True)).to_be_visible()
                audit('desktop_archived_episode_read_only')
                page.get_by_label('Saved episode', exact=True).select_option(episode['id'])

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
            softening = page.get_by_role('checkbox', name=re.compile('^Experimental material softening'))
            if softening.count() and softening.is_enabled():
                softening.check()
                expect(page.get_by_label('Experimental contact heat', exact=False)).to_be_visible()
                audit('desktop_physics_softening_controls')
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
            recorded = page.get_by_label('Recorded transition', exact=True)
            expect(recorded).to_be_enabled()
            recorded.focus()
            recorded.press('Home')
            expect(page.locator('.viewport-caption')).to_contain_text('Recording inspection')
            audit('mobile_recorded_episode_inspection')
            page.get_by_role('button', name='Return to latest', exact=True).click()
            page.get_by_role('button', name='Policy learning', exact=True).click()
            page.get_by_label('Policy checkpoint', exact=True).select_option(trained['id'])
            page.get_by_role('button', name='Policy evaluation', exact=True).click()
            expect(page.get_by_role('region', name='Held-out policy evaluation').get_by_role('rowheader', name='Learned policy', exact=True)).to_be_visible()
            audit('mobile_held_out_policy_evaluation')
            page.screenshot(path=str(args.output.with_name('policy-evaluation-mobile.png')), full_page=True)
            page.get_by_role('button', name='Response protocols', exact=True).click()
            if protocol_episode:
                page.get_by_label('Saved episode', exact=True).select_option(protocol_episode['id'])
                expect(page.locator('.protocol-state')).to_be_visible()
            expect(page.get_by_role('heading', name='Controlled response protocols', exact=True)).to_be_visible()
            audit('mobile_response_protocol_inspector')
            page.screenshot(path=str(args.output.with_name('response-protocol-mobile.png')), full_page=True)
            page.set_viewport_size({'width': 1440, 'height': 1050})
            workspace('Cohort experiments')
            expect(page.locator('.metrics-grid')).to_be_visible()
            page.screenshot(path=str(args.output.with_name('cohort-desktop.png')), full_page=True)
            report['javascript_errors'] = errors
            report['unexpected_record_mutations'] = mutations
            report['saved_record_ids'] = {'procedure': episode['id'], 'cohort': completed['id'], 'training': trained['id']}
        except Exception:
            page.screenshot(path=str(args.output.with_name('audit-failure.png')), full_page=True)
            raise
        finally:
            browser.close()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    failed = bool(report.get('javascript_errors')) or bool(report.get('unexpected_record_mutations')) or any(
        state['violations'] for state in report['states'].values())
    print(f'Report: {args.output}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
