from retraction.report import render_report
from retraction.simulation import run_experiment


def test_report_matches_heldout_results_and_discloses_model_limits():
    result = run_experiment({'n_patients': 40, 'seed': 42})
    report = render_report(result, 'example-id')
    assert 'Run: `example-id`' in report
    assert f'{100 * result["summary"]["personalization_gain"]:.1f} percentage points' in report
    assert 'Held out: 12' in report
    assert 'partly imposed' in report
    assert 'not externally preregistered' in report
    assert result['subgroups'][0]['label'] in report
    assert 'analytic-surrogate-1.0' in report


def test_single_class_predictor_report_shows_undefined():
    result = run_experiment({'n_patients': 40, 'force_limit_n': .001})
    assert result['probe']['auroc'] is None
    assert 'Held-out personalization AUROC: undefined' in render_report(result)
