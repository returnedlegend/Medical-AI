"""Independent checks of study invariants, reconstructed from exported records."""
from __future__ import annotations

import numpy as np
import pytest

from retraction import simulation
from retraction.analysis import _fit_evaluate


@pytest.fixture(scope='module')
def result():
    return simulation.run_experiment({'n_patients': 100, 'seed': 314})


def test_shared_strategy_is_selected_using_training_patients_only(result):
    train = [p for p in result['patients'] if p['split'] == 'train']
    successes = [sum(simulation.simulate_patient(p, strategy, result['thresholds'])['success']
                     for p in train) for strategy in result['strategies']]
    expected_index = int(np.argmax(successes))
    assert result['summary']['best_shared_strategy'] == result['strategies'][expected_index]


def test_exported_test_records_reconstruct_summary_and_all_categories(result):
    test = [p for p in result['patients'] if p['split'] == 'test']
    summary = result['summary']
    assert len(test) == summary['test_size']
    assert sum(summary['categories'].values()) == len(test)
    shared_count = sum(p['shared']['success'] for p in test)
    oracle_count = sum(p['oracle']['success'] for p in test)
    adapted = sum(not p['shared']['success'] and p['oracle']['success'] for p in test)
    assert summary['shared_success_rate'] == pytest.approx(shared_count / len(test))
    assert summary['oracle_success_rate'] == pytest.approx(oracle_count / len(test))
    assert summary['personalization_gain'] == pytest.approx(adapted / len(test))
    assert summary['personalization_gain'] == pytest.approx(
        summary['oracle_success_rate'] - summary['shared_success_rate'])
    assert all(p['category'] == ('bail_out' if not p['oracle']['success'] else
                                 'personalize' if not p['shared']['success'] else 'standard')
               for p in test)


def test_changing_held_out_patients_cannot_change_selected_shared_strategy(monkeypatch):
    config = {'n_patients': 60, 'seed': 777}
    baseline = simulation.run_experiment(config)
    test_ids = {p['id'] for p in baseline['patients'] if p['split'] == 'test'}
    original = simulation.generate_cohort

    def alternative_cohort(cfg):
        cohort = original(cfg)
        for p in cohort:
            if p['id'] in test_ids:
                p.update(attachment_length_mm=4.0, adhesion_count=12,
                         adhesion_stiffness_n_per_mm=0.5, distension=2.5,
                         preferred_direction_deg=90.0)
        return cohort

    monkeypatch.setattr(simulation, 'generate_cohort', alternative_cohort)
    changed = simulation.run_experiment(config)
    assert changed['summary']['best_shared_strategy'] == baseline['summary']['best_shared_strategy']
    for previous, current in zip(baseline['patients'], changed['patients']):
        if previous['split'] == 'train':
            assert previous == current


def test_every_success_obeys_all_three_constraints(result):
    for patient in result['patients']:
        for kind in ('standard', 'shared', 'oracle'):
            outcome = patient[kind]
            recomputed = simulation.simulate_patient(patient, outcome['strategy'], result['thresholds'])
            assert outcome == recomputed
            assert outcome['success'] == (outcome['force_ok'] and outcome['strain_ok'] and outcome['exposure_ok'])
    # The candidate set is held fixed across sensitivity analyses, so relaxed
    # constraints cannot reduce the patient-specific feasible success rate.
    rates = [profile['oracle_success_rate'] for profile in result['sensitivity']]
    assert rates == sorted(rates)


def test_probe_scaling_and_coefficients_ignore_heldout_features_and_labels():
    features = np.array([[0., 1., 2., 3.], [1., 2., 4., 5.],
                         [2., 1., 3., 7.], [3., 5., 1., 2.],
                         [4., 2., 7., 4.], [5., 1., 5., 6.]])
    labels = np.array([False, True, False, True, False, True])
    train, test = np.arange(4), np.arange(4, 6)
    baseline, _ = _fit_evaluate(features, labels, train, test, 42)
    changed_features = features.copy()
    changed_features[test] *= 1000
    changed_labels = labels.copy()
    changed_labels[test] = ~changed_labels[test]
    changed, _ = _fit_evaluate(changed_features, changed_labels, train, test, 42)
    assert baseline['coefficients'] == changed['coefficients']


def test_single_class_training_never_reports_invented_discrimination():
    metrics, predictions = _fit_evaluate(np.ones((6, 4)), np.zeros(6, dtype=bool),
                                         np.arange(4), np.arange(4, 6), 1)
    assert predictions is None
    assert metrics['auroc'] is None
    assert metrics['brier_score'] is None
    assert metrics['status'] == 'unavailable'
