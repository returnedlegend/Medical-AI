"""Scientific invariants and leakage checks for the research surrogate."""
import json
import unittest
import numpy as np

from retraction.simulation import (
    DEFAULT_CONFIG, STANDARD_STRATEGY, THRESHOLD_PROFILES,
    _mechanics, evaluate_grid, force_probe, generate_cohort,
    normalize_config, run_experiment, simulate_patient, strategy_grid,
)
from retraction.analysis import _fit_evaluate, analyze_subgroups, paired_bootstrap_ci


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        self.patient = {"attachment_length_mm": 20.0, "adhesion_count": 0,
                        "adhesion_stiffness_n_per_mm": 0.1, "distension": 1.0,
                        "preferred_direction_deg": 35.0}

    def test_zero_displacement_gives_zero_force_strain_and_exposure(self):
        outcome = simulate_patient(self.patient, {**STANDARD_STRATEGY, "distance_mm": 0})
        self.assertEqual((outcome["force_n"], outcome["strain"], outcome["exposure"]), (0.0, 0.0, 0.0))
        self.assertFalse(outcome["success"])

    def test_small_displacement_matches_linear_spring_limit(self):
        d = 1e-5
        force, strain, _ = _mechanics(self.patient, 0.75, 35, d)
        expected_extension = d * (0.42 + 0.58 * 0.75)
        expected_force = 1.6 / 20 * expected_extension
        self.assertAlmostEqual(force, expected_force, places=15)
        self.assertAlmostEqual(strain, expected_extension / 20)

    def test_force_and_strain_are_monotone_with_distance(self):
        distance = np.linspace(0, 30, 31)
        force, strain, exposure = _mechanics(self.patient, 0.75, 35, distance)
        self.assertTrue(np.all(np.diff(force) >= 0))
        self.assertTrue(np.all(np.diff(strain) >= 0))
        self.assertTrue(np.all((exposure >= 0) & (exposure <= 1)))

    def test_alignment_maximizes_exposure(self):
        _, _, aligned = _mechanics(self.patient, 0.75, 35, 12)
        _, _, misaligned = _mechanics(self.patient, 0.75, 75, 12)
        self.assertGreater(aligned, misaligned)

    def test_success_requires_all_three_constraints(self):
        permissive = {"force_limit_n": 20, "strain_limit": 3, "exposure_threshold": 0.01}
        base = simulate_patient(self.patient, STANDARD_STRATEGY, permissive)
        self.assertTrue(base["success"])
        for key, value in [("force_limit_n", base["force_n"] / 2),
                           ("strain_limit", base["strain"] / 2),
                           ("exposure_threshold", 1.0)]:
            self.assertFalse(simulate_patient(self.patient, STANDARD_STRATEGY, {**permissive, key: value})["success"])

    def test_scalar_and_vector_mechanics_agree(self):
        patients = generate_cohort({"n_patients": 40})
        grid = strategy_grid()
        ok, forces, strains, exposures = evaluate_grid(patients, grid, THRESHOLD_PROFILES["nominal"])
        for i, j in [(0, 0), (4, 31), (39, 74)]:
            scalar = simulate_patient(patients[i], grid[j])
            self.assertEqual(scalar["success"], bool(ok[i, j]))
            self.assertAlmostEqual(scalar["force_n"], forces[i, j])
            self.assertAlmostEqual(scalar["strain"], strains[i, j])
            self.assertAlmostEqual(scalar["exposure"], exposures[i, j])

    def test_probe_is_fixed_one_second_two_mm_ramp(self):
        probe = force_probe(self.patient)
        self.assertEqual(probe["times_s"][0], 0)
        self.assertEqual(probe["times_s"][-1], 1)
        self.assertEqual(probe["forces_n"][0], 0)
        self.assertGreater(probe["peak_force_n"], 0)
        endpoint = simulate_patient(self.patient, {**STANDARD_STRATEGY, "distance_mm": 2})
        self.assertAlmostEqual(probe["peak_force_n"], endpoint["force_n"])

    def test_probe_work_matches_analytic_spring_integral(self):
        # For an unadhered patient k=EA/L, e=beta*d. Integrate F(d)
        # analytically and compare to the sampled one-second probe work.
        probe = force_probe(self.patient)
        beta = 0.42 + 0.58 * 0.75
        k, length, d = 1.6 / 20, 20, 2
        exact_work = k * beta * d**2 / 2 + k * 1.3 * beta**3 * d**4 / (4 * length**2)
        self.assertAlmostEqual(probe["work_mj"], exact_work, delta=exact_work * 0.0001)

    def test_probe_exceedances_are_reported(self):
        severe = {**self.patient, "attachment_length_mm": 4.0, "adhesion_count": 12,
                  "adhesion_stiffness_n_per_mm": 0.5, "distension": 2.5}
        probe = force_probe(severe, {"force_limit_n": 0.01, "strain_limit": 0.01})
        self.assertFalse(probe["within_limits"])
        self.assertFalse(probe["force_ok"])
        self.assertFalse(probe["strain_ok"])
        self.assertEqual(probe["duration_s"], 1.0)  # no undocumented force stop

    def test_invalid_inputs_rejected(self):
        for config in [{"n_patients": 10}, {"n_patients": 40.5}, {"seed": -1},
                       {"threshold_profile": "made-up"}, {"force_limit_n": float("nan")},
                       {"attachment_min_mm": 40, "attachment_max_mm": 20},
                       {"attachment_min_mm": True}, {"force_limit_n": True},
                       {"distension_min": "0.8"}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_config(config)
        with self.assertRaises(ValueError):
            simulate_patient({**self.patient, "attachment_length_mm": -1}, STANDARD_STRATEGY)
        with self.assertRaises(ValueError):
            simulate_patient(self.patient, {**STANDARD_STRATEGY, "distance_mm": float("nan")})
        with self.assertRaises(ValueError):
            simulate_patient({**self.patient, "preferred_direction_deg": float("nan")}, STANDARD_STRATEGY)
        with self.assertRaises(ValueError):
            simulate_patient(self.patient, {**STANDARD_STRATEGY, "grasp_position": True})


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_experiment({"n_patients": 120, "seed": 42})

    def test_reproducible_and_json_serializable(self):
        second = run_experiment({"n_patients": 120, "seed": 42})
        self.assertEqual(self.result, second)
        json.dumps(self.result, allow_nan=False)
        self.assertNotEqual(generate_cohort({"seed": 1}), generate_cohort({"seed": 2}))

    def test_grid_oracle_contains_shared_and_partition_is_complete(self):
        r = self.result
        test = [p for p in r["patients"] if p["split"] == "test"]
        self.assertEqual(len(test), r["summary"]["test_size"])
        self.assertEqual(len(r["strategies"]), 75)
        for p in r["patients"]:
            if p["shared"]["success"]:
                self.assertTrue(p["oracle"]["success"])
            if p["oracle"]["success"]:
                self.assertTrue(p["oracle"]["force_ok"] and p["oracle"]["strain_ok"] and p["oracle"]["exposure_ok"])
        s = r["summary"]
        self.assertAlmostEqual(s["personalization_gain"], s["oracle_success_rate"] - s["shared_success_rate"])
        self.assertEqual(sum(s["categories"].values()), s["test_size"])
        self.assertAlmostEqual(s["bailout_rate"], 1 - s["oracle_success_rate"])

    def test_best_shared_selected_on_training_only(self):
        r = self.result
        train = [p for p in r["patients"] if p["split"] == "train"]
        ok, *_ = evaluate_grid(train, r["strategies"], r["thresholds"])
        expected = r["strategies"][int(np.argmax(ok.mean(axis=0)))]
        self.assertEqual(expected, r["summary"]["best_shared_strategy"])

    def test_successful_oracle_minimizes_force_within_grid(self):
        r = self.result
        for patient in r["patients"][:20]:
            if patient["oracle"]["success"]:
                ok, forces, _, _ = evaluate_grid([patient], r["strategies"], r["thresholds"])
                self.assertAlmostEqual(patient["oracle"]["force_n"], forces[0][ok[0]].min())

    def test_sensitivity_oracle_success_monotone_with_relaxed_limits(self):
        values = [x["oracle_success_rate"] for x in self.result["sensitivity"]]
        self.assertEqual(values, sorted(values))

    def test_subgroups_reconstruct_test_outcomes_and_partition_dimensions(self):
        r = self.result
        test = [p for p in r["patients"] if p["split"] == "test"]
        predicates = {
            ("attachment_length", "short"): lambda p: p["attachment_length_mm"] < 15,
            ("attachment_length", "medium"): lambda p: 15 <= p["attachment_length_mm"] < 25,
            ("attachment_length", "long"): lambda p: p["attachment_length_mm"] >= 25,
            ("adhesions", "none"): lambda p: p["adhesion_count"] == 0,
            ("adhesions", "mild"): lambda p: 1 <= p["adhesion_count"] <= 3,
            ("adhesions", "high"): lambda p: p["adhesion_count"] >= 4,
            ("distension", "lower"): lambda p: p["distension"] < 1.2,
            ("distension", "higher"): lambda p: p["distension"] >= 1.2,
        }
        for row in r["subgroups"]:
            selected = [p for p in test if predicates[(row["dimension"], row["group"])](p)]
            self.assertEqual(row["n"], len(selected))
            if selected:
                shared = np.mean([p["shared"]["success"] for p in selected])
                oracle = np.mean([p["oracle"]["success"] for p in selected])
                self.assertAlmostEqual(row["shared_success_rate"], shared)
                self.assertAlmostEqual(row["oracle_success_rate"], oracle)
                self.assertAlmostEqual(row["personalization_gain"], oracle - shared)
                self.assertAlmostEqual(row["bailout_rate"], 1 - oracle)
                self.assertTrue(0 <= row["gain_ci"][0] <= row["gain_ci"][1] <= 1)
        for dimension in {row["dimension"] for row in r["subgroups"]}:
            rows = [row for row in r["subgroups"] if row["dimension"] == dimension]
            self.assertEqual(sum(row["n"] for row in rows), len(test))
            self.assertAlmostEqual(sum(row["n"] * row["personalization_gain"] for row in rows if row["n"]) / len(test),
                                   r["summary"]["personalization_gain"])

    def test_metrics_cover_only_heldout_cases(self):
        probe = self.result["probe"]
        test = [p for p in self.result["patients"] if p["split"] == "test"]
        self.assertEqual(probe["test_count"], len(test))
        self.assertEqual(probe["test_positive_count"], sum(p["category"] == "personalize" for p in test))
        if probe["status"] == "ok":
            self.assertEqual(sum(b["count"] for b in probe["calibration"]), len(test))
            expected_brier = np.mean([(p["personalization_probability"] - int(p["category"] == "personalize"))**2 for p in test])
            self.assertAlmostEqual(probe["brier_score"], expected_brier)


class AnalysisTests(unittest.TestCase):
    def test_subgroup_boundaries_empty_groups_and_training_exclusion(self):
        base = {"split": "test", "attachment_length_mm": 15.0, "adhesion_count": 3,
                "distension": 1.2, "shared": {"success": False}, "oracle": {"success": True}}
        patients = [base, {**base, "attachment_length_mm": 25.0, "adhesion_count": 4},
                    {**base, "split": "train", "attachment_length_mm": 8.0,
                     "adhesion_count": 0, "distension": 0.8}]
        rows = analyze_subgroups(patients, 42)
        indexed = {(row["dimension"], row["group"]): row for row in rows}
        self.assertEqual(indexed[("attachment_length", "medium")]["n"], 1)
        self.assertEqual(indexed[("attachment_length", "long")]["n"], 1)
        self.assertEqual(indexed[("adhesions", "mild")]["n"], 1)
        self.assertEqual(indexed[("adhesions", "high")]["n"], 1)
        self.assertEqual(indexed[("distension", "higher")]["n"], 2)
        for key in [("attachment_length", "short"), ("adhesions", "none"), ("distension", "lower")]:
            row = indexed[key]
            self.assertEqual(row["n"], 0)
            for metric in ["shared_success_rate", "oracle_success_rate", "personalization_gain", "bailout_rate", "gain_ci"]:
                self.assertIsNone(row[metric])
        for row in rows:
            if row["n"]:
                self.assertEqual(row["gain_ci"], [1.0, 1.0])
        self.assertEqual(rows, analyze_subgroups(patients, 42))
        json.dumps(rows, allow_nan=False)
        empty = analyze_subgroups([], 42)
        self.assertEqual(len(empty), 8)
        self.assertTrue(all(row["n"] == 0 and row["gain_ci"] is None for row in empty))

    def test_test_labels_do_not_affect_model_fitting_or_predictions(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(100, 4))
        labels = features[:, 0] > 0
        train, test = np.arange(70), np.arange(70, 100)
        _, first = _fit_evaluate(features, labels, train, test, 1)
        changed = labels.copy()
        changed[test] = ~changed[test]
        _, second = _fit_evaluate(features, changed, train, test, 1)
        np.testing.assert_array_equal(first, second)

    def test_single_class_train_has_no_invented_probabilities(self):
        metrics, predictions = _fit_evaluate(np.zeros((40, 4)), np.zeros(40), np.arange(28), np.arange(28, 40), 1)
        self.assertIsNone(predictions)
        self.assertIsNone(metrics["auroc"])
        self.assertEqual(metrics["status"], "unavailable")

    def test_single_class_test_has_defined_brier_but_no_auroc(self):
        features = np.arange(160).reshape(40, 4)
        labels = np.array([i % 2 if i < 28 else 0 for i in range(40)])
        metrics, _ = _fit_evaluate(features, labels, np.arange(28), np.arange(28, 40), 1)
        self.assertIsNone(metrics["auroc"])
        self.assertIsNotNone(metrics["brier_score"])
        self.assertEqual(metrics["status"], "single_class_test")

    def test_bootstrap_degenerate_samples(self):
        self.assertEqual(paired_bootstrap_ci(np.zeros(20), 1), [0.0, 0.0])
        self.assertEqual(paired_bootstrap_ci(np.ones(20), 1), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
