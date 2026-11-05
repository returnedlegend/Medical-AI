"""Lifecycle, reward, injury and deterministic replay checks for the RL loop."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from retraction import environment
from retraction.environment import ACTION_NAMES, OBSERVATION_NAMES, TissueRetractionEnv, normalize_env_config


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract_and_complete_names(self):
        env = TissueRetractionEnv()
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), len(OBSERVATION_NAMES))
        self.assertEqual(len(ACTION_NAMES), 7)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.describe()["model_version"], info["environment_version"])
        json.dumps(info, allow_nan=False)

    def test_physics_fingerprint_matches_labeled_source_file_bytes(self):
        source = Path(environment.__file__).resolve().parent
        digest = hashlib.sha256(b"retraction-physics-source-v1\0")
        for name in ("environment.py", "tissue.py"):
            digest.update(f"retraction/{name}\0".encode("utf-8"))
            digest.update((source / name).read_bytes())
            digest.update(b"\0")
        self.assertEqual(environment.PHYSICS_FINGERPRINT, digest.hexdigest())
        self.assertEqual(environment.ENV_VERSION, "tissue-retraction-1.1")

    def test_running_instance_fingerprint_is_stable_without_file_reloads(self):
        env = TissueRetractionEnv()
        fingerprint = env.describe()["physics_fingerprint"]
        with patch("pathlib.Path.read_bytes", side_effect=AssertionError("Unexpected source rehash")):
            _, initial = env.reset(seed=42)
            _, _, _, _, stepped = env.step([0, 0, 0.55, 0, 0, 0, 0])
            self.assertEqual(env.describe()["physics_fingerprint"], fingerprint)
            self.assertEqual(initial["physics_fingerprint"], fingerprint)
            self.assertEqual(stepped["physics_fingerprint"], fingerprint)

    def test_missing_fingerprint_source_is_explicit_setup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Physics source fingerprint unavailable"):
                environment._compute_physics_fingerprint(Path(directory))

    def test_seeded_reset_and_action_replay_are_exact(self):
        first, second = TissueRetractionEnv(), TissueRetractionEnv()
        initial_a, info_a = first.reset(seed=42)
        initial_b, info_b = second.reset(seed=42)
        np.testing.assert_array_equal(initial_a, initial_b)
        self.assertEqual(info_a, info_b)
        actions = [[0, 0, 0.55, 0, 0, 0, 0], *[[0.35, 0, 0, 0, 0, 0, 0]] * 20]
        for action in actions:
            a, r1, t1, tr1, i1 = first.step(action)
            b, r2, t2, tr2, i2 = second.step(action)
            np.testing.assert_array_equal(a, b)
            self.assertEqual((r1, t1, tr1, i1), (r2, t2, tr2, i2))

    def test_explicit_patient_options_replay_and_reset_clears_injury(self):
        env = TissueRetractionEnv()
        patient = {"attachment_length_mm": 24, "adhesion_count": 2,
                   "adhesion_stiffness_n_per_mm": 0.08, "distension": 1.2}
        initial, initial_info = env.reset(seed=1, options={"patient": patient})
        env.step([0, 0, 1, 0, 0, 0, 0])
        for _ in range(5):
            env.step([0, 0, 0, 0, 0, 0, 0])
        self.assertGreater(env.tissue.damage.max(), 0)
        restored, info = env.reset(seed=99, options={"patient": patient})
        np.testing.assert_array_equal(initial, restored)
        self.assertEqual(initial_info, info)

    def test_grip_holds_on_zero_and_release_requires_negative_command(self):
        env = TissueRetractionEnv()
        env.reset(seed=42)
        env.step([0, 0, 0.55, 0, 0, 0, 0])
        self.assertIsNotNone(env.tissue.contact_index)
        env.step([0.2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(env.grip, 0.55)
        self.assertIsNotNone(env.tissue.contact_index)
        env.step([0, 0, -1, 0, 0, 0, 0])
        self.assertEqual(env.grip, 0)
        self.assertIsNone(env.tissue.contact_index)

    def test_handoff_is_terminal_and_does_not_award_success(self):
        env = TissueRetractionEnv()
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertFalse(info["severe_harm"])
        self.assertEqual(info["termination_reason"], "handoff")
        self.assertLess(reward, 0)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))

    def test_time_limit_truncates_and_rejects_more_steps(self):
        env = TissueRetractionEnv({"max_steps": 20})
        env.reset(seed=42)
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(np.zeros(7))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertFalse(info["success"])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))

    def test_known_baseline_achieves_success_once_and_reward_is_decomposed(self):
        env = TissueRetractionEnv()
        obs, _ = env.reset(seed=2)
        rewards = []
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(env.scripted_action(obs))
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            rewards.append(reward)
            if terminated or truncated:
                break
        self.assertTrue(info["success"])
        self.assertLess(info["state"]["max_damage"], 0.15)
        self.assertGreaterEqual(info["state"]["stable_exposure_steps"], env.config["success_hold_steps"])
        self.assertAlmostEqual(sum(rewards), info["cumulative_reward"])
        self.assertEqual(info["reward_components"]["terminal"], 15)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))

    def test_scripted_controller_uses_supplied_observation_and_relaxes_near_limits(self):
        env = TissueRetractionEnv()
        obs, _ = env.reset(seed=2)
        obs, *_ = env.step([0, 0, 0.55, 0, 0, 0, 0])
        other = TissueRetractionEnv()
        other.reset(seed=99)
        # Different environment internals cannot change a public-observation
        # baseline's action when it receives the same observation.
        np.testing.assert_array_equal(env.scripted_action(obs), other.scripted_action(obs))
        self.assertGreater(env.scripted_action(obs)[0], 0)
        strained = obs.copy()
        strained[OBSERVATION_NAMES.index("n03_strain")] = env.config["strain_limit"]
        self.assertLess(env.scripted_action(strained)[0], 0)
        force_limited = obs.copy()
        force_limited[OBSERVATION_NAMES.index("contact_force_n")] = env.config["force_limit_n"]
        self.assertLess(env.scripted_action(force_limited)[0], 0)
        self.assertEqual(env.describe()["scripted_controller_version"], "feedback-strain-guard-1.1")

    def test_exposure_reward_cannot_be_farmed_by_repeating_motion(self):
        env = TissueRetractionEnv()
        env.reset(seed=2)
        progress = 0
        env.step([0, 0, 0.55, 0, 0, 0, 0])
        for _ in range(3):
            for direction in [0.25] * 8 + [-0.25] * 8:
                _, _, term, trunc, info = env.step([direction, 0, 0, 0, 0, 0, 0])
                progress += info["reward_components"]["exposure_progress"]
                self.assertFalse(term or trunc)
        self.assertLessEqual(progress, 4 * info["state"]["peak_exposure"] + 1e-9)

    def test_cautery_harm_produces_physiology_penalty_and_terminal_failure(self):
        env = TissueRetractionEnv()
        env.reset(seed=42)
        # Move the separate cautery tool to the artery, without changing grasper.
        for _ in range(12):
            env.step([0, 0, 0, 1, 1 / 3, 0, 0])
        observed_events = []
        for _ in range(30):
            _, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 1, 0])
            observed_events.extend(event["type"] for event in info["events"])
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["severe_harm"])
        self.assertFalse(info["success"])
        self.assertEqual(info["termination_reason"], "severe_injury")
        self.assertGreater(info["state"]["physiology"]["blood_loss_ml"], 0)
        self.assertLess(info["state"]["physiology"]["mean_arterial_pressure_mmhg"], 90)
        self.assertLess(info["reward_components"]["critical_damage"], 0)
        self.assertLess(reward, -20)
        self.assertIn("bleeding_started", observed_events)

    def test_success_cannot_bypass_pending_thermal_injury(self):
        env = TissueRetractionEnv({"success_hold_steps": 1})
        env.reset(seed=2)
        # Controlled exposed state: establish that the successful baseline can
        # meet its target, then recreate the pre-success snapshot with a hot site.
        observed, _ = env.reset(seed=2)
        while True:
            if env.current_exposure >= env.config["exposure_target"]:
                break
            observed, _, terminated, truncated, _ = env.step(env.scripted_action(observed))
            if terminated or truncated:
                break
        self.assertGreaterEqual(env.current_exposure, env.config["exposure_target"])
        # Reset terminal bookkeeping only to construct this counterfactual
        # unit-test state; real users must reset after termination.
        env._ended = False
        env.termination_reason = None
        env.tissue.temperature[0] = 55.0
        env.stable_exposure_steps = 0
        _, reward, terminated, truncated, info = env.step(np.zeros(7))
        self.assertGreater(env.tissue.temperature[0], 43)
        self.assertFalse(info["success"])
        self.assertEqual(env.stable_exposure_steps, 0)
        self.assertEqual(info["reward_components"]["exposure_progress"], 0)

    def test_invalid_action_and_config_do_not_mutate_episode(self):
        env = TissueRetractionEnv()
        original, _ = env.reset(seed=42)
        for action in [[0] * 5, [0] * 8, [float("nan")] * 7, [1.01] * 7]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.step_count, 0)
            np.testing.assert_array_equal(original, env._observation())
        for config in [{"unknown": 1}, {"dt_s": float("nan")}, {"max_steps": True},
                       {"max_steps": 20.5}, {"randomize_anatomy": 1}, {"success_hold_steps": 50, "max_steps": 20}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_env_config(config)


if __name__ == "__main__":
    unittest.main()
