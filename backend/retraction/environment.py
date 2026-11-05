"""Gymnasium-compatible sequential tissue-response research environment.

Portable analytic spring/heat/injury surrogate. It is not a validated patient,
a clinical decision aid, a SOFA adapter, or a replacement for surgical training.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .simulation import validate_patient
from .tissue import KINDS, NODE_COUNT, NODE_SPECS, TissueModel

ENV_VERSION = "tissue-retraction-1.1"
PHYSICS_FINGERPRINT_ALGORITHM = "sha256-labeled-source-v1"
PHYSICS_SOURCE_FILES = ("environment.py", "tissue.py")


def _compute_physics_fingerprint(source_directory: Path | None = None) -> str:
    """Hash exact source bytes; bundled apps must ship these source artifacts.

    Labels and separators make the concatenation unambiguous without including
    machine-specific absolute paths. A missing file is a setup failure, never a
    made-up fingerprint or an inferred match based on the version string.
    """
    source_directory = Path(source_directory) if source_directory is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256(b"retraction-physics-source-v1\0")
    for name in PHYSICS_SOURCE_FILES:
        path = source_directory / name
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Physics source fingerprint unavailable: cannot read {path}. "
                "Restore the model source files or include both environment.py and tissue.py in the application bundle."
            ) from exc
        digest.update(f"retraction/{name}\0".encode("utf-8"))
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


# Capture once while the implementation is imported. Saving a checkpoint or
# inspecting a running episode must not silently pick up later filesystem edits.
PHYSICS_FINGERPRINT = _compute_physics_fingerprint()
SCRIPTED_CONTROLLER_VERSION = "feedback-strain-guard-1.1"
ACTION_NAMES = ["grasper_dx", "grasper_dy", "grip", "cautery_dx", "cautery_dy", "cautery_power", "handoff"]
DEFAULT_ENV_CONFIG = {
    "max_steps": 200, "dt_s": 0.1, "force_limit_n": 2.5, "strain_limit": 0.60,
    "exposure_target": 0.60, "success_hold_steps": 5,
    "injury_termination_threshold": 0.60, "critical_injury_threshold": 0.35,
    "blood_loss_termination_ml": 600.0, "randomize_anatomy": True,
}
ENV_CONFIG_LIMITS = {
    "max_steps": [20, 500], "dt_s": [0.02, 0.2], "force_limit_n": [0.2, 10],
    "strain_limit": [0.1, 2.0], "exposure_target": [0.2, 0.9], "success_hold_steps": [1, 50],
    "injury_termination_threshold": [0.2, 1.0], "critical_injury_threshold": [0.1, 0.9],
    "blood_loss_termination_ml": [10, 2000],
}
_BASE_OBS = [
    ("grasper_x", -30, 30), ("grasper_y", -30, 40), ("grip", 0, 1),
    ("cautery_x", -30, 30), ("cautery_y", -30, 40), ("cautery_power", 0, 1),
    ("contact_node_index", -1, NODE_COUNT - 1), ("contact_force_n", 0, 20),
    ("elapsed_s", 0, 100), ("step_count", 0, 500),
    ("exposure", 0, 1), ("peak_exposure", 0, 1), ("stable_exposure_steps", 0, 50),
    ("blood_loss_ml", 0, 5000), ("bile_leak_ml", 0, 5000),
    ("bleeding_rate_ml_per_s", 0, 200), ("bile_leak_rate_ml_per_s", 0, 100),
    ("heart_rate_bpm", 72, 180), ("mean_arterial_pressure_mmhg", 20, 90),
    ("attachment_length_mm", 4, 60), ("adhesion_count", 0, 12),
    ("adhesion_stiffness_n_per_mm", 0, 0.5), ("distension", 0.5, 2.5),
    ("dt_s", 0.02, 0.2), ("max_steps", 20, 500),
    ("force_limit_n", 0.2, 10), ("strain_limit", 0.1, 2),
    ("exposure_target", 0.2, 0.9), ("success_hold_steps", 1, 50),
    ("injury_termination_threshold", 0.2, 1), ("critical_injury_threshold", 0.1, 0.9),
    ("blood_loss_termination_ml", 10, 2000),
]
_NODE_OBS = [("rest_x", -15, 15), ("rest_y", -30, 30), ("x", -35, 35), ("y", -35, 45),
             ("mechanical_damage", 0, 1), ("thermal_dose", 0, 500), ("temperature_c", 37, 180),
             ("strain", 0, 30), ("coagulation", 0, 1)]
OBSERVATION_NAMES = [row[0] for row in _BASE_OBS] + [f"n{i:02d}_{name}" for i in range(NODE_COUNT) for name, _, _ in _NODE_OBS]


def normalize_env_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_ENV_CONFIG)
    if unknown:
        raise ValueError(f"Unknown environment configuration keys: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_ENV_CONFIG, **supplied}
    for key, (lower, upper) in ENV_CONFIG_LIMITS.items():
        value = result[key]
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{key} must be a finite number")
        if not math.isfinite(float(value)) or not lower <= float(value) <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
        if key in ("max_steps", "success_hold_steps"):
            if float(value) != int(value):
                raise ValueError(f"{key} must be an integer")
            result[key] = int(value)
        else:
            result[key] = float(value)
    if type(result["randomize_anatomy"]) is not bool:
        raise ValueError("randomize_anatomy must be a boolean")
    if result["success_hold_steps"] > result["max_steps"]:
        raise ValueError("success_hold_steps must not exceed max_steps")
    return result


class TissueRetractionEnv(gym.Env):
    """Fully observed fixed-step nonlinear tissue-response environment."""
    metadata = {"render_modes": ["ansi"], "render_fps": 10}
    action_names = ACTION_NAMES
    observation_names = OBSERVATION_NAMES

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        self.config = normalize_env_config(config)
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or ansi")
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTION_NAMES),), dtype=np.float32)
        bounds = _BASE_OBS + _NODE_OBS * NODE_COUNT
        self.observation_space = spaces.Box(np.array([lo for _, lo, _ in bounds], dtype=np.float32),
                                            np.array([hi for _, _, hi in bounds], dtype=np.float32), dtype=np.float32)
        self._reset_done = False
        self._ended = False

    def describe(self) -> dict[str, Any]:
        return {"version": ENV_VERSION, "model_version": ENV_VERSION,
                "physics_fingerprint": PHYSICS_FINGERPRINT,
                "physics_fingerprint_algorithm": PHYSICS_FINGERPRINT_ALGORITHM, "engine": "stateful-analytic-tissue-surrogate",
                "action_names": list(ACTION_NAMES), "action_low": self.action_space.low.tolist(),
                "action_high": self.action_space.high.tolist(), "observation_names": list(OBSERVATION_NAMES),
                "observation_low": self.observation_space.low.tolist(), "observation_high": self.observation_space.high.tolist(),
                "observation_size": len(OBSERVATION_NAMES), "node_count": NODE_COUNT,
                "scripted_controller_version": SCRIPTED_CONTROLLER_VERSION,
                "scripted_controller_development": "Proportional exposure feedback with observed strain/force guards; engineering-tuned on development seeds 0–19. Not tuned on the final evaluation cohort.",
                "default_config": deepcopy(DEFAULT_ENV_CONFIG), "config_limits": deepcopy(ENV_CONFIG_LIMITS),
                "config": dict(self.config),
                "action_semantics": "XY velocities at 10 mm/s maximum. Positive grip sets closure; grip < -0.2 releases; zero holds closure. Positive cautery power activates heat for this step; zero turns heat off. Handoff >0.5 ends the episode without implying recovery.",
                "objective": "Hold the illustrative exposure target for the configured consecutive steps with damage below 0.15, critical damage below 0.08, force/strain within the selected modeling limits, cautery off and all tissue at or below 43 Celsius.",
                "reward": {"record_exposure_gain": 4, "new_damage_per_node": -12,
                           "new_critical_damage_per_node": -40, "blood_loss_per_ml": -0.005,
                           "bile_leak_per_ml": -0.05, "step": -0.01, "success": 15,
                           "severe_injury": -20, "handoff": -2, "timeout": -1},
                "assumptions": self._assumptions()}

    @staticmethod
    def _assumptions() -> list[str]:
        return ["Educational/research analytic surrogate, not a validated patient or clinical tool. Not SOFA/FEM.",
                "Illustrative planar spring network and Gaussian heat source; no anatomical reconstruction, cutting/removal, fluid dynamics or inflammatory response.",
                "Injury, temperatures, doses, force/strain thresholds, flow rates and vital-sign equations are illustrative and not clinically calibrated.",
                "Bleeding, bile leakage and thermal damage are simplified governed states. Grasp and cautery contact occur in the modeled plane without tool depth.",
                "Full simulated tissue state is observable for Markov RL. This is more information than a real robot or surgeon observes.",
                "Injury lowers stiffness irreversibly. Heat dissipates but accumulated thermal damage does not heal during an episode.",
                "Exposure is a synthetic marker opening obscured by accumulated simulated blood/bile, not clinical critical view of safety.",
                "Success rewards simulated exposure only when modeled force/strain and injury conditions hold; handoff is a decision, not a simulated successful rescue."]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        if set(options) - {"patient"}:
            raise ValueError("Reset options only support patient")
        if "patient" in options:
            patient = dict(options["patient"])
            if set(patient) != {"attachment_length_mm", "adhesion_count", "adhesion_stiffness_n_per_mm", "distension"}:
                raise ValueError("Patient must contain exactly attachment_length_mm, adhesion_count, adhesion_stiffness_n_per_mm, distension")
            validate_patient(patient)
        elif self.config["randomize_anatomy"]:
            patient = {"attachment_length_mm": float(self.np_random.uniform(12, 35)),
                       "adhesion_count": int(self.np_random.integers(0, 5)),
                       "adhesion_stiffness_n_per_mm": float(self.np_random.uniform(0.03, 0.16)),
                       "distension": float(self.np_random.uniform(0.8, 1.6))}
        else:
            patient = {"attachment_length_mm": 24.0, "adhesion_count": 2,
                       "adhesion_stiffness_n_per_mm": 0.08, "distension": 1.2}
        self.patient = {key: int(value) if key == "adhesion_count" else float(value) for key, value in patient.items()}
        self.tissue = TissueModel(self.patient)
        self.grasper = np.array([6.0, 3.0])
        self.cautery = np.array([-16.0, -14.0])
        self.grip, self.power = 0.0, 0.0
        self.step_count, self.time_s = 0, 0.0
        self.current_exposure = self.tissue.exposure()
        self.peak_exposure = self.current_exposure
        self.stable_exposure_steps = 0
        self.termination_reason = None
        self.total_reward = 0.0
        self._reset_done, self._ended = True, False
        return self._observation(), self._info([], {}, 0.0)

    def _observation(self) -> np.ndarray:
        t, c = self.tissue, self.config
        values = [*self.grasper, self.grip, *self.cautery, self.power,
                  -1 if t.contact_index is None else t.contact_index, t.contact_force_n,
                  self.time_s, self.step_count, self.current_exposure, self.peak_exposure,
                  self.stable_exposure_steps, t.blood_loss_ml, t.bile_leak_ml,
                  t.bleeding_rate, t.bile_leak_rate, t.heart_rate, t.mean_arterial_pressure,
                  self.patient["attachment_length_mm"], self.patient["adhesion_count"],
                  self.patient["adhesion_stiffness_n_per_mm"], self.patient["distension"],
                  *[c[key] for key in ["dt_s", "max_steps", "force_limit_n", "strain_limit", "exposure_target",
                                       "success_hold_steps", "injury_termination_threshold", "critical_injury_threshold",
                                       "blood_loss_termination_ml"]]]
        for i in range(NODE_COUNT):
            values.extend([*t.rest[i], *t.positions[i], t.mechanical_damage[i], t.thermal_dose[i],
                           t.temperature[i], t.strain[i], t.coagulation[i]])
        result = np.asarray(values, dtype=np.float32)
        if not self.observation_space.contains(result):
            raise RuntimeError("Internal surrogate state exceeded declared observation bounds")
        return result

    def _info(self, events: list[dict[str, Any]], components: dict[str, float], reward: float) -> dict[str, Any]:
        t = self.tissue
        near = int(np.argmin(np.linalg.norm(t.positions - self.cautery, axis=1)))
        cautery_contact = f"n{near:02d}" if np.linalg.norm(t.positions[near] - self.cautery) <= 3 else None
        return {"environment_version": ENV_VERSION, "physics_fingerprint": PHYSICS_FINGERPRINT,
                "physics_fingerprint_algorithm": PHYSICS_FINGERPRINT_ALGORITHM, "engine": "stateful-analytic-tissue-surrogate",
                "state": {"time_s": self.time_s, "step_count": self.step_count,
                          "patient": dict(self.patient),
                          "tools": {"grasper": {"position_mm": self.grasper.tolist(), "grip": self.grip,
                                                "contact_node_id": None if t.contact_index is None else f"n{t.contact_index:02d}",
                                                "force_n": t.contact_force_n},
                                    "cautery": {"position_mm": self.cautery.tolist(), "power": self.power,
                                                "contact_node_id": cautery_contact}},
                          "tissue": t.snapshot(),
                          "physiology": {"blood_loss_ml": t.blood_loss_ml, "bleeding_rate_ml_per_s": t.bleeding_rate,
                                         "bile_leak_ml": t.bile_leak_ml, "bile_leak_rate_ml_per_s": t.bile_leak_rate,
                                         "heart_rate_bpm": t.heart_rate, "mean_arterial_pressure_mmhg": t.mean_arterial_pressure},
                          "exposure": self.current_exposure, "peak_exposure": self.peak_exposure,
                          "stable_exposure_steps": self.stable_exposure_steps,
                          "max_damage": float(t.damage.max()), "critical_damage": t.critical_damage(),
                          "termination_reason": self.termination_reason,
                          "limits": {key: value for key, value in self.config.items() if key not in ("randomize_anatomy",)}},
                "reward": reward, "reward_components": components, "cumulative_reward": self.total_reward,
                "events": events, "termination_reason": self.termination_reason,
                "success": self.termination_reason == "success",
                "severe_harm": bool(float(t.damage.max()) >= self.config["injury_termination_threshold"] or t.critical_damage() >= self.config["critical_injury_threshold"] or t.blood_loss_ml >= self.config["blood_loss_termination_ml"]),
                "assumptions": self._assumptions()}

    def step(self, action):
        if not self._reset_done:
            raise RuntimeError("Call reset before step")
        if self._ended:
            raise RuntimeError("Episode has ended; call reset before another step")
        try:
            a = np.asarray(action, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action must contain seven finite numeric values in [-1,1]") from exc
        if a.shape != (7,) or not np.isfinite(a).all() or (np.abs(a) > 1).any():
            raise ValueError("Action must contain seven finite numeric values in [-1,1]")
        events: list[dict[str, Any]] = []
        components = {"step": -0.01, "exposure_progress": 0.0, "tissue_damage": 0.0,
                      "critical_damage": 0.0, "bleeding": 0.0, "bile_leak": 0.0,
                      "force_excess": 0.0, "terminal": 0.0}
        self.step_count += 1
        if a[6] > 0.5:
            self.power = 0.0
            self.termination_reason = "handoff"
            components["terminal"] = -2.0
            events.append({"type": "handoff", "severity": "decision", "message": "Control handed off; existing injury is not reversed."})
        else:
            dt = self.config["dt_s"]
            self.time_s = self.step_count * dt
            self.grasper = np.clip(self.grasper + a[:2] * 10 * dt, [-30, -30], [30, 40])
            self.cautery = np.clip(self.cautery + a[3:5] * 10 * dt, [-30, -30], [30, 40])
            if a[2] < -0.2:
                self.grip = 0.0
                if self.tissue.release_contact():
                    events.append({"type": "release", "severity": "info", "message": "Grasper contact released."})
            elif a[2] > 0:
                self.grip = float(a[2])
            self.power = float(max(0, a[5]))
            if self.tissue.acquire_contact(self.grasper, self.grip):
                events.append({"type": "grasp", "severity": "info", "node_id": f"n{self.tissue.contact_index:02d}", "message": "Grasper contact established."})
            old_damage = self.tissue.damage.copy()
            old_bleeding, old_leak = self.tissue.bleeding_rate, self.tissue.bile_leak_rate
            change = self.tissue.step(dt, self.grasper, self.grip, self.cautery, self.power,
                                      self.config["force_limit_n"], self.config["strain_limit"])
            damage = self.tissue.damage
            for i in range(NODE_COUNT):
                for boundary, name in [(0.05, "injury_started"), (0.15, "injury_warning"), (0.35, "major_injury"), (0.9, "tear")]:
                    if old_damage[i] < boundary <= damage[i]:
                        events.append({"type": name, "severity": "danger" if boundary >= 0.15 else "warning",
                                       "node_id": f"n{i:02d}", "tissue_kind": self.tissue.kinds[i],
                                       "message": f"{self.tissue.kinds[i]} irreversible modeled injury crossed {boundary:.0%}."})
            if old_bleeding == 0 and self.tissue.bleeding_rate > 0:
                events.append({"type": "bleeding_started", "severity": "danger", "message": "Modeled vessel injury is causing blood loss."})
            if old_leak == 0 and self.tissue.bile_leak_rate > 0:
                events.append({"type": "bile_leak_started", "severity": "danger", "message": "Modeled duct injury is causing bile leakage."})
            if change["contact_torn"]:
                events.append({"type": "grasp_lost", "severity": "danger", "message": "Irreversible tissue damage caused contact failure."})
            self.current_exposure = self.tissue.exposure()
            new_peak = max(self.peak_exposure, self.current_exposure)
            # Credit only records while injury conditions are acceptable; tearing
            # soft tissue to improve marker opening cannot earn progress reward.
            acceptable = (damage.max() < 0.15 and self.tissue.critical_damage() < 0.08 and
                          self.tissue.contact_force_n <= self.config["force_limit_n"] and
                          self.tissue.strain.max() <= self.config["strain_limit"] and
                          self.power == 0 and float(self.tissue.temperature.max()) <= 43.0)
            components["exposure_progress"] = 4 * (new_peak - self.peak_exposure) if acceptable else 0.0
            self.peak_exposure = new_peak
            self.stable_exposure_steps = self.stable_exposure_steps + 1 if acceptable and self.current_exposure >= self.config["exposure_target"] else 0
            increment = change["damage_increment"]
            components["tissue_damage"] = -12 * float(increment.sum())
            components["critical_damage"] = -40 * float(sum(increment[i] for i, kind in enumerate(self.tissue.kinds) if kind != "gallbladder"))
            components["bleeding"] = -0.005 * change["blood_loss_increment_ml"]
            components["bile_leak"] = -0.05 * change["bile_leak_increment_ml"]
            components["force_excess"] = -0.1 * dt * max(0, self.tissue.contact_force_n / self.config["force_limit_n"] - 1)
            if (float(damage.max()) >= self.config["injury_termination_threshold"] or
                    self.tissue.critical_damage() >= self.config["critical_injury_threshold"]):
                self.termination_reason = "severe_injury"
                components["terminal"] = -20.0
            elif self.tissue.blood_loss_ml >= self.config["blood_loss_termination_ml"]:
                self.termination_reason = "blood_loss"
                components["terminal"] = -20.0
            elif self.stable_exposure_steps >= self.config["success_hold_steps"]:
                self.termination_reason = "success"
                components["terminal"] = 15.0
            elif self.step_count >= self.config["max_steps"]:
                self.termination_reason = "time_limit"
                components["terminal"] = -1.0
            if self.termination_reason:
                events.append({"type": "episode_ended", "severity": "info" if self.termination_reason == "success" else "warning",
                               "message": f"Episode ended: {self.termination_reason}."})
        reward = float(sum(components.values()))
        self.total_reward += reward
        terminated = self.termination_reason is not None and self.termination_reason != "time_limit"
        truncated = self.termination_reason == "time_limit"
        self._ended = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(events, components, reward)

    def scripted_action(self, observation: np.ndarray) -> np.ndarray:
        """Simple transparent feedback baseline using only the public observation."""
        state = dict(zip(OBSERVATION_NAMES, np.asarray(observation).tolist()))
        action = np.zeros(7, dtype=np.float32)
        if state["contact_node_index"] < 0:
            # Start near the neck marker; return there if a user moved away.
            action[0] = float(np.clip((6 - state["grasper_x"]) / 2, -0.5, 0.5))
            action[1] = float(np.clip((3 - state["grasper_y"]) / 2, -0.5, 0.5))
            action[2] = 0.55
        else:
            # Controller parameters were engineering-tuned on development seeds
            # 0–19, not on a final evaluation cohort. No hidden tissue/oracle
            # state is used; every quantity comes from the supplied observation.
            error = state["exposure_target"] + 0.005 - state["exposure"]
            action[0] = float(np.clip(1.2 * error, 0, 0.30))
            maximum_strain = max(state[f"n{i:02d}_strain"] for i in range(NODE_COUNT))
            if (maximum_strain > 0.95 * state["strain_limit"] or
                    state["contact_force_n"] > 0.95 * state["force_limit_n"]):
                action[0] = -0.03
        return action

    def render(self):
        if not self._reset_done:
            return "TissueRetractionEnv: reset required"
        return (f"t={self.time_s:.1f}s exposure={self.current_exposure:.3f} "
                f"injury={self.tissue.damage.max():.3f} blood_loss={self.tissue.blood_loss_ml:.1f}ml "
                f"status={self.termination_reason or 'active'}")
