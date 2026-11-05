"""Reproducible tool-response interventions for the public tissue environment.

These controllers issue ordinary actions; they cannot write anatomy, injury,
physiology or rewards. They are software response experiments, not treatment
plans, learned policies, or clinically calibrated procedures.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any

import numpy as np

from .environment import ACTION_NAMES, DEFAULT_ENV_CONFIG, TissueRetractionEnv
from .tissue import NODE_COUNT

PROTOCOL_VERSION = "tool-response-protocols-1.0"
MAX_PROTOCOL_STEPS = 200
SCOPE = "Controlled synthetic tissue-response experiment; not clinical instructions or a trained RL policy."

# Durations and tool commands belong to the intervention, never the tissue model.
_COMMON_PARAMETERS = {
    "maximum_axis_speed_mm_per_s": 10.0,
    "release_command": -1.0,
    "arrival_tolerance_mm": 0.05,
    "maximum_steps": MAX_PROTOCOL_STEPS,
    "action_names": list(ACTION_NAMES),
}
_CATALOG = [
    {
        "id": "gentle-traction",
        "title": "Gentle traction baseline",
        "description": "Acquire the nearest gallbladder site, then apply exposure feedback with observed strain and force guards.",
        "parameters": {"jaw_closure": 0.55, "maximum_traction_command": 0.30,
                       "exposure_proportional_gain": 1.2, "exposure_offset": 0.005,
                       "limit_guard_fraction": 0.95, "relaxation_command": -0.03,
                       "cautery_power": 0.0},
    },
    {
        "id": "held-jaw-crush",
        "title": "Held-jaw compression",
        "description": "Close fully on the nearest gallbladder site and hold without traction; observe accumulated mechanical injury.",
        "parameters": {"jaw_closure": 1.0, "hold_duration_s": 4.0,
                       "observation_duration_s": 2.0, "cautery_power": 0.0},
    },
    {
        "id": "arterial-thermal-contact",
        "title": "Arterial thermal contact",
        "description": "Move unpowered to the upper arterial site, then apply a bounded heat exposure; measure injury and blood loss.",
        "parameters": {"target_kind": "artery", "target_selection": "greatest observed y coordinate",
                       "cautery_power": 1.0, "heat_duration_s": 0.8,
                       "observation_duration_s": 2.0, "jaw_closure": 0.0},
    },
    {
        "id": "duct-thermal-contact",
        "title": "Bile-duct thermal contact",
        "description": "Move unpowered to the median common-duct site, then apply the same heat exposure; measure injury and bile leakage.",
        "parameters": {"target_kind": "common_bile_duct", "target_selection": "median observed y coordinate",
                       "cautery_power": 1.0, "heat_duration_s": 0.8,
                       "observation_duration_s": 2.0, "jaw_closure": 0.0},
    },
    {
        "id": "remote-heat-control",
        "title": "Remote heat negative control",
        "description": "Apply the same heat command at the initial remote tool position; measure whether modeled tissue responds without nearby heat contact.",
        "parameters": {"cautery_power": 1.0, "heat_duration_s": 0.8,
                       "observation_duration_s": 2.0, "jaw_closure": 0.0,
                       "target_selection": "initial cautery position; no motion"},
    },
]
_BY_ID = {row["id"]: row for row in _CATALOG}


def scenario_catalog() -> list[dict[str, Any]]:
    """Return independent JSON metadata, including the unchanged model defaults."""
    return [{**deepcopy(row), "protocol_version": PROTOCOL_VERSION, "scope": SCOPE,
             "seed": 2, "config": deepcopy(DEFAULT_ENV_CONFIG),
             "parameters": {**deepcopy(_COMMON_PARAMETERS), **deepcopy(row["parameters"])}}
            for row in _CATALOG]


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


class ScenarioController:
    """Bounded online controller with JSON state for durable API continuation.

    ``next_action`` reads only public ``info['state']``. Its counters describe
    issued actions, not successful transitions: callers persist ``state_dict``
    only after their normal environment/service step succeeds. ``phase`` after
    a non-None return labels that very action, not the next action.
    """

    def __init__(self, scenario_id: str, *, max_steps: int = MAX_PROTOCOL_STEPS):
        if not isinstance(scenario_id, str) or scenario_id not in _BY_ID:
            raise ValueError(f"Unknown response protocol: {scenario_id}")
        self.scenario_id = scenario_id
        self.max_steps = _integer(max_steps, "max_steps", 1, MAX_PROTOCOL_STEPS)
        self.steps = 0
        self.phase = "ready"
        self.target_node_id: str | None = None
        self.exposure_steps = 0
        self.observation_steps = 0
        self.completed = False
        self.completion_reason: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        row = next(row for row in scenario_catalog() if row["id"] == self.scenario_id)
        return {**row, "max_steps": self.max_steps, "target_node_id": self.target_node_id}

    def state_dict(self) -> dict[str, Any]:
        return {"protocol_version": PROTOCOL_VERSION, "scenario_id": self.scenario_id,
                "max_steps": self.max_steps, "steps": self.steps, "phase": self.phase,
                "target_node_id": self.target_node_id, "exposure_steps": self.exposure_steps,
                "observation_steps": self.observation_steps, "completed": self.completed,
                "completion_reason": self.completion_reason}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ScenarioController:
        if not isinstance(state, dict) or state.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Response protocol state has an unsupported protocol version")
        expected = {"protocol_version", "scenario_id", "max_steps", "steps", "phase", "target_node_id",
                    "exposure_steps", "observation_steps", "completed", "completion_reason"}
        if set(state) != expected:
            raise ValueError("Response protocol state has missing or unknown fields")
        result = cls(state["scenario_id"], max_steps=state["max_steps"])
        for key in ("steps", "exposure_steps", "observation_steps"):
            setattr(result, key, _integer(state[key], key, 0, result.max_steps))
        phases = {"ready", "approach", "grasp", "traction", "strain-guard", "compress",
                  "heat", "observe", "complete", "episode-ended", "step-budget"}
        if not isinstance(state["phase"], str) or state["phase"] not in phases or type(state["completed"]) is not bool:
            raise ValueError("Response protocol state has invalid phase or completion status")
        if state["completion_reason"] not in (None, "protocol-duration", "episode-ended", "step-budget"):
            raise ValueError("Response protocol state has invalid completion reason")
        node = state["target_node_id"]
        if node is not None and (not isinstance(node, str) or node not in {f"n{i:02d}" for i in range(NODE_COUNT)}):
            raise ValueError("Response protocol state has invalid target node id")
        if result.exposure_steps + result.observation_steps > result.steps:
            raise ValueError("Response protocol counters exceed issued actions")
        result.phase = state["phase"]
        result.target_node_id = node
        result.completed = state["completed"]
        result.completion_reason = state["completion_reason"]
        return result

    def _finish(self, reason: str) -> None:
        self.completed = True
        self.completion_reason = reason
        self.phase = {"protocol-duration": "complete"}.get(reason, reason)

    def _select_target(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if self.scenario_id == "remote-heat-control":
            return None
        nodes = state["tissue"]["nodes"]
        if self.target_node_id is None:
            if self.scenario_id == "arterial-thermal-contact":
                selected = max((n for n in nodes if n["kind"] == "artery"),
                               key=lambda n: (n["position_mm"][1], n["id"]))
            elif self.scenario_id == "duct-thermal-contact":
                candidates = sorted((n for n in nodes if n["kind"] == "common_bile_duct"),
                                    key=lambda n: (n["position_mm"][1], n["id"]))
                selected = candidates[len(candidates) // 2]
            else:
                grasper = np.asarray(state["tools"]["grasper"]["position_mm"], dtype=float)
                selected = min((n for n in nodes if n["kind"] == "gallbladder"),
                               key=lambda n: (float(np.linalg.norm(np.asarray(n["position_mm"]) - grasper)), n["id"]))
            self.target_node_id = selected["id"]
        return next(n for n in nodes if n["id"] == self.target_node_id)

    def next_action(self, observation: Any, info: dict[str, Any]) -> np.ndarray | None:
        """Issue one 7D action, or None when the intervention/episode has stopped."""
        del observation  # Interface matches the Gym loop; decisions use its public structured state.
        if self.completed:
            return None
        if info.get("termination_reason"):
            self._finish("episode-ended")
            return None
        if self.steps >= self.max_steps:
            self._finish("step-budget")
            return None
        state = info["state"]
        dt = float(state["limits"]["dt_s"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Public state must contain a positive finite timestep")
        parameters = _BY_ID[self.scenario_id]["parameters"]
        target = self._select_target(state)
        action = np.zeros(len(ACTION_NAMES), dtype=np.float32)

        if self.scenario_id == "gentle-traction":
            grasper = state["tools"]["grasper"]
            if grasper["contact_node_id"] is None:
                delta = np.asarray(target["position_mm"]) - np.asarray(grasper["position_mm"])
                if float(np.linalg.norm(delta)) > _COMMON_PARAMETERS["arrival_tolerance_mm"]:
                    action[:2] = np.clip(delta / (10 * dt), -1, 1)
                    action[2] = -1  # Open while moving; do not acquire an intermediate site.
                    self.phase = "approach"
                else:
                    action[2] = parameters["jaw_closure"]
                    self.phase = "grasp"
            else:
                error = state["limits"]["exposure_target"] + parameters["exposure_offset"] - state["exposure"]
                action[0] = np.clip(parameters["exposure_proportional_gain"] * error, 0,
                                    parameters["maximum_traction_command"])
                strain = max(node["strain"] for node in state["tissue"]["nodes"])
                guard = parameters["limit_guard_fraction"]
                if strain > guard * state["limits"]["strain_limit"] or grasper["force_n"] > guard * state["limits"]["force_limit_n"]:
                    action[0] = parameters["relaxation_command"]
                    self.phase = "strain-guard"
                else:
                    self.phase = "traction"
        else:
            tool_name = "grasper" if self.scenario_id == "held-jaw-crush" else "cautery"
            if target is not None and self.exposure_steps == 0:
                delta = np.asarray(target["position_mm"]) - np.asarray(state["tools"][tool_name]["position_mm"])
                if float(np.linalg.norm(delta)) > _COMMON_PARAMETERS["arrival_tolerance_mm"]:
                    action[:2] = np.clip(delta / (10 * dt), -1, 1) if tool_name == "grasper" else 0
                    if tool_name == "cautery":
                        action[3:5] = np.clip(delta / (10 * dt), -1, 1)
                    action[2] = -1
                    self.phase = "approach"
                    self.steps += 1
                    return action
            duration = parameters.get("hold_duration_s", parameters.get("heat_duration_s"))
            exposure_count = math.ceil(duration / dt - 1e-12)
            observation_count = math.ceil(parameters["observation_duration_s"] / dt - 1e-12)
            if self.exposure_steps < exposure_count:
                if self.scenario_id == "held-jaw-crush":
                    action[2] = 1.0 if self.exposure_steps == 0 else 0.0
                    self.phase = "compress"
                else:
                    action[2] = -1
                    action[5] = parameters["cautery_power"]
                    self.phase = "heat"
                self.exposure_steps += 1
            elif self.observation_steps < observation_count:
                action[2] = -1  # Release compression and keep cautery off during observation.
                self.phase = "observe"
                self.observation_steps += 1
            else:
                self._finish("protocol-duration")
                return None
        self.steps += 1
        return action


def create_scenario_controller(scenario_id: str, *, max_steps: int = MAX_PROTOCOL_STEPS,
                               state: dict[str, Any] | None = None) -> ScenarioController:
    if state is None:
        return ScenarioController(scenario_id, max_steps=max_steps)
    controller = ScenarioController.from_state(state)
    if controller.scenario_id != scenario_id or controller.max_steps != max_steps:
        raise ValueError("Response protocol state does not match requested id or step budget")
    return controller


def summarize_scenario(initial_info: dict[str, Any], transitions: list[dict[str, Any]],
                       controller: ScenarioController) -> dict[str, Any]:
    """Compute outcomes solely from actual recorded simulator transitions."""
    final = transitions[-1]["info"] if transitions else initial_info
    states = [initial_info["state"], *(step["info"]["state"] for step in transitions)]
    reward_components: Counter = Counter()
    events: Counter = Counter()
    for step in transitions:
        reward_components.update(step["info"]["reward_components"])
        events.update(event["type"] for event in step["info"]["events"])
    ended = bool(final.get("termination_reason"))
    status = "truncated" if transitions and transitions[-1]["truncated"] else "terminated" if ended else "protocol-complete" if controller.completion_reason == "protocol-duration" else "step-budget"
    return {"status": status, "termination_reason": final.get("termination_reason"),
            "protocol_completion_reason": controller.completion_reason,
            "success": bool(final.get("success", False)), "severe_harm": bool(final.get("severe_harm", False)),
            "steps": len(transitions), "duration_s": states[-1]["time_s"] - states[0]["time_s"],
            "cumulative_reward": float(sum(step["reward"] for step in transitions)),
            "reward_components": dict(reward_components), "event_counts": dict(events),
            "max_damage": states[-1]["max_damage"], "critical_damage": states[-1]["critical_damage"],
            "peak_temperature_c": max(n["temperature_c"] for s in states for n in s["tissue"]["nodes"]),
            "peak_force_n": max(s["tools"]["grasper"]["force_n"] for s in states),
            "peak_strain": max(n["strain"] for s in states for n in s["tissue"]["nodes"]),
            "final_exposure": states[-1]["exposure"], "physiology": deepcopy(states[-1]["physiology"]),
            "injured_nodes": [{"id": n["id"], "kind": n["kind"], "damage": n["damage"],
                               "mechanical_damage": n["mechanical_damage"], "thermal_damage": n["thermal_damage"]}
                              for n in states[-1]["tissue"]["nodes"] if n["damage"] > 0]}


def run_scenario(scenario_id: str, *, seed: int = 2, config: dict[str, Any] | None = None,
                 patient: dict[str, Any] | None = None, max_steps: int = MAX_PROTOCOL_STEPS) -> dict[str, Any]:
    """Execute an isolated real environment; API sessions use the controller instead.

    Complete observations, events, reward components and actions are retained for
    replay. No success label or patient response is inferred from the case name.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    controller = create_scenario_controller(scenario_id, max_steps=max_steps)
    environment = TissueRetractionEnv(config)
    try:
        observation, info = environment.reset(seed=seed, options={"patient": patient} if patient is not None else None)
        initial = {"observation": observation.tolist(), "info": deepcopy(info)}
        transitions = []
        while (action := controller.next_action(observation, info)) is not None:
            observation, reward, terminated, truncated, info = environment.step(action)
            transitions.append({"step": controller.steps, "phase": controller.phase, "action": action.tolist(),
                                "observation": observation.tolist(), "reward": float(reward),
                                "terminated": bool(terminated), "truncated": bool(truncated), "info": deepcopy(info)})
            if terminated or truncated:
                controller.next_action(observation, info)  # Mark the controller stopped without issuing an action.
                break
        return {"id": scenario_id, "title": controller.metadata["title"], "scope": SCOPE,
                "protocol_version": PROTOCOL_VERSION, "environment": environment.describe(),
                "config": dict(environment.config), "seed": seed, "patient": deepcopy(initial["info"]["state"]["patient"]),
                "parameters": controller.metadata["parameters"], "max_steps": max_steps,
                "initial": initial, "actions": [step["action"] for step in transitions],
                "transitions": transitions, "controller_state": controller.state_dict(),
                "outcome": summarize_scenario(initial["info"], transitions, controller)}
    finally:
        environment.close()
