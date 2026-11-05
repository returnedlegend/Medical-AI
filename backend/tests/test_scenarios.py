"""Causal and replay checks for real, bounded tissue-response interventions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from retraction.environment import DEFAULT_ENV_CONFIG, TissueRetractionEnv
from retraction.scenarios import (
    MAX_PROTOCOL_STEPS, PROTOCOL_VERSION, ScenarioController,
    create_scenario_controller, run_scenario, scenario_catalog,
)


@pytest.fixture(scope="module")
def cases():
    return {row["id"]: run_scenario(row["id"], seed=2) for row in scenario_catalog()}


def test_catalog_records_unchanged_config_and_independent_metadata():
    catalog = scenario_catalog()
    assert len(catalog) == 5
    assert len({case["id"] for case in catalog}) == 5
    assert all(case["config"] == DEFAULT_ENV_CONFIG for case in catalog)
    assert all(case["protocol_version"] == PROTOCOL_VERSION for case in catalog)
    assert all(case["parameters"]["maximum_steps"] == 200 for case in catalog)
    catalog[0]["config"]["force_limit_n"] = 99
    catalog[0]["parameters"]["action_names"].append("fake")
    assert scenario_catalog()[0]["config"] == DEFAULT_ENV_CONFIG
    assert len(scenario_catalog()[0]["parameters"]["action_names"]) == 7


@pytest.mark.parametrize("scenario_id", [case["id"] for case in scenario_catalog()])
def test_actions_are_real_bounded_finite_transitions_and_replay_exactly(cases, scenario_id):
    result = cases[scenario_id]
    assert 0 < len(result["actions"]) <= MAX_PROTOCOL_STEPS
    json.dumps(result, allow_nan=False)
    env = TissueRetractionEnv(result["config"])
    try:
        observation, info = env.reset(seed=result["seed"], options={"patient": result["patient"]})
        assert observation.tolist() == result["initial"]["observation"]
        assert info == result["initial"]["info"]
        for action, recorded in zip(result["actions"], result["transitions"], strict=True):
            assert np.asarray(action).shape == (7,)
            assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.tolist() == recorded["observation"]
            assert (reward, terminated, truncated, info) == (
                recorded["reward"], recorded["terminated"], recorded["truncated"], recorded["info"])
        outcome = result["outcome"]
        assert outcome["cumulative_reward"] == sum(t["reward"] for t in result["transitions"])
        assert outcome["max_damage"] == info["state"]["max_damage"]
        assert outcome["physiology"] == info["state"]["physiology"]
        assert outcome["steps"] == info["state"]["step_count"]
    finally:
        env.close()


@pytest.mark.parametrize("scenario_id", [case["id"] for case in scenario_catalog()])
def test_json_controller_continuation_matches_uninterrupted_execution(cases, scenario_id):
    result = cases[scenario_id]
    controller = create_scenario_controller(scenario_id)
    prior = result["initial"]
    for recorded in result["transitions"]:
        # Simulate an API process reconstructing the controller for every action.
        controller = ScenarioController.from_state(json.loads(json.dumps(controller.state_dict())))
        action = controller.next_action(prior["observation"], prior["info"])
        assert action.tolist() == recorded["action"]
        assert controller.phase == recorded["phase"]
        assert controller.steps == recorded["step"]
        prior = recorded
    assert controller.next_action(prior["observation"], prior["info"]) is None
    stopped = controller.state_dict()
    assert controller.next_action(prior["observation"], prior["info"]) is None
    assert controller.state_dict() == stopped
    assert stopped == result["controller_state"]


def test_gentle_baseline_is_low_injury_without_promising_success_for_every_patient(cases):
    result = cases["gentle-traction"]
    assert result["outcome"]["success"]  # Explicit reproducibility case, seed 2 only.
    assert result["outcome"]["max_damage"] == 0
    assert result["outcome"]["physiology"]["blood_loss_ml"] == 0
    assert result["outcome"]["physiology"]["bile_leak_ml"] == 0
    assert all(action[5] == 0 for action in result["actions"])
    assert 0.55 in [action[2] for action in result["actions"]] or any(
        abs(action[2] - 0.55) < 1e-6 for action in result["actions"])


def test_held_jaw_causes_mechanical_injury_without_movement_or_heat(cases):
    result = cases["held-jaw-crush"]
    compressions = [step for step in result["transitions"] if step["phase"] == "compress"]
    assert compressions[0]["action"][2] == 1
    assert all(step["action"] == [0] * 7 for step in compressions[1:])
    assert all(step["info"]["state"]["tools"]["grasper"]["grip"] == 1 for step in compressions)
    assert result["outcome"]["max_damage"] > cases["gentle-traction"]["outcome"]["max_damage"]
    assert result["outcome"]["severe_harm"]
    assert result["outcome"]["reward_components"]["tissue_damage"] < 0
    assert result["outcome"]["reward_components"]["terminal"] == -20
    assert result["outcome"]["event_counts"]["major_injury"] > 0
    for node in compressions[-1]["info"]["state"]["tissue"]["nodes"]:
        assert node["thermal_damage"] == 0 and node["thermal_dose"] == 0


def test_arterial_heat_contact_causes_recorded_blood_loss_and_vital_response(cases):
    result = cases["arterial-thermal-contact"]
    control = cases["remote-heat-control"]
    assert result["parameters"]["cautery_power"] == control["parameters"]["cautery_power"]
    assert result["parameters"]["heat_duration_s"] == control["parameters"]["heat_duration_s"]
    outcome = result["outcome"]
    assert outcome["physiology"]["blood_loss_ml"] > control["outcome"]["physiology"]["blood_loss_ml"]
    assert outcome["physiology"]["bleeding_rate_ml_per_s"] > 0
    assert outcome["physiology"]["heart_rate_bpm"] > 72
    assert outcome["physiology"]["mean_arterial_pressure_mmhg"] < 90
    assert outcome["physiology"]["bile_leak_ml"] == 0
    assert outcome["event_counts"]["bleeding_started"] == 1
    assert outcome["reward_components"]["critical_damage"] < 0
    assert outcome["reward_components"]["bleeding"] < 0
    assert outcome["termination_reason"] == "severe_injury"
    assert outcome["critical_damage"] >= result["config"]["critical_injury_threshold"]
    assert outcome["physiology"]["blood_loss_ml"] < result["config"]["blood_loss_termination_ml"]
    assert not outcome["success"]  # Critical injury, not a fabricated bulk-loss failure.


def test_duct_contact_causes_leak_without_fabricating_arterial_bleeding(cases):
    outcome = cases["duct-thermal-contact"]["outcome"]
    assert outcome["physiology"]["bile_leak_ml"] > 0
    assert outcome["physiology"]["bile_leak_rate_ml_per_s"] > 0
    assert outcome["physiology"]["blood_loss_ml"] == 0
    assert outcome["event_counts"]["bile_leak_started"] == 1
    assert outcome["reward_components"]["bile_leak"] < 0
    assert outcome["reward_components"]["critical_damage"] < 0
    assert outcome["severe_harm"] and not outcome["success"]


def test_remote_heat_has_no_tissue_response_at_the_unchanged_initial_pose(cases):
    result = cases["remote-heat-control"]
    assert result["outcome"]["status"] == "protocol-complete"
    assert result["outcome"]["termination_reason"] is None
    assert not result["outcome"]["success"]
    assert result["outcome"]["max_damage"] == 0
    assert result["outcome"]["peak_temperature_c"] == 37
    assert result["outcome"]["event_counts"] == {}
    assert sum(action[5] > 0 for action in result["actions"]) == 8
    assert all(action[3:5] == [0, 0] for action in result["actions"])


@pytest.mark.parametrize("scenario_id", ["held-jaw-crush", "arterial-thermal-contact", "duct-thermal-contact"])
def test_recorded_damage_and_dose_never_recover(cases, scenario_id):
    result = cases[scenario_id]
    prior = result["initial"]["info"]["state"]["tissue"]["nodes"]
    for step in result["transitions"]:
        current = step["info"]["state"]["tissue"]["nodes"]
        for old, new in zip(prior, current, strict=True):
            for field in ("damage", "mechanical_damage", "thermal_dose", "thermal_damage"):
                assert new[field] >= old[field]
        prior = current


def test_one_pulse_then_cooling_does_not_heal_the_actual_environment(cases):
    # Replay only the unpowered approach plus one pulse, then change the
    # intervention to unpowered observation. The underlying model is unchanged.
    env = TissueRetractionEnv()
    try:
        observation, info = env.reset(seed=2)
        for step in cases["arterial-thermal-contact"]["transitions"]:
            observation, reward, terminated, truncated, info = env.step(step["action"])
            if step["phase"] == "heat":
                break
        target_id = cases["arterial-thermal-contact"]["controller_state"]["target_node_id"]
        hot = next(n for n in info["state"]["tissue"]["nodes"] if n["id"] == target_id)
        assert hot["thermal_damage"] > 0
        first_damage = hot["damage"]
        for _ in range(20):
            observation, reward, terminated, truncated, info = env.step(np.zeros(7))
            if terminated or truncated:
                break
        cooled = next(n for n in info["state"]["tissue"]["nodes"] if n["id"] == target_id)
        assert cooled["temperature_c"] < hot["temperature_c"]
        assert cooled["damage"] >= first_damage
        assert cooled["stiffness_n_per_mm"] <= hot["stiffness_n_per_mm"]
    finally:
        env.close()


def test_budget_and_horizon_stops_do_not_infer_expected_outcomes():
    limited = run_scenario("arterial-thermal-contact", max_steps=1)
    assert len(limited["actions"]) == 1
    assert limited["outcome"]["status"] == "step-budget"
    assert limited["outcome"]["physiology"]["blood_loss_ml"] == 0
    assert limited["outcome"]["termination_reason"] is None
    timeout = run_scenario("duct-thermal-contact", config={"max_steps": 20, "dt_s": 0.02})
    assert timeout["outcome"]["status"] == "truncated"
    assert timeout["outcome"]["termination_reason"] == "time_limit"
    assert timeout["outcome"]["physiology"]["bile_leak_ml"] == 0
    assert not timeout["outcome"]["severe_harm"]


@pytest.mark.parametrize("budget", [0, 201, True, 1.5, "20"])
def test_invalid_step_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="max_steps"):
        create_scenario_controller("gentle-traction", max_steps=budget)


def test_invalid_controller_state_is_rejected():
    state = create_scenario_controller("gentle-traction").state_dict()
    for replacement in ({"protocol_version": "future"}, {"steps": -1}, {"exposure_steps": 1},
                        {"completed": "yes"}, {"phase": "heal"}, {"target_node_id": "not-a-node"},
                        {"target_node_id": "n999"}):
        with pytest.raises(ValueError):
            ScenarioController.from_state({**deepcopy(state), **replacement})
    with pytest.raises(ValueError, match="Unknown response"):
        create_scenario_controller("nonexistent")
    with pytest.raises(ValueError, match="does not match"):
        create_scenario_controller("held-jaw-crush", state=state)
    with pytest.raises(ValueError, match="nonnegative integer"):
        run_scenario("gentle-traction", seed=True)


@pytest.mark.parametrize("scenario_id,command,sham_value", [
    ("arterial-thermal-contact", 5, 0.0),
    ("duct-thermal-contact", 5, 0.0),
    ("held-jaw-crush", 2, 0.55),
])
def test_matched_path_and_duration_controls_isolate_tool_command(cases, scenario_id, command, sham_value):
    actual = cases[scenario_id]
    env = TissueRetractionEnv(actual["config"])
    try:
        observation, info = env.reset(seed=actual["seed"], options={"patient": actual["patient"]})
        rewards = []
        for recorded in actual["transitions"]:
            action = list(recorded["action"])
            if action[command] > 0:
                action[command] = sham_value
            observation, reward, terminated, truncated, info = env.step(action)
            assert not (terminated or truncated)
            rewards.append(reward)
        assert info["state"]["max_damage"] == 0
        assert info["state"]["physiology"]["blood_loss_ml"] == 0
        assert info["state"]["physiology"]["bile_leak_ml"] == 0
        assert sum(rewards) > actual["outcome"]["cumulative_reward"]
    finally:
        env.close()


def test_release_after_subthreshold_compression_preserves_acquired_injury(cases):
    env = TissueRetractionEnv()
    try:
        observation, info = env.reset(seed=2)
        compressions = 0
        for recorded in cases["held-jaw-crush"]["transitions"]:
            observation, reward, terminated, truncated, info = env.step(recorded["action"])
            if recorded["phase"] == "compress":
                compressions += 1
            if compressions == 10:
                break
        acquired = info["state"]["max_damage"]
        assert 0 < acquired < env.config["injury_termination_threshold"]
        for _ in range(20):
            observation, reward, terminated, truncated, info = env.step([0, 0, -1, 0, 0, 0, 0])
            assert not (terminated or truncated)
            assert info["state"]["max_damage"] >= acquired
            assert info["state"]["tools"]["grasper"]["contact_node_id"] is None
    finally:
        env.close()
