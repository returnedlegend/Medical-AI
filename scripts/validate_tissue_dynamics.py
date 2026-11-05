#!/usr/bin/env python3
"""Offline numerical checks for the frozen analytic tissue surrogate.

No engine sources are modified. Closed-form checks verify implementation of
specified equations; numerical trajectory references do not validate biology.
Run: PYTHONPATH=backend .venv/bin/python scripts/validate_tissue_dynamics.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np

from retraction.environment import ENV_VERSION, PHYSICS_FINGERPRINT, TissueRetractionEnv
from retraction.tissue import TissueModel

TIMESTEPS = (0.02, 0.1, 0.2)
REFERENCE_DT = 0.005
PATIENT = {"attachment_length_mm": 24.0, "adhesion_count": 2,
           "adhesion_stiffness_n_per_mm": 0.08, "distension": 1.2}


def intervals(duration: float, dt: float) -> int:
    count = round(duration / dt)
    if not math.isclose(count * dt, duration, abs_tol=1e-10):
        raise ValueError("Duration must contain an integer number of timesteps")
    return count


def advance(model: TissueModel, dt: float, grasper=None, closure=0.0, cautery=None, power=0.0):
    return model.step(dt, np.array([6.0, 3.0]) if grasper is None else np.asarray(grasper), closure,
                      np.array([-25.0, 35.0]) if cautery is None else np.asarray(cautery), power, 2.5, 0.6)


def cooling_checks() -> list[dict[str, Any]]:
    """Exact Newton cooling and its positive-temperature-excess integral."""
    rows = []
    initial, ambient, tau, duration = 80.0, 37.0, 3.0, 6.0
    crossing = tau * math.log((initial - ambient) / (43 - ambient))
    upper = min(duration, crossing)
    exact_dose = ((ambient - 43) * upper + (initial - ambient) * tau *
                  (1 - math.exp(-upper / tau))) / 100
    for dt in TIMESTEPS:
        tissue = TissueModel(PATIENT)
        tissue.temperature[8] = initial  # prescribed initial condition, not a surgical action
        errors = []
        for step in range(intervals(duration, dt)):
            advance(tissue, dt)
            time = (step + 1) * dt
            exact = ambient + (initial - ambient) * math.exp(-time / tau)
            errors.append(abs(tissue.temperature[8] - exact))
        rows.append({"dt_s": dt, "duration_s": duration, "initial_temperature_c": initial,
                     "final_temperature_c": float(tissue.temperature[8]),
                     "exact_final_temperature_c": ambient + (initial - ambient) * math.exp(-duration / tau),
                     "max_temperature_error_c": float(max(errors)), "final_temperature_error_c": float(errors[-1]),
                     "thermal_dose": float(tissue.thermal_dose[8]), "exact_thermal_dose": exact_dose,
                     "thermal_dose_error": float(abs(tissue.thermal_dose[8] - exact_dose)),
                     "reference": "T=37+43*exp(-t/3); dose is integral max(T-43,0)/100 dt"})
    return rows


def component_force_checks() -> list[dict[str, Any]]:
    """Isolate one spring/contact degree of freedom and compare exact ODE."""
    rows = []
    for injured in (False, True):
        tissue = TissueModel(PATIENT)
        i, closure, pull_mm, duration, dt = 3, 0.55, 0.5, 4.0, 0.02
        if injured:
            tissue.acquire_contact(tissue.rest[i], 1.0)
            for _ in range(12):
                advance(tissue, 0.1, grasper=tissue.rest[i], closure=1.0)
            tissue.release_contact()
        damage = float(tissue.damage[i])
        # Controlled component fixture: remove neighbor springs to expose a
        # single analytically solvable contact+restoring spring, not a patient.
        tissue.edges = np.empty((0, 2), dtype=int)
        tissue.rest_lengths = np.empty(0)
        tissue.acquire_contact(tissue.rest[i], closure)
        initial_position = tissue.positions[i].copy()
        rest_k = 0.08 * (1 - 0.9 * damage)  # EA*D/L at the reference patient/site
        contact_k = 1.1 * closure * (1 - 0.8 * damage)
        equilibrium = contact_k / (rest_k + contact_k) * pull_mm
        exact_u = equilibrium * (1 - math.exp(-(rest_k + contact_k) * duration / 0.65))
        exact_force = contact_k * (pull_mm - exact_u)
        target = tissue.rest[i] + [pull_mm, 0]
        for _ in range(intervals(duration, dt)):
            advance(tissue, dt, grasper=target, closure=closure)
        observed_u = float(tissue.positions[i, 0] - initial_position[0])
        rows.append({"fixture": "previously_crushed" if injured else "intact", "dt_s": dt,
                     "duration_s": duration, "damage_before": damage, "damage_after": float(tissue.damage[i]),
                     "restoring_stiffness_n_per_mm": float(tissue.stiffness[i]),
                     "contact_stiffness_n_per_mm": contact_k,
                     "observed_displacement_mm": observed_u, "exact_displacement_mm": exact_u,
                     "displacement_error_mm": abs(observed_u - exact_u),
                     "observed_force_n": tissue.contact_force_n, "exact_force_n": exact_force,
                     "force_error_n": abs(tissue.contact_force_n - exact_force),
                     "relative_force_error": abs(tissue.contact_force_n - exact_force) / exact_force,
                     "reference": "du/dt=(kc*pull-(kr+kc)*u)/0.65; injury held naturally constant below activation limits"})
    return rows


def blood_loss_checks() -> list[dict[str, Any]]:
    """Conservation and exact pressure-dependent leak ODE before saturation."""
    rows = []
    duration, damage, coefficient = 30.0, 0.3, 55 * (0.3 - 0.08)
    # B'=C(1-B/2000) as long as neither pressure clamp nor reservoir limit binds.
    exact_loss = 2000 * (1 - math.exp(-coefficient * duration / 2000))
    for dt in TIMESTEPS:
        tissue = TissueModel(PATIENT)
        tissue.mechanical_damage[16] = damage  # prescribed persistent arterial wound
        integrated_flow = 0.0
        max_balance_error = 0.0
        minimum_loss_increment = float("inf")
        for _ in range(intervals(duration, dt)):
            before = tissue.blood_loss_ml
            advance(tissue, dt)
            integrated_flow += tissue.bleeding_rate * dt
            max_balance_error = max(max_balance_error, abs(tissue.blood_loss_ml - integrated_flow))
            minimum_loss_increment = min(minimum_loss_increment, tissue.blood_loss_ml - before)
        pressure_exact = 90 * (1 - exact_loss / 2000)
        heart_rate_exact = 72 + 180 * exact_loss / 5000
        rows.append({"dt_s": dt, "duration_s": duration, "arterial_damage": damage,
                     "blood_loss_ml": tissue.blood_loss_ml, "exact_blood_loss_ml": exact_loss,
                     "blood_loss_error_ml": abs(tissue.blood_loss_ml - exact_loss),
                     "max_discrete_mass_balance_error_ml": max_balance_error,
                     "minimum_loss_increment_ml": minimum_loss_increment,
                     "last_interval_flow_ml_per_s": tissue.bleeding_rate,
                     "initial_flow_ml_per_s": coefficient,
                     "mean_arterial_pressure_mmhg": tissue.mean_arterial_pressure,
                     "exact_pressure_mmhg": pressure_exact,
                     "pressure_error_mmhg": abs(tissue.mean_arterial_pressure - pressure_exact),
                     "heart_rate_bpm": tissue.heart_rate, "exact_heart_rate_bpm": heart_rate_exact,
                     "pressure_floor_reached": tissue.mean_arterial_pressure <= 20,
                     "reference": "B=2000*(1-exp(-C*t/2000)), C=55*(0.3-0.08); no healing, no coagulation, no saturation"})
    return rows


def fixed_pull(patient: dict[str, Any], dt: float) -> dict[str, Any]:
    duration, velocity = 4.0, 1.0  # same physical command at every resolution
    tissue = TissueModel(patient)
    start = tissue.rest[3].copy()
    tissue.acquire_contact(start, 0.55)
    samples = []
    for step in range(intervals(duration, dt)):
        time = (step + 1) * dt
        advance(tissue, dt, grasper=start + [velocity * time, 0], closure=0.55)
        if math.isclose(time, round(time), abs_tol=1e-9):
            samples.append({"time_s": time, "positions": tissue.positions.copy(),
                            "force_n": tissue.contact_force_n, "exposure": tissue.exposure()})
    return {"samples": samples, "max_damage": float(tissue.damage.max()),
            "max_strain": float(tissue.strain.max()), "final_force_n": tissue.contact_force_n,
            "final_exposure": tissue.exposure()}


def trajectory_checks(patient_count: int, first_seed: int) -> dict[str, Any]:
    rows = []
    errors: dict[float, list[dict[str, float]]] = {dt: [] for dt in TIMESTEPS}
    for seed in range(first_seed, first_seed + patient_count):
        env = TissueRetractionEnv()
        env.reset(seed=seed)
        patient = dict(env.patient)
        reference = fixed_pull(patient, REFERENCE_DT)
        patient_rows = []
        for dt in TIMESTEPS:
            result = fixed_pull(patient, dt)
            rms, maximum, force, exposure = [], [], [], []
            for current, fine in zip(result["samples"], reference["samples"]):
                delta = current["positions"] - fine["positions"]
                distances = np.linalg.norm(delta, axis=1)
                rms.append(float(np.sqrt(np.mean(distances**2))))
                maximum.append(float(distances.max()))
                force.append(abs(current["force_n"] - fine["force_n"]))
                exposure.append(abs(current["exposure"] - fine["exposure"]))
            metrics = {"max_sample_rms_position_error_mm": max(rms),
                       "max_sample_node_position_error_mm": max(maximum),
                       "max_sample_force_error_n": max(force),
                       "max_sample_exposure_error": max(exposure),
                       "max_damage": result["max_damage"], "final_force_n": result["final_force_n"],
                       "final_exposure": result["final_exposure"]}
            errors[dt].append(metrics)
            patient_rows.append({"dt_s": dt, **metrics})
        rows.append({"seed": seed, "patient": patient, "results": patient_rows})
    aggregate = []
    for dt in TIMESTEPS:
        aggregate.append({"dt_s": dt, "patient_count": patient_count,
                          **{f"worst_{key}": max(row[key] for row in errors[dt]) for key in errors[dt][0]
                             if key.startswith("max_sample_") or key == "max_damage"},
                          "maximum_final_force_n": max(row["final_force_n"] for row in errors[dt]),
                          "minimum_final_exposure": min(row["final_exposure"] for row in errors[dt]),
                          "maximum_final_exposure": max(row["final_exposure"] for row in errors[dt]),
                          "mean_max_sample_rms_position_error_mm": float(np.mean([
                              row["max_sample_rms_position_error_mm"] for row in errors[dt]]))})
    return {"duration_s": 4, "grasper_velocity_mm_per_s": [1, 0], "closure": 0.55,
            "reference_dt_s": REFERENCE_DT, "sample_times_s": [1, 2, 3, 4],
            "reference_kind": "Fine numerical integration of the same surrogate, not independent physical ground truth.",
            "input_sampling": "Tool position is the same constant-velocity trajectory sampled and held per action interval; differences include input sample/hold and integration error, with no policy decisions.",
            "aggregate": aggregate, "patients": rows}


def cautery_stop_checks() -> list[dict[str, Any]]:
    # At the stationary contact site and before the temperature cap:
    # T(t)=37+480*(1-exp(-t/3)); integrate T-43 after its first crossing.
    crossing = -3 * math.log(474 / 480)
    dose_threshold = -0.8 * math.log(1 - 0.35)
    def exact_heating_dose(time):
        if time <= crossing:
            return 0.0
        return (474 * (time - crossing) + 1440 * (math.exp(-time / 3) - math.exp(-crossing / 3))) / 100
    lower, upper = crossing, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if exact_heating_dose(middle) < dose_threshold:
            lower = middle
        else:
            upper = middle
    exact_injury_time = (lower + upper) / 2
    exact_event_temperature = 37 + 480 * (1 - math.exp(-exact_injury_time / 3))
    if exact_event_temperature >= 180:
        raise RuntimeError("Analytic heating reference crossed the model temperature clamp")
    rows = []
    for dt in TIMESTEPS:
        movement_s, planned_heating_s = 1.2, 5.0
        env = TissueRetractionEnv({"dt_s": dt, "max_steps": intervals(movement_s + planned_heating_s, dt),
                                  "randomize_anatomy": False})
        _, _ = env.reset(seed=42)
        for _ in range(intervals(movement_s, dt)):
            _, _, terminated, truncated, _ = env.step([0, 0, 0, 1, 1 / 3, 0, 0])
            if terminated or truncated:
                raise RuntimeError("Unexpected termination during unpowered tool positioning")
        heating_steps = 0
        for _ in range(intervals(planned_heating_s, dt)):
            _, _, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 1, 0])
            heating_steps += 1
            if terminated or truncated:
                break
        rows.append({"dt_s": dt, "planned_heating_duration_s": planned_heating_s,
                     "actual_heating_duration_s": heating_steps * dt,
                     "continuous_heating_injury_time_s": exact_injury_time,
                     "heating_stop_time_error_s": heating_steps * dt - exact_injury_time,
                     "absolute_heating_stop_time_error_s": abs(heating_steps * dt - exact_injury_time),
                     "continuous_event_temperature_c": exact_event_temperature,
                     "damage_threshold_overshoot": info["state"]["critical_damage"] - 0.35,
                     "completed_planned_heating": heating_steps == intervals(planned_heating_s, dt),
                     "termination_time_s": env.time_s, "termination_reason": info["termination_reason"],
                     "terminated": terminated, "truncated": truncated,
                     "max_damage": info["state"]["max_damage"],
                     "critical_damage": info["state"]["critical_damage"],
                     "max_temperature_c": float(env.tissue.temperature.max()),
                     "blood_loss_ml": env.tissue.blood_loss_ml,
                     "bleeding_rate_ml_per_s": env.tissue.bleeding_rate,
                     "mean_arterial_pressure_mmhg": env.tissue.mean_arterial_pressure,
                     "return": info["cumulative_reward"],
                     "caution": "Stops are evaluated at action boundaries. Different stopping times and damage overshoot are retained; no rescaling to a common successful outcome."})
    return rows


def run_validation(patient_count=20, first_seed=10000) -> dict[str, Any]:
    if not 1 <= patient_count <= 100 or first_seed < 0:
        raise ValueError("patient_count must be 1..100 and first_seed nonnegative")
    cooling = cooling_checks()
    force = component_force_checks()
    blood = blood_loss_checks()
    trajectory = trajectory_checks(patient_count, first_seed)
    cautery = cautery_stop_checks()
    # These are implementation verification checks against independently solved
    # equations, not empirical tolerances for an actual organ or surgery.
    checks = {
        "cooling_errors_decrease_with_dt": cooling[0]["max_temperature_error_c"] < cooling[1]["max_temperature_error_c"] < cooling[2]["max_temperature_error_c"],
        "cooling_smallest_dt_error_below_point1c": cooling[0]["max_temperature_error_c"] < 0.1,
        "force_component_relative_error_below_1pct": all(row["relative_force_error"] < 0.01 for row in force),
        "component_injury_lowers_stiffness_and_force": force[1]["restoring_stiffness_n_per_mm"] < force[0]["restoring_stiffness_n_per_mm"] and force[1]["observed_force_n"] < force[0]["observed_force_n"],
        "component_injury_did_not_change_during_reference_test": all(row["damage_before"] == row["damage_after"] for row in force),
        "blood_conservation_residual_below_1e_minus8ml": all(row["max_discrete_mass_balance_error_ml"] < 1e-8 for row in blood),
        "blood_loss_monotone_and_flow_reduces": all(row["minimum_loss_increment_ml"] >= 0 and row["last_interval_flow_ml_per_s"] < row["initial_flow_ml_per_s"] for row in blood),
        "blood_errors_decrease_with_dt": blood[0]["blood_loss_error_ml"] < blood[1]["blood_loss_error_ml"] < blood[2]["blood_loss_error_ml"],
        "gentle_trajectory_has_no_damage": all(row["worst_max_damage"] == 0 for row in trajectory["aggregate"]),
        "gentle_position_errors_decrease_with_dt": all(
            trajectory["aggregate"][i]["worst_max_sample_rms_position_error_mm"] <
            trajectory["aggregate"][i + 1]["worst_max_sample_rms_position_error_mm"] for i in range(2)),
        "cautery_event_time_errors_below_one_step": all(row["absolute_heating_stop_time_error_s"] < row["dt_s"] for row in cautery),
        "cautery_stops_reported_as_failures": all(row["terminated"] and row["termination_reason"] == "severe_injury" and not row["completed_planned_heating"] for row in cautery),
    }
    return {"purpose": "Numerical implementation verification, not biological or clinical validation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment_version": ENV_VERSION, "physics_fingerprint": PHYSICS_FINGERPRINT,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "patient_seeds": [first_seed, first_seed + patient_count - 1],
            "all_implementation_checks_passed": all(checks.values()), "checks": checks,
            "cooling": cooling, "force_component": force, "blood_loss": blood,
            "fixed_command_trajectory": trajectory, "cautery_terminal_sensitivity": cautery,
            "limitations": ["All constants, damage thresholds and physiological responses are illustrative and unvalidated.",
                            "The network has 23 lumped sites; no FEM, geometric mesh convergence or patient anatomy validation is performed.",
                            "Analytic references check the stated equations, not their applicability to human tissue.",
                            "Fine numerical references share the same model and are not independent physics validation.",
                            "Timestep-sensitive injury stops occur only at action boundaries; coarser intervals can overshoot damage thresholds.",
                            "The prescribed vessel wound and isolated spring fixture are controlled mathematical tests, not surgical interventions.",
                            "The arterial response coefficients are fixed across patients; only the gentle gallbladder pull varies the sampled anatomy."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/numerical-validation.json"))
    args = parser.parse_args()
    report = run_validation(args.patients, args.first_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "all_implementation_checks_passed": report["all_implementation_checks_passed"],
                      "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if report["all_implementation_checks_passed"] else 1)


if __name__ == "__main__":
    main()
