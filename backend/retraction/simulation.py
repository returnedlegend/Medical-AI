"""Deterministic analytic retraction surrogate, not a surgical physics simulator.

The model is deliberately small and inspectable: a nonlinear spring law gives
force and strain; an angular projection gives a synthetic exposure-area proxy.
Its parameters are illustrative assumptions, not fitted tissue measurements.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

MODEL_VERSION = "analytic-surrogate-1.0"
THRESHOLD_PROFILES = {
    "conservative": {"force_limit_n": 1.8, "strain_limit": 0.45, "exposure_threshold": 0.65},
    "nominal": {"force_limit_n": 2.5, "strain_limit": 0.60, "exposure_threshold": 0.60},
    "permissive": {"force_limit_n": 3.2, "strain_limit": 0.75, "exposure_threshold": 0.55},
}

DEFAULT_CONFIG = {
    "n_patients": 240,
    "seed": 42,
    "threshold_profile": "nominal",
    "attachment_min_mm": 8.0,
    "attachment_max_mm": 35.0,
    "adhesion_max_count": 6,
    "adhesion_stiffness_min": 0.03,
    "adhesion_stiffness_max": 0.20,
    "distension_min": 0.8,
    "distension_max": 1.8,
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number, not a boolean or string")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(config or {})
    if "cohort_size" in supplied and "n_patients" not in supplied:
        supplied["n_patients"] = supplied.pop("cohort_size")
    result = {**DEFAULT_CONFIG, **supplied}
    n = result["n_patients"]
    seed = result["seed"]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 40 <= n <= 2000:
        raise ValueError("n_patients must be an integer from 40 to 2000")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 to 4294967295")
    if result["threshold_profile"] not in THRESHOLD_PROFILES:
        raise ValueError("Unknown threshold_profile; choose conservative, nominal, or permissive")
    for lo, hi, lower, upper in [
        ("attachment_min_mm", "attachment_max_mm", 4, 60),
        ("adhesion_stiffness_min", "adhesion_stiffness_max", 0, 0.5),
        ("distension_min", "distension_max", 0.5, 2.5),
    ]:
        if not (lower <= _finite_number(result[lo], lo) <= _finite_number(result[hi], hi) <= upper):
            raise ValueError(f"{lo} and {hi} must be ordered in [{lower}, {upper}]")
    a = result["adhesion_max_count"]
    if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= a <= 12:
        raise ValueError("adhesion_max_count must be an integer from 0 to 12")
    thresholds = dict(THRESHOLD_PROFILES[result["threshold_profile"]])
    for key in thresholds:
        if key in supplied and supplied[key] is not None:
            thresholds[key] = _finite_number(supplied[key], key)
    if not (0 < thresholds["force_limit_n"] <= 20):
        raise ValueError("force_limit_n must be in (0, 20]")
    if not (0 < thresholds["strain_limit"] <= 3):
        raise ValueError("strain_limit must be in (0, 3]")
    if not (0 < thresholds["exposure_threshold"] <= 1):
        raise ValueError("exposure_threshold must be in (0, 1]")
    result.update(thresholds)
    return result


def strategy_grid() -> list[dict[str, Any]]:
    return [
        {"id": f"g{int(g * 100)}-a{angle}-d{distance}", "grasp_position": g,
         "direction_deg": angle, "distance_mm": distance}
        for g in (0.35, 0.55, 0.75)
        for angle in (15, 35, 55, 75, 95)
        for distance in (4, 8, 12, 16, 20)
    ]


STANDARD_STRATEGY = {"id": "g75-a35-d12", "grasp_position": 0.75,
                     "direction_deg": 35, "distance_mm": 12}


def generate_cohort(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = normalize_config(config)
    rng = np.random.default_rng(config["seed"])
    patients = []
    for i in range(config["n_patients"]):
        length = float(rng.uniform(config["attachment_min_mm"], config["attachment_max_mm"]))
        count = int(rng.integers(0, config["adhesion_max_count"] + 1))
        stiffness = float(rng.uniform(config["adhesion_stiffness_min"], config["adhesion_stiffness_max"]))
        distension = float(rng.uniform(config["distension_min"], config["distension_max"]))
        preferred = float(np.clip(25 + 0.95 * (35 - length) + 4.5 * count - 9 * (distension - 1), 10, 90))
        patients.append({"id": f"P{i + 1:04d}", "attachment_length_mm": length,
                         "adhesion_count": count, "adhesion_stiffness_n_per_mm": stiffness,
                         "distension": distension, "preferred_direction_deg": preferred})
    return patients


def validate_patient(patient: dict[str, Any]) -> None:
    for key, lo, hi in [("attachment_length_mm", 4, 60), ("adhesion_count", 0, 12),
                        ("adhesion_stiffness_n_per_mm", 0, 0.5), ("distension", 0.5, 2.5)]:
        value = _finite_number(patient[key], key)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} must be finite and in [{lo}, {hi}]")
    if "preferred_direction_deg" in patient and not 0 <= _finite_number(patient["preferred_direction_deg"], "preferred_direction_deg") <= 180:
        raise ValueError("preferred_direction_deg must be in [0, 180]")
    if float(patient["adhesion_count"]) != int(patient["adhesion_count"]):
        raise ValueError("adhesion_count must be an integer")


def _mechanics(patient: dict[str, Any], grasp: np.ndarray | float,
               angle: np.ndarray | float, distance: np.ndarray | float) -> tuple[Any, Any, Any]:
    length = patient["attachment_length_mm"]
    count = patient["adhesion_count"]
    adhesion_k = patient["adhesion_stiffness_n_per_mm"]
    distension = patient["distension"]
    preferred = patient.get("preferred_direction_deg", float(np.clip(
        25 + 0.95 * (35 - length) + 4.5 * count - 9 * (distension - 1), 10, 90)))
    mismatch = np.deg2rad(np.asarray(angle) - preferred)
    # Alignment is smooth and positive on the tested angular grid.
    alignment = np.clip(np.cos(mismatch), 0, 1) ** 2
    extension = np.asarray(distance) * (0.42 + 0.58 * np.asarray(grasp))
    strain = extension / length
    # EA=1.6 N is illustrative. Adhesion springs act in parallel and engage
    # more strongly when pulling across their preferred direction.
    stiffness = 1.6 * distension / length + count * adhesion_k * (0.28 + 0.72 * (1 - alignment))
    force = stiffness * extension * (1 + 1.3 * strain**2)
    opening = np.maximum(0, np.asarray(distance) * (0.40 + 0.80 * np.asarray(grasp)) * alignment
                         / (0.80 + 0.25 * distension) - 0.38 * count)
    # Normalized projected triangle area: base=12 mm, reference height=15 mm.
    exposure = np.clip(opening / 15.0, 0, 1)
    return force, strain, exposure


def simulate_patient(patient: dict[str, Any], strategy: dict[str, Any],
                     thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_patient(patient)
    g, a, d = (_finite_number(strategy[k], k) for k in ("grasp_position", "direction_deg", "distance_mm"))
    if not (math.isfinite(g) and 0 <= g <= 1 and math.isfinite(a) and 0 <= a <= 180
            and math.isfinite(d) and 0 <= d <= 30):
        raise ValueError("Strategy needs grasp_position in [0,1], direction_deg in [0,180], distance_mm in [0,30]")
    limits = {**THRESHOLD_PROFILES["nominal"], **(thresholds or {})}
    limits = {key: _finite_number(value, key) for key, value in limits.items()}
    if any(value <= 0 for value in limits.values()) or limits["exposure_threshold"] > 1:
        raise ValueError("Thresholds must be positive and exposure_threshold at most 1")
    f, s, e = _mechanics(patient, g, a, d)
    force, strain, exposure = float(f), float(s), float(e)
    force_ok = force <= limits["force_limit_n"]
    strain_ok = strain <= limits["strain_limit"]
    exposure_ok = exposure >= limits["exposure_threshold"]
    return {"success": bool(force_ok and strain_ok and exposure_ok), "force_n": force,
            "strain": strain, "exposure": exposure, "force_ok": bool(force_ok),
            "strain_ok": bool(strain_ok), "exposure_ok": bool(exposure_ok),
            "projected_area_mm2": exposure * 90.0, "strategy": dict(strategy)}


def force_probe(patient: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """A fixed 2-mm, one-second ramp tug; no oracle information in features."""
    validate_patient(patient)
    limits = {**THRESHOLD_PROFILES["nominal"], **(thresholds or {})}
    times = np.linspace(0, 1, 21)
    distance = 2.0 * times
    forces, strains, _ = _mechanics(patient, 0.75, 35, distance)
    return {"times_s": times.tolist(), "forces_n": forces.tolist(),
            "peak_force_n": float(forces[-1]),
            "stiffness_n_per_mm": float(forces[1] / distance[1]),
            "nonlinearity_ratio": float(forces[-1] / (2 * forces[10]) - 1) if forces[10] else 0.0,
            "work_mj": float(np.trapezoid(forces, distance)),
            "duration_s": 1.0, "distance_mm": 2.0, "peak_strain": float(strains[-1]),
            "force_ok": bool(forces[-1] <= limits["force_limit_n"]),
            "strain_ok": bool(strains[-1] <= limits["strain_limit"]),
            "within_limits": bool(forces[-1] <= limits["force_limit_n"] and strains[-1] <= limits["strain_limit"]),
            "protocol": "Fixed ramp with no force-triggered stop; limit violations retained and reported."}


def evaluate_grid(patients: list[dict[str, Any]], strategies: list[dict[str, Any]],
                  limits: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grasps = np.array([s["grasp_position"] for s in strategies])
    angles = np.array([s["direction_deg"] for s in strategies])
    distances = np.array([s["distance_mm"] for s in strategies])
    forces, strains, exposures = [], [], []
    for patient in patients:
        force, strain, exposure = _mechanics(patient, grasps, angles, distances)
        forces.append(force); strains.append(strain); exposures.append(exposure)
    f, s, e = np.asarray(forces), np.asarray(strains), np.asarray(exposures)
    success = (f <= limits["force_limit_n"]) & (s <= limits["strain_limit"]) & (e >= limits["exposure_threshold"])
    return success, f, s, e


def _oracle_indices(success: np.ndarray, forces: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    # On success prefer lower force, then higher exposure. On infeasibility
    # return maximum exposure ONLY AMONG force/strain-safe candidates in caller.
    return np.array([int(np.flatnonzero(row)[np.argmin(forces[i, row])])
                     if row.any() else int(np.argmax(exposures[i]))
                     for i, row in enumerate(success)])


def run_experiment(config: dict[str, Any] | None = None,
                   progress_callback: Callable[..., Any] | None = None) -> dict[str, Any]:
    from .analysis import analyze_probe, analyze_subgroups, paired_bootstrap_ci
    cfg = normalize_config(config)
    patients, strategies = generate_cohort(cfg), strategy_grid()
    limits = {k: cfg[k] for k in THRESHOLD_PROFILES["nominal"]}
    rng = np.random.default_rng(cfg["seed"] + 1)
    shuffled = rng.permutation(len(patients))
    train_size = int(len(patients) * 0.7)
    train_idx, test_idx = shuffled[:train_size], shuffled[train_size:]
    train_set = set(train_idx.tolist())
    if progress_callback:
        progress_callback(0.1, "Generated synthetic cohort")
    success, forces, strains, exposures = evaluate_grid(patients, strategies, limits)
    # Tie break is grid order, fixed before inspecting held-out cases.
    shared_idx = int(np.argmax(success[train_idx].mean(axis=0)))
    standard_idx = next(i for i, s in enumerate(strategies) if s["id"] == STANDARD_STRATEGY["id"])
    oracle_idx = _oracle_indices(success, forces, exposures)
    feasible = (forces <= limits["force_limit_n"]) & (strains <= limits["strain_limit"])
    for i in range(len(patients)):
        if not success[i].any():
            # If no success exists show the safest feasible maximum-exposure
            # candidate, retaining failure status; zero-distance not in grid.
            safe = np.flatnonzero(feasible[i])
            oracle_idx[i] = int(safe[np.argmax(exposures[i, safe])]) if len(safe) else int(np.argmin(forces[i]))
    oracle_success = success.any(axis=1)
    shared_success = success[:, shared_idx]
    standard_success = success[:, standard_idx]
    needs_personalization = oracle_success & ~shared_success
    bailout = ~oracle_success
    probes = [force_probe(p, limits) for p in patients]
    probe_result, predictions, bailout_predictions = analyze_probe(
        probes, needs_personalization, bailout, train_idx, test_idx, cfg["seed"])
    records = []
    for i, patient in enumerate(patients):
        def outcome(j: int) -> dict[str, Any]:
            return {"success": bool(success[i, j]), "force_n": float(forces[i, j]),
                    "strain": float(strains[i, j]), "exposure": float(exposures[i, j]),
                    "force_ok": bool(forces[i, j] <= limits["force_limit_n"]),
                    "strain_ok": bool(strains[i, j] <= limits["strain_limit"]),
                    "exposure_ok": bool(exposures[i, j] >= limits["exposure_threshold"]),
                    "projected_area_mm2": float(exposures[i, j] * 90), "strategy": strategies[j]}
        records.append({**patient, "split": "train" if i in train_set else "test",
                        "category": "bail_out" if bailout[i] else "personalize" if needs_personalization[i] else "standard",
                        "standard": outcome(standard_idx), "shared": outcome(shared_idx),
                        "oracle": outcome(int(oracle_idx[i])), "probe": probes[i],
                        "personalization_probability": float(predictions[i]) if predictions is not None else None,
                        "bailout_probability": float(bailout_predictions[i]) if bailout_predictions is not None else None})
    if progress_callback:
        progress_callback(0.7, "Evaluated finite strategy grid and held-out force-probe models")
    summary = {"n_patients": len(patients), "train_size": train_size, "test_size": len(test_idx),
               "strategy_count": len(strategies), "standard_success_rate": float(standard_success[test_idx].mean()),
               "shared_success_rate": float(shared_success[test_idx].mean()),
               "oracle_success_rate": float(oracle_success[test_idx].mean()),
               "personalization_gain": float(needs_personalization[test_idx].mean()),
               "bailout_rate": float(bailout[test_idx].mean()),
               "probe_violation_rate": float(np.mean([not probes[i]["within_limits"] for i in test_idx])),
               "personalization_gain_ci": paired_bootstrap_ci(needs_personalization[test_idx], cfg["seed"]),
               "best_shared_strategy": strategies[shared_idx],
               "categories": {"standard": int(shared_success[test_idx].sum()),
                              "personalize": int(needs_personalization[test_idx].sum()),
                              "bail_out": int(bailout[test_idx].sum())}}
    sensitivity = []
    for name, profile in THRESHOLD_PROFILES.items():
        ok = (forces <= profile["force_limit_n"]) & (strains <= profile["strain_limit"]) & (exposures >= profile["exposure_threshold"])
        shared = int(np.argmax(ok[train_idx].mean(axis=0)))
        oracle = ok.any(axis=1)
        sensitivity.append({"profile": name, **profile,
                            "shared_success_rate": float(ok[test_idx, shared].mean()),
                            "oracle_success_rate": float(oracle[test_idx].mean()),
                            "personalization_gain": float((oracle[test_idx] & ~ok[test_idx, shared]).mean()),
                            "bailout_rate": float((~oracle[test_idx]).mean()),
                            "best_shared_strategy": strategies[shared]})
    result = {"engine": "analytic-surrogate", "model_version": MODEL_VERSION, "config": cfg,
              "thresholds": limits, "summary": summary, "patients": records, "strategies": strategies,
              "probe": probe_result, "sensitivity": sensitivity,
              "subgroups": analyze_subgroups(records, cfg["seed"]),
              "methodology": {
                  "scope": "Exploratory synthetic research only. Not LapGym/SOFA, not patient data, and not evidence of clinical safety.",
                  "oracle": "Exhaustive optimum over 75 discrete strategies, not a continuous or clinical optimum. Lowest-force successful strategy selected; if none succeeds, show the highest-exposure force/strain-feasible strategy or lowest force if none feasible.",
                  "split": "Seeded 70% training / 30% held-out test, assigned before strategy selection and model fitting. Summary, confidence intervals, calibration and AUROC use held-out cases only.",
                  "shared_strategy": "Best single strategy chosen by training-set success rate; ties resolved by predefined grid order. Canonical standard strategy is also reported separately.",
                  "mechanics": "Illustrative nonlinear parallel springs: extension=d(0.42+0.58g); strain=extension/L; k=1.6D/L + count*adhesion_k*(0.28+0.72*(1-alignment)); force=k*extension*(1+1.3*strain^2). Alignment=cos(angle-preferred_angle)^2 clipped at zero.",
                  "exposure": "Synthetic normalized projected triangle area (reference 90 mm²); no anatomy, camera rendering, segmentation, or clinical CVS verification.",
                  "probe": "Fixed 2 mm ramp in 1 second at grasp=0.75 and angle=35 degrees. Features derived only from this response; no anatomy or oracle outcomes used as input. Noise-free observations are optimistic.",
                  "early_warning_target": "Shared strategy fails but at least one grid strategy succeeds. This target is separate from grid infeasibility (bail-out).",
                  "subgroups": "Descriptive held-out strata with fixed illustrative cutoffs: attachment <15 / 15–<25 / ≥25 mm; adhesions 0 / 1–3 / ≥4; distension <1.2 / ≥1.2. These are not clinical grades. Each dimension partitions test cases; dimensions overlap. Bootstrap intervals are unadjusted for multiple comparisons and can be unstable in small strata. Empty strata report null rates and intervals.",
                  "uncertainty": "Paired patient bootstrap, 1000 replicates, 95% percentile interval, shared strategy held fixed; does not include simulator/model uncertainty or selection variability.",
                  "limitations": ["Constants and thresholds are illustrative, unvalidated, and not calibrated to real tissue.", "Fixed fundus support is assumed; no mesh, dynamics, surgical anatomy, bleeding or inflammation.", "Bail-out means infeasible in this finite surrogate grid, not clinical inability to retract safely.", "Preferred direction is explicitly computed from anatomy parameters; personalization heterogeneity is partly imposed by model construction. Results exercise the analysis pipeline, not the biological hypothesis.", "Fixed probe ramps are not force-stopped. Cases exceeding the selected force or strain limits remain in model training/evaluation and are flagged in the data.", "Probe model is logistic regression; reinforcement learning and five-agent disagreement are not implemented.", "Threshold sensitivity is exploratory, not a preregistered experiment; no mesh-convergence claim."]}}
    if progress_callback:
        progress_callback(1.0, "Experiment complete")
    return result
