"""Leakage-aware, held-out statistics for the analytic surrogate."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["peak_force_n", "stiffness_n_per_mm", "nonlinearity_ratio", "work_mj"]


def paired_bootstrap_ci(difference: np.ndarray, seed: int, replicates: int = 1000) -> list[float]:
    rng = np.random.default_rng(seed + 2)
    values = np.asarray(difference, dtype=float)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _fit_evaluate(features: np.ndarray, target: np.ndarray, train_idx: np.ndarray,
                  test_idx: np.ndarray, seed: int) -> tuple[dict[str, Any], np.ndarray | None]:
    train_y = target[train_idx].astype(int)
    test_y = target[test_idx].astype(int)
    base = {"train_positive_count": int(train_y.sum()), "test_positive_count": int(test_y.sum()),
            "train_count": len(train_idx), "test_count": len(test_idx),
            "auroc": None, "brier_score": None, "ece": None, "calibration": [], "roc": [],
            "coefficients": [], "status": "unavailable", "reason": None}
    if len(np.unique(train_y)) < 2:
        return {**base, "reason": "Training target has one class; no classifier or probability estimates fitted."}, None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    model.fit(features[train_idx], train_y)
    predictions = model.predict_proba(features)[:, 1]
    probs = predictions[test_idx]
    bins = []
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        n = int(selected.sum())
        predicted = float(probs[selected].mean()) if n else None
        observed = float(test_y[selected].mean()) if n else None
        bins.append({"lower": float(lo), "upper": float(hi), "count": n,
                     "mean_predicted": predicted, "observed_rate": observed})
        if n:
            ece += n / len(test_y) * abs(predicted - observed)
    auroc = None
    roc = []
    if len(np.unique(test_y)) == 2:
        auroc = float(roc_auc_score(test_y, probs))
        fpr, tpr, _ = roc_curve(test_y, probs)
        roc = [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
    flags = probs >= 0.5
    true_positives = int((flags & (test_y == 1)).sum())
    metrics = {**base, "status": "ok" if auroc is not None else "single_class_test",
               "reason": None if auroc is not None else "Held-out target has one class; AUROC is undefined.",
               "auroc": auroc, "brier_score": float(brier_score_loss(test_y, probs)), "ece": float(ece),
               "calibration": bins, "roc": roc,
               "coefficients": [{"feature": name, "coefficient": float(coef)} for name, coef in
                                zip(FEATURE_NAMES, model.named_steps["logisticregression"].coef_[0])],
               "handoff": {"threshold": 0.5, "flagged_count": int(flags.sum()),
                           "flagged_fraction": float(flags.mean()),
                           "precision": true_positives / int(flags.sum()) if flags.any() else None,
                           "recall": true_positives / int(test_y.sum()) if test_y.sum() else None,
                           "true_positives": true_positives,
                           "false_positives": int((flags & (test_y == 0)).sum()),
                           "false_negatives": int((~flags & (test_y == 1)).sum()),
                           "true_negatives": int((~flags & (test_y == 0)).sum())}}
    return metrics, predictions


def analyze_probe(probes: list[dict[str, Any]], needs_personalization: np.ndarray,
                  bailout: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                  seed: int) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    features = np.array([[probe[key] for key in FEATURE_NAMES] for probe in probes])
    primary, predictions = _fit_evaluate(features, needs_personalization, train_idx, test_idx, seed)
    infeasibility, bailout_predictions = _fit_evaluate(features, bailout, train_idx, test_idx, seed)
    return {**primary, "bailout": infeasibility, "model": "logistic_regression",
            "feature_names": FEATURE_NAMES, "decision_threshold": 0.5,
            "interpretation": "Held-out prediction of personalization need; probabilities are synthetic model estimates, not clinical confidence. Flagging does not simulate a successful surgeon intervention."}, predictions, bailout_predictions


def analyze_subgroups(patients: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Describe held-out outcomes in fixed, illustrative anatomy strata.

    The cutoffs below are declared independently of outcomes, are not clinical
    grades, and are never selected to maximize observed personalization gains.
    Intervals are descriptive patient-bootstrap intervals conditional on the
    training-selected shared strategy; they are not adjusted for multiplicity.
    Each dimension partitions the same test cohort, so dimensions overlap.
    """
    test = [patient for patient in patients if patient["split"] == "test"]
    strata = [
        ("attachment_length", "short", "Short attachment (<15 mm)",
         lambda p: p["attachment_length_mm"] < 15),
        ("attachment_length", "medium", "Medium attachment (15–<25 mm)",
         lambda p: 15 <= p["attachment_length_mm"] < 25),
        ("attachment_length", "long", "Long attachment (≥25 mm)",
         lambda p: p["attachment_length_mm"] >= 25),
        ("adhesions", "none", "No adhesions (0)",
         lambda p: p["adhesion_count"] == 0),
        ("adhesions", "mild", "1–3 adhesion tethers",
         lambda p: 1 <= p["adhesion_count"] <= 3),
        ("adhesions", "high", "≥4 adhesion tethers",
         lambda p: p["adhesion_count"] >= 4),
        ("distension", "lower", "Lower distension (<1.2×)",
         lambda p: p["distension"] < 1.2),
        ("distension", "higher", "Higher distension (≥1.2×)",
         lambda p: p["distension"] >= 1.2),
    ]
    results = []
    for index, (dimension, group, label, predicate) in enumerate(strata):
        selected = [patient for patient in test if predicate(patient)]
        row = {"dimension": dimension, "group": group, "label": label,
               "n": len(selected), "shared_success_rate": None,
               "oracle_success_rate": None, "personalization_gain": None,
               "bailout_rate": None, "gain_ci": None}
        if selected:
            shared = np.array([patient["shared"]["success"] for patient in selected], dtype=float)
            oracle = np.array([patient["oracle"]["success"] for patient in selected], dtype=float)
            gain = oracle - shared
            row.update(shared_success_rate=float(shared.mean()),
                       oracle_success_rate=float(oracle.mean()),
                       personalization_gain=float(gain.mean()),
                       bailout_rate=float((1 - oracle).mean()),
                       gain_ci=paired_bootstrap_ci(gain, seed + 100 + index))
        results.append(row)
    return results
