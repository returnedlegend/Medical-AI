# Analytic model reference

Model version: `analytic-surrogate-1.0`. The source of truth is `backend/retraction/simulation.py`; every result exports the model version and methodology.

## Patient and strategy parameters

A synthetic patient has attachment length **L** in mm, adhesion count **n**, adhesion stiffness **kₐ** in N/mm, and distension factor **D**. Defaults are independent uniform draws: L ∈ [8,35], n ∈ {0,…,6}, kₐ ∈ [.03,.20], D ∈ [.8,1.8]. These are illustrative parameter distributions, not clinical prevalence estimates.

The preferred direction θ* is imposed:

```
θ* = clip(25 + .95(35 − L) + 4.5n − 9(D − 1), 10, 90) degrees
```

A strategy has grasp fraction **g**, direction **θ** in degrees, and displacement **d** in mm. Grasp is a normalized lever parameter; it does not map to a validated anatomical landmark. Fundus support is held fixed by assumption.

```
alignment = max(0, cos(θ − θ*))²
extension = d(.42 + .58g)
strain = extension / L
stiffness = 1.6D/L + n kₐ(.28 + .72(1 − alignment))
force = stiffness × extension × (1 + 1.3 × strain²)
opening = max(0, d(.40 + .80g) × alignment / (.80 + .25D) − .38n)
exposure = clip(opening / 15, 0, 1)
projected_area = 90 × exposure mm²
```

Angles are converted to radians for cosine. The 1.6 coefficient has units of force, making stiffness N/mm. Displacement times force integrates to N·mm = mJ. The area is a capped normalized planar proxy, not a rendered triangle or anatomical visibility measurement.

All coefficients are hand-specified. In particular, the imposed preferred direction means a positive personalization benefit is partly designed into the model. A null or positive result cannot resolve the original tissue-mechanics research question without independent model validation.

## Success and thresholds

A strategy succeeds if exposure ≥ its threshold, force ≤ its limit, and strain ≤ its limit. Equality passes. The displacement ramp is monotonic under this constitutive law, so endpoint force/strain are their maxima for the quasi-static pull.

| Profile | Force limit (N) | Strain limit | Exposure minimum |
|---|---:|---:|---:|
| Conservative | 1.8 | .45 | .65 |
| Nominal | 2.5 | .60 | .60 |
| Permissive | 3.2 | .75 | .55 |

None is a validated tissue safety threshold. Because all three change together, these profiles show joint sensitivity, not the individual effect of one threshold.

The reference strategy uses g=.75, θ=35°, d=12 mm. The tested grid uses g ∈ {.35,.55,.75}, θ ∈ {15,35,55,75,95}°, d ∈ {4,8,12,16,20} mm. Shared-strategy ties use this fixed grid order. A patient-specific successful strategy is selected by lowest force; if no strategy succeeds, the display shows the maximum-exposure force/strain-feasible candidate, or minimum force when none is feasible, retaining failure status.

## Force probe and prediction

The same g=.75, θ=35° ramp from 0 to 2 mm over one second is sampled at 21 equally spaced points. Time is a protocol label; this is a quasi-static spring model without inertia, viscosity, relaxation, contact, or measurement noise. Features are peak force, initial secant stiffness, nonlinear curvature ratio, and trapezoidal work.

Probe force and strain limit checks are reported under the active thresholds. Observations with violations are retained and counted; the model does not implement an instrument stop controller. Treat them as a protocol feasibility limitation, not permission to probe such tissue.

A seeded 70/30 split fixes training and held-out patients. A scaler and logistic regression fit only training probes. The target is **the chosen shared strategy fails AND at least one grid strategy succeeds**. A second logistic model predicts grid infeasibility. Probabilities on training patients are in-sample; use held-out records and metrics for evaluation.

AUROC is undefined for one-class test targets. When training has only one class, no classifier is fit and probabilities remain null. Brier score and ten-bin expected calibration error evaluate held-out probabilities; empty bins are exported with zero count and null means. A fixed .5 classification threshold yields a confusion matrix. No confidence threshold is tuned on held-out data and no successful human intervention is assumed.

## Uncertainty and reproducibility

The primary outcome is a paired difference: the fraction of held-out patients where grid search succeeds but the shared strategy fails. Both methods use the same patients. A seeded 1,000-replicate percentile bootstrap reports a 95% interval conditional on the chosen shared strategy and model. The interval does not account for training strategy selection, model assumptions, or simulator fidelity.

Patient seeds, split assignment, strategies, thresholds, probe responses, patient results, and methodology are exported. Study bundles also fingerprint source files and capture their protocol before execution. There is no mesh, so mesh-convergence validation is not implemented. Mechanics tests verify internal equation consistency and dimensional edge cases, not physical accuracy.
