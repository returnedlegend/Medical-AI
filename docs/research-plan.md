# Research plan and MVP boundaries

## Origin and research question

This project implements an exploratory prototype based on the user-supplied proposal, “Personalizing Robotic Gallbladder Retraction Using Simulated Strategy Search and Force Probing to Improve Safe Surgical Exposure.” The proposal describes a future LapGym/SOFA study. This MVP is a lower-cost analytic workbench that supports pipeline development and hypothesis exploration before a validated soft-tissue simulator is available.

The primary question is how much safe exposure improves when a retraction strategy is chosen separately for each synthetic patient instead of applying one strategy to all patients. Secondary questions ask whether early force responses identify patients needing adaptation and patients for whom the available strategy grid fails.

## What the MVP establishes

The application must run the complete pipeline using reproducible random seeds, preserve inputs and model metadata, distinguish training from test patients, expose per-patient outcomes, and export results for independent inspection. Unit, API, and browser checks establish software behavior. They do not validate the tissue model.

A successful exposure proxy requires the geometric visibility threshold AND both force and strain constraints. Grid bail-out means no evaluated strategy meets all three; it does not mean every possible surgical strategy would fail. Likewise, the per-patient grid optimum is an upper bound within this tested grid and model, not a clinically meaningful oracle.

## Evaluation discipline

1. Generate synthetic patients using an explicit seed and bounded parameter ranges.
2. Split patients into disjoint training and test partitions before strategy selection or classifier fitting.
3. Select the best shared grid strategy using training outcomes only, using a deterministic tie-break.
4. Evaluate the fixed shared strategy and per-patient grid strategy on the same held-out patients. Report their paired success difference in percentage points.
5. Label a patient as needing personalization only if the shared strategy fails and a grid strategy succeeds.
6. Fit feature transforms and the early-warning classifier using training patients only. Probe features must come from the same fixed tug for every patient, not from the patient-specific winning strategy.
7. Report held-out AUROC only when both classes occur, and reliability/calibration with sample counts. Undefined metrics stay undefined rather than becoming a passing score.
8. Report uncertainty from held-out patients with a seeded bootstrap. These intervals describe sampling variability in the synthetic population, not uncertainty in the physical assumptions.
9. Test sensitivity to three named force/strain presets. Threshold selection after inspecting results is exploratory and must not be described as preregistration.
10. Keep crashes distinct from unsuccessful retractions. Export the configuration, model version, selected strategy, and patient-level results.

The web workbench permits repeated exploratory runs. It is not itself a preregistration system. A formal study should freeze its final cohort distributions, seeds, exclusion rules, thresholds, mesh checks, and analysis plan in a time-stamped external registration before running confirmatory experiments.

## Model-design dependence

The surrogate explicitly computes a preferred pull direction from attachment length, adhesion count, and distension. Alignment with that preferred direction affects force and exposure. Therefore, patient-dependent strategy benefit is partly imposed by the equations. A positive gain or strong probe AUROC demonstrates behavior of this chosen model and the analysis pipeline; it cannot establish that real tissue exhibits the same effect. Noise-free force features can also make early prediction optimistically informative.

## Beyond the MVP

- Integrate LapGym/SOFA through a separate simulator adapter with actual meshes and camera rendering.
- Review constitutive assumptions, numerical integration, force units, and landmark visibility with a mechanics mentor and surgeon.
- Verify forces against analytic mechanics and check at two mesh resolutions.
- Add independently trained reinforcement-learning seeds, confidence estimates, and a fixed handoff policy.
- Evaluate domain shift, simulated inflammation and bleeding, and model discrepancy.

No RL agent is claimed to be trained by the current prototype. The logistic early-warning model is a supervised baseline. A handoff or abstention curve is an evaluation of model confidence, not a working surgical control or clinical handoff system.
