# Research plan and MVP boundaries

## Origin and research question

This project implements an exploratory prototype based on the user-supplied proposal, “Personalizing Robotic Gallbladder Retraction Using Simulated Strategy Search and Force Probing to Improve Safe Surgical Exposure.” The proposal describes a future LapGym/SOFA study. The current MVP is a native desktop application with three separate research components: an analytic cohort strategy study, a portable stateful tissue/RL laboratory, and optional SOFA mesh diagnostics. Their mechanics, tasks and success rates are different; none supplies validated human tissue behavior.

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

The desktop workbench permits repeated exploratory runs. The numbered discipline above describes the analytic cohort study. The sequential RL experiments additionally freeze training-source snapshots, action/observation encoders, complete fitting/validation seed histories and declarative checkpoints. Their cross-run validation selection is disclosed. The separate [fixed-policy final plan](final-evaluation-plan.md) defines a paired assessment with immutable inputs and a complete prior-use exclusion audit; repository records are not a substitute for an external clinical preregistration.

## Stateful body and trained controllers

The portable laboratory implements actual tool actions, observations, rewards and terminal states. Mechanical/thermal injury persists, changes restoring stiffness, and can produce bleeding or bile leakage with simplified physiological feedback. The [equations and limitations](tissue-response.md) and [controlled tool-response protocols](response-protocols.md) document this causal loop. The 3D depth/surface rendering is illustrative; the numerical body is a 23-site model.

NumPy Double DQN and tabular Double-Q controllers have been trained from observed transitions. [Studies 001](rl-study-001.md), [002](rl-study-002.md) and [003](rl-study-003.md) retain all 35 runs and 105,000 online training episodes on paired synthetic seed streams. Those are repeated episodes, not that many independent patients. Larger encodings did not demonstrate improved performance in the recorded comparisons, and harmful/zero-success runs remain available. Scripted demonstrations, continuous scripted control and learned policies are labeled separately. Policy playback keeps weights fixed; manual episode logs do not silently become training demonstrations.

The [completed fixed-policy assessment](rl-final-001.md) compares twelve frozen checkpoints and three common controls on the same 128 previously reserved patients. All 1,920 rollouts and independent audits are retained, with no subsequent checkpoint replacement. Its synthetic performance and uncertainty do not establish the proposal's anatomical or clinical hypothesis.

## Model-design dependence

The surrogate explicitly computes a preferred pull direction from attachment length, adhesion count, and distension. Alignment with that preferred direction affects force and exposure. Therefore, patient-dependent strategy benefit is partly imposed by the equations. A positive gain or strong probe AUROC demonstrates behavior of this chosen model and the analysis pipeline; it cannot establish that real tissue exhibits the same effect. Noise-free force features can also make early prediction optimistically informative.

## Beyond the MVP

- Replace or calibrate the portable dynamics using an independently validated simulator adapter. The existing optional SOFA mesh/camera diagnostic is not yet that adapter.
- Review constitutive assumptions, numerical integration, force units, and landmark visibility with a mechanics mentor and surgeon.
- Resolve the [organ mesh-refinement contact/preload mismatch](mesh-resolution.md), mass distribution and discretization error before claiming convergence. An independent tetrahedral stiffness check agrees with SOFA, but the coarse cantilever remains substantially too stiff relative to a beam approximation.
- Expand independent training seeds and assess richer state/history representations, continuous control and justified handoff criteria. Existing checkpoints and negative results must remain frozen when new experiments begin.
- Evaluate domain shift, simulated inflammation and bleeding, and model discrepancy.

The logistic early-warning model remains a separate supervised cohort baseline. The portable RL agents are real trained research controllers, but the optional SOFA scene does not yet have a learned surgical policy or the proposal's full patient-specific CVS task. Modeled handoff terminates a synthetic episode; it is not a clinical handoff system. Positive analytic personalization gains cannot be combined with sequential RL scores to claim the original surgical hypothesis is established.
