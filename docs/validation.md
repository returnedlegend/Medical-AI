# MVP validation record

The Windows desktop MVP and the development environment were exercised on 2026-11-05. These checks establish software behavior and recorded numerical behavior; they do not validate biological accuracy or surgical safety. The portable tissue engine, analytic cohort model and optional SOFA diagnostics remain separate.

## Desktop application and body response

- A self-contained Windows x64 application and per-user NSIS installer were built. The packaged native Qt window, actual procedure controls, offline Help, exports, reference policy, real DQN/Double-Q training, service shutdown and durable replay were exercised.
- Native Windows workflow [42082267757](https://github.com/returnedlegend/Medical-AI/actions/runs/42082267757) passed all three jobs for checkpoint `9f9db8a`: Windows app/model/GUI checks, exact verified Debian NSIS assembly, and installation/model checks/uninstallation on a fresh Windows runner.
- Uninstalling preserved saved cohort, learning, procedure and optional-physics databases byte for byte, along with unrelated files in root and nested installation directories. Installation was also exercised under a path containing spaces.
- Five native lifecycle/resource tests passed, including preserving verbatim license bytes and keeping the window and study lock present while active work finishes. All 18 pinned notice hashes were checked in source, build snapshots, and installed diagnosis.
- Controlled response protocols passed through the actual API: gentle traction, sustained jaw compression, arterial thermal contact, bile-duct thermal contact and remote-heat control. Damage, losses and action provenance persisted exactly through a real service restart. A protocol can finish while an episode remains active; this is not reported as surgical success.
- The 3D renderer uses actual XY body state with illustrative depth/shading. WebGL loss and unavailable graphics fall back to the live XY view. Browser rendering checks verify visible tissue/blood pixels and frozen terminal/recorded frames. Native software-rendered checks do not establish 3D compatibility on every Windows graphics driver.

## Automated and browser checks

- A full `scripts/check.sh` invocation completed **321 backend tests and 19 subtests**, followed by the TypeScript and production build. The final report component separately passed **11 focused tests** using retained genuine development records and synthetic statistical fixtures. One upstream Starlette/httpx deprecation warning remains.
- The core environment is installed from `backend/requirements.lock`; native packaging has separately pinned requirements. The final app interface builds successfully. Vite reports the lazy Three.js chunk exceeds 500 KiB; that warning is retained.
- Procedure Chromium regression: **25 checks**, including real tool actions/harm, persistent records/export, actual trained-policy execution, model/condition gating, keyboard focus, read-only history, offline and transient recovery, stale async response handling, and mobile layout. No unhandled JavaScript errors.
- Cohort Chromium regression: **17 checks**, including all 75 strategy points, live custom pulls, patient switching, verified downloads, history, held-out tables, keyboard/modal navigation, error/offline recovery and mobile layout.
- Automated axe checks passed **29 workspace states and seven additional recovery states** without violations or unintended mutations. Canvas/SVG contrast and other incomplete findings remain manual review items; this is not comprehensive accessibility certification.
- Existing setup and startup scripts were exercised, including a clean isolated service, all five response protocols, a real 40-patient/75-strategy cohort, and exact durable-state recovery after restart. Cloud installation/restart instructions are saved separately in the environment configuration draft; that does not establish publication or validation of a newly restored snapshot.

## Learning and numerical evidence

The three frozen exploratory studies retain all **35 runs**, 105,000 online episode records and 3,000 assisted demonstration trajectories. These are repeated episodes on paired synthetic patient streams, not 105,000 independent patients. Across the studies there are 450 distinct validation patient seeds. Harmful, zero-success and negative-return results remain in the complete archives. Larger reactive encodings did not demonstrate improved performance in their recorded paired comparisons; the bundled reference remains explicitly validation-selected.

The separately reserved final comparison has completed once: **12 fixed policies plus three shared baselines on 128 patients, totaling 1,920 rollouts**. Its controller set, 60,565 prior-use/conservative patient exclusions, sources and checkpoint hashes were frozen and independently reviewed before execution. Final results are assessed without retraining or reference replacement. The completed artifacts, full report and independent audit are published separately; no selected validation score is relabeled as an independent test.

- [Tissue equations and reward feedback](tissue-response.md)
- [Controlled tool-response protocols](response-protocols.md)
- [Portable timestep and thermal diagnostics](numerical-validation.md)
- [All-seed validation figures](validation-figures.md), [study 001](rl-study-001.md), [study 002](rl-study-002.md), and [study 003](rl-study-003.md)
- [Final fixed-policy results](rl-final-001.md), [evaluation design](final-evaluation-plan.md), and [report verification](fixed-policy-reporting.md)
- [Development portability receipt](policy-portability.md): all 15 fixed V1/V2/V3 development trajectories matched exactly between Linux Python and actual Windows Python under Wine. That small test is not a general cross-platform equivalence proof or native Windows GUI validation.

## Optional SOFA scene

The isolated Python 3.10.22/SOFA 24.06 runtime uses a checksum-verified official release and pinned upstream assets. Actual rendered reset, neutral stepping, scripted grasp and bounded pull jobs ran. The derived-connectivity wrapper preserves original tetrahedra/vertices and supplies the exact missing faces/edges. Corrected liver topology has 273 tetrahedra, 639 unique faces and 463 unique edges; gallbladder topology has 1,063 tetrahedra, 2,490 unique faces and 1,792 unique edges.

The optional damage prototype updates actual per-element Young moduli using the vector-valued tetrahedral force field. Its elementary displacement check agrees with analytic elasticity. This does not validate tissue rupture, thermal constitutive behavior or surgical forces.

The mesh-resolution investigation retains negative results: organ grasp/preload changed after refinement, so its comparison is uncontrolled. A three-resolution cantilever agrees with independent sparse tetrahedral stiffness assembly but remains substantially stiffer than a beam approximation. Neither runtime success, positive element orientation nor agreement between discrete implementations proves continuum accuracy. Component states reported as `Undefined` are not relabeled `Valid`.

See [SOFA integration](sofa-integration.md), [topology diagnostics](physics-diagnostics.md), [material softening](sofa-softening.md), and [mesh resolution](mesh-resolution.md). The optional scene still lacks the proposal's validated patient parameterization, matched dual-retractor/CVS task, calibrated limits and learned SOFA surgical policy. The portable RL controllers do not close those gaps.

## Retained evidence

Reusable tests and scripts are in `backend/tests/`, `desktop/tests/` and `scripts/`. Representative actual screenshots are in `docs/images/`; standalone figures and numerical receipts are in `docs/assets/` and `docs/validation/`. Complete research records are in `research-artifacts/` with checksums and payload manifests. Native builds and smoke/install evidence are downloadable from the linked GitHub workflows. Temporary local logs are additional evidence and are not assumed to survive a fresh cloud task.
