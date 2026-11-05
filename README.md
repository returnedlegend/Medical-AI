# Retraction Laboratory

A desktop research application for studying gallbladder retraction, simulated tissue injury, and reinforcement-learning feedback. The application has an academic simulation interface: a live 3D procedure view, tool controls, patient/tissue state, reward traces, episode records, and reproducible experiments.

The portable procedure engine is a **stateful mechanistic surrogate**, not validated human physiology. A separate optional SOFA diagnostic runs actual deformable meshes. Both identify their model and limitations explicitly.

## Windows desktop application

The primary target is a self-contained **Windows x64 `.exe`**, with a per-user installer and a portable ZIP. Python, Node, and a separate browser are not required to run the packaged application. The native Qt window starts its own private authenticated local backend and stores studies in the operating system's user-data directory.

The [Windows desktop workflow](https://github.com/returnedlegend/Medical-AI/actions/workflows/desktop.yml) builds the executable and records packaged checks. Download its `RetractionLab-windows-x64` artifact after a successful run, then run the included installer or extract the portable package and launch `RetractionLab.exe`. Keep the portable application's supporting files together. See [desktop installation and build details](docs/desktop.md) for exact artifact names, verification, and platform limits.

Local build outputs live under ignored `artifacts/desktop/`; compiled runtimes and patient/session data are not committed to GitHub. The development log distinguishes locally built Windows PE files, Wine checks, and native Windows CI results rather than assuming they are equivalent.

## Procedure laboratory

1. Create a seeded episode. Select **Grasper** or **Cautery**.
2. Move with the tool buttons, arrow keys, or a bounded viewport target. Close/release the grasper or activate a cautery pulse.
3. Observe actual state transitions: deformation, force, heat, accumulated injury, bleeding or bile leakage, and explicitly synthetic vital signs.
4. Inspect each reward component and the episode trace. Injury persists after tool release or cooling and changes future stiffness and mechanics.
5. Run the transparent scripted baseline or train and replay an experimental learned policy. The interface distinguishes them and shows held-out success, harm, handoff, and return separately.
6. Export the episode JSON, including configuration, actions, responses, events, and model provenance. A reset creates a new episode and preserves the previous record.

Shortcuts: **G** grasper, **C** cautery, arrow keys move the selected tool, **Space** grasp/pulse, **Escape** pause automatic playback. A 2D node view is available alongside the illustrative 3D rendering.

**Tool/body documentation:** [tissue-response.md](docs/tissue-response.md) defines contact, crushing, strain, thermal dose, irreversible softening, vessel/duct injury, bleeding/leakage, synthetic physiology, rewards, and termination. The same mechanics drive manual control, API transitions, training, and rendering. Documentation is also available within the application and native Help menu.

## Reinforcement-learning scope

The Gymnasium environment exposes reset/step/observation/reward/termination. Policies learn from observed transitions; JSON checkpoints contain declarative parameters and weights, not executable pickle data. Training seeds and evaluation seeds are recorded and separated. Playback reports changed environment conditions and can create an episode matching a checkpoint's settings.

The research agents are experimental. Some development runs learned to hand off or caused substantial harm rather than solving the task; these outcomes are retained and documented. A successful scripted baseline, a supervised demonstration component, and a trained RL policy are different results. See [the training reference](docs/rl-training.md).

## Cohort experiments and real-physics diagnostics

The cohort module retains seeded patient distributions, exhaustive 75-strategy sweeps, training-selected shared strategies, held-out personalization benefit, force-probe prediction/calibration, subgroup analysis, threshold sensitivity, and JSON/CSV/research-report exports. Its analytic exposure model is described in [model.md](docs/model.md).

The optional Linux x86_64 SOFA runtime renders actual gallbladder/liver meshes, scripted grasping, bounded motions, and raw telemetry. An independently audited wrapper derives complete connectivity from unchanged upstream tetrahedra. The experimental material-softening variant changes actual element stiffness in response to strain/heat and has an elementary elasticity reference. It is separate from the portable RL engine and does not yet implement the proposal's full patient-specific CVS study. See [SOFA setup](docs/sofa-integration.md), [connectivity diagnosis](docs/physics-diagnostics.md), and [material-softening experiments](docs/sofa-softening.md).

## Run from source

Requires Python 3.12 and Node.js 22/24 for development. No GPU, external database, clinical data, or application API key is needed for the portable laboratory.

```bash
./scripts/setup.sh
./scripts/start.sh
```

The development server serves the complete interface and API on port 8000; API documentation is at `/docs`. It is a local development option; the Windows distribution uses the native embedded window. For hot reloading:

```bash
# Terminal 1, repository root
cd backend
../.venv/bin/python -m uvicorn retraction.main:app --reload --host 127.0.0.1 --port 8000
# Terminal 2, repository root
npm --prefix frontend run dev
```

Use one API coordinator process. Studies and episodes persist in `.data/` for the source launcher. `RETRACTION_DATA_DIR`, `RETRACTION_VENV_DIR`, `HOST`, and `PORT` override source-launcher defaults. The packaged application uses per-user writable paths instead. Core dependency versions are locked in `backend/requirements.lock` and `frontend/package-lock.json`; desktop dependencies are listed separately.

## Validation and headless research

```bash
./scripts/check.sh
# After starting the app, with optional Playwright/Chromium installed:
.venv/bin/python scripts/procedure_browser_smoke.py
.venv/bin/python scripts/browser_smoke.py
.venv/bin/python frontend/scripts/audit_accessibility.py
```

Native desktop lifecycle/resource tests and packaged diagnostics are documented in [desktop.md](docs/desktop.md). Browser tests perform real tool actions and verify saved state, exports, policy training, harm, mobile layout, and error recovery. Automated accessibility results retain incomplete/manual findings separately.

Headless cohort studies remain available from `backend/`:

```bash
../.venv/bin/python -m retraction --patients 240 --seed 42 --output ../.data/example.json
../.venv/bin/python -m retraction.study --seeds 42 43 44 --output ../.data/study-001
```

Study bundles save their protocol and source fingerprints before execution and refuse to overwrite previous results. They are exploratory local records, not external preregistration. See [the research plan](docs/research-plan.md) and [validation record](docs/validation.md). The proposal's validated anatomy, calibrated tissue/force limits, mesh-convergence study, and clinically meaningful outcomes remain research milestones.
