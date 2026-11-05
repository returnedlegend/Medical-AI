# Optional LapGym / SOFA physics sandbox

The main experiment engine remains `analytic-surrogate-1.0`. Its results are not
SOFA outputs. The optional sandbox runs a separate Python process using actual
upstream LapGym gallbladder and liver geometry. It demonstrates infrastructure,
rendering, a scripted grasp, and actuator-driven mesh motion; it does **not**
implement or validate the proposal's critical-view-of-safety study.

## Inspected upstream and runtime

- [`ScheiklP/sofa_env`](https://github.com/ScheiklP/sofa_env/tree/85bf7e05dd088b824794dda0046679df13b13e6e),
  pinned commit `85bf7e05dd088b824794dda0046679df13b13e6e`.
- Its [`GraspLiftTouchEnv`](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/sofa_env/scenes/grasp_lift_touch/grasp_lift_touch_env.py)
  grasps the infundibulum, lifts a gallbladder to reveal a target sphere, and uses
  a cauter instrument to touch the target. This is a useful starting point for
  the proposed scene; a gallbladder environment and volumetric meshes **already
  exist**. A new anatomical simulator need not start from scratch.
- [`setup.py`](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/setup.py)
  selects the SOFA **24.06.00** prebuilt distribution and requires Python **3.10**
  and x86_64. The application uses Python 3.12, so this runtime is intentionally
  isolated.
- [Official SOFA release](https://github.com/sofa-framework/sofa/releases/tag/v24.06.00):
  `SOFA_v24.06.00_Linux.zip`, 305,487,512 bytes. Its published and locally verified
  SHA256 is `9d515e2f25f657c744821be8a5361e22803c18947b33af7a0b357c259202236a`.
- Current tested runtime: Debian 13, Linux x86_64, managed CPython 3.10.22,
  SOFA 24.06.00 with its bundled SofaPython3, NumPy 1.26.4, Gymnasium 0.29.1,
  pyglet 1.5.21, and CPU-accessible EGL/OpenGL. Exact Python packages are recorded
  separately in `scripts/requirements-sofa.lock`.

No SOFA native libraries or upstream meshes are committed to this repository.
The upstream `sofa_env` source license is MIT (copyright Paul Scheikl, 2023);
retain its license and review upstream asset and bundled dependency licensing
when redistributing. The optional installer retains the upstream license.

## Reproduce the checked workflow

The default runtime directory is `/workspace/.sofa-runtime`; change
`RETRACTION_SOFA_DIR` to use another location. Managed Python is retained under
`/workspace/.toolchains`, with uv's cache under `/workspace/.cache/uv`.

Prerequisites are `uv`, `git`, `curl`, `tar`, `timeout`, and system EGL/OpenGL
libraries. Required downloads use the normal configured HTTPS proxy and TLS
verification: GitHub/release-assets, PyPI, and files.pythonhosted.org. The checked
cloud environment already permits these destinations. Its command sandbox needs
the normal approved escalation for downloads and graphics-backed subprocesses.

```bash
cd /workspace/Medical-AI
bash scripts/setup-sofa.sh
bash scripts/run-sofa-smoke.sh --seed 42 --steps 3
bash scripts/run-sofa-smoke.sh --seed 42 --start-grasped --steps 10 \
  --insertion -0.2 --output /workspace/.sofa-runtime/trajectory.json
```

The setup script preserves the core application environment, verifies the SOFA
archive before extracting, preserves native-library symlinks and executable
permissions, retrieves upstream source by full commit, and runs an actual
headless smoke check followed by the strict derived-connectivity audit. Re-running it on the prepared runtime was exercised.
No separate Git checkout is retained inside the runtime source directory.

A second check assembled a fresh temporary native runtime and fetched/archived
the pinned upstream source, reusing only the already downloaded release archive
and existing Python dependency environment. Its real headless reset and three
steps completed with zero SOFA errors, and input asset hashes matched the first
runtime. The temporary runtime was removed afterward. This verifies fresh native
extraction and source setup; it is not a claim of a second clean dependency install
or validation in a newly published cloud task.

The official archive includes an absolute zlib symlink in the unused CImg plugin
that points to `/lib/x86_64-linux-gnu/libz.so.1.2.11`, an Ubuntu-specific path
absent on the tested Debian host. Fresh extraction deliberately omits that one
unused broken link, rather than substituting a library with an assumed ABI.
The checked scene does not load CImgPlugin. Other archive links escaping the
runtime are rejected. This workflow does not claim all bundled SOFA plugins work.

The worker uses a 0.01-second timestep. Commands `--pan`, `--tilt`, `--spin`,
`--insertion`, and `--jaw` are upstream normalized gripper velocities, each
restricted to [-0.5,0.5]. The cauter receives a neutral action. Upstream maximum
velocities are [15,15,25,25,20] in degrees/second, except insertion in
millimeters/second. Thus insertion -0.2 requests -5 mm/s, with actual realized
poses reported separately. These bounds limit command magnitude; they are not
tissue-force limits or a safety controller. Upstream state/workspace constraints
can further restrict motion. `--start-grasped` requests upstream's scripted
grasp during reset and caps its attempts. It is not an independently trained
agent. Steps are bounded to 1–100 and the shell wrapper has a 300-second timeout;
the API applies its own shorter worker timeout.

The default `--topology-mode derived` uses `scripts/sofa_scene.py` to clear the
inherited triangle and edge arrays before initialization, allowing SOFA to
reconstruct complete lower-dimensional topology from the unchanged tetrahedra.
`--topology-mode upstream` reproduces the original scene and its diagnostics.
Neither mode modifies the upstream source files or geometry on disk.

Each output prefix produces JSON, a final PNG, and an `-initial.png`. JSON records
seed, upstream commit, runtime versions, actual poses, per-step mesh
displacements, raw internal-force diagnostics, rendered shape, and explicit
limitations. It also exports the actual topology arrays, component states,
worker/wrapper/dependency-lock SHA256s, and the six input mesh SHA256s.
`status: passed` means numerical/IO smoke assertions completed.
Inspect `diagnostic_state` and `diagnostics`; it is **not** a mechanics-validity
or surgical-success label.

## What was observed

A headless reset and three neutral steps produced a nonblank 256×256 RGB image
and finite state/force diagnostics on a 367-vertex gallbladder mesh. A subsequent
run used the upstream scripted grasp and ten insertion commands at -0.2 with a
0.01-second timestep: the grasp remained established, the realized insertion
state changed, and maximum vertex displacement from the post-reset baseline was
approximately 0.487 scene length units. The run completed in about 8.4 seconds
on this instance. Timings are observations, not performance guarantees.

The unmodified scene emits **six topology errors** on the tested build:
inconsistent liver edge/triangle lists and missing tetrahedron triangle
adjacency in the liver and gallbladder. These are recorded by an actual
`Sofa.Helper.MessageHandler` in the result, which sets
`diagnostic_state: needs_review`. The derived topology adapter was then run
through the same scripted grasp and ten pull steps: **zero SOFA errors** remained.
The generated arrays were independently audited against all tetrahedral faces
and edges, with no missing, duplicate, or extra elements. Liver counts are 273
tetrahedra, 639 triangles, and 463 edges; gallbladder counts are 1,063 tetrahedra,
2,490 triangles, and 1,792 edges. Point counts and tetrahedral connectivity are
unchanged. Component states are `Undefined`, **not** `Valid`; this is not a
claim that SOFA has certified the components or that the mechanics are validated.
The resulting `diagnostic_state: infrastructure_checked` means the tested
workflow completed without logged errors.

Nonfatal warnings also include camera defaults,
display colors, and VTK metadata. Initial shader setup can emit additional
messages. Finite mesh motion and a rendered image do not resolve these errors.
The adapter repairs the identified connectivity representation problem; it does
not establish physical fidelity. See `physics-diagnostics.md` and the repeatable
mesh audit for the exact checked invariants. Do not use this sandbox for research
conclusions until mechanics have been checked independently.

The upstream `STATE` observation has a dtype mismatch with its declared
float32 Gymnasium space in the inspected configuration. The worker uses the
upstream-supported `RGB` mode, for which the image-space assertion passes. It
does not silently cast state observations to claim that contract was validated.

## Missing pieces for the proposed experiment

1. Extend the topology checks and demonstrate stable mechanics. Check a small
   analytical reference, time-step sensitivity, force balance, and two mesh
   resolutions before fitting or comparing techniques. Separate numerical
   failure and timeout outcomes from unsuccessful exposure.
2. Introduce explicit attachment length, adhesion tethers, and distension in a
   custom scene. The current upstream gallbladder has hard-coded attachment
   indices and rest-shape springs; these do not implement the three proposed
   patient factors. Distension needs a chosen physical model and calibration,
   rather than equating a geometric scale with pressure.
3. Add the fixed fundus assistant and controllable neck grasp/retraction protocol.
   Upstream provides a gripper and a cauter, not the proposal's two-retractor
   protocol. Preserve a consistent baseline and reset state for every strategy.
4. Define cystic-duct/common-bile-duct landmarks and a camera-aware projected
   triangle visibility metric with occlusion checks. Upstream target-sphere
   visibility is not a CVS proxy. No cystic/common-bile-duct geometry or CVS
   verification was identified in this scene's six supplied mesh files.
5. Define tool/tissue interaction forces and strain. Upstream
   `get_internal_force_magnitude()` sums magnitudes across mesh vertices, and
   its reward feature multiplies that sum by 1e-9. It is **not** a calibrated
   grasp force in newtons and must not be compared with the analytic model's
   nominal force thresholds.
6. Run deterministic protocol sweeps in isolated subprocesses, then connect a
   versioned physical-simulator adapter to the experiment interface. Upstream
   documents one SOFA simulation per process. Do not run multiple scenes in the
   threaded analytic job worker.

This path can reuse the existing cohort, held-out analysis, export, and dashboard
infrastructure, while maintaining distinct simulator versions and provenance.
Reinforcement learning and multi-agent uncertainty should follow a working,
validated scene and Parts A/B; they are not established by this physics smoke.
