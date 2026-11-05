# Experimental SOFA material-softening adapter

The portable procedure laboratory and its RL policies use the explicitly labeled lumped spring/heat surrogate documented in `tissue-response.md`. This separate optional experiment demonstrates that persistent injury can also change **actual SOFA FEM mechanics**, rather than merely changing a plot or material color.

## Implemented mechanism

`backend/retraction/sofa_damage.py` measures each tetrahedron's deformation gradient against a fixed reference configuration. The largest singular value gives rotation-invariant incremental tensile strain. Peak strain is retained; unloading cannot reset it. The illustrative mechanical damage law is a clipped linear ramp between strain .25 and .75. Thermal contact uses a compact spatial heat source, exponential cooling toward 37°C, and irreversible accumulation above an illustrative 50°C threshold. These numbers are uncalibrated engineering parameters, not tissue safety limits.

Mechanical and thermal damage combine as `D = 1 − (1 − D_mechanical)(1 − D_thermal)`. Requested element stiffness becomes `E = E0 × (1 − .95D)`, preserving 5% residual stiffness for numerical stability. This is material softening, not a crack, cut, removed element, or validated rupture.

The upstream `FastTetrahedralCorotationalForceField` exposes cached scalar material data. Simply assigning its modulus and calling `reinit()` did **not** change measured compliance in the tested release; repeatedly calling `init()` caused topology-handler errors. The adapter rejects this unsupported update path. `scripts/sofa_damage_scene.py` instead selects stock vector-valued `TetrahedronFEMForceField` before scene initialization. Its per-element modulus update and `reinit()` were checked against an actual elasticity reference. The upstream meshes and installed source are not changed.

## Checked elementary mechanics

```bash
bash scripts/check-sofa-damage.sh
```

The saved reference fixes three nodes of a unit tetrahedron, sets Poisson ratio zero, and applies the same small axial force before and after controlled injury preconditioning. For this constrained affine case `k = EV/L²` and `u = F/k`. The intact displacement agreed within .08% of the analytic value, and the injured displacement within .12%. A .5 damage state reduced E from 1000 to 525 scene modulus units and increased displacement from approximately .000600 to .001142 scene length units. No SOFA errors were recorded.

This tests material-update behavior and a simple elastic element; it does not validate organ tissue laws.

## Actual gallbladder scene

```bash
bash scripts/run-sofa-smoke.sh --damage-model --seed 42 --start-grasped \
  --steps 10 --insertion -0.2 --output /workspace/.sofa-runtime/damage-small-pull.json
```

The worker uses corrected tetrahedral connectivity, changes to the explicit experimental FEM variant, and constructs damage relative to the **post-reset/post-grasp** vertex configuration. Each actual simulation step computes incremental strain and updates element stiffness for the next step. A bounded ten-step pull completed with zero SOFA errors. One of 1,063 elements crossed the illustrative onset; maximum damage reached approximately .116 and its requested modulus fell from 1500 to 1335. Volume-weighted whole-organ damage was much smaller. The JSON records both, to avoid presenting a local maximum as widespread tissue injury.

Optional `--cautery-power` places the illustrative heat source at the actual upstream cauter tip. Heating falls to zero outside its contact radius; no proximity means no burn. There is no mesh cutting, thermal PDE, vascular fluid model, or clinical interpretation. Inverted elements are treated as numerical failure, not successful injury simulation. This experimental SOFA variant is separate from the portable RL environment and is not presently a trained-policy backend.
