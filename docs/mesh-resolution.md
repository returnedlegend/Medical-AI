# Mesh-resolution diagnostics and unresolved limits

The optional SOFA diagnostics now include an audited gallbladder refinement,
an attempted two-resolution organ trajectory, and a controlled three-resolution
linear-elastic cantilever fixture. The organ comparison is **not controlled**
because automatic grasp/reset bindings changed. The cantilever solutions match
an independent discrete finite-element assembly but remain far from a classical
beam approximation. Neither result establishes mesh convergence, tissue-model
accuracy, clinical exposure or surgical safety.

The portable 23-site tissue model, its reinforcement-learning reward and its
physics fingerprint were not changed by these diagnostics. Only development
seed 42 was used for organ resets. No reserved final RL patient was evaluated.

## Audited conforming refinement

[`refine_sofa_mesh.py`](../scripts/refine_sofa_mesh.py) inserts one shared midpoint
for every unique tetrahedral edge, retaining all original vertices first and in
exact order. Each tetrahedron is divided into eight children. The central
octahedron uses a fixed diagonal, and negative child orientations are reordered
positive. Shared-face subdivisions agree across neighboring parents.

The upstream gallbladder asset is the unmodified file from `sofa_env` commit
`85bf7e05dd088b824794dda0046679df13b13e6e`, SHA-256
`31bec8202724efe406474dea416a5e15e798caefccaff84e0570d57e116c3bdc`.
The generated fine VTK is stored outside the checkout and upstream source,
under `/workspace/.sofa-runtime/mesh-resolution/`. The generator refuses to
write into the input asset directory or overwrite an existing output file.

| Geometry check | Coarse organ | Refined organ |
| --- | ---: | ---: |
| Vertices | 367 | 2,159 |
| Tetrahedra | 1,063 | 8,504 |
| Boundary triangles | 728 | 2,912 |
| Total volume, source units³ | 34,122.545207 | 34,122.545207 |
| Nonpositive tetrahedra | 0 | 0 |
| Nonmanifold faces / boundary edges | 0 / 0 | 0 / 0 |
| Minimum mean-ratio shape quality | 0.016898 | 0.008797 |
| Median mean-ratio shape quality | 0.369891 | 0.350595 |

All original points round-trip exactly through the generated ASCII VTK. The
worst per-parent relative volume discrepancy is about `9.02e-15`, and the fine
boundary exactly matches subdivision of the derived coarse boundary. The
upstream unused vertex 314 remains unused and is retained to preserve indices.

Shape quality is `12(3V)^(2/3) / sum(edge_length²)`, equal to one for a regular
tetrahedron and approaching zero for a sliver. The fixed-diagonal refinement
**worsens the minimum quality**. More elements are not automatically better
shaped, and positive volume/manifold incidence does not rule out poor
conditioning, intersections or inaccurate anatomy.

## Attempted organ trajectory: controlled comparison blocked

[`sofa_mesh_resolution_check.py`](../scripts/sofa_mesh_resolution_check.py) uses a
project-owned scene wrapper that changes only the gallbladder volume-mesh path.
The upstream source, liver, visual/collision meshes and hard-coded collision
attachment indices remain unchanged. Both topology containers derive faces and
edges from tetrahedra, as described in [the topology diagnosis](physics-diagnostics.md).

Both headless SOFA 24.06 workers reset with seed 42, establish the upstream
scripted grasp, and complete ten identical gentle insertion commands at a
0.01-second step: 0.1 seconds of measured motion. Both retain their grasp and
have finite, non-inverted states with zero recorded SOFA errors. Their component
states are `Undefined`, not an affirmative `Valid` certification. A first
sandboxed graphics attempt reported shader errors and stopped; it was retained,
not counted as a pass. The supported external graphics execution then completed
without those errors, with no shader or mechanics workaround.

Despite equal original rest positions, collision rest positions, attachment
indices and initial gripper pose, the automatic reset associates the grasping
springs with different jaw sphere indices. The same nonzero tissue collision
index appears in a different entry of the jaw mapping. Thus the tool/tissue
constraint pairing changes. The already preloaded original-node states differ
by RMS **1.53109 source scene units** before measured motion starts.

The observed original-node incremental displacement difference after ten steps
is RMS **0.10391** and maximum **0.42796 source scene units**. Those numbers mix
mesh resolution with changed preload/contact binding and cannot isolate a mesh
error. The command writes `controlled_comparison_ready=false`, status
`uncontrolled_contact_reset_changes`, and exits nonzero for this comparison.
It does not convert two successful runtime smokes into a convergence pass.

For completeness, common-coarse-element incremental tensile strain estimates
have maxima about 0.2713 and 0.0515 in the coarse/fine runs; the fine native
mesh maximum is about 0.1092. These differences are **confounded** by the changed
initial/contact state and sampling. Native maxima also sample different element
counts. They are not comparable force/strain safety conclusions.

Additional physical-discretization concerns remain:

- Upstream `UniformMass` keeps total mass 5 fixed but changes per-vertex mass
  distribution when the point count changes.
- The same collision surfaces/attachment sites map barycentrically into
  different volume bases, changing load localization.
- Scripted reset includes contact and settling; identical seeds alone do not
  establish identical preloaded anatomy.
- Summed nodal internal-force magnitude changes meaning with resolution and is
  deliberately excluded from this comparison.

A controlled organ study would need explicit matched physical grasp/attachment
constraints, a consistent density/mass model, matched reference/loading history,
and solver/error assessment. These unresolved conditions are a concrete blocker
to interpreting this organ run as mesh convergence.

## Controlled three-resolution cantilever fixture

A separate static component fixture avoids contact, cameras and organ reset.
The rectangular beam is 10 × 1 × 1 in declared consistent model units, fixed on
the entire x=0 face, with Young modulus 1,500 and Poisson ratio 0.3. Constant
transverse traction is integrated over the x=10 face; triangle areas distribute
forces to their vertices. Total force `[0, −0.01, 0]` and moment about the fixed
face center match across all meshes. Density 0.001 gives total volume-lumped
mass 0.01 on every mesh. Mass does not affect this static solve, but its
consistent construction is recorded.

The initial mesh has four axial cells, each split into six tetrahedra. Two
successive conforming refinements produce 24, 192 and 1,536 tetrahedra. Fixed
boundary geometry, integrated force/moment, material and total mass are checked
explicitly. All three static states settle, with no SOFA errors or inverted
elements. There is no surgical task or injury policy in this fixture.

| Tetrahedra | Vertices | Area-weighted tip deflection magnitude | Difference from beam approximation |
| --- | ---: | ---: | ---: |
| 24 | 20 | 0.002639 | 90.1% lower |
| 192 | 81 | 0.005543 | 79.2% lower |
| 1,536 | 425 | 0.009263 | 65.3% lower |

For scale, Euler–Bernoulli theory gives
`delta = F L³ / (3 E I) = 0.026667`, with `I = b h³/12`. This is a slender-beam
approximation, not the exact 3D elasticity solution for the clamped/end-loaded
fixture. A simple rectangular-section shear correction, using `G=E/[2(1+nu)]`
and shear factor 5/6, is about 0.000208, under 1% of the bending estimate; omitted
shear alone does not explain the large deflection shortfall.

Coarse linear tetrahedra can over-stiffen a bending field, and unresolved
cross-section behavior, possible bending/shear locking and 3D clamp/end effects
need further investigation. Near-incompressible volumetric locking is not
assumed at Poisson ratio 0.3. Refinement increases compliance, but successive
deflection changes grow from about 0.00290 to 0.00372; these data do not establish
an asymptotic convergence regime or an error bound. No beam-calibrated parameter
has been substituted to hide the discrepancy.

### Independent discrete reference

The diagnostic independently assembles linear isotropic P1 tetrahedral stiffness
matrices in SciPy, using analytic shape-function gradients, engineering shear
components and Lamé coefficients. It solves the free-DOF sparse system with the
same integrated traction and clamped surface. This implementation does not call
SOFA's element-force or stiffness routines.

Across the three meshes, the maximum nodal difference between SOFA and this
independent discrete solution is below **2.8e-13**, and the free-DOF equilibrium
residual is below **1e-13**. The matching discrete solutions make a simple
load/solver wiring mistake less likely. They verify the specified discretized
linear-elastic problem, **not** its continuum accuracy or applicability to real
tissue. Agreement between implementations sharing constitutive assumptions does
not validate those assumptions.

## Reproduce and inspect

After the optional runtime is installed, generate a new refinement from the
unchanged upstream file:

```bash
/workspace/.sofa-runtime/venv/bin/python scripts/refine_sofa_mesh.py \
  --input /workspace/.sofa-runtime/source/sofa_env/scenes/grasp_lift_touch/meshes/gallbladder_volumetric.vtk \
  --output /workspace/.sofa-runtime/mesh-resolution/gallbladder-fine.vtk
```

The generator refuses an existing path. Reuse the already audited fine mesh or
choose a fresh output name when separately regenerating; the organ checker
expects the audited default fine-mesh path shown above. Then run in the same
supported graphics context as the existing headless SOFA worker:

```bash
/workspace/.sofa-runtime/venv/bin/python scripts/sofa_mesh_resolution_check.py \
  --output /workspace/.sofa-runtime/mesh-resolution/new-organ-check --steps 10

/workspace/.sofa-runtime/venv/bin/python scripts/sofa_mesh_resolution_check.py \
  --cantilever --output /workspace/.sofa-runtime/mesh-resolution/new-cantilever-check
```

The organ command is expected to return a blocked controlled-comparison status
for the recorded binding mismatch. The cantilever command checks matching
conditions and independent discrete agreement; it does not assert that beam
error is small. All output directories must be new. Worker logs, mesh arrays,
JSON and organ images remain under `.sofa-runtime/mesh-resolution/`, outside the
checkout and upstream source. The [compact diagnostic report](validation/sofa-mesh-resolution.json)
records the completed measurements and limitations.

These diagnostics are additional research evidence, including unfavorable
results. They do not fulfill the proposal's clinical or anatomical validation,
prove convergence of the gallbladder scene, or alter the frozen RL task.
