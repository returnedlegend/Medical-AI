# Upstream SOFA topology diagnosis

The unmodified upstream `grasp_lift_touch` scene renders and takes neutral steps,
but that does **not** establish a valid mechanical topology. The baseline run
logged errors that correspond to missing adjacency buffers. In SOFA 24.06's
implementation, the missing-edge branch explicitly marks the component
`Invalid`. Treat baseline image generation as an infrastructure smoke check,
not physics validation or a successful retraction experiment.

## Scope and provenance

These observations concern `sofa_env` commit
`85bf7e05dd088b824794dda0046679df13b13e6e`, its two unmodified volumetric VTK
assets, and the official SOFA `v24.06.00` Linux release using Python 3.10.22.
Upstream `setup.py` specifies that same SOFA version, so the evidence does not
suggest an unsupported-version mismatch. Deprecation, camera and color warnings
are separate from the topology errors described below.

The baseline artifacts were `/workspace/.sofa-runtime/smoke.json`, `smoke.log`
and `smoke.png`. Their successful RGB/finite-state checks are narrower than a
mechanical validation. Source and asset inspection was read-only.

Asset SHA-256 values:

- `liver_volumetric.vtk`:
  `40d1df54f2f4e7a8b01c7d3c2b91d5651b7e6c4765d11544bea5694afec25e6a`
- `gallbladder_volumetric.vtk`:
  `31bec8202724efe406474dea416a5e15e798caefccaff84e0570d57e116c3bdc`

## What the source and data show

The upstream `DeformableObject` constructor in
`sofa_env/sofa_templates/deformable.py` links the whole mesh loader into a
`TetrahedronSetTopologyContainer` using `src=loader.getLinkPath()`. The VTK files
contain tetrahedra **and** explicit triangles. Those triangles describe the
surface twice, once per orientation, rather than the complete collection of
tetrahedral faces required by the dynamic topology container.

In the official SOFA code:

- [`TetrahedronSetTopologyContainer::createTrianglesInTetrahedronArray`](https://github.com/sofa-framework/sofa/blob/v24.06.00/Sofa/Component/Topology/Container/Dynamic/src/sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.cpp#L298)
  generates the complete triangle collection only when no triangles are
  provided. Otherwise it searches the supplied triangles for every tetrahedral
  face. On a missing face it clears `m_trianglesInTetrahedron` and returns.
- The same class derives edges from tetrahedra. A supplied triangle can
  therefore ask for an edge that is not present in any tetrahedron.
- [`TriangleSetTopologyContainer::createEdgesInTriangleArray`](https://github.com/sofa-framework/sofa/blob/v24.06.00/Sofa/Component/Topology/Container/Dynamic/src/sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.cpp#L291)
  logs the observed missing-edge error, clears `m_edgesInTriangle`, and sets
  `d_componentState` to `Invalid`. This is a real topology-construction failure,
  not a harmless warning string.

A read-only parser decoded the legacy big-endian binary VTK `POINTS`, `CELLS`
and `CELL_TYPES` arrays. It enumerated sorted tetrahedral faces/edges,
checked index bounds and repeated vertices, computed signed tetrahedron volumes,
counted face and boundary-edge incidences, and traversed vertex connectivity.

| Audit item | Liver | Gallbladder |
| --- | ---: | ---: |
| Points | 98 | 367 |
| Tetrahedra | 273 | 1,063 |
| Explicit triangle cells | 376 | 1,456 |
| Unique explicit triangles | 188 | 728 |
| Unique faces derived from tetrahedra | 639 | 2,490 |
| Boundary faces derived from tetrahedra | 186 | 728 |
| Interior faces derived from tetrahedra | 453 | 1,762 |
| Unique edges derived from tetrahedra | 463 | 1,792 |
| Derived faces absent from supplied triangles | 452 | 1,762 |
| Supplied triangles that are not any tetrahedron's face | 1 | 0 |
| Unused points in tetrahedral connectivity | 0 | 1 |

Every unique explicit triangle occurs exactly twice with opposite orientation.
All derived boundary faces are present in the explicit triangles, but interior
faces are missing. Specifically, the logged missing face `[15,43,49]` in liver
and `[345,158,15]` in gallbladder each belongs to **two** tetrahedra: they are
interior faces omitted from the supplied triangle list.

Liver has an additional concrete inconsistency: supplied triangle `[5,87,91]`
is not a face of any tetrahedron. Its edge `[87,91]` is not an edge of any
tetrahedron, which exactly explains the logged missing-edge error. The other
extra supplied liver triangle, `[4,42,63]`, is an interior tetrahedral face.
Gallbladder point 314 is unused by the tetrahedral connectivity; no coincident
point coordinates were found in either file. No effect of that unused point on
mass or dynamics has been established here.

The tetrahedral arrays have no out-of-range indices, repeated-vertex cells,
duplicate tetrahedra or zero-volume tetrahedra. Under the inspected vertex
ordering, all signed volumes are positive. Every tetrahedral face has incidence
one or two, and every edge on the derived boundary has incidence two. Liver's
98 used vertices form one connected component; gallbladder's 366 used vertices
form one component. These checks support a narrower diagnosis of inconsistent
explicit lower-dimensional connectivity; they do **not** prove absence of
self-intersections, element-quality problems, poor conditioning or mechanical
inaccuracy.

## Tested scene-level correction

The project-owned [`scripts/sofa_scene.py`](../scripts/sofa_scene.py) wrapper
constructs the upstream scene, then clears the inherited triangle and edge
arrays **before** SOFA initialization. SOFA consequently derives its dynamic
lower-dimensional connectivity from the unchanged tetrahedra. Upstream source
and assets remain unmodified.

The corrected `trajectory-derived.json` run used a scripted upstream grasp,
seed 42, a 0.01-second timestep and ten gripper insertion commands of −0.2.
Independent inspection of the exported topology arrays found:

| Corrected-scene check | Liver | Gallbladder |
| --- | ---: | ---: |
| Vertex count unchanged | Yes | Yes |
| Tetrahedra and their index order unchanged | Yes | Yes |
| Derived triangles | 639 | 2,490 |
| Derived edges | 463 | 1,792 |
| Missing/extra/duplicate derived faces | 0 / 0 / 0 | 0 / 0 / 0 |
| Missing/extra/duplicate derived edges | 0 / 0 / 0 | 0 / 0 / 0 |
| Reported component state | Undefined | Undefined |

That run captured **zero errors and zero topology errors**, with six other
warnings. The corrected containers report `Undefined`, not `Invalid`, but also
**not explicitly `Valid`**. This state must not be presented as an affirmative
SOFA validation result. Complete exported connectivity and absence of the
previous errors support the narrower conclusion that the identified adjacency
construction failures were corrected.

A same-protocol comparison against `trajectory-smallstep.json` found exactly
equal initial and final gallbladder vertices and all per-step diagnostic records.
The largest final coordinate difference was zero. Grasp remained established
through all ten steps; mean final vertex displacement was about 0.111 and maximum
about 0.487 in upstream scene units. This short comparison shows no observed
change to that trajectory; it does not establish stability for every action,
patient geometry, timestep or material setting.

## Reproduce the read-only audit

After installing the optional physics runtime, run from the repository root:

```bash
/workspace/.sofa-runtime/venv/bin/python scripts/audit_sofa_meshes.py \
  --runtime /workspace/.sofa-runtime \
  --output /workspace/.sofa-runtime/mesh-audit-source.json
```

To independently compare a physics worker's exported topology arrays against the
unchanged VTK tetrahedra, provide its result file:

```bash
/workspace/.sofa-runtime/venv/bin/python scripts/audit_sofa_meshes.py \
  --scene-json /workspace/.sofa-runtime/trajectory-derived.json \
  --baseline-scene-json /workspace/.sofa-runtime/trajectory-smallstep.json \
  --output /workspace/.sofa-runtime/mesh-audit-derived.json \
  --strict-scene
```

`--baseline-scene-json` is optional and compares the protocol fields, initial and
final vertices, and per-step records. Omit it for a new smoke run without a
corresponding baseline. `--scene-json` expects a current worker result containing
`topology` arrays; older results lacking those arrays can only serve as a
trajectory baseline. The audit uses NumPy, available in the optional runtime,
and never imports SOFA, initializes a scene or writes into third-party source.
It reports observed inconsistencies rather than treating successful report
generation as proof that all checks passed. With `--strict-scene`, it requires
`--scene-json` and exits 1 **after emitting the report** if the scene is not in
derived mode, face/edge connectivity differs, tetrahedral index order changes,
a component is explicitly `Invalid`, or recorded SOFA errors are nonzero.
Missing diagnostic/component observations also fail this gate. `Undefined`
remains explicitly distinct from `Valid`; passing the gate establishes the
limited connectivity checks, not physical model validity. The source assets'
known explicit-triangle defects are still reported separately and are not
mistaken for a failure of a correctly derived runtime topology.

The saved source-only and scene-comparison commands were exercised against the
artifacts named above. A strict audit of `trajectory-final.json` also passed,
writing `/workspace/.sofa-runtime/mesh-audit-final-strict.json`: complete matching
connectivity, zero errors, and both component states still `Undefined`.

The script checks combinatorial topology and the original VTK element geometry;
it does not check SOFA's private adjacency buffers or prove self-intersection
absence, material accuracy, force units or convergence. Rendered frames, finite
state and a successful grasp/short trajectory remain smoke checks. They do not
validate force limits, tissue material laws, anatomical exposure/CVS scoring,
cohort generation or the research proposal's scientific conclusions.
