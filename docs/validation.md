# MVP validation record

Validated in the managed Linux x86_64 environment on 2026-11-05. These checks establish software and runtime behavior, not biological accuracy or surgical safety.

## Core workflow

- Clean project virtual environment installed from PyPI; all resolved backend versions saved in `backend/requirements.lock`.
- `scripts/setup.sh` exercised end to end, including frozen npm installation and production build.
- `scripts/check.sh`: 95 backend tests passed, including nine unittest subtests. One upstream Starlette/httpx deprecation warning remains; no tests were skipped to obtain this result.
- TypeScript checks and Vite production build passed. Frontend dependencies audited with zero reported vulnerabilities after upgrading Vite to 6.4.5.
- Actual CLI execution and overwrite protection tested for single experiments and study bundles. A 15-run bundle (five seeds × three threshold profiles, 400 patients each) completed, preserving its protocol and per-patient records under local `.data/validation-study`.
- Automated axe-core 4.10.3 accessibility audit: zero WCAG 2 A/AA and 2.1 A/AA violations across ten desktop/mobile states. Manual/incomplete findings remain separate; this does not assert comprehensive accessibility certification.
- Main Chromium browser suite: 17 checks passed, including actual cohort creation, all 75 strategy points, live custom-pull mechanics, patient-switch cancellation, held-out subgroup tables, verified CSV/JSON/Markdown downloads, history/reload, configuration reuse, modal keyboard navigation, explicit mocked error/offline recovery, and 390px mobile layout. No unhandled JavaScript errors.

## Optional physical scene

- Official SOFA 24.06 Linux archive SHA-256 checked before loading native code; separate Python 3.10.22 and pinned upstream source/packages retained outside the application environment.
- Real RGB reset, neutral stepping, and scripted grasp plus bounded actuator motion executed successfully. Real initial/final rendered PNGs and finite 367-vertex gallbladder telemetry saved.
- Independent VTK audit identified actual upstream lower-topology defects. Derived connectivity wrapper produces exactly the faces and edges implied by unchanged tetrahedra. Audit checked no missing/extra/duplicate faces or edges and preserved original tetrahedron order and vertex coordinates.
- Corrected liver: 273 tetrahedra, 639 unique faces, 463 unique edges. Corrected gallbladder: 1,063 tetrahedra, 2,490 unique faces, 1,792 unique edges.
- Corrected short grasp/pull trajectory matched the original trajectory’s numerical records, with zero SOFA errors. Nonfatal warnings remain. Component states are `Undefined`, not explicitly `Valid`; no physical-validation claim follows.
- Optional Chromium physics suite: seven checks passed using real neutral and scripted-grasp jobs, rendered images, structured diagnostics, persistent separate history, mobile layout, and mocked missing-runtime handling. No JavaScript errors.

- Installer repeatability passed, including a fresh native extraction and pinned source archive; the fresh-path check reused the previously verified release archive and installed Python dependency environment. The final saved installer also passed its strict scene-connectivity gate.

## Evidence and limits

The reusable tests live in `backend/tests/`, `scripts/browser_smoke.py`, `scripts/physics_browser_smoke.py`, and `scripts/audit_sofa_meshes.py`. Browser evidence remains under `/tmp/retraction-browser-smoke` and `/tmp/retraction-physics-browser` on this instance; these directories may not survive a new task. A representative screenshot is checked into `docs/images/workbench.png`.

CI is configured to run the backend/build checks and core browser/accessibility suites. Local execution does not establish that a remote GitHub Actions run or a new-task snapshot restoration has passed. Optional physical-scene assets are deliberately not downloaded by lightweight core CI.

The analytic model’s heterogeneity is partly imposed, and the physical scene lacks the proposed patient parameterization, dual-retractor protocol, CVS landmarks, calibrated interaction-force limits, mesh-convergence validation, and learned RL agents. These remain research/engineering milestones rather than completed study results.
