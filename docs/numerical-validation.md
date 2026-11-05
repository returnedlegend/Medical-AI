# Numerical implementation verification

The frozen 23-site tissue surrogate has been checked against closed-form
solutions of several **stated model equations**, and a constant-command pull has
been compared across timesteps. This tests numerical implementation and exposes
timestep sensitivity. It does not validate those equations against human tissue,
human physiology, clinical limits, or surgical outcomes.

The executed [compact report](validation/tissue-numerics-1.1.json) records
`tissue-retraction-1.1`, Python/NumPy versions and physics fingerprint
`99bcc146975fff6221c65ee927d4bc38373a18069e56ce8cd45bf7a75c984898`.
The script reads the engine and does not modify its source, model constants,
reward, thresholds or learned policies.

## Reproduce

From the repository root:

```bash
PYTHONPATH=backend .venv/bin/python scripts/validate_tissue_dynamics.py \
  --patients 20 --first-seed 10000 \
  --output artifacts/numerical-validation.json
```

On Windows PowerShell, set `$env:PYTHONPATH="backend"` and use
`.venv\Scripts\python.exe`. The script needs the ordinary backend dependencies;
SOFA and a GPU are not required. It writes quantitative results and exits
nonzero if the stated implementation checks fail. Those checks are numerical
engineering criteria, not clinical acceptance tests. The full generated report
includes all per-patient results; the checked-in example retains their aggregate
and records the seed range, 10000–10019.

## Free cooling and accumulated dose

A controlled initial condition sets one stationary site to 80°C, with ambient
37°C and no powered tool. The model specifies Newton cooling with a 3-second
time constant. Its exact solution is

`T(t) = 37 + 43 exp(−t/3)`.

The independent reference for accumulated dose integrates
`max(T(t)−43,0)/100` analytically until the temperature crosses 43°C. For a
6-second observation window the exact final temperature is 42.819417°C and the
exact accumulated dose is 0.755501 in the model's normalized dose units. Cooling
and dose remain deliberately illustrative constitutive assumptions.

| Action timestep | Maximum temperature error | Final temperature error | Dose absolute error |
| --- | ---: | ---: | ---: |
| 0.02 s | 0.052876°C | 0.038839°C | 0.006216 |
| 0.10 s | 0.267371°C | 0.195058°C | 0.031050 |
| 0.20 s | 0.542448°C | 0.392268°C | 0.061945 |

Errors shrink approximately in proportion to timestep, consistent with the
implemented first-order update. In particular, the default 0.1-second timestep
is not exact: it underestimates the dose in this cooling test by about 4.1%.
Passing the check does not establish a clinically acceptable thermal error.

## Injury changes force and stiffness

A controlled component fixture isolates one restoring spring and its grasper
contact spring by removing neighboring springs **only in the offline test
instance**. One copy is intact; the other receives 1.2 seconds of excessive jaw
closure, creating persistent modeled injury 0.264. Both then receive the same
0.5-mm tool displacement at closure 0.55, below further injury activation.

For this fixture, the exact overdamped solution is independently solvable:

`du/dt = [kc·pull − (kr+kc)u] / 0.65`,

`u(t) = kc·pull/(kr+kc) · [1−exp(−(kr+kc)t/0.65)]`,

`F(t) = kc[ pull − u(t) ]`.

The implementation is compared with that solution after 4 seconds at a 0.02-s
action timestep. Injury is verified unchanged during the comparison, so the
analytic constant-coefficient assumption remains applicable.

| Fixture | Restoring stiffness | Observed force | Exact force | Relative force error |
| --- | ---: | ---: | ---: | ---: |
| Intact | 0.080000 N/mm | 0.039239 N | 0.039274 N | 0.0891% |
| Previously crushed | 0.060992 N/mm | 0.034707 N | 0.034750 N | 0.1216% |

The injured fixture has 23.76% lower restoring stiffness and lower transmitted
force. Its contact coupling also weakens, so a stiffness reduction need not mean
larger tool-driven displacement. This verifies that stored injury changes the
next mechanical response; it does not verify real tissue failure laws.

## Constant-command trajectory sensitivity

Twenty seeded patients vary attachment length, adhesion parameters and
distension. Each starts with contact at the lower-right marker, closure 0.55,
and receives the same **1 mm/s lateral velocity for 4 seconds**. There are no
policy decisions. States are compared at matched physical times 1, 2, 3 and
4 seconds against a 0.005-second numerical reference using the same equations.
Tool targets are sampled and held per action interval, so the differences
include both command sampling and integration error.

| Action timestep | Worst RMS position difference across sites | Worst single-site difference | Worst force difference | Worst exposure difference |
| --- | ---: | ---: | ---: | ---: |
| 0.02 s | 0.001392 mm | 0.006190 mm | 0.003740 N | 0.000611 |
| 0.10 s | 0.008715 mm | 0.038650 mm | 0.023354 N | 0.003810 |
| 0.20 s | 0.017620 mm | 0.077879 mm | 0.047054 N | 0.007666 |

All twenty gentle trajectories remained below injury activation. These small
numerical differences describe this particular low-force trajectory, not every
possible action or patient. The fine numerical reference shares the same model
and is not independent physical ground truth; residual error below 0.005 s has
not been bounded here.

The model has **23 lumped material sites, not a volumetric FEM mesh**. This is
timestep sensitivity, not organ geometry/mesh convergence, and it does not
satisfy the research proposal's requested multi-resolution tissue validation.

## Loss conservation and pressure-to-flow feedback

A prescribed persistent artery injury of 0.3 is used to test the loss equations
without heating, coagulation or moving tools. Before any pressure floor or
reservoir limit binds, the model reduces to

`dB/dt = C(1−B/2000)`, where `C = 55(0.3−0.08) = 12.1 ml/s`.

The exact solution is `B(t)=2000[1−exp(−Ct/2000)]`. At 30 seconds this gives
331.963509 ml lost, with model pressure 75.061642 mmHg. The product `Ct/2000`
is dimensionless. The report compares cumulative loss, pressure and heart rate
against this analytic reference and independently integrates the reported
interval flow to check discrete volume accounting.

| Action timestep | Blood-loss absolute error at 30 s | Pressure absolute error | Maximum discrete balance residual |
| --- | ---: | ---: | ---: |
| 0.02 s | 0.018318 ml | 0.000824 mmHg | 0 ml |
| 0.10 s | 0.091616 ml | 0.004123 mmHg | 0 ml |
| 0.20 s | 0.183301 ml | 0.008249 mmHg | 0 ml |

Loss was monotone, and the declining model pressure reduced interval flow from
12.1 to about 10.1 ml/s. No floor/cap was active. The reported flow is the rate
used over the last interval; its pressure input is the previous interval's
state. These are checks of the specified toy feedback law and bookkeeping,
**not human hemodynamic validation**. The controlled injury fixture is a
mathematical initial condition, not a simulated surgical intervention.

## Cautery stopping times differ with timestep

The public Gymnasium loop moves an unpowered cautery tool to the same arterial
site in 1.2 seconds, then requests five seconds of full power. The configured
injury stop ends every run early. No steps are taken after termination, and the
runs are not stretched or relabeled to imply equal exposure time or success.

At the stationary contact site, before the model's temperature cap, the exact
continuous heating law is `T(t)=37+480[1−exp(−t/3)]`. Integrating its positive
excess over 43°C and solving `1−exp(−dose/0.8)=0.35` gives a continuous injury
threshold time of **0.723369 seconds** and temperature 139.842404°C. The
reference remains below the 180°C cap, so that branch does not invalidate the
calculation.

| Action timestep | Actual powered duration | Difference from continuous threshold time | Injury at stop | Blood lost at stop |
| --- | ---: | ---: | ---: | ---: |
| 0.02 s | 0.72 s | −0.003369 s | 0.356162 | 2.783461 ml |
| 0.10 s | 0.70 s | −0.023369 s | 0.376048 | 3.649071 ml |
| 0.20 s | 0.80 s | +0.076631 s | 0.497631 | 7.996271 ml |

All runs end as **severe injury**, not success or completed five-second heating.
The 0.2-s interval overshoots the 0.35 injury threshold by about 0.148; the
0.02-s interval overshoots it by about 0.006. Thermal integration and checking
stops only at action boundaries both matter. Event timing and final injury are
therefore materially timestep-sensitive even though the gentle pull is much
less sensitive. This limitation is retained in the report; the frozen engine
has not been silently changed to hide it.

The critical artery coefficients do not vary with the gallbladder's sampled
anatomy in this model. This cautery experiment consequently uses one reference
patient, rather than presenting twenty identical arterial runs as evidence of
patient diversity.

## Interpretation boundary

The report's passing engineering checks support implemented conservation,
closed-form equation consistency and the measured timestep behavior. They do
not establish calibrated damage, correct anatomical exposure, clinical safety,
spatial convergence, robust autonomous surgery or biological truth. In
particular, policy evaluation must retain its environment version, source
fingerprint and timestep; results from different stop discretizations should
not be pooled as if they came from one unchanged task.
