# Retraction Lab

A computational research workbench for **“Does one retraction fit all?”**: explore whether simulated gallbladder retraction benefits from patient-specific strategy selection, and whether an early force probe predicts who needs adaptation.

The MVP implements an **analytic surrogate**, not LapGym/SOFA. It uses synthetic tissue parameters and a geometric exposure proxy. Its outputs are exploratory model results; they do not establish surgical safety or efficacy.

## MVP scope

- Seeded virtual patients varying attachment length, adhesion burden, and distension.
- Finite grasp-point, pull-direction, and distance sweeps with force and strain constraints.
- Compare a shared strategy selected on training patients with the best grid strategy for each held-out patient.
- Estimate personalization benefit, characterize grid bail-out cases, and evaluate a one-second force-probe classifier on held-out patients.
- Interactive cohort dashboard, patient inspection, saved run history, and downloadable results.

Implementation and launch instructions are being added alongside the first working milestone. See [the research plan](docs/research-plan.md) for the scope and study safeguards.
