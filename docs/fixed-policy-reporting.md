# Reporting a completed fixed-policy evaluation

The report generator reads completed records. It never imports the simulator,
policy loader, trainer, or evaluator, and cannot launch a rollout or select a
replacement checkpoint.

After the evaluation owner has produced a completed bundle:

```bash
python scripts/report_frozen_evaluation.py \
  --evaluation .data/final-evaluation-001 \
  --output .data/final-evaluation-001-report \
  --plots
```

The evaluation path is an example; use the owner's actual completed directory.
The output must be a new directory. Matplotlib is optional figure tooling;
NumPy and the standard library suffice without `--plots`.

With the default design, the generator automatically reads the prespecified
[12-policy design](../examples/final-evaluation-001-design.json),
[policy hash freeze](../examples/final-evaluation-001-policy-freeze.json), and
[crossed-analysis supplement](../examples/final-evaluation-001-crossed-analysis.json).
Alternative development fixtures can specify `--design`; explicit
`--policy-freeze` and `--crossed-analysis` paths remain available. The generator
retains design order followed by three baseline controls. It never sorts by
performance, adds a validation-selected alias as another controller, or changes
the existing reference policy.

Before creating output it verifies completion-marker and protocol/result hashes,
copied policy bytes and canonical checkpoint/result hashes, the prespecified
policy freeze, environment/source/exclusion identity, patient alignment,
rollout-log contents, episode totals, outcome aggregates, Wilson intervals,
and the complete set of seeded paired-return bootstrap intervals.

Outputs are `report.md`, `controllers.csv`, `paired-return-contrasts.csv`, and
`report-complete.json`. Optional `fixed-policies.svg` and PNG show every fixed
policy and baseline in prespecified order. The completion manifest records input
hashes, generator hash, output hashes, and analysis plans. A missing report
completion manifest means report generation did not complete.

Individual-policy intervals are conditional on fixed checkpoints: Wilson 95%
bounds for success/severe injury and patient-only percentile bootstrap intervals
for return, using the frozen evaluation seed and replicate count. They exclude
training-run and tissue-model uncertainty. Every controller uses the same
patients; 1,920 rollouts from 15 controllers on 128 patients are not 1,920
independent patients. Pairwise intervals are unadjusted for multiplicity.

The prespecified study-003 supplement forms five paired V3-minus-V2 outcome
rows crossed with the shared patient cohort. Each replicate draws five paired
training-seed indices and **one common patient-index vector**, then takes their
Cartesian-product mean. Return, success, and severe injury share the same
indices. CSVs retain crossed contrasts, each paired-seed descriptor, and the
two fixed-policy set descriptors. The report does not pool 640 observations
per arm into independent binomial patients or assign a pooled Wilson interval.
These descriptive intervals can be unstable with five training seeds, exclude
model/clinical uncertainty, and are unadjusted across outcomes and comparisons.

Tests use the retained genuine development-evaluation fixture and independent
synthetic interaction matrices. Report development does not execute the reserved
final cohort. The research owner controls any final evaluation and release.
