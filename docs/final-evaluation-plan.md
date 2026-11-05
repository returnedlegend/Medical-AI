# Fixed-policy final evaluation plan

**Status: completed once on 2026-11-05 at 18:28:24 UTC.** The [final report](rl-final-001.md)
and complete archive preserve all 1,920 rollouts and independent audits. Seeds
20000–20127 are now consumed final data. Do not reuse them as an untouched test
for a tuned or replacement policy. The pre-outcome JSON design, policy freeze
and crossed-analysis plan remain unchanged; the preparation instructions below
are retained as the historical plan, not authorization for another run.

The evaluator is implemented and tested on **development patients only**.
The reserved final patient seeds are 20000–20127, a cohort of 128. Do not execute
that cohort until the parent/root task explicitly authorizes the final run after
policy/method freeze and independent exclusion review. No reserved final
patients were used to develop or test this evaluation script.

The objective is a paired comparison of **fixed** policies and public baselines
in the same synthetic tissue environment. It is not further training, a new
model-selection contest, or a claim of clinical performance.

The [prespecified design for evaluation 001](../examples/final-evaluation-001-design.json)
includes the existing live V1 reference, study 002's validation-selected V2,
and every paired V2/V3 policy from study 003's five training seeds 201–205.
That is twelve fixed policies and three shared baselines on the 128 reserved
patients: 1,920 episodes. The V3 validation-selected candidate is an alias of
one included policy. The release reference remains unchanged regardless of
final outcomes. This controller set was recorded before study 003 finished;
the completed file hashes and exclusion audit must still be frozen before
the root authorizes any final rollout.

## Before the reserved run

1. Freeze the complete run/reference JSON files to evaluate, their labels, the
   simulator fingerprint, the environment configuration, and the algorithm
   decoder versions. Include the already-selected V1/V2 examples and any
   prespecified all-seed policies; do not choose this set after observing final
   patient outcomes.
2. Construct a distinct JSON list containing the union of **all prior fitting,
   demonstration, development, previously inspected and validation patient
   seeds**, across all relevant studies and engineering runs. Retain the study
   protocols and the source records used to construct that list. The root task
   must independently verify that the reserved 128 seeds are disjoint from this
   complete union, not just from the selected policy's training run.
3. Save the reserved seed list and exclusion union as fixed input files. Record
   the evaluation labels, baseline RNG seed 99173, paired-bootstrap RNG seed
   99174, and 2,000 resamples before rollouts. Every compared policy must declare
   the exact same normalized environment configuration, including timestep and
   horizon. Different timesteps must be separate experiments rather than pooled
   as an unchanged surgical task.
4. Obtain the root task's explicit go-ahead. Passing development tests or reaching
   a clock time is not authorization to consume the reserved cohort.

The script checks that the exclusion list covers every input policy's recorded
fitting/demo/validation history and rejects any evaluation overlap. It cannot
independently discover omitted studies or undocumented earlier inspections.
Completeness of the externally assembled union is therefore an explicit root
review responsibility, recorded as a limitation rather than silently assumed.

## Evaluate, without training or selecting

From the repository root, **only after authorization**, the command has this
form (replace paths with the frozen inputs):

```bash
PYTHONPATH=backend .venv/bin/python scripts/evaluate_frozen_policies.py \
  --policy selected_v1=path/to/frozen-v1-reference.json \
  --policy selected_v2=path/to/frozen-v2-reference.json \
  --seeds-file path/to/reserved-final-seeds.json \
  --exclusion-seeds-file path/to/all-prior-used-seeds.json \
  --output .data/final-fixed-policy-evaluation-001 \
  --phase final
```

Additional `--policy label=path` arguments add prespecified policies. Labels are
lowercase letters/digits/underscore/hyphen, beginning with a letter. Inputs are
JSON only, with no pickle or executable model payload. Each file is limited to
32 MiB; at most 32 policies and 256 MiB total input are allowed. Complete
training-result JSON and bundled-reference JSON are supported; a bare checkpoint
without fitting/validation provenance is rejected.

For Windows PowerShell, set `$env:PYTHONPATH="backend"` and run
`.venv\Scripts\python.exe`. The evaluator uses the portable 23-site analytic
model, not the optional SOFA runtime.

The evaluator validates declarative policy arrays, environment version, physics
fingerprint, observation/action specifications and exact environment settings.
It compiles each neural or reactive-table policy once and uses greedy Q outputs
without a scripted fallback. A narrow private feature-encoding adapter is used
for both policy types; the decoder source hash is recorded with the run.

Public baselines run once per patient, independent of how many policies are
being compared:

- `baseline:random` samples the canonical declared action library uniformly,
  including handoff.
- `baseline:random_no_handoff` samples that same library after excluding every
  primitive whose handoff component exceeds 0.5.
- `baseline:scripted` uses the versioned public-observation feedback controller,
  without accessing hidden tissue state or an oracle.

Random streams are fixed per patient and baseline mode using
`SeedSequence([99173, patient_seed, mode])`, with mode 1 for random and mode 2 for
random without handoff. They do not depend on a policy's training seed, label,
evaluation order or number of competing policies. All controllers reset the same
patient seeds, and the evaluator verifies identical initial patient anatomy.
The full canonical action library and scripted-controller version are saved in
the protocol.

## Immutable setup and evidence of completion

The output directory must be new. Existing directories and files are never
overwritten. Before the first environment reset, the evaluator copies the exact
policy input bytes and writes/flushes its protocol, including:

- policy file and checkpoint SHA-256 values;
- source hashes for the evaluator, neural/table decoders, environment and tissue;
- simulator fingerprint, Python/NumPy versions and normalized environment;
- patient and exclusion seed lists with canonical hashes;
- baseline streams/action library and bootstrap settings;
- phase (`development` or `final`) and interpretation limits.

During execution, each completed episode is appended to `episodes.jsonl`.
`results.json` contains actual per-patient returns, success, severe harm,
handoff, termination/truncation, simulated injury, losses, exposure, force and
temperature maxima, reward-component totals and final-state hashes.

`complete.marker` is written **only after** all expected episodes complete,
paired-anatomy and count checks pass, and final results are saved. Its result and
protocol hashes identify the completed artifacts. A numerical exception or
other rollout failure instead leaves the protocol/logs and `failed.json`, with
no completion marker; it is not silently recoded as a completed safe episode.
Do not use a partial run as the final comparison or describe a zero-episode run
as verification.

## Report the paired outcomes

For every fixed policy and baseline, report the actual episode count, observed
success and severe-harm counts, rates and 95% Wilson intervals, mean return,
injury and losses. With zero observed severe harm in 128 modeled patients, the
95% Wilson upper bound remains approximately **2.91%**; zero observed events do
not imply zero future risk.

Each controller pair has an explicitly labeled positive and negative controller
and a paired-patient mean-return difference. A common bootstrap sample of
patient indices is used for every contrast, preserving pairing. Read the stored
sign labels; do not assume every contrast is policy minus baseline. Intervals
condition on the fixed trained checkpoints and synthetic patient generator.
They exclude training-run variability, constitutive-model error and clinical
generalization uncertainty. Pairwise intervals are unadjusted for multiple
comparisons and must not be presented as a familywise-confirmed superiority
claim.

The evaluator has no ranking, best-policy selection or reference-policy export
step. If final outcomes are unfavorable, retain and report them. Any subsequent
tuning consumes this cohort as development data and requires a new untouched
cohort for a later final evaluation. Publishing a new reference selected from
these outcomes would be model selection, not evaluation of the previously frozen
reference.

## Development verification already performed

Tests train a tiny 20-episode reactive-table fixture, then evaluate only seeds
4000000000–4000000004. Its generated fitting/evaluation streams are checked for
reserved-cohort exclusion before fixture training. Tests cover real paired
rollouts, no training calls from the evaluator, immutable protocol/input hashes
before reset, common baseline RNG, identical-policy zero paired difference,
prior-use overlap and missing-exclusion refusal, configuration/fingerprint
mismatch, output non-overwrite, numerical failure without completion, nonzero
zero-harm confidence bounds, bounded declarative inputs, and an actual
`--phase development` CLI run. These checks exercised real outcomes, not a
conditional shortcut that skipped the simulator.

The development checks above preceded the separately authorized final run;
they are not final-cohort results. The completed assessment is reported in the
linked final report and is not a new authorization to execute the cohort.
All results concern unvalidated analytic tissue and toy physiology, not patient
care or proven surgical competence.
