# Frozen exploratory RL study 002

This study compares two demonstration-assisted reactive tabular Double-Q encodings and an unassisted V2 ablation in the illustrative stateful tissue simulator. Assisted V2 had a higher observed mean return and success rate than assisted V1, but the paired return interval includes zero: these five training seeds do **not establish V2 superiority**. The continuous scripted controller had the highest observed mean success. The unassisted arm includes severe-injury failures. These synthetic results do not validate tissue mechanics or autonomous surgery.

All three arms used training seeds 101–105, 3,000 online episodes per seed, a 200-step horizon at 0.1 seconds, gamma .99, tabular learning rate .15, reward scale 10, and the same 28 discrete controls. Both assisted arms received 100 additional training-only quantized scripted trajectories per seed, observed-return initialization, and one demonstration replay update per online step. The unassisted arm received no demonstrations or demonstration replay. The online episode budget is equal; the total observed transitions and learning assistance differ.

Each training seed had 30 validation patients, paired across all three arms and all controllers. Different training seeds had separate patient sets, giving 150 distinct validation patients. Baseline records repeat across arm directories and represent the same 150 patients; they are reported once below. The 15,500 fitting-patient seeds and 150 validation-patient seeds are disjoint across the entire study. An independent set-intersection check found no overlap with the parent-owned reserved final cohort 20000–20127. No reserved final patient was run or inspected for this report.

## All learned runs

Each row contains 30 validation episodes. Severe injury and maximum injury refer to the simulator's modeled damage metrics.

| Learner | Training seed | Validation success | Severe injury | Mean maximum injury | Mean return |
| --- | --- | --- | --- | --- | --- |
| V1 assisted | 101 | 0.0% | 0.0% | 0.01152 | -1.39651 |
| V1 assisted | 102 | 40.0% | 0.0% | 0.00315 | 5.73373 |
| V1 assisted | 103 | 0.0% | 0.0% | 0.00000 | -3.00000 |
| V1 assisted | 104 | 46.7% | 0.0% | 0.00169 | 6.88561 |
| V1 assisted | 105 | 23.3% | 0.0% | 0.00335 | 2.83046 |
| V2 assisted | 101 | 50.0% | 0.0% | 0.00314 | 7.45275 |
| V2 assisted | 102 | 63.3% | 0.0% | 0.00053 | 9.97738 |
| V2 assisted | 103 | 33.3% | 0.0% | 0.02107 | 4.05949 |
| V2 assisted | 104 | 36.7% | 0.0% | 0.00398 | 5.03136 |
| V2 assisted | 105 | 6.7% | 0.0% | 0.00901 | -1.29407 |
| V2 unassisted | 101 | 0.0% | 0.0% | 0.17648 | -5.72372 |
| V2 unassisted | 102 | 0.0% | 0.0% | 0.00266 | -2.25000 |
| V2 unassisted | 103 | 0.0% | 23.3% | 0.19535 | -10.68275 |
| V2 unassisted | 104 | 46.7% | 0.0% | 0.00551 | 6.80202 |
| V2 unassisted | 105 | 26.7% | 0.0% | 0.02899 | 2.99817 |

## Pooled means across five training seeds

All runs have equal validation sample sizes, so these pooled means also equal the means across the five training seeds. Each learner has 150 validation episodes on the same 150 patients.

| Controller | Success | Severe injury | Mean maximum injury | Mean return | 95% hierarchical return interval |
| --- | --- | --- | --- | --- | --- |
| V1 assisted | 33/150 (22.0%) | 0/150 (0.0%) | 0.00394 | 2.21066 | [-1.2142, 5.8145] |
| V2 assisted | 57/150 (38.0%) | 0/150 (0.0%) | 0.00755 | 5.04538 | [1.4742, 8.4416] |
| V2 unassisted | 22/150 (14.7%) | 7/150 (4.7%) | 0.08180 | -1.77126 | [-7.1851, 3.8870] |
| Continuous scripted | 85/150 (56.7%) | 0/150 (0.0%) | 0.00008 | 8.79955 | [6.7601, 11.0177] |
| Uniform random | 0/150 (0.0%) | 0/150 (0.0%) | 0.00000 | -2.20372 | [-2.2550, -2.1524] |
| Random without handoff | 0/150 (0.0%) | 0/150 (0.0%) | 0.00062 | -2.54007 | [-2.6384, -2.4441] |

The unassisted arm succeeded on 22/150 cases and caused seven observed severe injuries (4.67%). All seven occurred in seed 103, which had zero successes, 7/30 severe injuries (23.3%), and mean return -10.68275. Seeds 101 and 102 also had zero successes; seed 101's mean maximum injury was .17648 despite zero observed severe injuries. Zero severe injury alone does not establish useful procedure control. Uniform random always handed off in this cohort and never succeeded; its low injury is not successful control.

## Paired return contrasts

| Contrast | Mean return difference | 95% paired hierarchical interval |
| --- | --- | --- |
| V2 assisted minus V1 assisted | 2.83472 | [-1.7687, 7.2101] |
| V2 unassisted minus V1 assisted | -3.98192 | [-7.2501, -0.7045] |
| V2 unassisted minus V2 assisted | -6.81664 | [-13.7899, 0.5007] |

Intervals use 2,000 bootstrap replicates: resample the five training seeds, then patients within each selected seed. The contrasts resample the within-patient return differences, preserving the recorded arm pairing. The assisted V2 minus V1 interval [-1.7687, 7.2101] includes zero; the positive estimate alone is insufficient evidence of superiority. The unassisted-minus-V1 interval is below zero, while the unassisted-minus-assisted-V2 interval includes zero. These exploratory intervals reflect only recorded training-seed and patient variability, exclude tissue-model uncertainty, and can be unstable with five training seeds. Per-run binary rates and Wilson intervals remain in the full JSON results.

## Validation-selected candidate

The frozen rule required zero observed severe injury, then ranked validation success, lower mean maximum injury, and higher mean return. Fourteen of 15 runs were eligible under that convention, including zero-success runs. It selected **assisted V2 seed 102**: 19/30 successes (63.3%), zero observed severe injuries, mean maximum injury .00052547, and mean return 9.97738. The continuous scripted controller succeeded on 21/30 of these same patients.

This selection used the reported validation outcomes, so the selected candidate's score is optimistic and must not represent typical learning performance. Assisted V2's mean success across seeds was 38.0%, and its seed 105 succeeded on only 2/30 patients. Zero observed severe injuries in 30 cases still has a Wilson upper 95% bound of about 11.35%; the eligibility rule is not proof of safety. Candidate inference uses its Q-table without a scripted fallback. The study candidate is preserved separately in [stateful-policy-1.1-study-002.json](../examples/stateful-policy-1.1-study-002.json); this report does not promote it to the application's default or seal it using final-cohort results.

## Reduced state and numerical limits

V1 maps public measurements into 1,200 bins using grasp status, exposure error, force/limit ratio, maximum strain/limit ratio, and maximum mechanical damage. V2 expands this to 12,000 bins: acquisition/grasp context distinguishes the exposure-marker node from other nodes, damage combines mechanical damage and thermal dose, and extra flags indicate temperature above 43 C and remaining time at or below 20%. V2 still compresses the observations and has no memory of prior transitions. Different anatomy, spatial deformation, thermal distributions, and histories can share the same reactive bin. Neither encoding is a full Markov state, and adding bins does not resolve all state aliasing.

Both encodings reconstruct physical measurements from normalized floating-point observations, round the four scalar bin comparisons to six decimal places, and use right-sided boundary insertion. This reduces the known zero-exposure reconstruction ambiguity; it does not guarantee cross-platform numerical equivalence. V2 also uses nearest-node selection, a 3 mm acquisition boundary, a .9 combined-damage intactness boundary, temperature and time thresholds, and a rounded contact index. Near thresholds or ties, floating-point differences can change bins, chosen controls, and subsequent trajectories. Source identity is not a general numerical or clinical validity guarantee.

## Frozen source and records

The model is `tissue-retraction-1.1`, with exact physics fingerprint `99bcc146975fff6221c65ee927d4bc38373a18069e56ce8cd45bf7a75c984898`. All tissue mechanics, harm weights, and rewards remain fixed illustrative assumptions. No physics or reward coefficient changed for this comparison.

The four declared source hashes were independently checked against `git show` at commit `f4f65b8b327b04e0d1169e50bfa778ef5763160d` and against the root and all three per-arm frozen source snapshots. All matched. The protocol hash also matches the summary. Every policy file matches the checkpoint embedded in its result file, and every one of the 15 result files contains all 3,000 learning-curve episode records.

- [Exact frozen protocol](../examples/rl-study-002-protocol.json)
- [Exact full summary and fitting/validation seed unions](../examples/rl-study-002-summary.json)
- [Complete study archive](../research-artifacts/rl-study-002-full.zip) and [archive SHA-256](../research-artifacts/rl-study-002-full.zip.sha256)
- [Study 001](rl-study-001.md), [learning methods](rl-training.md), and [illustrative tissue equations](tissue-response.md)

The archive preserves all 15 checkpoints, complete learning curves and raw comparator episodes in the result files, exact source snapshots, protocol, summary, configurations, completion markers, CSVs, and logs. It includes harmful and zero-success runs. Sorted ZIP entries use fixed 1980 timestamps; `MANIFEST-SHA256.json` records every payload file's hash, and every manifest hash was verified from the completed archive. The archive is a complete development/validation record, not a final-test result or a clinical controller.

## Reproduce

Use the recorded source commit and locked dependencies, then extract the archive into a separate directory (shown as `study002-records` below). The archived arm configurations preserve the original settings. Each command creates a fresh output directory and refuses to overwrite earlier records:

```bash
.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config study002-records/tabular_v1-config.json --seeds 101 102 103 104 105 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-002-v1

.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config study002-records/tabular_v2-config.json --seeds 101 102 103 104 105 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-002-v2

.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config study002-records/tabular_v2_unassisted-config.json --seeds 101 102 103 104 105 \
  --episodes 3000 --demonstration-episodes 0 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-002-v2-unassisted
```

These are reproduction instructions only; this report and archive were built from completed frozen records without retraining or running final patients. Dependency/platform floating-point differences can change results near decision thresholds.
