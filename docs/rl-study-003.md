# Frozen exploratory RL study 003

The larger V3 reactive representation did **not show evidence of improvement** over V2 in this paired five-training-seed study. V3 had 47/150 validation successes (31.3%) and one observed severe-injury outcome, versus V2's 53/150 successes (35.3%) and zero observed severe injuries. Its mean return was lower, but the paired V3-minus-V2 interval includes zero and substantial differences in both directions. These results neither establish V2 superiority nor equivalence. The continuous scripted comparator had the highest observed mean success. All ten runs, including failures, are reported and archived.

These are development/validation measurements in an illustrative stateful tissue simulator. They do not validate tissue mechanics, clinical safety, or autonomous surgical competence. Neither candidate replaces the application's bundled V1 reference.

## All learned runs

Each row contains 30 validation episodes. Injury denotes the simulator's modeled damage, not a calibrated clinical endpoint.

| Learner | Training seed | Validation success | Severe injury | Mean maximum injury | Mean return |
| --- | --- | --- | --- | --- | --- |
| V2 assisted | 201 | 50.0% | 0.0% | 0.02519 | 6.83128 |
| V2 assisted | 202 | 0.0% | 0.0% | 0.00539 | -1.22655 |
| V2 assisted | 203 | 0.0% | 0.0% | 0.00000 | -3.00000 |
| V2 assisted | 204 | 76.7% | 0.0% | 0.00221 | 12.00251 |
| V2 assisted | 205 | 50.0% | 0.0% | 0.00528 | 7.26137 |
| V3 assisted | 201 | 30.0% | 0.0% | 0.00709 | 3.95360 |
| V3 assisted | 202 | 46.7% | 0.0% | 0.00101 | 7.22621 |
| V3 assisted | 203 | 6.7% | 3.3% | 0.03582 | -1.25043 |
| V3 assisted | 204 | 0.0% | 0.0% | 0.00000 | -3.52873 |
| V3 assisted | 205 | 73.3% | 0.0% | 0.00036 | 11.49945 |

V2 seeds 202 and 203 never succeeded. V3 seed 203 produced the study's one observed severe-injury outcome (1/30), and V3 seed 204 handed off in all 30 cases without succeeding. Low modeled injury alone is not useful exposure control. The archive retains these runs and every negative-return learning-curve episode.

## Pooled means across five training seeds

Every learned arm contains 150 validation episodes on the same 150 distinct patients. Equal sample sizes make pooled means equal to means across training seeds. Baseline records are identical across the two arm directories and are counted once here.

| Controller | Success | Severe injury | Mean maximum injury | Mean return | 95% hierarchical return interval |
| --- | --- | --- | --- | --- | --- |
| V2 assisted | 53/150 (35.3%) | 0/150 (0.0%) | 0.00761 | 4.37372 | [-0.4262, 9.2173] |
| V3 assisted | 47/150 (31.3%) | 1/150 (0.7%) | 0.00885 | 3.58002 | [-1.4457, 8.6132] |
| Continuous scripted | 95/150 (63.3%) | 0/150 (0.0%) | 0.00007 | 9.93174 | [7.5090, 12.2688] |
| Uniform random | 0/150 (0.0%) | 0/150 (0.0%) | 0.00000 | -2.18582 | [-2.2417, -2.1373] |
| Random without handoff | 1/150 (0.7%) | 0/150 (0.0%) | 0.00480 | -2.54770 | [-2.9785, -2.0244] |

Uniform random handed off in every case and never succeeded. Its zero observed injury must not be interpreted as successful procedure control. The continuous scripted controller is not a learned policy and uses continuous feedback commands, while the learned arms select from a finite primitive library.

## Paired return contrast

| Contrast | Mean return difference | 95% paired hierarchical interval |
| --- | --- | --- |
| V3 assisted minus V2 assisted | -0.79370 | [-8.6668, 5.6443] |

The recorded interval uses 2,000 bootstrap replicates with RNG seed 98911: resample the five training seeds, then the 30 patients within each sampled seed. The contrast resamples corresponding within-patient V3-minus-V2 returns, preserving both training-seed and patient pairing. The 2.5th and 97.5th percentiles form the interval. With only five training seeds it is unstable, excludes model uncertainty, and provides no evidence that the representation change improves performance. Per-run binary Wilson intervals and patient-level return intervals remain in the full JSON results.

## Validation-selected candidates

The predeclared rule requires zero observed severe injury, then ranks validation success, lower mean maximum injury, and higher mean return. Nine of ten runs qualify under that convention, including zero-success runs. It selects **V2 seed 204** globally: 23/30 successes (76.7%), zero observed severe injuries, mean maximum injury .00221219, and mean return 12.00251. The same rule within V3 selects **seed 205**: 22/30 successes (73.3%), zero observed severe injuries, mean maximum injury .00036065, and mean return 11.49945.

Both selections use the reported validation outcomes and therefore have optimistic selection bias. Their success rates must not stand in for the corresponding five-seed means of 35.3% and 31.3%. For either candidate, zero severe injuries in 30 cases still has a Wilson upper 95% bound of about 11.35%; eligibility is not proof of safety. Candidate inference uses the recorded Q-table, without a scripted fallback.

- [Global V2 validation candidate](../examples/stateful-policy-1.1-study-003.json)
- [Prespecified within-V3 validation candidate](../examples/stateful-policy-1.1-study-003-v3.json)

The candidates remain separate study artifacts. No reserved final patient was used to choose or tune them, and this report does not promote a policy or use final-cohort outcomes.

## Design and limits

The two arms use training seeds 201–205, 3,000 online episodes per seed, 100 additional training-only quantized scripted demonstration trajectories, a 200-step horizon at 0.1 seconds, gamma .99, tabular learning rate .15, reward scale 10, and the same 28 discrete control primitives. Demonstrations choose the nearest declared primitive to the continuous script; the continuous scripted evaluation comparator is a different controller. Both arms initialize action values from demonstration returns and perform one demonstration replay update per online step. The recorded neural-network learning rate, observation-encoding option, demonstration margin, and demonstration weight are unused DQN settings for these tabular arms. This is an equal online-episode and demonstration budget, not an unconstrained continuous-control comparison.

Each training seed uses 30 validation patients, paired across both encodings and all controllers. Different training seeds use different patient sets. There are 15,500 distinct fitting-patient seeds and 150 distinct validation-patient seeds, disjoint across the entire study. An independent set-intersection check finds no overlap of either union with reserved final seeds 20000–20127. The comparator records repeat across arm directories and must be counted once, rather than as independent replicates. Reserved final patients do not enter training, validation selection, or this development report.

V2 has 12,000 reactive bins. It reconstructs public physical measurements through its normalized Box transformation, rounds scalar bin measurements to six decimals, and uses right-sided threshold insertion. V3 has 34,560 bins and changes several things together: it uses clipped raw public float32 observations promoted to float64, adds force/strain ratio 1.0 and critical-site injury .08 boundaries, changes acquisition ordering to nearest-node-then-reject-damage, and makes its late-horizon comparison explicit. This is a representation and numerical-convention comparison; it does not isolate the causal effect of one new feature.

V3 quantizes scalar bin comparisons to integer micro-units with nearest-even rounding. Quantized exposure, force, and strain equality enter the lower bin; quantized damage equality enters the higher bin. This convention does not recover the simulator's internal float64 values exactly. Acquisition, temperature, and other direct comparisons remain floating point; trajectories can diverge near thresholds or ties. Source identity and boundary tests do not guarantee cross-platform numerical identity.

Both encodings use explicit simulator-derived priors: node 03 is the exposure marker, acquisition distance is 3 mm, intactness requires combined injury below .9, and heat above 43°C is flagged. V3 also names critical nodes 16–22 and includes the model's .08 critical-injury acceptance boundary. These are public schematic model features, not clinical tissue knowledge. Acceptance boundaries differ from severe-injury termination thresholds. Many tool positions, stable-hold progress, physiological details, anatomy, and histories still share a bin; neither representation is a complete Markov state or a validated surgical controller.

## Frozen source and records

The simulator is `tissue-retraction-1.1`, with mechanics fingerprint `99bcc146975fff6221c65ee927d4bc38373a18069e56ce8cd45bf7a75c984898`. No tissue equation, harm weight, or reward coefficient changed for this comparison. The four declared source hashes match exact Git bytes at commit `21f0e35e351f177619ab2c665ec3f12996bc365b`, the root source snapshot, and both per-arm snapshots.

- [Exact frozen protocol](../examples/rl-study-003-protocol.json)
- [Exact audited summary and fitting/validation seed unions](../examples/rl-study-003-summary.json)
- [Complete ten-run archive](../research-artifacts/rl-study-003-full.zip) and [archive SHA-256](../research-artifacts/rl-study-003-full.zip.sha256)
- [Study 002](rl-study-002.md), [learning methods](rl-training.md), and [illustrative tissue equations](tissue-response.md)

The archive preserves all ten checkpoints, all 3,000 learning-curve records per run including negative returns, raw comparator episodes, exact source snapshots, protocols, audited summary, configurations, completion markers, CSVs, and logs. Every standalone checkpoint equals the checkpoint embedded in its result JSON. Sorted ZIP entries use fixed 1980 timestamps and a per-file `MANIFEST-SHA256.json`; the package is built twice from identical inputs and compared byte for byte, then every archived manifest entry is checked. It is a development/validation record, not final-test evidence or the application's live policy.

## Reproduce

Use the recorded source commit and locked dependencies. Extract the archive into a separate directory, shown as `study003-records`; retain the original arm configurations. Each command creates fresh records and refuses to overwrite an earlier output:

```bash
.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config study003-records/tabular_v2-config.json --seeds 201 202 203 204 205 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-003-v2

.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config study003-records/tabular_v3-config.json --seeds 201 202 203 204 205 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-003-v3
```

These commands are reproduction instructions only. This report and ZIP package were assembled from completed records without running a policy or reserved final patients. Dependency/platform floating-point differences can change decisions near boundaries.
