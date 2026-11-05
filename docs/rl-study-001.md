# Frozen exploratory RL study 001

This pilot compares two **demonstration-assisted** reinforcement-learning baselines in the illustrative stateful tissue simulator. It does not validate tissue mechanics or autonomous surgery. Across five training seeds, the continuous scripted controller succeeded more often and caused less modeled severe injury than either learned method.

Both learners used 3,000 online episodes plus 100 training-only quantized scripted trajectories per seed, a 200-step horizon at 0.1 seconds, gamma .99, and the same 28 declared controls. Training seeds were 43–47. Each seed had 30 separate validation patients, paired between algorithms and all controllers. The neural method used Double-DQN with the recorded large-margin demonstration assistance; the tabular method used 1,200 reactive bins, observed-return initialization and Double-Q TD updates. The source and complete hyperparameters were frozen before training. No physics or reward coefficient changed during this study.

The 15,500 fitting-patient seeds and 150 validation-patient seeds are disjoint across the entire study. Neither union intersects the parent-owned reserved final cohort 20000–20127. Those reserved patients were not run or inspected for this report.

## All learned runs

| Learner | Training seed | Validation success | Severe injury | Mean maximum injury | Mean return |
| --- | --- | --- | --- | --- | --- |
| Double-DQN | 43 | 3.3% | 76.7% | 0.52681 | -26.64 |
| Double-DQN | 44 | 3.3% | 0.0% | 0.28812 | -6.14 |
| Double-DQN | 45 | 3.3% | 0.0% | 0.04790 | -1.81 |
| Double-DQN | 46 | 36.7% | 0.0% | 0.01770 | 4.74 |
| Double-DQN | 47 | 3.3% | 66.7% | 0.41760 | -23.60 |
| Tabular Double-Q | 43 | 66.7% | 0.0% | 0.00001 | 10.61 |
| Tabular Double-Q | 44 | 0.0% | 0.0% | 0.01200 | -1.40 |
| Tabular Double-Q | 45 | 50.0% | 0.0% | 0.00502 | 7.39 |
| Tabular Double-Q | 46 | 36.7% | 6.7% | 0.06623 | 2.50 |
| Tabular Double-Q | 47 | 0.0% | 0.0% | 0.00208 | -1.28 |

## Means across five training seeds

| Controller | Success | Severe injury | Mean return | 95% hierarchical return interval |
| --- | --- | --- | --- | --- |
| Double-DQN | 10.0% | 28.7% | -10.69 | [-21.46, 0.04] |
| Tabular Double-Q | 30.7% | 1.3% | 3.57 | [-0.63, 7.98] |
| Continuous scripted | 62.0% | 0.0% | 9.75 | [8.28, 11.10] |
| Uniform random | 0.0% | 0.0% | -2.19 | [-2.24, -2.14] |
| Random without handoff | 0.0% | 0.0% | -2.57 | [-2.70, -2.45] |

Intervals use 2,000 bootstrap replicates, first resampling the five training seeds and then patients within each selected seed. They exclude model uncertainty and do not establish clinical safety. The random policies often fail to engage tissue or terminate through handoff; their low injury is not successful procedure control. Per-run binary rates include Wilson intervals in the JSON results.

## Bundled reference and selection bias

The fixed validation rule selected **tabular seed 43**: 20/30 successes, no observed severe injury, mean maximum injury about .0000106, and mean return 10.61. The continuous scripted controller also succeeded on 20/30 of these same patients. This chosen run performs much better than the 30.7% average success of tabular training, which includes two zero-success runs and one run with severe injuries. Its selected validation score is optimistic; it must not represent typical learning performance.

Zero observed severe injuries in 30 cases still has a Wilson upper 95% bound near 11%. This reference is an inspectable synthetic example, not a safety-certified or clinically competent controller. Its inference uses its learned Q-table alone; there is no scripted fallback.

The complete reference result includes all 3,000 episode records, actual weights, all fitting/validation seeds and raw comparator episodes. Source hashes match commit `79bcf567399bccc07805ec976829a519a064c90d`. The environment version is `tissue-retraction-1.1`; its exact physics fingerprint and the reference's whole-file SHA-256 are recorded in provenance and pinned by the application.

- [Frozen protocol](../examples/rl-study-001-protocol.json)
- [Audited full summary and all seed unions](../examples/rl-study-001-summary.json)
- [Complete bundled reference](../examples/stateful-policy-1.1.json)
- [Learning methods and known failure modes](rl-training.md)
- [Implemented tissue equations and tool responses](tissue-response.md)

## Reproduce

Use the recorded source commit and its locked dependencies. Each command creates a fresh output directory and never overwrites an earlier study:

```bash
.venv/bin/python scripts/train_agent.py --algorithm dqn \
  --config examples/rl-study-001-common-config.json --seeds 43 44 45 46 47 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-001-dqn

.venv/bin/python scripts/train_agent.py --algorithm tabular_q \
  --config examples/rl-study-001-common-config.json --seeds 43 44 45 46 47 \
  --episodes 3000 --demonstration-episodes 100 --eval-episodes 30 \
  --max-steps 200 --output .data/reproduce-study-001-tabular
```

The common JSON configuration above records gamma .99; all other effective settings are in the frozen protocol. If using the historical source checkout, copy that small configuration file from this report. Dependency/platform floating-point differences can change trajectories near decision thresholds; the source identity alone is not a cross-platform numerical guarantee.
