This ZIP preserves a genuine completed development fixture from
`test_real_paired_evaluation_writes_protocol_before_rollouts_and_never_trains`.
It contains one fixed tabular V2 policy and three baselines on five development
patients (4,000,000,000–4,000,000,004), with a 20-step horizon. The tiny policy
was trained in the existing fixture with training seed 923, 20 online episodes,
zero demonstrations, and five separate validation patients.

The retained evaluator run was supplied at
`/tmp/retraction-evaluator-development-proof-1811/test_real_paired_evaluation_wr0/paired`.
No additional training streams, simulator rollouts, or reserved final patients
were used to create the reporting fixture. Its completion marker and copied
input hashes are checked by the report generator. ZIP paths are relative and
timestamps are fixed at 1980 for reproducible fixture bytes.

This is test data, not a final research result. Synthetic crossed-bootstrap
interaction matrices in the reporting tests are explicitly separate from it.
