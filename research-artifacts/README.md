# Complete exploratory research artifacts

`rl-study-001-full.zip` contains all ten runs from the first paired five-seed DQN/tabular study: successful and unsuccessful learned policies, full learning curves, protocols, exact training-source snapshots, and a per-file SHA-256 manifest. All patients and outcomes are synthetic.

`rl-study-002-full.zip` contains all 15 runs from the paired five-seed assisted V1/V2 tabular comparison and unassisted V2 ablation. It preserves every checkpoint, complete learning curves and raw comparator episodes, exact frozen sources, protocols, summary, configurations, logs, and a per-file SHA-256 manifest. Its [study report](../docs/rl-study-002.md) includes all zero-success and harmful runs, paired return intervals, and validation-selection bias. The assisted V2-minus-V1 return interval includes zero and does not establish superiority.

Verify the archive with the accompanying `.sha256` file before extraction. The archive is a research record, not the Windows application installer. Checkpoints are declarative JSON; none is a clinically validated controller. The study's weak average performance, harmful runs, and validation-selection bias are described in [the study report](../docs/rl-study-001.md).

The application's bundled reference is a single explicitly validation-selected example. The complete archives preserve the other results so that selecting an example does not conceal failures. The study 002 candidate remains separate from the application's default. No reserved final-test cohort is included in either development/validation study.
