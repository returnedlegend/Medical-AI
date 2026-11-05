# Reinforcement-learning baselines

The learning module trains a small **NumPy Double-DQN**, using actual action → state transition → reward observations from the stateful tissue-response environment. It does not train on real patients or claim clinical competence.

An alternative **tabular Double-Q** baseline deliberately uses a smaller, reactive representation. Select `algorithm: "dqn"` (the default) or `algorithm: "tabular_q"`. Both receive the same actual environment rewards and the same declared tool-control primitives; neither calls a scripted controller during learned-policy inference.

The environment accepts seven continuous controls. DQN selects from a declared, finite library of control primitives: hold, gentle grasp, release, eight full-speed grasper directions, four finer grasper directions, four cautery movement directions, several slower lateral grasper movements, a low-power cautery pulse, and handoff. The exact seven-element vectors and names are exported in every checkpoint. This is a discrete-action wrapper around the continuous simulator, not unrestricted robot-control optimization. Cautery is an available harmful affordance; the exposure task does not require it.

## Demonstration assistance and initial exploration failure

The default training protocol first collects 40 transparent scripted trajectories using **training-only seeds**. Their actions are mapped to the nearest declared primitive. These are actual environment trajectories with observed rewards, not oracle Q-values or held-out patient labels. The network performs 1,000 warm-start TD updates from those transitions. Subsequent batches contain half retained demonstration transitions and half online exploration transitions. Demonstration batches sample action groups uniformly so a single initial grasp is not overwhelmed by hundreds of movement/hold examples.

An auxiliary large-margin action loss also encourages the demonstrated action to exceed other Q-values. Its default margin is .1 in scaled Q-value units and its coefficient is .5; `demonstration_margin` and `demonstration_weight` record these choices. This follows the large-margin idea in [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732) (Hester et al., AAAI 2018), but is a small custom implementation rather than a reproduction of that paper's complete algorithm. The TD loss still uses observed environment rewards. The action labels are additional imitation guidance and must not be omitted when describing the method.

This is **demonstration-assisted reinforcement learning**. It must not be presented as learning the procedure from unassisted exploration. Set `demonstration_episodes: 0` (or CLI `--demonstration-episodes 0`) for the unassisted ablation. The final deployed action is chosen by the learned Q-network; it does not call the scripted controller.

In an initial engineering run with 1,000 episodes and unassisted uniform exploration, the learner converged to immediate handoff on every evaluated case. It achieved no exposure successes. A replay-only demonstration variant timed out, and an early large-margin variant using the former scripted controller caused severe injury on 20% of 30 engineering cases. These are substantive failures, not evidence of clinical safety. They motivated improved controller demonstrations, more precise declared motion primitives, and observation scaling. No reward coefficient or tissue mechanism was changed to force a favorable learning result.

The current scripted controller, `feedback-strain-guard-1.1`, uses measured exposure error and measured force/strain guards. It was engineering-tuned on development seeds 0–19. Its provenance is part of the environment description. Quantizing a continuous teacher can change its behavior: before the smaller lateral primitives were added, the quantized demonstrations succeeded on only 13/40 training trajectories; adding .30, .15, .05 and .025 lateral controls raised this diagnostic to 23/40. A quantified scripted comparator must therefore distinguish its original continuous controls from a quantized teacher.

Repeated engineering checks reuse the training seed 42 evaluation stream. Those checks are **development data**, despite being excluded from gradients within each individual run. They cannot support an untouched final evaluation claim. Full failed checkpoints and engineering metrics are retained locally in `.data/rl-*`; a separately frozen protocol and fresh patient/training seeds are required for any final benchmark.

## Learning algorithm

The policy is a two-layer neural Q-function with 64 tanh hidden units. A replay buffer stores observed transitions. A separate target network is refreshed every 100 optimization updates; online Q-values select the next action and the target network evaluates it. Updates occur every four environment steps after 128 transitions, using batches of 64, Huber TD loss, Adam, and a global gradient norm limit of 10.

Training exploration decreases linearly from ε=1 to ε=.05 during the first 70% of episodes. The remaining training episodes retain .05 exploration. Optimizer rewards are divided by 10 for numerical conditioning; exported episode returns and evaluation returns retain the environment's original reward scale. A finite-horizon termination or truncation has no bootstrapped future value. This treatment assumes the remaining horizon is part of the observation.

`observation_encoding: "characteristic"` retains all 239 public measurements and appends nine explicit features: contact presence, exposure error, force/limit, maximum strain/limit, maximum mechanical injury, modeled thermal injury derived from current thermal dose, temperature rise, remaining-horizon fraction, and stable-hold fraction. These use the supplied observation alone. They do not query simulator internals, an optimal angle, future injury, a future reward, or a patient oracle.

Raw measurements use recorded characteristic scales where broad Gym bounds would hide early injury signals: strain/.6, thermal dose/.8, temperature rise above 37°C divided by 6°C, mechanical damage/.15, and force/2.5 N. Other coordinates retain their Box-derived affine scales. All encoded values clip to [-5,5]. The transform version, full feature names, centers and scales are declarative checkpoint data, validated against the supported implementation. `observation_encoding: "box"` retains the original Box transformation clipped to [-1,1] for an explicit encoding ablation; older JSON checkpoints also retain that original behavior.

The checkpoint records original observation bounds/names, the transform, exact effective environment configuration, action primitives, algorithm version, tissue-model version, architecture, and numeric weights. JSON decoding never executes Python or unpickles code. Structural loading checks finite arrays and bounded dimensions; inference rejects incompatible tissue-model versions and observation/action signatures. A different horizon, limit, or manually overridden patient is a distribution change and is shown by the application as research playback outside the recorded training configuration.

## Evaluation discipline

Each training run generates patient seeds from a dedicated seeded random stream. Held-out patient seeds come from a separate stream and are explicitly checked against the training seed set. The held-out set is never used for gradient updates, exploration decisions, checkpoint selection, or tuning a success threshold.

The final policy, random primitive controller, a random controller with handoff excluded, and the explicit continuous scripted baseline run on exactly the same held-out patient seeds. Random control has its own reproducible action stream. The second random comparator matters because otherwise handoff often ends random episodes before harmful contact or meaningful progress.

Results include raw per-episode outcomes, return, success, severe harm, handoff, maximum tissue damage, and episode length. Binary rates include Wilson 95% intervals; zero observed severe injury in 30 episodes still has an upper interval near 11%, not proof of zero risk. A 1,000-replicate paired bootstrap reports learned-minus-comparator differences in return, success, severe harm and maximum damage against each comparator. Return improvements alone do not establish less harm; the outcome metrics must be inspected separately.

A single run represents one initialization/training seed. Use multiple independent runs to assess algorithm stability. The intervals within a run reflect synthetic patient sampling, not training-seed uncertainty, tissue-model uncertainty, or generalization to real anatomy.

## Reactive tabular baseline

The original `tabular_state_encoding: "v1"` bins five current public measurements: contact presence; exposure-target error (edges −.02, 0, .01, .03, .08, .2, .4); force/limit (edges .5, .85, .95, 1.05); maximum strain/limit (the same four edges); and maximum mechanical injury (edges .05, .15). This creates 1,200 bins, each with one value per declared action. The observations are reconstructed from their Box normalization and rounded to six decimals only for those bin comparisons, which prevents floating-point reconstruction from moving an exact zero across a boundary. This version remains the default for historical configuration compatibility, and the sealed study-001 reference uses it.

This representation **is not Markov**: different anatomy, thermal histories, tool positions and tissue deformations can share a bin. It is an interpretable reactive comparison, not a claim that these five measurements describe the patient sufficiently. In particular, the state can alias a displaced ungrasped tool with the initial contact location, and thermal risk is represented only through its effect on later measured rewards/outcomes. Such limitations must remain visible beside any benchmark result.

Two independent Q-tables start at zero. If demonstrations are enabled, their observed discounted returns initialize the state/action entries they visited; this is a sample mean of actual trajectory returns, not an invented expert reward bonus. Each subsequent real online transition updates one table at random, while the other table evaluates the first table's greedy next action. Terminal/truncated transitions do not bootstrap. `tabular_learning_rate` defaults to .15; γ and the ε schedule are shared with DQN. One retained demonstration transition also receives a TD update per online step when assistance is enabled.

Tabular training does not use a neural network, gradients, the characteristic neural encoding, or the DQN large-margin imitation loss. Its metadata explicitly records these unused DQN settings. Checkpoints export the average Q-table and a fixed, validated bin specification as JSON; the two optimization tables and online replay state are not exported. Changing the bin specification requires a version change.

The optional `tabular_state_encoding: "v2"` addresses specific aliases observed in study 001. It distinguishes five acquisition contexts: an ungrasped tool far from intact tissue, an ungrasped tool near node03, an ungrasped tool near another node, contact with node03, and contact with another node. Near means within the model's 3-mm acquisition distance; intact means current combined injury below .9. All positions, contact identity, mechanical injury and thermal dose come from the public observation. Node03 is the publicly defined schematic exposure marker, not a hidden optimal grasp/angle label.

Version2 retains the exposure-error/force/strain edges, replaces the mechanical-only injury feature with maximum combined mechanical/thermal injury using the documented material equation, and adds a current-temperature-above-43°C bit plus a remaining-horizon-at-or-below-20% bit. This gives 12,000 states. These features distinguish displaced tools, pending heat, cooled thermal injury and near-timeout states without consulting future outcomes. Other anatomy, contact sites and deformation histories still alias; this remains a partial reactive policy.

V2 does not change the tool library, tissue equations, reward, Double-Q update, demonstration initialization or exploration schedule. The original v1 decoder remains available and replays the sealed reference exactly. A separately frozen study compares both encodings and an unassisted v2 ablation on fresh training/patient streams; improvement is a hypothesis rather than a condition for publishing the result.

## Reward-feedback and reward-hacking review

Each optimizer sees the reward from the tool action that actually ran. Irreversible damage increments and modeled bleeding/bile losses lower that reward; accumulated injury also changes later mechanics and therefore later observations. Terminal severe injury receives its own penalty. This is the harm → changed state/reward → subsequent action/value update loop; changing only a visualization would not provide it.

Several controls address obvious simulator exploits. Exposure reward is paid only for a new record, so oscillating repeatedly across the same opening earns no extra progress. Injury/force/strain conditions gate progress and success. Success also requires cautery to be off and temperatures no higher than 43°C, preventing an immediate success bonus while a pending thermal injury is still heating tissue. Injury is irreversible within an episode, and ending or handing off does not report a simulated rescue or successful exposure.

These controls do not prove the reward is complete. Immediate handoff, doing nothing, overpulling for short-term exposure, and delayed injury are observed failure modes. Severe-injury rate, maximum damage, episode outcomes and the scripted comparator remain necessary alongside reward. The negative engineering variants demonstrate why a policy should not be promoted solely because it trained or because its mean return increased.

A small integration stress test checks the causal connection to learning with the actual simulator. At seed 42, repeatedly issuing `[1,0,.55,0,0,0,0]` ends after 24 actions with modeled maximum injury about .670 and return −32.64; 200 all-zero actions yield no injury, timeout, and return −3.00. A terminal-return Q update prefers the latter trajectory. Removing only the measured harm penalties and severe-injury terminal penalty reverses that learned preference. This is a two-trajectory test of reward dependence, **not a successful surgical policy or a cohort-level safety result**.

## Reproduce a study

After installing the core environment:

```bash
.venv/bin/python scripts/train_agent.py --seeds 42 43 44 45 46 \
  --episodes 1000 --eval-episodes 30 --max-steps 200 \
  --output .data/rl-study-001
```

For the reactive baseline, add `--algorithm tabular_q`; select the newer representation explicitly with `--tabular-state-encoding v2`. Keep the algorithm, effective limits, horizon, source fingerprints, primitive library, bin/feature specification and exact patient seeds with each result bundle. The [frozen first study](rl-study-001.md) records all ten runs, including failures, and the validation-selected bundled reference.

The new output directory captures the complete protocol, source SHA-256 fingerprints and exact source snapshots before training, then JSON results and declarative policies for each training seed plus `summary.csv`. An absent `complete.json` means the bundle did not finish. Existing output directories are never overwritten. The study-summary helper audits frozen source hashes, configuration equality, paired patients, checkpoint compatibility and cross-run fitting/validation disjointness before selecting or exporting a bundled example.

The API runs one training worker at a time. Progress callbacks occur during demonstration collection, optimization, training and every evaluation episode, so cancellation remains cooperative throughout the job. Cancelled/failed jobs are not promoted as completed policies. Checkpoints contain inference weights; optimizer/replay state is not exported, so they are not resumable training snapshots.

All model parameters and harm thresholds are inspectable simulation assumptions. The companion [tissue-response documentation](tissue-response.md) specifies what actions change mechanically, which injury states are irreversible, and which signals determine reward/termination. They require expert validation before any clinical interpretation.
