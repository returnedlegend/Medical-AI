# Controlled tool-response protocols

The desktop Procedure laboratory includes five reproducible interventions in
[`scenarios.py`](../backend/retraction/scenarios.py). Each intervention issues
ordinary seven-dimensional tool actions to the same `TissueRetractionEnv` used
by manual control and reinforcement learning. The model produces every tissue
position, injury value, loss, vital sign, event and reward. No response frame or
outcome is supplied by the protocol.

These are **synthetic software experiments**, not instructions for performing a
medical procedure. The geometry, material constants, injury thresholds, heat
response and fluid/vital-sign equations are illustrative and have not been
clinically calibrated. The controllers are fixed, transparent scripts; executing
one is not training or evaluating an RL policy. The complete model assumptions
and reward equations are documented in [tissue-response.md](tissue-response.md).

## Reproducibility and defaults

Protocol version is `tool-response-protocols-1.0`. It is distinct from environment
version `tissue-retraction-1.1` and the exact source fingerprint recorded with each
transition. The protocol module does not change the frozen environment or tissue
implementation, material constants, reward weights or termination thresholds.

The catalog recommends seed **2** and records every default environment setting:

| Setting | Value |
| --- | ---: |
| Maximum environment steps | 200 |
| Timestep | 0.1 s |
| Force limit | 2.5 N |
| Strain limit | 0.60 |
| Exposure target | 0.60 |
| Consecutive target intervals | 5 |
| Maximum injury stop | 0.60 |
| Critical artery/duct injury stop | 0.35 |
| Blood-loss stop | 600 ml |
| Seeded anatomy sampling | Enabled |

The actual sampled patient values, complete normalized configuration, seed,
protocol parameters, model identity, initial state, issued actions and every
resulting observation/info/reward are retained by `run_scenario()`. The desktop
service instead executes the same online controller through its normal durable
step path. This preserves ordinary action/reward records and replay provenance.
An explicit patient can be passed to the Python runner; it is reset through the
public environment API rather than written into the tissue state.

All action components remain in `[-1, 1]`, ordered as:

```
[grasper_dx, grasper_dy, grip, cautery_dx, cautery_dy, cautery_power, handoff]
```

A movement component of one corresponds to **10 mm/s on that axis**. Simultaneous
x/y movement can therefore have a larger resultant speed. Approach commands are
`clip((target_position - tool_position)/(10 * dt), -1, 1)` on each axis, with a
0.05-mm arrival tolerance. The grasper is explicitly opened with command `-1`
during approaches and thermal interventions. Handoff is always zero. Motion
and target selection use the current public tool and node positions, including
seeded gallbladder geometry; controllers do not access private tissue arrays.

Each controller issues at most 200 actions, or a lower requested budget. The
model's own termination/truncation always takes priority. Fixed durations become
`ceil(duration / dt)` intervals, so realized durations may round up by less than
one timestep when a different timestep is explicitly selected. Both the actual
configuration and action count remain recorded. A short horizon/budget can stop
an intervention before contact or injury; its outcome must then show that fact.

## Intervention definitions

| ID | Target and commanded intervention | Planned observation |
| --- | --- | --- |
| `gentle-traction` | Approach the gallbladder node nearest the initial grasper. Close to 0.55. Use rightward command `clip(1.2 * (target_exposure + 0.005 - exposure), 0, 0.30)`; when observed maximum strain or contact force exceeds 95% of its configured limit, use a small leftward command of −0.03. Cautery remains off. | Continue until the model stops or the controller budget is exhausted. |
| `held-jaw-crush` | Approach the nearest gallbladder node. Set jaw closure to 1.0 once, then issue zero jaw/movement commands to **hold** that closure for up to 4 s without traction. | Release with −1 and observe for up to 2 s if the model has not already stopped. |
| `arterial-thermal-contact` | With cautery off, approach the artery node with greatest observed y coordinate. Apply cautery command 1.0 for up to 0.8 s without moving. | Switch heat off and observe for up to 2 s if the model remains active. |
| `duct-thermal-contact` | With cautery off, approach the middle common-bile-duct node sorted by observed y coordinate. Apply the same cautery command and duration. | The same unpowered observation period. |
| `remote-heat-control` | Leave cautery at its initial remote position and apply the same command 1.0 for 0.8 s. No tool motion. | Switch heat off and observe for 2 s. |

The gentle controller is an exposure-feedback baseline with public force/strain
guards, not a guarantee of low injury or completion on every patient. Its gains
mirror the existing transparent engineering baseline; they were not fitted to a
new held-out research cohort.

At seed 2, the selected target IDs are `n03` for the gallbladder, `n16` for the
artery and `n21` for the common duct. Those identifiers are results of public
geometry selection, not direct state injection. The schematic arterial and duct
sites are independent model nodes. There is no tool depth, real contact geometry
or fluid simulation. The initial remote cautery pose is outside the model's
7.2-mm heat support, which is why this particular negative control has no tissue
heating in the current model.

## How tool use becomes body response and RL feedback

A held closure above 0.75 adds monotonic local mechanical injury even when tool
velocity is zero. At full closure the illustrative crush term is `0.22 * dt` per
interval at the contacted site. Injury reduces future restoring stiffness and
contact coupling. Zero jaw command holds the previous closure; it does not open
the grasper. The compression protocol makes that otherwise easy-to-miss stateful
behavior visible.

Cautery delivers the existing local Gaussian heat source. Recorded temperature
and thermal dose rise; accumulated dose produces irreversible thermal injury.
Heat is local, so equal power applied remotely does not heat nodes. Turning power
off permits cooling but does not subtract dose or repair injury, and residual
heat can continue adding injury while cooling.

Injury at arterial nodes drives the model's bleeding rate and cumulative blood
loss. The volume-feedback equations then alter pressure and heart rate. Injury
at duct nodes drives a separate leak rate and cumulative bile leakage. Both
losses can reduce the projected exposure proxy. The visible response therefore
comes from the material and physiology states used by the RL observation, not a
separate animation flag.

The same canonical step also produces structured threshold/onset events and
reward components: new material injury, additional artery/duct injury, blood or
bile loss, force excess, time and terminal outcome. The protocol runner summarizes
these actual components and event counts; it never assigns a penalty merely
because the selected protocol has an adverse name.

**Critical injury usually ends the arterial/duct demonstrations before large
fluid volumes accumulate.** A small measured loss with `severe_injury` is the
actual failure outcome. The protocol does not raise thresholds, force additional
steps after termination, or replace the measured loss with a dramatic amount.
A protocol that finishes its planned actions while the model remains active is
labeled `protocol-complete`, not successful surgery. Its remaining state can be
inspected or advanced with ordinary control.

## Recorded reference outcomes

The following are actual runs of the unchanged environment at seed 2 with the
configuration above. Values are rounded for presentation; exported transitions
retain the measured values. These are reproducibility diagnostics, not clinical
performance evidence or a promise for different settings.

| Intervention | Actions / simulated time | Actual model stop | Final maximum injury | Blood loss | Bile leak | Cumulative reward |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Gentle traction | 103 / 10.3 s | `success` | 0 | 0 ml | 0 ml | 16.3873 |
| Held-jaw compression | 29 / 2.9 s | `severe_injury` | 0.6160 | 0 ml | 0 ml | −27.6820 |
| Arterial heat contact | 19 / 1.9 s | `severe_injury` | 0.3760 | 3.6491 ml | 0 ml | −39.7627 |
| Duct heat contact | 31 / 3.1 s | `severe_injury` | 0.3760 | 0 ml | 0.3676 ml | −39.8829 |
| Remote heat control | 28 / 2.8 s | No environment stop; protocol complete | 0 | 0 ml | 0 ml | −0.2800 |

The arterial run records approximately 14.9541 ml/s final bleeding rate,
72.1314 bpm heart rate and 89.8358 mmHg mean arterial pressure. The duct run
records approximately 1.4802 ml/s final leak rate. These small numeric changes
are the model's actual response before its critical-injury stop.

The remote heat protocol is a spatial negative control, not a complete matched
time/path experiment: it omits the contact approach. Automated checks additionally
replay the exact arterial/duct paths and durations with power replaced by zero,
and the compression path with closure replaced by 0.55. These matched command
controls show no injury/loss under this reference seed while the original
commands do. This separates the commanded insult from time and navigation in
the tested model; it does not establish biological causality in real patients.

## Python interface and durable continuation

From the repository root:

```bash
PYTHONPATH=backend .venv/bin/python - <<'PY'
import json
from retraction.scenarios import run_scenario

result = run_scenario("arterial-thermal-contact", seed=2)
print(json.dumps(result["outcome"], indent=2))
with open("arterial-response.json", "w", encoding="utf-8") as output:
    json.dump(result, output, allow_nan=False)
PY
```

Windows PowerShell can set `$env:PYTHONPATH="backend"` and use
`.venv\Scripts\python.exe`. The desktop application does not require these
commands to run the cases.

`scenario_catalog()` returns independent JSON metadata. For an existing service
session, call `create_scenario_controller(id, max_steps=200)` and repeatedly call
`controller.next_action(observation, info)`. A non-None result is an ordinary
NumPy seven-vector. After the normal step succeeds, persist
`controller.state_dict()` together with that action and its response. `phase`
and `steps` then describe the **current issued action**, so source metadata can
name its protocol, phase and one-based protocol action index. Do not persist the
advanced controller if the environment/service step fails.

Reconstruct it with `ScenarioController.from_state(saved_json)` or
`create_scenario_controller(id, max_steps=saved_budget, state=saved_json)`.
The protocol version, ID, budget, counters, phase and target identifier are
validated. The next decision requires the current public state, not a hidden
simulator handle. Stop on model termination/truncation or when the controller
returns `None`; repeated calls after completion issue no actions. On a model
stop, calling `next_action` with its final info marks the controller ended without
advancing the environment.

[`test_scenarios.py`](../backend/tests/test_scenarios.py) checks exact transition
replay, JSON continuation after every action, bounded actions/steps, finite
records, actual event/reward decomposition, spatial and matched-command negative
controls, nondecreasing injury/dose, cooling without healing, persistent injury
after jaw release, and truthful short-budget/horizon outcomes. Run it from
`backend` with `../.venv/bin/python -m pytest tests/test_scenarios.py -q`.
