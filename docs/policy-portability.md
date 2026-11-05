# Development policy portability check

Linux CPython 3.12.14 and actual Windows CPython 3.12.15 under Wine produced
exactly equal trajectories for fifteen fixed development cases: the published
V1, study-002 V2 and study-003 V3 policies, each on seeds 2, 17, 42, 74 and 100.
Both used NumPy 2.3.5 and the same recorded source and policy bytes.

Every action index, reward and body snapshot contributes to the trajectory
SHA-256. All fifteen hashes, final states, outcomes and accumulated returns
matched exactly. Seed 42 timed out for every policy; those unsuccessful outcomes
are retained. No training or reserved final patient was executed by this check.
The [compact receipt](validation/policy-portability-development.json) records
the actual measurements, interpreter versions, input hashes and limitations.

This is a development check on five already-used patient seeds. It does not
prove equivalence across all patients, processors, dependency versions or
floating-point thresholds. Windows Python under Wine also does not replace the
separate native Windows GUI, installation and model checks in the desktop CI.

## Reproduce

From the same source checkout and locked dependencies on Linux:

```bash
.venv/bin/python scripts/check_policy_portability.py --output linux-policy-check.json
```

On Windows, using an environment with the same core requirements:

```powershell
.venv\Scripts\python.exe scripts/check_policy_portability.py --output windows-policy-check.json
```

Copy the two JSON reports to one machine, then compare them:

```bash
python scripts/check_policy_portability.py \
  --compare linux-policy-check.json windows-policy-check.json \
  --output policy-comparison.json
```

The collector fixes its development seeds and never accepts an arbitrary final
cohort. The comparator verifies the policy bytes, numerical dependency version,
source hashes, model fingerprint and exact case order before comparing results.
It exits nonzero on a mismatch and never overwrites an existing report.
