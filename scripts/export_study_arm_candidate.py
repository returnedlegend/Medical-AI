#!/usr/bin/env python3
"""Export a validation-only arm candidate under a previously declared rule.

The complete study is audited first. This does not train, evaluate patients,
change the live reference, or read independent final-evaluation outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from summarize_rl_study import audit_and_summarize, canonical_sha, write_json


def export_candidate(study, arm, source_commit, reserved):
    protocol = json.loads((study / "protocol.json").read_text())
    rule = protocol.get("per_arm_candidate_rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("The frozen protocol did not declare per-arm candidate selection")
    if protocol.get("selection_require_zero_severe_harm") is not True:
        raise ValueError("This exporter supports only the declared zero-observed-severe-harm eligibility rule")
    summary, _ = audit_and_summarize(study, source_commit, reserved)
    rows = [row for row in summary["runs"] if row["arm"] == arm]
    if not rows:
        raise ValueError("Arm does not exist in the audited frozen protocol")
    eligible = [row for row in rows if row["learned"]["severe_harm_rate"] == 0]
    if not eligible:
        raise ValueError("No policy in this arm met the predeclared eligibility rule")
    chosen = min(eligible, key=lambda row: (-row["learned"]["success_rate"],
                                          row["learned"]["severe_harm_rate"],
                                          row["learned"]["mean_max_damage"],
                                          -row["learned"]["mean_return"]))
    result = json.loads((study / chosen["source"]).read_text())
    return {"format": "retraction-bundled-reference-v1", "result": result,
            "provenance": {"created_at": datetime.now(timezone.utc).isoformat(),
                           "study_protocol_sha256": summary["protocol_sha256"],
                           "source_sha256": summary["source_sha256"],
                           "source_commit": summary["source_commit"],
                           "selection_rule": protocol["selection_rule"],
                           "per_arm_candidate_rule": rule, "validation_only": True,
                           "policy_source_run": {key: chosen[key] for key in ("arm", "algorithm", "training_seed")},
                           "selection_eligibility": {"require_zero_observed_severe_harm": True,
                                                     "eligible_runs": len(eligible), "total_runs": len(rows),
                                                     "status": "selected_within_declared_arm"},
                           "checkpoint_sha256": canonical_sha(result["checkpoint"]),
                           "result_sha256": canonical_sha(result),
                           "reserved_patient_exclusion": summary["reserved_patient_exclusion"],
                           "scientific_note": "Validation-selected within a prespecified arm; optimistic selection estimate. No independent final outcomes were read. This candidate does not replace the live bundled reference."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reserved-seeds-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        example = export_candidate(args.study, args.arm, args.source_commit,
                                   json.loads(args.reserved_seeds_file.read_text()))
        write_json(args.output, example)
        print(json.dumps(example["provenance"]["policy_source_run"]))
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
