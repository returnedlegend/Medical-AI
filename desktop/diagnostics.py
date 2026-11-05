"""An actual packaged-service check independent of graphical drivers.

This deliberately reports its limited scope: a service check is not evidence
that a native browser window rendered successfully.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
import platform
import sys
import tempfile
import time

from desktop.runtime import LocalServer, resource_root


def wait_for_run(server: LocalServer, route: str, timeout: float = 90) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = server.request(route)
        if record["status"] == "completed":
            return record
        if record["status"] == "failed":
            raise RuntimeError(record.get("error") or "Diagnostic research job failed")
        time.sleep(.05)
    raise TimeoutError("Diagnostic research job exceeded its time limit")


def diagnose(output: Path, data_directory: Path | None = None) -> int:
    evidence = {"scope": "packaged-service-and-models", "status": "running",
                "ui_validation": "not_requested", "packaged": bool(getattr(sys, "frozen", False)),
                "platform": platform.platform(), "checks": {}}
    temporary = tempfile.TemporaryDirectory(prefix="retraction-diagnostic-") if data_directory is None else None
    directory = data_directory or Path(temporary.name)
    server = LocalServer(directory)
    try:
        from PySide6.QtCore import qVersion
        from retraction import __version__
        from retraction.environment import ENV_VERSION, PHYSICS_FINGERPRINT, PHYSICS_FINGERPRINT_ALGORITHM
        evidence["version"] = __version__
        evidence["qt_version"] = qVersion()
        evidence["model_version"] = ENV_VERSION
        evidence["physics_fingerprint"] = PHYSICS_FINGERPRINT
        evidence["physics_fingerprint_algorithm"] = PHYSICS_FINGERPRINT_ALGORITHM
        if evidence["packaged"]:
            build = json.loads((resource_root() / "build-info.json").read_text(encoding="utf-8"))
            assert build["physics_fingerprint"] == PHYSICS_FINGERPRINT, "Bundled physics fingerprint differs from build provenance"
            for name in ("environment.py", "tissue.py"):
                digest = hashlib.sha256((resource_root() / "retraction" / name).read_bytes()).hexdigest()
                assert digest == build["source_sha256"][f"backend/retraction/{name}"], f"Bundled {name} differs from recorded source"
            evidence["checks"]["physics_source_provenance"] = True
        server.start()
        evidence["checks"]["health"] = server.request("/api/health")["status"] == "ok"
        evidence["checks"]["frontend_resources"] = b"<html" in server.request_bytes("/").lower()
        documents = ("tissue-response.md", "rl-training.md", "model.md", "desktop.md")
        evidence["checks"]["offline_documents"] = all(len((resource_root() / "docs" / name).read_text(encoding="utf-8")) > 100 for name in documents)
        cohort = server.request("/api/runs", {"n_patients": 40, "seed": 812})
        wait_for_run(server, f"/api/runs/{cohort['id']}")
        results = server.request(f"/api/runs/{cohort['id']}/results")
        assert len(results["patients"]) == 40
        evidence["checks"]["cohort"] = {"patients": 40, "export_bytes": len(server.request_bytes(f"/api/runs/{cohort['id']}/export?format=json"))}
        session = server.request("/api/procedure/sessions", {"seed": 42})
        for action in ([0, 0, .55, 0, 0, 0, 0], [.15, 0, .55, 0, 0, 0, 0], [0, .15, .55, 0, 0, 0, 0]):
            session = server.request(f"/api/procedure/sessions/{session['id']}/step", {"action": action, "expected_revision": session["revision"]})
        assert session["revision"] == 3 and session["status"] == "active"
        # Restart the actual service, replay an active persisted episode, and
        # make a further real transition rather than only re-reading a row.
        server.stop()
        server = LocalServer(directory)
        server.start()
        restored = server.request(f"/api/procedure/sessions/{session['id']}")
        assert restored["revision"] == 3
        session = server.request(f"/api/procedure/sessions/{session['id']}/step", {"action": [0, 0, 0, 0, 0, 0, 1], "expected_revision": 3})
        assert session["status"] == "terminated" and session["revision"] == 4
        exported = server.request(f"/api/procedure/sessions/{session['id']}/export")
        assert len(exported["transitions"]) == 4
        evidence["checks"]["procedure"] = {"transitions": 4, "status": session["status"], "restart_replay": True}
        learning = server.request("/api/learning/runs", {"training_seed": 7, "episodes": 20, "eval_episodes": 5, "max_steps": 20, "demonstration_episodes": 0})
        wait_for_run(server, f"/api/learning/runs/{learning['id']}")
        training = server.request(f"/api/learning/runs/{learning['id']}/results")
        assert len(training["learning_curve"]) == 20 and training["checkpoint"]
        policy_session = server.request(f"/api/learning/runs/{learning['id']}/session", {"seed": 193})
        stepped = server.request(f"/api/procedure/sessions/{policy_session['id']}/policy-step", {"training_run_id": learning["id"], "expected_revision": 0})
        assert stepped["revision"] == 1
        evidence["checks"]["learning"] = {"algorithm": training["algorithm"], "episodes": 20, "held_out_episodes": 5, "policy_transition": True}
        assert all(value is not False for value in evidence["checks"].values())
        evidence["status"] = "passed"
    except Exception as error:
        logging.exception("Packaged service diagnosis failed")
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
    finally:
        server.stop()
        if temporary is not None:
            temporary.cleanup()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0 if evidence["status"] == "passed" else 1
