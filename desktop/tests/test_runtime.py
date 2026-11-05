"""Private local service and packaged-resource contracts without a GUI driver."""

import json
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from desktop import runtime


def test_resource_root_source_and_frozen(tmp_path, monkeypatch):
    assert runtime.resource_root() == ROOT
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert runtime.resource_root() == tmp_path
    with pytest.raises(RuntimeError, match="interface is missing"):
        runtime.frontend_directory()
    (tmp_path / "frontend/dist").mkdir(parents=True)
    (tmp_path / "frontend/dist/index.html").write_text("desktop")
    assert runtime.frontend_directory() == tmp_path / "frontend/dist"


def test_private_server_health_static_requests_and_stop(tmp_path):
    web = tmp_path / "bundled-frontend"
    web.mkdir()
    (web / "index.html").write_text("<html><body>desktop resources served</body></html>")
    data = tmp_path / "persistent-user-data"
    service = runtime.LocalServer(data, web_directory=web)
    service.start()
    url = service.url
    port = int(url.rsplit(":", 1)[1])
    try:
        assert service.request("/api/health")["status"] == "ok"
        assert b"desktop resources served" in service.request_bytes("/")
        assert (data / "runs.sqlite3").is_file()
        opener = build_opener(ProxyHandler({}))
        with pytest.raises(HTTPError) as rejected:
            opener.open(url + "/api/health", timeout=2)
        assert rejected.value.code == 403
        with pytest.raises(ValueError):
            service.request("//example.com")
        created = service.request("/api/runs", {"n_patients": 40, "seed": 621})
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            record = service.request(f"/api/runs/{created['id']}")
            if record["status"] in ("completed", "failed"):
                break
            time.sleep(.05)
        assert record["status"] == "completed"
        exported = json.loads(service.request_bytes(f"/api/runs/{created['id']}/export?format=json"))
        assert len(exported["results"]["patients"]) == 40
        session = service.request("/api/procedure/sessions", {"seed": 42})
        for _ in range(3):
            session = service.request(f"/api/procedure/sessions/{session['id']}/step", {"action": [0, 0, .55, 0, 0, 0, 0], "expected_revision": session["revision"]})
        assert session["revision"] == 3
    finally:
        service.stop()
    with socket.socket() as connection:
        connection.settimeout(1)
        assert connection.connect_ex(("127.0.0.1", port)) != 0
    # A new native window resumes the same durable user study directory.
    reopened = runtime.LocalServer(data, web_directory=web)
    try:
        reopened.start()
        assert reopened.request("/api/runs")["runs"][0]["status"] == "completed"
        restored = reopened.request(f"/api/procedure/sessions/{session['id']}")
        assert restored["revision"] == 3
        handoff = reopened.request(f"/api/procedure/sessions/{session['id']}/step", {"action": [0, 0, 0, 0, 0, 0, 1], "expected_revision": 3})
        assert handoff["status"] == "terminated"
        assert handoff["revision"] == 4
    finally:
        reopened.stop()
