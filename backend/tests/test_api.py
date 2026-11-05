"""Exercise durable job lifecycle, validation, failures, and scientific exports."""

import csv
import io
import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from retraction import main
from retraction.storage import Store


def finish(client, run_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.01)
    pytest.fail("Run never finished")


@pytest.fixture
def client(tmp_path, monkeypatch):
    def engine(config, progress):
        progress(0.5)
        return {
            "summary": {"n_patients": config["n_patients"], "oracle_success_rate": 0.5},
            "patients": [{"id": "p-001", "attachment_mm": 12, "oracle": {"success": True}}],
            "methodology": {"engine": "test-double"},
        }

    monkeypatch.setattr(main, "execute_experiment", engine)
    with TestClient(main.create_app(tmp_path)) as api:
        yield api


def test_lifecycle_exports_and_health(client):
    assert client.get("/api/health").json()["clinical_use"] is False
    assert client.get("/api/presets").json()["default_config"]["n_patients"] == 240
    response = client.post("/api/runs", json={"n_patients": 60, "seed": 9})
    assert response.status_code == 202
    run_id = response.json()["id"]
    record = finish(client, run_id)
    assert record["status"] == "completed"
    assert record["progress"] == 1
    assert record["summary"]["n_patients"] == 60
    assert client.get("/api/runs").json()["runs"][0]["id"] == run_id
    result = client.get(f"/api/runs/{run_id}/results").json()
    assert result["patients"][0]["oracle"]["success"] is True
    export = client.get(f"/api/runs/{run_id}/export?format=json")
    assert export.json()["run"]["config"]["seed"] == 9
    assert export.json()["results"] == result
    assert "attachment;" in export.headers["content-disposition"]
    table = client.get(f"/api/runs/{run_id}/export?format=csv")
    row = next(csv.DictReader(io.StringIO(table.text)))
    assert row["run_id"] == run_id
    assert row["seed"] == "9"
    assert row["attachment_mm"] == "12"


@pytest.mark.parametrize("config", [
    {"n_patients": 0}, {"n_patients": 2001}, {"seed": -1},
    {"threshold_profile": "clinical"}, {"force_limit_n": 0},
    {"strain_limit": -0.1}, {"exposure_threshold": 1.1},
    {"attachment_min_mm": 40, "attachment_max_mm": 8},
    {"distension_min": 2, "distension_max": 1},
    {"unknown_field": 7},
])
def test_invalid_configs_rejected(client, config):
    assert client.post("/api/runs", json=config).status_code == 422
    assert client.get("/api/runs").json()["runs"] == []


def test_missing_run_and_invalid_ids(client):
    assert client.get(f"/api/runs/{uuid4()}").status_code == 404
    assert client.get("/api/runs/not-a-uuid").status_code == 422
    assert client.get("/api/runs?limit=500").status_code == 422


def test_failed_simulation_is_reported(tmp_path, monkeypatch):
    def broken(config, progress):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(main, "execute_experiment", broken)
    with TestClient(main.create_app(tmp_path)) as client:
        run_id = client.post("/api/runs", json={}).json()["id"]
        record = finish(client, run_id)
        assert record["status"] == "failed"
        assert record["error"]
        assert client.get(f"/api/runs/{run_id}/results").status_code == 409
        assert client.get(f"/api/runs/{run_id}/export").status_code == 409


def test_persistence_and_interrupted_recovery(tmp_path):
    store = Store(tmp_path)
    complete = store.create({"seed": 42})
    interrupted = store.create({"seed": 43})
    store.update(complete["id"], status="completed", progress=1, summary={"success": True}, result={"patients": []})
    with TestClient(main.create_app(tmp_path)) as client:
        assert client.get(f"/api/runs/{complete['id']}").json()["status"] == "completed"
        assert client.get(f"/api/runs/{complete['id']}/results").json() == {"patients": []}
        record = client.get(f"/api/runs/{interrupted['id']}").json()
        assert record["status"] == "failed"
        assert "restarted" in record["error"]


def test_preview_zero_displacement_and_validation(client):
    response = client.post("/api/preview", json={"strategy": {"distance_mm": 0}})
    assert response.status_code == 200
    preview = response.json()
    assert preview["outcome"]["force_n"] == 0
    assert preview["outcome"]["strain"] == 0
    assert preview["outcome"]["exposure"] == 0
    assert preview["outcome"]["success"] is False
    assert len(preview["probe"]["times_s"]) == 21
    assert preview["probe"]["forces_n"][0] == 0
    assert client.post("/api/preview", json={"patient": {"attachment_length_mm": 0}}).status_code == 422
    assert client.post("/api/preview", json={"strategy": {"direction_deg": 190}}).status_code == 422
    assert client.post("/api/preview", json={"patient": {"adhesion_count": 1.5}}).status_code == 422


@pytest.mark.parametrize("config", [
    {"attachment_min_mm": 4, "attachment_max_mm": 60},
    {"adhesion_stiffness_min": 0, "adhesion_stiffness_max": 0.5},
    {"distension_min": 0.5, "distension_max": 2.5},
    {"strain_limit": 3},
    {"attachment_min_mm": 10, "attachment_max_mm": 10},
])
def test_accepted_api_bounds_are_supported_by_engine(config):
    from retraction.simulation import generate_cohort, normalize_config, simulate_patient, STANDARD_STRATEGY

    accepted = main.RunConfig(**config).model_dump(exclude_none=True)
    normalized = normalize_config(accepted)
    patients = generate_cohort(normalized)
    for patient in patients:
        assert isinstance(simulate_patient(patient, STANDARD_STRATEGY)["success"], bool)


def test_real_simulation_completes_and_persists(tmp_path):
    with TestClient(main.create_app(tmp_path)) as client:
        response = client.post("/api/runs", json={"n_patients": 40, "seed": 10})
        assert response.status_code == 202
        run_id = response.json()["id"]
        record = finish(client, run_id)
        assert record["status"] == "completed", record
        results = client.get(f"/api/runs/{run_id}/results").json()
        assert len(results["patients"]) == 40
        assert results["engine"] == "analytic-surrogate"
        assert results["summary"]["oracle_success_rate"] >= results["summary"]["shared_success_rate"]
        assert client.get(f"/api/runs/{run_id}/export?format=json").status_code == 200
        assert len(list(csv.DictReader(io.StringIO(client.get(f"/api/runs/{run_id}/export?format=csv").text)))) == 40


def test_active_queue_limit_and_incomplete_exports(tmp_path, monkeypatch):
    release = threading.Event()

    def blocked(config, progress):
        assert release.wait(timeout=10), "Test did not release blocked worker"
        return {"summary": {}, "patients": []}

    monkeypatch.setattr(main, "execute_experiment", blocked)
    with TestClient(main.create_app(tmp_path)) as client:
        try:
            run_ids = []
            for _ in range(8):
                response = client.post("/api/runs", json={})
                assert response.status_code == 202
                run_ids.append(response.json()["id"])
            assert client.post("/api/runs", json={}).status_code == 429
            assert client.get(f"/api/runs/{run_ids[0]}/results").status_code == 409
            assert client.get(f"/api/runs/{run_ids[0]}/export").status_code == 409
        finally:
            release.set()
        assert all(finish(client, run_id)["status"] == "completed" for run_id in run_ids)


def test_nonfinite_numbers_are_rejected(client):
    assert client.post("/api/runs", content='{"force_limit_n": NaN}', headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/preview", content='{"strategy":{"distance_mm": Infinity}}', headers={"Content-Type": "application/json"}).status_code == 422


def test_patient_grid_reproduces_saved_oracle(tmp_path):
    with TestClient(main.create_app(tmp_path)) as client:
        run = client.post('/api/runs', json={'n_patients': 40, 'seed': 42}).json()
        for _ in range(100):
            record = client.get(f'/api/runs/{run["id"]}').json()
            if record['status'] in ('completed', 'failed'):
                break
            time.sleep(.02)
        assert record['status'] == 'completed'
        results = client.get(f'/api/runs/{run["id"]}/results').json()
        patient = results['patients'][0]
        response = client.get(f'/api/runs/{run["id"]}/patients/{patient["id"]}/strategies')
        assert response.status_code == 200
        grid = response.json()
        assert len(grid['strategies']) == 75
        assert any(s['success'] for s in grid['strategies']) == patient['oracle']['success']
        selected = next(s for s in grid['strategies'] if s['strategy']['id'] == patient['oracle']['strategy']['id'])
        assert selected['force_n'] == pytest.approx(patient['oracle']['force_n'])
        assert selected['exposure'] == pytest.approx(patient['oracle']['exposure'])
        assert client.get(f'/api/runs/{run["id"]}/patients/missing/strategies').status_code == 404
        report = client.get(f'/api/runs/{run["id"]}/report')
        assert report.status_code == 200
        assert 'attachment;' in report.headers['content-disposition']
        assert 'analytic-surrogate-1.0' in report.text
        assert 'not externally preregistered' in report.text
