"""Optional physics adapter lifecycle, process bounds, and honest diagnostics."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from retraction import physics
from retraction.storage import Store


def create_test_app(directory):
    @asynccontextmanager
    async def lifespan(app):
        physics.initialize_physics(app, directory)
        yield
        physics.shutdown_physics(app)
    app = FastAPI(lifespan=lifespan)
    app.include_router(physics.router)
    return app


def stub_result(config, output_path, **kwargs):
    output_path.with_suffix('.png').write_bytes(physics.PNG_SIGNATURE + b'test-frame')
    output_path.with_name('result-initial.png').write_bytes(physics.PNG_SIGNATURE + b'initial-frame')
    return {'steps': [{'step': 1, 'internal_force_sum_raw': 123.0}],
            'diagnostics': {'error_count': 2, 'warning_count': 1},
            'diagnostic_state': 'needs_review', 'model_ready': True,
            'sofa_version': '24.06.00', 'gallbladder_vertex_count': 367}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(physics, 'runtime_status', lambda: {'available': True, 'purpose': physics.PURPOSE, 'model_ready': False})
    monkeypatch.setattr(physics, 'execute_worker', stub_result)
    with TestClient(create_test_app(tmp_path)) as client:
        yield client


def finish(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f'/api/physics/runs/{run_id}').json()
        if record['status'] in ('completed', 'failed'):
            return record
        time.sleep(0.01)
    pytest.fail('Physics run did not finish')


def test_complete_lifecycle_separate_store_and_images(client):
    response = client.post('/api/physics/runs', json={'steps': 4, 'seed': 123, 'insertion': -0.2, 'start_grasped': True})
    assert response.status_code == 202
    run_id = response.json()['id']
    complete = finish(client, run_id)
    assert complete['status'] == 'completed'
    assert complete['config']['insertion'] == -0.2
    assert complete['summary']['model_ready'] is False
    assert complete['summary']['purpose'] == physics.PURPOSE
    assert complete['summary']['error_count'] == 2
    assert complete['summary']['diagnostic_state'] == 'needs_review'
    result = client.get(f'/api/physics/runs/{run_id}/results').json()
    assert result['model_ready'] is False
    assert result['image_url'] == f'/api/physics/runs/{run_id}/image'
    assert result['initial_image_url'].endswith('frame=initial')
    image = client.get(result['image_url'])
    assert image.status_code == 200
    assert image.headers['content-type'] == 'image/png'
    assert image.content.startswith(physics.PNG_SIGNATURE)
    assert client.get(result['initial_image_url']).content.endswith(b'initial-frame')
    assert client.get('/api/physics/runs').json()['runs'][0]['id'] == run_id
    jobs = client.app.state.physics
    assert jobs.store.path == jobs.directory / 'runs.sqlite3'
    assert jobs.directory.name == 'physics'


@pytest.mark.parametrize('payload', [
    {'seed': -1}, {'seed': 4294967296}, {'seed': True}, {'steps': 0}, {'steps': 51}, {'steps': 1.5},
    {'pan': 0.6}, {'tilt': -0.6}, {'spin': 'NaN'}, {'insertion': 'Infinity'}, {'jaw': -1},
    {'trajectory': 'neutral'}, {'output': '/tmp/escape'},
])
def test_invalid_controls_are_rejected_before_queueing(client, payload):
    assert client.post('/api/physics/runs', json=payload).status_code == 422
    assert client.get('/api/physics/runs').json()['runs'] == []


def test_optional_runtime_missing_returns_status_and503(tmp_path, monkeypatch):
    monkeypatch.setenv('RETRACTION_SOFA_DIR', str(tmp_path / 'missing'))
    with TestClient(create_test_app(tmp_path)) as client:
        status = client.get('/api/physics/status').json()
        assert status['available'] is False
        assert status['model_ready'] is False
        if status['supported_platform']:
            assert 'setup-sofa.sh' in status['reason']
        else:
            assert 'Linux x86_64 only' in status['reason']
        assert client.post('/api/physics/runs', json={}).status_code == 503
        assert client.get('/api/physics/runs').json()['runs'] == []


@pytest.mark.parametrize(('system', 'machine'), [('win32', 'AMD64'), ('darwin', 'arm64'), ('linux', 'aarch64')])
def test_optional_runtime_explicitly_rejects_unsupported_platforms(monkeypatch, system, machine):
    monkeypatch.setattr(physics.sys, 'platform', system)
    monkeypatch.setattr(physics.platform, 'machine', lambda: machine)
    status = physics.runtime_status()
    assert status['available'] is False
    assert status['supported_platform'] is False
    assert 'Linux x86_64 only' in status['reason']
    assert status['model_ready'] is False


def test_queue_capacity_and_incomplete_artifact_requests(tmp_path, monkeypatch):
    release = threading.Event()
    def slow(config, output_path, **kwargs):
        assert release.wait(5)
        return stub_result(config, output_path)
    monkeypatch.setattr(physics, 'execute_worker', slow)
    monkeypatch.setattr(physics, 'runtime_status', lambda: {'available': True})
    with TestClient(create_test_app(tmp_path)) as client:
        try:
            first = client.post('/api/physics/runs', json={})
            second = client.post('/api/physics/runs', json={})
            assert first.status_code == second.status_code == 202
            run_id = first.json()['id']
            assert client.post('/api/physics/runs', json={}).status_code == 429
            assert client.get(f'/api/physics/runs/{run_id}/results').status_code == 409
            assert client.get(f'/api/physics/runs/{run_id}/image').status_code == 409
        finally:
            release.set()
        assert finish(client, first.json()['id'])['status'] == 'completed'
        assert finish(client, second.json()['id'])['status'] == 'completed'


def test_failed_worker_reports_failure_without_claiming_results(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise physics.WorkerFailure('Diagnostic timed out; process group stopped.')
    monkeypatch.setattr(physics, 'runtime_status', lambda: {'available': True})
    monkeypatch.setattr(physics, 'execute_worker', broken)
    with TestClient(create_test_app(tmp_path)) as client:
        run_id = client.post('/api/physics/runs', json={}).json()['id']
        record = finish(client, run_id)
        assert record['status'] == 'failed'
        assert 'timed out' in record['error']
        assert record['summary'] is None
        assert client.get(f'/api/physics/runs/{run_id}/results').status_code == 409


def test_restart_recovers_interrupted_and_retains_completed_runs(tmp_path):
    store = Store(tmp_path / 'physics')
    incomplete = store.create({'seed': 9})
    complete = store.create({'seed': 10})
    store.update(complete['id'], status='completed', result={'model_ready': False}, summary={'model_ready': False})
    with TestClient(create_test_app(tmp_path)) as client:
        assert client.get(f"/api/physics/runs/{incomplete['id']}").json()['status'] == 'failed'
        assert client.get(f"/api/physics/runs/{complete['id']}/results").json() == {'model_ready': False}
        assert client.get(f"/api/physics/runs/{complete['id']}/image").status_code == 404
        assert client.get(f'/api/physics/runs/{uuid4()}').status_code == 404
        assert client.get('/api/physics/runs/not-a-uuid').status_code == 422
        assert client.get('/api/physics/runs?limit=500').status_code == 422
        assert client.get(f"/api/physics/runs/{complete['id']}/image?frame=../../etc/passwd").status_code == 422


def test_worker_command_uses_fixed_flags_and_bounded_values(tmp_path):
    command = physics.worker_command({'start_grasped': True, 'insertion': -0.5}, tmp_path / 'result.json')
    assert command[:2] == ['bash', str(physics.PROJECT_ROOT / 'scripts/run-sofa-smoke.sh')]
    assert '--start-grasped' in command
    assert command[command.index('--insertion') + 1] == '-0.5'
    assert str(tmp_path / 'result.json') == command[command.index('--output') + 1]
    with pytest.raises(ValueError):
        physics.worker_command({'jaw': 3}, tmp_path / 'result.json')


def test_subprocess_success_parses_diagnostics_and_removes_absolute_paths(tmp_path, monkeypatch):
    output = tmp_path / 'result.json'
    payload = {'steps': [{'step': 1}], 'image_path': '/etc/passwd', 'initial_image_path': '/etc/passwd',
               'diagnostics': {'error_count': 1, 'warning_count': 0,
                               'messages': [{'type': 'Error', 'sender': 'Mesh', 'message': str(tmp_path) + '/mesh bad'}]}}
    error_message = f'[ERROR] Bad topology at {tmp_path}/mesh'
    script = f"import json,pathlib; p=pathlib.Path({str(output)!r}); p.write_text({json.dumps(payload)!r}); p.with_suffix('.png').write_bytes({physics.PNG_SIGNATURE!r}); print({error_message!r}); print('[WARNING] Review mesh')"
    monkeypatch.setattr(physics, 'worker_command', lambda *args: [sys.executable, '-c', script])
    result = physics.execute_worker({}, output)
    assert result['diagnostic_state'] == 'needs_review'
    assert result['model_ready'] is False
    assert result['diagnostics']['error_count'] == 1  # same event is not double-counted
    assert result['diagnostics']['warning_count'] == 1
    assert '<run>' in result['diagnostics']['excerpts'][0]
    assert str(tmp_path) not in json.dumps(result['diagnostics'])
    assert 'image_path' not in result and 'initial_image_path' not in result


@pytest.mark.skipif(not hasattr(os, 'killpg'), reason='POSIX process-group signals are unavailable on Windows; the optional SOFA worker supports Linux x86_64 only.')
def test_timeout_kills_process_group_and_reaps_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(physics, 'worker_command', lambda *args: [sys.executable, '-c', 'import time; time.sleep(30)'])
    monkeypatch.setattr(physics, 'WORKER_TIMEOUT_SECONDS', 0.05)
    original = os.killpg
    signals = []
    def killpg(pid, signum):
        signals.append((pid, signum))
        return original(pid, signum)
    monkeypatch.setattr(physics.os, 'killpg', killpg)
    started = []
    finished = []
    with pytest.raises(physics.WorkerFailure, match='exceeded'):
        physics.execute_worker({}, tmp_path / 'result.json', process_started=started.append, process_finished=finished.append)
    assert started == finished
    assert len(started) == 1
    assert signals == [(started[0].pid, signal.SIGKILL)]
    assert started[0].poll() is not None


def test_nonfinite_worker_artifacts_are_rejected(tmp_path, monkeypatch):
    output = tmp_path / 'result.json'
    script = f"import pathlib; p=pathlib.Path({str(output)!r}); p.write_text('{{\"steps\":[{{\"force\":NaN}}]}}'); p.with_suffix('.png').write_bytes({physics.PNG_SIGNATURE!r})"
    monkeypatch.setattr(physics, 'worker_command', lambda *args: [sys.executable, '-c', script])
    with pytest.raises(physics.WorkerFailure, match='invalid telemetry'):
        physics.execute_worker({}, output)


def test_worker_review_flag_is_preserved_without_log_errors(tmp_path, monkeypatch):
    output = tmp_path / 'result.json'
    payload = {'steps': [{'step': 1}], 'diagnostic_state': 'needs_review',
               'diagnostics': {'error_count': 0, 'warning_count': 0}}
    script = f"import pathlib; p=pathlib.Path({str(output)!r}); p.write_text({json.dumps(payload)!r}); p.with_suffix('.png').write_bytes({physics.PNG_SIGNATURE!r})"
    monkeypatch.setattr(physics, 'worker_command', lambda *args: [sys.executable, '-c', script])
    result = physics.execute_worker({}, output)
    assert result['diagnostic_state'] == 'needs_review'
    assert result['model_ready'] is False


@pytest.mark.skipif(not hasattr(os, 'killpg'), reason='POSIX process-group signals are unavailable on Windows; the optional SOFA worker supports Linux x86_64 only.')
def test_finished_process_is_not_signaled(monkeypatch):
    class Completed:
        pid = 123456
        def poll(self):
            return 0
    def forbidden(*args):
        pytest.fail('A reaped process must not be signaled')
    monkeypatch.setattr(physics.os, 'killpg', forbidden)
    physics.kill_process_group(Completed())


def test_experimental_damage_controls_are_explicit_and_bounded(tmp_path):
    from retraction.physics import worker_command, PhysicsConfig
    from pydantic import ValidationError
    config = PhysicsConfig(damage_model=True, cautery_power=.25)
    command = worker_command(config.model_dump(), tmp_path / 'out.json')
    assert '--damage-model' in command
    assert command[command.index('--cautery-power') + 1] == '0.25'
    with pytest.raises(ValidationError):
        PhysicsConfig(cautery_power=.5)
    with pytest.raises(ValidationError):
        PhysicsConfig(damage_model=True, cautery_power=1.1)
