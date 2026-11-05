"""Offline reference import: genuine tiny training, sealed JSON, no overwrites."""
from contextlib import asynccontextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from retraction import environment, procedure, reference
from retraction.learning import train_policy
from retraction.storage import Store


@pytest.fixture(scope='module')
def actual_training_result():
    return train_policy({'episodes': 20, 'eval_episodes': 5, 'max_steps': 20,
                         'demonstration_episodes': 0, 'training_seed': 923})


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def bundled(tmp_path, monkeypatch, actual_training_result):
    bundle = tmp_path / 'application-bundle'
    monkeypatch.setattr(sys, '_MEIPASS', str(bundle), raising=False)
    path = bundle / 'examples' / reference.REFERENCE_RESOURCE
    path.parent.mkdir(parents=True)
    source = Path(environment.__file__).parent
    result = copy.deepcopy(actual_training_result)
    payload = {'format': reference.REFERENCE_FORMAT, 'result': result,
               'provenance': {'created_at': '2026-11-05T00:00:00+00:00',
                              'study_protocol_sha256': 'a' * 64,
                              'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                                for name in ('learning.py', 'tabular.py', 'environment.py', 'tissue.py')},
                              'selection_rule': 'Test fixture only: use the single genuine tiny training run.',
                              'validation_only': True,
                              'policy_source_run': {'algorithm': 'dqn', 'training_seed': 923},
                              'checkpoint_sha256': sha(result['checkpoint']), 'result_sha256': sha(result),
                              'scientific_note': 'Test fixture, not the shipped reference or a validated policy.'}}

    def save(value=payload, *, seal=True):
        contents = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        path.write_bytes(contents)
        if seal:
            monkeypatch.setattr(reference, 'REFERENCE_SHA256', hashlib.sha256(contents).hexdigest())
        return hashlib.sha256(contents).hexdigest()

    save()
    return path, payload, save


def app(directory):
    @asynccontextmanager
    async def lifespan(application):
        procedure.initialize_procedure(application, directory)
        yield
        procedure.shutdown_procedure(application)
    application = FastAPI(lifespan=lifespan)
    application.include_router(procedure.router)
    application.include_router(procedure.learning_router)
    return application


def test_packaged_reference_seeds_once_and_acts_through_normal_api(tmp_path, bundled):
    path, payload, _ = bundled
    assert reference.reference_path() == path
    directory = tmp_path / 'data'
    with TestClient(app(directory)) as client:
        metadata = client.get('/api/learning/reference').json()
        assert metadata['available'] is True
        run_id = metadata['run_id']
        listed = client.get('/api/learning/runs').json()['runs']
        assert len(listed) == 1 and listed[0]['id'] == run_id
        assert listed[0]['status'] == 'completed'
        assert listed[0]['summary']['label'] == 'Bundled reference'
        assert listed[0]['summary']['origin']['kind'] == 'bundled-reference'
        assert client.get(f'/api/learning/runs/{run_id}/results').json() == payload['result']
        matched = client.post(f'/api/learning/runs/{run_id}/session', json={'seed': 149})
        assert matched.status_code == 201, matched.text
        session = matched.json()
        assert session['config']['max_steps'] == 20
        assert session['policy_context']['out_of_distribution'] is False
        acted = client.post(f"/api/procedure/sessions/{session['id']}/policy-step",
                            json={'training_run_id': run_id, 'expected_revision': 0})
        assert acted.status_code == 200 and acted.json()['revision'] == 1
        original = client.app.state.learning.store.get(run_id, include_result=True)
    with TestClient(app(directory)) as restarted:
        assert restarted.get('/api/learning/reference').json()['run_id'] == run_id
        assert len(restarted.get('/api/learning/runs').json()['runs']) == 1
        assert restarted.app.state.learning.store.get(run_id, include_result=True) == original


def test_reference_does_not_overwrite_user_studies(tmp_path, bundled):
    store = Store(tmp_path / 'learning')
    _, payload, _ = bundled
    user = store.create(payload['result']['config'])
    store.update(user['id'], status='completed', progress=1, summary={'label': 'User study'}, result=payload['result'])
    original = store.get(user['id'], include_result=True)
    first = reference.seed_reference(store)
    second = reference.seed_reference(store)
    assert first == second and first['available'] is True
    assert len(store.list()) == 2
    assert store.get(user['id'], include_result=True) == original


def test_reference_identifier_collision_never_overwrites_existing_record(tmp_path, bundled):
    store = Store(tmp_path)
    seeded = reference.seed_reference(store)
    assert seeded['available']
    store.update(seeded['run_id'], summary={'label': 'Preserve this record'}, result={'user': 'retained'})
    original = store.get(seeded['run_id'], include_result=True)
    rejected = reference.seed_reference(store)
    assert rejected['available'] is False
    assert 'no record was overwritten' in rejected['reason']
    assert store.get(seeded['run_id'], include_result=True) == original


@pytest.mark.parametrize('kind', ['file_hash', 'checkpoint_hash', 'model_version', 'physics_fingerprint', 'physics_source', 'overlapping_seeds', 'incorrect_aggregate', 'malformed_episode', 'nonboolean_outcome'])
def test_invalid_reference_is_explicitly_unavailable_and_not_inserted(tmp_path, monkeypatch, bundled, kind):
    _, payload, save = bundled
    if kind == 'file_hash':
        monkeypatch.setattr(reference, 'REFERENCE_SHA256', '0' * 64)
    else:
        if kind == 'checkpoint_hash':
            payload['provenance']['checkpoint_sha256'] = '0' * 64
        elif kind == 'model_version':
            payload['result']['checkpoint']['model_version'] = 'tissue-retraction-1.0'
        elif kind == 'physics_fingerprint':
            payload['result']['checkpoint']['physics_fingerprint'] = '0' * 64
        elif kind == 'physics_source':
            payload['provenance']['source_sha256']['tissue.py'] = '0' * 64
        elif kind == 'overlapping_seeds':
            payload['result']['metadata']['test_seeds'][0] = payload['result']['metadata']['training_seeds'][0]
        elif kind == 'incorrect_aggregate':
            payload['result']['evaluation']['learned']['mean_return'] += 1
        elif kind == 'malformed_episode':
            payload['result']['evaluation']['learned']['episodes'][0] = []
        else:
            payload['result']['evaluation']['learned']['episodes'][0]['success'] = 'false'
        if kind != 'checkpoint_hash':
            payload['provenance']['checkpoint_sha256'] = sha(payload['result']['checkpoint'])
        payload['provenance']['result_sha256'] = sha(payload['result'])
        save()
    store = Store(tmp_path / 'runs')
    status = reference.seed_reference(store)
    assert status['available'] is False
    assert status['reason']
    assert store.list() == []


def test_missing_unsealed_and_malformed_optional_reference_do_not_block_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / 'bundle'), raising=False)
    directory = tmp_path / 'data'
    with TestClient(app(directory)) as client:
        assert client.get('/api/learning/reference').json()['available'] is False
        assert client.post('/api/procedure/sessions', json={}).status_code == 201
    path = reference.reference_path()
    path.parent.mkdir(parents=True)
    path.write_text('{malformed')
    monkeypatch.setattr(reference, 'REFERENCE_SHA256', None)
    with TestClient(app(directory)) as client:
        assert 'not been sealed' in client.get('/api/learning/reference').json()['reason']
    monkeypatch.setattr(reference, 'REFERENCE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    with TestClient(app(directory)) as client:
        assert client.get('/api/learning/reference').json()['available'] is False
        assert client.get('/api/learning/runs').json()['runs'] == []


def test_development_resource_path_is_fixed_and_has_no_request_path_input(monkeypatch):
    monkeypatch.delattr(sys, '_MEIPASS', raising=False)
    assert reference.reference_path() == Path(reference.__file__).resolve().parents[2] / 'examples' / reference.REFERENCE_RESOURCE
