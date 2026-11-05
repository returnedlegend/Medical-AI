"""Durable interactive episodes, concurrency, restoration, and policy orchestration."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import sqlite3
import threading
import time
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from retraction import procedure


ZERO = [0.] * 7


class DeterministicEnvironment:
    """Small test double; actual mechanics are tested separately by the engine."""
    def __init__(self, config):
        self.config = config
        self.closed = False
    def describe(self):
        return {'action_names': ['gx', 'gy', 'grip', 'cx', 'cy', 'power', 'handoff']}
    def reset(self, *, seed, options=None):
        self.seed = seed
        self.n = 0
        self.position = (seed % 13) / 13
        self.injury = 0.
        self.options = options
        self.done = False
        return self.observation(), self.info()
    def observation(self):
        return [self.position, self.injury, float(self.n)]
    def info(self, *, action=None):
        harmful = action is not None and action[5] > 0
        return {'state': {'tool_x': self.position, 'thermal_injury': self.injury},
                'reward_components': {'progress': self.position, 'injury_penalty': -self.injury},
                'events': [{'type': 'thermal_injury', 'step': self.n}] if harmful else [],
                'termination_reason': 'handoff' if self.done else None,
                'patient': self.options.get('patient') if self.options else None}
    def step(self, action):
        if self.done:
            raise RuntimeError('Episode finished')
        self.n += 1
        self.position += action[0] * .1
        self.injury += max(action[5], 0) * .1
        terminated = action[6] > .5
        truncated = self.n >= self.config['max_steps'] and not terminated
        self.done = terminated or truncated
        return self.observation(), self.position - self.injury, terminated, truncated, self.info(action=action)
    def close(self):
        self.closed = True


def normalize(config):
    if set(config) - {'max_steps'}:
        raise ValueError('Unknown environment setting')
    steps = config.get('max_steps', 8)
    if not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError('max_steps out of bounds')
    return {'max_steps': steps}


def create_test_app(directory):
    @asynccontextmanager
    async def lifespan(app):
        procedure.initialize_procedure(app, directory)
        yield
        procedure.shutdown_procedure(app)
    app = FastAPI(lifespan=lifespan)
    app.include_router(procedure.router)
    app.include_router(procedure.learning_router)
    return app


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(procedure, 'make_environment', lambda config: DeterministicEnvironment(config))
    monkeypatch.setattr(procedure, 'normalize_environment_config', normalize)
    monkeypatch.setattr(procedure, 'current_engine_version', lambda: 'test-engine-1')
    monkeypatch.setattr(procedure, 'current_physics_fingerprint', lambda: 'test-physics-fingerprint-1')


@pytest.fixture
def client(tmp_path, engine):
    with TestClient(create_test_app(tmp_path)) as client:
        yield client


def create_session(client, **config):
    response = client.post('/api/procedure/sessions', json=config)
    assert response.status_code == 201, response.text
    return response.json()


def step(client, session_id, action=ZERO, **options):
    return client.post(f'/api/procedure/sessions/{session_id}/step', json={'action': action, **options})


def test_create_step_and_export_preserve_actual_transitions(client):
    initial = create_session(client, seed=7, patient={'attachment_length_mm': 12})
    assert initial['revision'] == 0
    assert initial['status'] == 'active'
    assert initial['clinical_use'] is False
    assert initial['mode'] == 'synthetic-tissue-research'
    assert initial['config']['max_steps'] == 8
    assert initial['info']['patient']['attachment_length_mm'] == 12
    action = [.5, 0, 0, 0, 0, .8, 0]
    response = step(client, initial['id'], action, expected_revision=0)
    assert response.status_code == 200
    current = response.json()
    assert current['revision'] == 1
    assert current['state']['thermal_injury'] == pytest.approx(.08)
    assert current['reward_components']['injury_penalty'] == pytest.approx(-.08)
    assert current['episode_reward'] == current['reward']
    assert current['events'][0]['type'] == 'thermal_injury'
    assert client.get(f"/api/procedure/sessions/{initial['id']}").json() == current
    export = client.get(f"/api/procedure/sessions/{initial['id']}/export")
    assert export.status_code == 200
    payload = export.json()
    assert 'attachment;' in export.headers['content-disposition']
    assert payload['session'] == current
    assert payload['replay']['seed'] == 7
    assert payload['transitions'][0]['action'] == action
    assert payload['transitions'][0]['source'] == {'type': 'manual'}
    assert payload['transitions'][0]['reward'] == current['reward']
    listed = client.get('/api/procedure/sessions').json()['sessions'][0]
    assert listed['id'] == initial['id'] and 'observation' not in listed


@pytest.mark.parametrize('payload', [
    {'action': [0] * 5}, {'action': [0] * 8}, {'action': [0] * 6 + [1.1]},
    {'action': [0] * 6 + [-1.1]}, {'action': [0] * 6 + ['NaN']},
    {'action': [0] * 6 + [True]}, {'action': ZERO, 'expected_revision': -1},
    {'action': ZERO, 'output': '/tmp/invalid'},
])
def test_invalid_actions_never_advance_episode(client, payload):
    initial = create_session(client)
    assert client.post(f"/api/procedure/sessions/{initial['id']}/step", json=payload).status_code == 422
    assert client.get(f"/api/procedure/sessions/{initial['id']}").json()['revision'] == 0


@pytest.mark.parametrize('payload', [{'seed': -1}, {'seed': True}, {'seed': 4294967296},
                                      {'config': {'unknown': 5}}, {'patient': {'adhesion_count': 3.5}}])
def test_invalid_initial_configuration_is_not_persisted(client, payload):
    assert client.post('/api/procedure/sessions', json=payload).status_code == 422
    assert client.get('/api/procedure/sessions').json()['sessions'] == []


def test_optimistic_revision_rejects_stale_and_simultaneous_actions(client):
    initial = create_session(client)
    session_id = initial['id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: step(client, session_id, expected_revision=0), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 409]
    latest = client.get(f'/api/procedure/sessions/{session_id}').json()
    assert latest['revision'] == 1
    stale = step(client, session_id, expected_revision=0)
    assert stale.status_code == 409
    assert stale.json()['detail']['current_revision'] == 1
    assert len(client.get(f'/api/procedure/sessions/{session_id}/export').json()['transitions']) == 1


def test_terminal_and_closed_sessions_reject_steps_and_reset_preserves_old_episode(client):
    initial = create_session(client, seed=19)
    handoff = [0.] * 6 + [1.]
    terminated = step(client, initial['id'], handoff).json()
    assert terminated['terminated'] is True and terminated['status'] == 'terminated'
    assert step(client, initial['id']).status_code == 409
    reset = client.post(f"/api/procedure/sessions/{initial['id']}/reset", json={'expected_revision': 1, 'seed': 20})
    assert reset.status_code == 201
    current = reset.json()
    assert current['id'] != initial['id']
    assert current['reset_from_id'] == initial['id']
    assert current['revision'] == 0 and current['seed'] == 20
    old = client.get(f"/api/procedure/sessions/{initial['id']}/export").json()
    assert old['session']['status'] == 'closed'
    assert len(old['transitions']) == 1
    assert client.delete(f"/api/procedure/sessions/{current['id']}?expected_revision=0").status_code == 200
    assert step(client, current['id']).status_code == 409
    assert client.delete(f"/api/procedure/sessions/{current['id']}").status_code == 200


def test_truncation_stops_at_configured_horizon(client):
    initial = create_session(client, config={'max_steps': 2})
    assert step(client, initial['id']).json()['status'] == 'active'
    final = step(client, initial['id']).json()
    assert final['status'] == 'truncated' and final['truncated'] is True
    assert step(client, initial['id']).status_code == 409


def test_restart_restores_seed_and_actions_exactly(tmp_path, engine):
    with TestClient(create_test_app(tmp_path)) as first:
        initial = create_session(first, seed=99)
        for action in ([.2, 0, 0, 0, 0, 0, 0], [-.4, 0, 0, 0, 0, .3, 0]):
            before = step(first, initial['id'], action).json()
    with TestClient(create_test_app(tmp_path)) as restored:
        saved = restored.get(f"/api/procedure/sessions/{initial['id']}").json()
        assert saved == before
        after = step(restored, initial['id'], [.1, 0, 0, 0, 0, 0, 0], expected_revision=2)
        assert after.status_code == 200, after.text
        result = after.json()
        assert result['revision'] == 3
        assert result['state']['thermal_injury'] == before['state']['thermal_injury']
        assert result['state']['tool_x'] == pytest.approx(before['state']['tool_x'] + .01)


def test_persistence_failure_evicts_mutated_environment_and_replays(client, monkeypatch):
    initial = create_session(client)
    service = client.app.state.procedure
    original = service.store.append
    monkeypatch.setattr(service.store, 'append', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('Disk unavailable')))
    with pytest.raises(OSError, match='Disk unavailable'):
        step(client, initial['id'], [.5, 0, 0, 0, 0, 0, 0])
    assert initial['id'] not in service.cache
    monkeypatch.setattr(service.store, 'append', original)
    after = step(client, initial['id'], [.5, 0, 0, 0, 0, 0, 0]).json()
    assert after['revision'] == 1
    assert after['state']['tool_x'] == pytest.approx(initial['state']['tool_x'] + .05)


def test_engine_version_change_refuses_replay_but_keeps_export(client, monkeypatch):
    initial = create_session(client)
    monkeypatch.setattr(procedure, 'current_engine_version', lambda: 'changed-engine-2')
    assert step(client, initial['id']).status_code == 409
    assert client.get(f"/api/procedure/sessions/{initial['id']}/export").status_code == 200


def test_original_schema_legacy_episode_can_close_or_reset_without_old_physics(tmp_path, engine, monkeypatch):
    with TestClient(create_test_app(tmp_path)) as client:
        initial = create_session(client, seed=61)
        saved = step(client, initial['id'], [.3, 0, 0, 0, 0, 0, 0]).json()
    database = tmp_path / 'procedure' / 'episodes.sqlite3'
    with sqlite3.connect(database) as db:
        db.execute("UPDATE sessions SET engine_version='tissue-retraction-1.0'")
        db.execute('ALTER TABLE sessions DROP COLUMN policy_context')
        db.execute('ALTER TABLE sessions DROP COLUMN physics_fingerprint')
    monkeypatch.setattr(procedure, 'current_engine_version', lambda: 'tissue-retraction-1.1')
    with TestClient(create_test_app(tmp_path)) as restored:
        url = f"/api/procedure/sessions/{initial['id']}"
        legacy = restored.get(url).json()
        assert legacy['physics_fingerprint'] is None
        assert legacy['engine_compatible'] is False
        assert 'another engine version' in legacy['read_only_reason']
        assert legacy['state'] == saved['state'] and legacy['revision'] == 1
        assert step(restored, initial['id']).status_code == 409
        # Closing is a metadata operation, independent of replay compatibility.
        closed = restored.delete(url, params={'expected_revision': 1})
        assert closed.status_code == 200 and closed.json()['status'] == 'closed'
        exported = restored.get(f'{url}/export').json()
        assert exported['replay']['engine_version'] == 'tissue-retraction-1.0'
        assert exported['replay']['physics_fingerprint'] is None
        assert exported['transitions'][0]['action'] == [.3, 0, 0, 0, 0, 0, 0]
        reset = restored.post(f'{url}/reset', json={'seed': 62, 'expected_revision': 1})
        assert reset.status_code == 201, reset.text
        new = reset.json()
        assert new['id'] != initial['id'] and new['reset_from_id'] == initial['id']
        assert new['engine_version'] == 'tissue-retraction-1.1'
        assert new['physics_fingerprint'] == procedure.current_physics_fingerprint()
        assert new['engine_compatible'] is True and new['read_only_reason'] is None
        assert new['revision'] == 0 and new['seed'] == 62
        assert restored.get(url).json()['status'] == 'closed'
        assert step(restored, new['id']).status_code == 200


@pytest.mark.parametrize('fingerprint', [None, 'different-source-fingerprint'])
def test_same_version_missing_or_changed_physics_blocks_every_action_but_allows_reset(client, trainer, fingerprint):
    session = create_session(client)
    run_id = completed_training(client)
    with client.app.state.procedure.store.connect() as db:
        db.execute('UPDATE sessions SET physics_fingerprint=? WHERE id=?', (fingerprint, session['id']))
    url = f"/api/procedure/sessions/{session['id']}"
    saved = client.get(url).json()
    assert saved['engine_version'] == procedure.current_engine_version()
    assert saved['engine_compatible'] is False
    assert step(client, session['id']).status_code == 409
    assert client.post(f'{url}/scripted-step', json={}).status_code == 409
    assert client.post(f'{url}/policy-step', json={'training_run_id': run_id}).status_code == 409
    assert client.get(url).json()['revision'] == 0
    assert client.get(f'{url}/export').status_code == 200
    reset = client.post(f'{url}/reset', json={'expected_revision': 0})
    assert reset.status_code == 201
    assert reset.json()['physics_fingerprint'] == procedure.current_physics_fingerprint()
    assert reset.json()['engine_compatible'] is True
    assert client.get(url).json()['status'] == 'closed'


def test_fingerprint_guard_precedes_cached_environment_reuse(client, monkeypatch):
    session = create_session(client)
    assert session['id'] in client.app.state.procedure.cache
    monkeypatch.setattr(procedure, 'current_physics_fingerprint', lambda: 'new-physics-fingerprint')
    rejected = step(client, session['id'])
    assert rejected.status_code == 409
    assert 'different physics source' in rejected.json()['detail']
    assert client.get(f"/api/procedure/sessions/{session['id']}/export").json()['transitions'] == []


def test_cache_is_bounded_and_session_quota_requires_explicit_close(client, monkeypatch):
    monkeypatch.setattr(procedure, 'MAX_CACHED_ENVIRONMENTS', 2)
    monkeypatch.setattr(procedure, 'MAX_ACTIVE_SESSIONS', 3)
    sessions = [create_session(client) for _ in range(3)]
    assert len(client.app.state.procedure.cache) == 2
    assert client.post('/api/procedure/sessions', json={}).status_code == 429
    assert step(client, sessions[0]['id']).status_code == 200
    assert client.delete(f"/api/procedure/sessions/{sessions[0]['id']}").status_code == 200
    assert client.post('/api/procedure/sessions', json={}).status_code == 201


def test_archived_version_and_fingerprint_episodes_do_not_exhaust_active_quota(client, monkeypatch):
    monkeypatch.setattr(procedure, 'MAX_ACTIVE_SESSIONS', 2)
    legacy_version, legacy_source = [create_session(client) for _ in range(2)]
    with client.app.state.procedure.store.connect() as db:
        db.execute('UPDATE sessions SET engine_version=? WHERE id=?', ('old-version', legacy_version['id']))
        db.execute('UPDATE sessions SET physics_fingerprint=? WHERE id=?', ('old-source', legacy_source['id']))
    current = [create_session(client) for _ in range(2)]
    assert client.post('/api/procedure/sessions', json={}).status_code == 429
    assert all(item['engine_compatible'] for item in current)
    for archived in (legacy_version, legacy_source):
        saved = client.get(f"/api/procedure/sessions/{archived['id']}").json()
        assert saved['status'] == 'active' and saved['engine_compatible'] is False
        assert client.get(f"/api/procedure/sessions/{archived['id']}/export").status_code == 200
    assert len(client.get('/api/procedure/sessions').json()['sessions']) == 4


def finish_training(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f'/api/learning/runs/{run_id}').json()
        if record['status'] in ('completed', 'failed'):
            return record
        time.sleep(.01)
    pytest.fail('Training did not finish')


@pytest.fixture
def trainer(monkeypatch):
    def normalize(config):
        if set(config) - {'episodes'} or not isinstance(config.get('episodes', 20), int) or config.get('episodes', 20) < 20:
            raise ValueError('Invalid training configuration')
        return {'episodes': config.get('episodes', 20)}
    def train(config, progress_callback):
        progress_callback(.5, 'Training')
        return {'algorithm': 'test-policy', 'model_version': 'test-1',
                'checkpoint': {'action': [.2, 0, 0, 0, 0, 0, 0],
                               'environment_config': {'max_steps': 8}, 'max_steps': 8},
                'learning_curve': [{'episode': 1, 'return': 1}],
                'metadata': {'training_seeds': [42]},
                'evaluation': {'learned': {'mean_return': 1}}}
    monkeypatch.setattr(procedure, 'normalize_training_config', normalize)
    monkeypatch.setattr(procedure, 'train_policy', train)
    monkeypatch.setattr(procedure, 'trained_action', lambda checkpoint, observation: checkpoint['action'])
    monkeypatch.setattr(procedure, 'validate_policy_compatibility', lambda checkpoint, description: None)


def test_learning_lifecycle_checkpoint_playback_and_exports(client, trainer):
    created = client.post('/api/learning/runs', json={'episodes': 20})
    assert created.status_code == 202
    run_id = created.json()['id']
    complete = finish_training(client, run_id)
    assert complete['status'] == 'completed'
    assert complete['summary']['episodes_completed'] == 1
    result = client.get(f'/api/learning/runs/{run_id}/results').json()
    assert result['checkpoint']['action'][0] == .2
    initial = create_session(client)
    played = client.post(f"/api/procedure/sessions/{initial['id']}/policy-step", json={'training_run_id': run_id, 'expected_revision': 0})
    assert played.status_code == 200
    assert played.json()['revision'] == 1
    transition = client.get(f"/api/procedure/sessions/{initial['id']}/export").json()['transitions'][0]
    assert transition['source']['type'] == 'trained-policy'
    assert transition['source']['training_run_id'] == run_id
    context = played.json()['policy_context']
    assert context['out_of_distribution'] is False
    assert context['session_seed_in_training'] is True
    assert transition['source']['policy_context'] == context
    assert transition['action'] == result['checkpoint']['action']
    assert client.get(f'/api/learning/runs/{run_id}/export').json()['run']['result'] == result
    assert client.post(f'/api/learning/runs/{run_id}/cancel').status_code == 409
    assert client.post('/api/learning/runs', json={'episodes': 1}).status_code == 422


def completed_training(client):
    created = client.post('/api/learning/runs', json={'episodes': 20})
    assert created.status_code == 202
    run_id = created.json()['id']
    assert finish_training(client, run_id)['status'] == 'completed'
    return run_id


def test_policy_distribution_preview_and_playback_report_horizon_and_patient_override(client, trainer):
    run_id = completed_training(client)
    session = create_session(client, seed=43, config={'max_steps': 12}, patient={'attachment_length_mm': 24})
    url = f"/api/procedure/sessions/{session['id']}"
    preview = client.get(f'{url}/policy-context?training_run_id={run_id}')
    assert preview.status_code == 200
    context = preview.json()
    assert context['out_of_distribution'] is True
    assert context['comparison_available'] is True
    assert context['patient_override'] is True
    assert context['session_seed_in_training'] is False
    assert context['differing_parameters'] == [{'parameter': 'max_steps', 'training': 8, 'session': 12}]
    assert len(context['reasons']) == 2
    assert client.get(url).json()['revision'] == 0
    played = client.post(f'{url}/policy-step', json={'training_run_id': run_id}).json()
    assert played['revision'] == 1
    assert played['policy_context'] == context
    assert client.get(url).json()['policy_context'] == context
    assert client.get(f'{url}/export').json()['transitions'][0]['source']['policy_context'] == context


def test_matched_policy_episode_uses_recorded_horizon_and_keeps_context_on_restart(tmp_path, engine, trainer):
    with TestClient(create_test_app(tmp_path)) as client:
        run_id = completed_training(client)
        response = client.post(f'/api/learning/runs/{run_id}/session', json={'seed': 91})
        assert response.status_code == 201, response.text
        session = response.json()
        assert session['config'] == {'max_steps': 8}
        assert session['seed'] == 91 and session['patient'] is None
        assert session['policy_context']['out_of_distribution'] is False
        assert session['policy_context']['training_environment'] == session['config']
        first = client.post(f"/api/procedure/sessions/{session['id']}/policy-step", json={'training_run_id': run_id}).json()
    with TestClient(create_test_app(tmp_path)) as restored:
        assert restored.get(f"/api/procedure/sessions/{session['id']}").json() == first
        resumed = step(restored, session['id'], expected_revision=1)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()['revision'] == 2
        assert resumed.json()['policy_context'] == first['policy_context']


def test_legacy_checkpoint_environment_is_normalized_for_matched_session(client, trainer):
    run_id = completed_training(client)
    store = client.app.state.learning.store
    result = store.get(run_id, include_result=True)['result']
    result['checkpoint'].pop('environment_config')
    result['checkpoint']['environment'] = {}
    result['checkpoint']['max_steps'] = 13
    store.update(run_id, result=result)
    matched = client.post(f'/api/learning/runs/{run_id}/session', json={'seed': 7})
    assert matched.status_code == 201
    assert matched.json()['config']['max_steps'] == 13
    assert matched.json()['policy_context']['out_of_distribution'] is False
    # Session creation accepts a seed only; callers cannot hide config changes
    # behind the "matched" route or override the recorded patient distribution.
    assert client.post(f'/api/learning/runs/{run_id}/session', json={'seed': 7, 'config': {'max_steps': 5}}).status_code == 422
    assert client.post(f'/api/learning/runs/{run_id}/session', json={'seed': True}).status_code == 422


def test_missing_or_inconsistent_checkpoint_environment_never_claims_a_match(client, trainer):
    run_id = completed_training(client)
    store = client.app.state.learning.store
    result = store.get(run_id, include_result=True)['result']
    result['checkpoint'].pop('environment_config')
    result['checkpoint'].pop('max_steps')
    store.update(run_id, result=result)
    session = create_session(client)
    played = client.post(f"/api/procedure/sessions/{session['id']}/policy-step", json={'training_run_id': run_id})
    assert played.status_code == 200
    assert played.json()['policy_context']['out_of_distribution'] is True
    assert played.json()['policy_context']['comparison_available'] is False
    assert client.post(f'/api/learning/runs/{run_id}/session', json={}).status_code == 422
    result['checkpoint'].update(environment_config={'max_steps': 8}, max_steps=13)
    store.update(run_id, result=result)
    before = len(client.get('/api/procedure/sessions').json()['sessions'])
    assert client.post(f'/api/learning/runs/{run_id}/session', json={}).status_code == 422
    assert len(client.get('/api/procedure/sessions').json()['sessions']) == before


def test_episode_database_before_policy_context_migrates_without_losing_replay(tmp_path, engine):
    with TestClient(create_test_app(tmp_path)) as client:
        session = create_session(client, seed=76)
        saved = step(client, session['id'], [.2, 0, 0, 0, 0, 0, 0]).json()
    # Reproduce the immediately preceding released schema, preserving all
    # original session columns and the already committed transition log.
    with sqlite3.connect(tmp_path / 'procedure' / 'episodes.sqlite3') as db:
        db.execute('ALTER TABLE sessions DROP COLUMN policy_context')
    with TestClient(create_test_app(tmp_path)) as restored:
        assert restored.get(f"/api/procedure/sessions/{session['id']}").json() == saved
        resumed = step(restored, session['id'], expected_revision=1)
        assert resumed.status_code == 200
        assert resumed.json()['revision'] == 2


def test_fingerprint_migration_preserves_identity_actually_recorded_in_real_snapshot(tmp_path):
    with TestClient(create_test_app(tmp_path)) as client:
        session = create_session(client, seed=83)
        saved = step(client, session['id']).json()
        fingerprint = saved['physics_fingerprint']
        assert len(fingerprint) == 64
        assert fingerprint == saved['info']['physics_fingerprint']
        assert client.get(f"/api/procedure/sessions/{session['id']}/export").json()['replay']['physics_fingerprint'] == fingerprint
    with sqlite3.connect(tmp_path / 'procedure' / 'episodes.sqlite3') as db:
        db.execute('ALTER TABLE sessions DROP COLUMN physics_fingerprint')
    with TestClient(create_test_app(tmp_path)) as restored:
        migrated = restored.get(f"/api/procedure/sessions/{session['id']}").json()
        assert migrated == saved
        assert migrated['engine_compatible'] is True
        assert step(restored, session['id'], expected_revision=1).status_code == 200


def test_learning_queue_capacity_and_cooperative_cancellation(client, trainer, monkeypatch):
    def blocked(config, progress_callback):
        while True:
            progress_callback(.1)
            time.sleep(.01)
    monkeypatch.setattr(procedure, 'train_policy', blocked)
    first = client.post('/api/learning/runs', json={}).json()
    second = client.post('/api/learning/runs', json={}).json()
    assert client.post('/api/learning/runs', json={}).status_code == 429
    assert client.get(f"/api/learning/runs/{first['id']}/results").status_code == 409
    assert client.post(f"/api/learning/runs/{second['id']}/cancel").status_code == 200
    assert finish_training(client, second['id'])['status'] == 'failed'
    assert client.post(f"/api/learning/runs/{first['id']}/cancel").status_code == 200
    cancelled = finish_training(client, first['id'])
    assert cancelled['status'] == 'failed'
    assert 'cancelled' in cancelled['error'].lower()


def test_unknown_ids_validation_and_nonfinite_json_conversion(client):
    assert client.get(f'/api/procedure/sessions/{uuid4()}').status_code == 404
    assert client.get('/api/procedure/sessions/not-a-uuid').status_code == 422
    assert client.get('/api/procedure/sessions?limit=201').status_code == 422
    assert client.get(f'/api/learning/runs/{uuid4()}').status_code == 404
    with pytest.raises(ValueError, match='nonfinite'):
        procedure.json_safe({'force': float('nan')})


def test_actual_tissue_heat_harm_and_reward_survive_api_restart(tmp_path):
    """Exercise real coupled body state through REST, not the test double."""
    with TestClient(create_test_app(tmp_path)) as client:
        initial = create_session(client, seed=42, config={'randomize_anatomy': False,
                                'injury_termination_threshold': 1., 'critical_injury_threshold': .9})
        session_id = initial['id']
        spec = client.get('/api/procedure/spec').json()
        assert spec['observation_size'] == len(initial['observation'])
        assert len(spec['action_names']) == 7
        assert initial['state']['max_damage'] == 0
        # Move only the cautery in the illustrative workspace, then apply heat.
        for action in [[0, 0, 0, 1, 1, 0, 0]] * 4 + [[0, 0, 0, 1, 0, 0, 0]] * 8:
            assert step(client, session_id, action).status_code == 200
        after_movement = client.get(f'/api/procedure/sessions/{session_id}').json()
        assert after_movement['state']['max_damage'] == 0
        for _ in range(5):
            heated = step(client, session_id, [0, 0, 0, 0, 0, 1, 0]).json()
        assert heated['state']['max_damage'] > .1
        assert heated['state']['physiology']['blood_loss_ml'] > 0
        assert heated['reward_components']['critical_damage'] < 0
        assert heated['reward'] == pytest.approx(sum(heated['reward_components'].values()))
        assert heated['info']['success'] is False
        export = client.get(f'/api/procedure/sessions/{session_id}/export').json()
        assert len(export['transitions']) == 17
        assert any(event['type'] == 'bleeding_started' for transition in export['transitions'] for event in transition['events'])
    # Persisted snapshots can be read without replay, but the next action must
    # restore all heat, irreversible injury, physiology, and tool state exactly.
    with TestClient(create_test_app(tmp_path)) as client:
        assert client.get(f'/api/procedure/sessions/{session_id}').json() == heated
        resumed = step(client, session_id, ZERO, expected_revision=17)
        assert resumed.status_code == 200, resumed.text
        current = resumed.json()
        assert current['revision'] == 18
        assert current['state']['max_damage'] >= heated['state']['max_damage']
        assert current['state']['physiology']['blood_loss_ml'] >= heated['state']['physiology']['blood_loss_ml']


def test_actual_scripted_feedback_step_is_identified_as_baseline(tmp_path):
    with TestClient(create_test_app(tmp_path)) as client:
        initial = create_session(client, seed=11)
        response = client.post(f"/api/procedure/sessions/{initial['id']}/scripted-step", json={'expected_revision': 0})
        assert response.status_code == 200
        state = response.json()
        assert state['state']['tools']['grasper']['contact_node_id'] is not None
        transition = client.get(f"/api/procedure/sessions/{initial['id']}/export").json()['transitions'][0]
        assert transition['source']['type'] == 'scripted-feedback-baseline'
        assert len(transition['action']) == 7


def test_actual_small_training_checkpoint_can_control_current_session(tmp_path):
    with TestClient(create_test_app(tmp_path)) as client:
        response = client.post('/api/learning/runs', json={'episodes': 20, 'eval_episodes': 5, 'max_steps': 20,
                                                         'environment': {'force_limit_n': 1.75}})
        assert response.status_code == 202
        run_id = response.json()['id']
        completed = finish_training(client, run_id)
        assert completed['status'] == 'completed', completed
        trained = client.get(f'/api/learning/runs/{run_id}/results').json()
        assert trained['metadata']['environment_steps'] > 0
        assert trained['metadata']['gradient_updates'] > 0
        assert set(trained['metadata']['training_seeds']).isdisjoint(trained['metadata']['test_seeds'])
        assert len(trained['evaluation']['learned']['episodes']) == 5
        session = create_session(client, seed=123)
        acted = client.post(f"/api/procedure/sessions/{session['id']}/policy-step", json={'training_run_id': run_id, 'expected_revision': 0})
        assert acted.status_code == 200, acted.text
        assert acted.json()['revision'] == 1
        assert acted.json()['policy_context']['out_of_distribution'] is True
        assert {item['parameter'] for item in acted.json()['policy_context']['differing_parameters']} == {'max_steps', 'force_limit_n'}
        transition = client.get(f"/api/procedure/sessions/{session['id']}/export").json()['transitions'][0]
        assert len(transition['action']) == 7
        assert transition['source']['training_run_id'] == run_id
        matched = client.post(f'/api/learning/runs/{run_id}/session', json={'seed': 124})
        assert matched.status_code == 201, matched.text
        assert matched.json()['config']['max_steps'] == 20
        assert matched.json()['config']['force_limit_n'] == 1.75
        assert matched.json()['policy_context']['out_of_distribution'] is False


def test_paginated_event_and_transition_history_can_resume_after_reload(client):
    initial = create_session(client)
    session_id = initial['id']
    for action in (ZERO, [0, 0, 0, 0, 0, .5, 0], ZERO):
        assert step(client, session_id, action).status_code == 200
    first = client.get(f'/api/procedure/sessions/{session_id}/events?limit=1').json()
    assert first['events'] == []
    assert first['next_revision'] == 1 and first['has_more'] is True
    second = client.get(f'/api/procedure/sessions/{session_id}/events?after_revision=1&limit=1').json()
    assert second['events'][0]['type'] == 'thermal_injury'
    assert second['events'][0]['revision'] == 2
    assert second['next_revision'] == 2
    last = client.get(f'/api/procedure/sessions/{session_id}/transitions?after_revision=2').json()
    assert len(last['transitions']) == 1 and last['transitions'][0]['revision'] == 3
    assert last['has_more'] is False
    assert client.get(f'/api/procedure/sessions/{session_id}/transitions?limit=201').status_code == 422
