"""Controlled response protocols use durable ordinary episode transitions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from retraction import procedure


def application(directory):
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
def client(tmp_path):
    with TestClient(application(tmp_path)) as client:
        yield client


def create(client, scenario='remote-heat-control', **config):
    response = client.post(f'/api/procedure/scenarios/{scenario}/session', json=config)
    assert response.status_code == 201, response.text
    return response.json()


def advance(client, record, **options):
    return client.post(f"/api/procedure/sessions/{record['id']}/protocol-step",
                       json={'expected_revision': record['revision'], **options})


def test_catalog_requires_explicit_creation_and_records_effective_configuration(client):
    catalog = client.get('/api/procedure/scenarios').json()['scenarios']
    assert len(catalog) == 5
    assert all('not clinical' in item['scope'] for item in catalog)
    assert client.get('/api/procedure/sessions').json()['sessions'] == []
    record = create(client, seed=17, config={'max_steps': 40}, patient={'attachment_length_mm': 20})
    protocol = record['protocol']
    assert record['revision'] == 0 and record['state']['max_damage'] == 0
    assert record['seed'] == 17 and record['config']['max_steps'] == 40
    assert protocol['environment_config'] == record['config']
    assert protocol['patient_override'] is True
    assert protocol['protocol_step'] == 0 and protocol['phase'] == 'ready'
    assert protocol['controller_state']['max_steps'] == 40
    assert client.post('/api/procedure/scenarios/unknown/session', json={}).status_code == 404
    assert client.post('/api/procedure/scenarios/remote-heat-control/session', json={'seed': -1}).status_code == 422


@pytest.mark.parametrize('scenario', ['gentle-traction', 'held-jaw-crush', 'arterial-thermal-contact',
                                     'duct-thermal-contact', 'remote-heat-control'])
def test_real_protocol_actions_persist_outcomes_and_provenance(client, scenario):
    record = create(client, scenario)
    assert record['seed'] == 2
    for _ in range(201):
        response = advance(client, record)
        assert response.status_code == 200, response.text
        record = response.json()
        if record['protocol']['completed']:
            break
    else:
        pytest.fail('Controlled protocol exceeded its bounded step budget')
    export = client.get(f"/api/procedure/sessions/{record['id']}/export").json()
    transitions = export['transitions']
    assert len(transitions) == record['revision'] == record['protocol']['protocol_step']
    for index, transition in enumerate(transitions, start=1):
        source = transition['source']
        assert source['type'] == 'response-protocol' and source['scenario_id'] == scenario
        assert source['version'] == record['protocol']['version']
        assert source['protocol_step'] == source['controller_state']['steps'] == index
        assert len(transition['action']) == 7
    assert record['source'] == transitions[-1]['source']
    if scenario == 'gentle-traction':
        assert record['info']['success'] is True and record['state']['max_damage'] == 0
    elif scenario == 'held-jaw-crush':
        assert record['state']['max_damage'] > .6
    elif scenario == 'arterial-thermal-contact':
        assert record['state']['physiology']['blood_loss_ml'] > 0
    elif scenario == 'duct-thermal-contact':
        assert record['state']['physiology']['bile_leak_ml'] > 0
    else:
        assert record['status'] == 'active'
        assert record['protocol']['status'] == 'completed'
        assert record['state']['max_damage'] == 0
        assert record['protocol']['completion_reason'] == 'protocol-duration'
    # Completion bookkeeping does not synthesize a tissue action or success.
    assert advance(client, record).json() == record


def test_controller_phase_and_heat_counter_survive_service_restart(tmp_path):
    with TestClient(application(tmp_path)) as client:
        record = create(client)
        for _ in range(7):
            record = advance(client, record).json()
        assert record['protocol']['controller_state']['exposure_steps'] == 7
    with TestClient(application(tmp_path)) as restored:
        saved = restored.get(f"/api/procedure/sessions/{record['id']}").json()
        assert saved == record
        eighth = advance(restored, saved).json()
        assert eighth['protocol']['controller_state']['exposure_steps'] == 8
        assert eighth['protocol']['phase'] == 'heat'
        ninth = advance(restored, eighth).json()
        assert ninth['protocol']['phase'] == 'observe'
        assert ninth['protocol']['controller_state']['exposure_steps'] == 8
        assert ninth['protocol']['controller_state']['observation_steps'] == 1
        transitions = restored.get(f"/api/procedure/sessions/{record['id']}/export").json()['transitions']
        assert transitions[7]['action'][5] == 1 and transitions[8]['action'][5] == 0


def test_protocol_revision_race_has_one_action_and_one_durable_controller_increment(client):
    initial = create(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: advance(client, initial), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 409]
    saved = client.get(f"/api/procedure/sessions/{initial['id']}").json()
    assert saved['revision'] == saved['protocol']['protocol_step'] == 1
    assert saved['protocol']['controller_state']['exposure_steps'] == 1


def test_protocol_counter_is_rolled_back_with_failed_transition_persistence(client, monkeypatch):
    initial = create(client)
    store = client.app.state.procedure.store
    original = store.append
    monkeypatch.setattr(store, 'append', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('Disk unavailable')))
    with pytest.raises(OSError, match='Disk unavailable'):
        advance(client, initial)
    saved = client.get(f"/api/procedure/sessions/{initial['id']}").json()
    assert saved['revision'] == saved['protocol']['protocol_step'] == 0
    monkeypatch.setattr(store, 'append', original)
    resumed = advance(client, saved).json()
    assert resumed['revision'] == resumed['protocol']['protocol_step'] == 1
    assert resumed['state']['time_s'] == pytest.approx(.1)


def test_manual_intervention_marks_protocol_interrupted_and_reset_starts_fresh_case(client):
    initial = create(client)
    record = advance(client, initial).json()
    manual = client.post(f"/api/procedure/sessions/{record['id']}/step",
                         json={'action': [0] * 7, 'expected_revision': record['revision']}).json()
    assert manual['protocol']['status'] == 'interrupted'
    assert manual['protocol']['completed'] is True
    assert manual['source'] == {'type': 'manual'}
    assert manual['revision'] == 2 and manual['protocol']['protocol_step'] == 1
    assert advance(client, manual).status_code == 409
    reset = client.post(f"/api/procedure/sessions/{manual['id']}/reset", json={'expected_revision': 2})
    assert reset.status_code == 201
    fresh = reset.json()
    assert fresh['reset_from_id'] == manual['id'] and fresh['id'] != manual['id']
    assert fresh['protocol']['scenario_id'] == manual['protocol']['scenario_id']
    assert fresh['revision'] == fresh['protocol']['protocol_step'] == 0
    assert fresh['protocol']['status'] == 'active'
    assert client.get(f"/api/procedure/sessions/{manual['id']}").json()['status'] == 'closed'


def test_closing_incomplete_case_marks_interruption_without_a_synthetic_tissue_action(client):
    initial = create(client)
    progressed = advance(client, initial).json()
    response = client.delete(f"/api/procedure/sessions/{initial['id']}", params={'expected_revision': 1})
    assert response.status_code == 200
    closed = response.json()
    assert closed['status'] == 'closed'
    assert closed['protocol']['status'] == 'interrupted' and closed['protocol']['completed'] is True
    assert closed['protocol']['protocol_step'] == progressed['protocol']['protocol_step'] == 1
    assert closed['revision'] == 1 and closed['state'] == progressed['state']
    assert len(client.get(f"/api/procedure/sessions/{initial['id']}/export").json()['transitions']) == 1


def test_protocol_controller_rejects_changed_physics_without_advancing(client):
    initial = create(client)
    with client.app.state.procedure.store.connect() as db:
        db.execute('UPDATE sessions SET physics_fingerprint=? WHERE id=?', ('old-physics', initial['id']))
    rejected = advance(client, initial)
    assert rejected.status_code == 409
    assert client.get(f"/api/procedure/sessions/{initial['id']}").json()['revision'] == 0


def test_saved_protocol_version_is_readable_but_requires_reset_before_current_controller(client):
    initial = create(client)
    service = client.app.state.procedure
    record = service.require(initial['id'])
    protocol = initial['protocol']
    protocol['version'] = 'old-response-protocol'
    protocol['controller_state']['protocol_version'] = 'old-response-protocol'
    service.store.update_protocol(record, protocol)
    assert advance(client, initial).status_code == 409
    saved = client.get(f"/api/procedure/sessions/{initial['id']}").json()
    assert saved['revision'] == 0 and saved['protocol']['version'] == 'old-response-protocol'
    assert client.get(f"/api/procedure/sessions/{initial['id']}/export").status_code == 200
    reset = client.post(f"/api/procedure/sessions/{initial['id']}/reset", json={'expected_revision': 0})
    assert reset.status_code == 201
    assert reset.json()['protocol']['version'] != 'old-response-protocol'
    assert reset.json()['protocol']['protocol_step'] == 0
