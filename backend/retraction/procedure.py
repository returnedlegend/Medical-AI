"""Durable, revisioned live tissue episodes and bounded local policy training.

Episode restoration replays validated actions from an explicit seed/configuration.
No executable or pickle state is accepted or stored. All metrics describe a
synthetic research environment, never a patient or a clinical safety claim.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import math
from pathlib import Path
import sqlite3
import threading
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .storage import Store, timestamp

logger = logging.getLogger(__name__)
API_VERSION = 'procedure-api-1'
MAX_ACTIVE_SESSIONS = 32
MAX_CACHED_ENVIRONMENTS = 16
MAX_EPISODE_TRANSITIONS = 1000


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays while rejecting nonfinite simulation data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Environment returned nonfinite data')
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if hasattr(value, 'item'):
        return json_safe(value.item())
    raise ValueError(f'Environment returned unsupported data type: {type(value).__name__}')


def encoded(value: Any) -> str:
    return json.dumps(json_safe(value), allow_nan=False, separators=(',', ':'), sort_keys=True)


def make_environment(config: dict):
    from .environment import TissueRetractionEnv
    return TissueRetractionEnv(config=config)


def normalize_environment_config(config: dict) -> dict:
    from .environment import normalize_env_config
    return json_safe(normalize_env_config(config))


def current_engine_version() -> str:
    from . import environment
    return str(getattr(environment, 'ENV_VERSION', 'tissue-environment-1'))


def current_physics_fingerprint() -> str:
    from .environment import PHYSICS_FINGERPRINT
    return PHYSICS_FINGERPRINT


def engine_incompatibility(record: dict) -> str | None:
    if record['engine_version'] != current_engine_version():
        return 'Saved episode uses another engine version. Read or export it, or reset to create a new current-model episode.'
    if not record.get('physics_fingerprint'):
        return 'Saved episode has no recorded physics fingerprint. Read or export it, or reset to create a new episode.'
    if record['physics_fingerprint'] != current_physics_fingerprint():
        return 'Saved episode uses different physics source. Read or export it, or reset to create a new current-model episode.'
    return None


def normalize_training_config(config: dict) -> dict:
    from .learning import normalize_training_config as normalize
    return json_safe(normalize(config))


def train_policy(config: dict, progress_callback):
    from .learning import train_policy as train
    return train(config, progress_callback=progress_callback)


def validate_policy_compatibility(checkpoint: dict, description: dict):
    from .learning import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(checkpoint, description)


def trained_action(checkpoint: dict, observation: list) -> list[float]:
    from .learning import policy_action
    return json_safe(policy_action(checkpoint, observation))


class PatientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    attachment_length_mm: float = Field(default=24, ge=4, le=60)
    adhesion_count: int = Field(default=2, ge=0, le=12, strict=True)
    adhesion_stiffness_n_per_mm: float = Field(default=0.1, ge=0, le=0.5)
    distension: float = Field(default=1.2, ge=0.5, le=2.5)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=4294967295, strict=True)
    config: dict[str, Any] = Field(default_factory=dict)
    patient: PatientConfig | None = None


Control = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False, strict=True)]


class StepConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    action: list[Control] = Field(min_length=7, max_length=7)
    expected_revision: int | None = Field(default=None, ge=0, strict=True)


class RevisionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    expected_revision: int | None = Field(default=None, ge=0, strict=True)


class ResetConfig(RevisionConfig):
    seed: int | None = Field(default=None, ge=0, le=4294967295, strict=True)


class PolicyStepConfig(RevisionConfig):
    training_run_id: UUID


class MatchedPolicySessionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=4294967295, strict=True)


class ScenarioSessionConfig(SessionConfig):
    seed: int | None = Field(default=None, ge=0, le=4294967295, strict=True)


def protocol_progress(controller, previous: dict | None = None, *, episode_finished=False) -> dict:
    state = json_safe(controller.state_dict())
    return {**(previous or {}), 'scenario_id': controller.scenario_id,
            'version': state['protocol_version'], 'phase': controller.phase,
            'protocol_step': controller.steps, 'completed': bool(controller.completed),
            'completion_reason': controller.completion_reason, 'target_node_id': state['target_node_id'],
            'status': 'episode-finished' if episode_finished else 'completed' if controller.completed else 'active',
            'controller_state': state, 'metadata': json_safe(controller.metadata)}


def scenario_definition(scenario_id: str) -> dict:
    from .scenarios import scenario_catalog
    definition = next((item for item in scenario_catalog() if item['id'] == scenario_id), None)
    if definition is None:
        raise HTTPException(404, 'Controlled response protocol not found')
    return definition


def create_protocol_session(service, scenario_id: str, config: ScenarioSessionConfig, *, reset_from_id=None):
    from .scenarios import MAX_PROTOCOL_STEPS, create_scenario_controller
    definition = scenario_definition(scenario_id)
    try:
        settings = normalize_environment_config({**definition['config'], **config.config})
        seed = definition['seed'] if config.seed is None else config.seed
        controller = create_scenario_controller(scenario_id, max_steps=min(settings['max_steps'], MAX_PROTOCOL_STEPS))
        protocol = protocol_progress(controller, {'environment_config': settings, 'configured_seed': seed,
                                                  'patient_override': config.patient is not None})
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error
    return service.create(SessionConfig(seed=seed, config=settings, patient=config.patient),
                          reset_from_id=reset_from_id, protocol=protocol)


def training_environment_config(training: dict) -> dict | None:
    """Recover the effective training settings, including the episode horizon.

    New checkpoints record all normalized settings. Older checkpoints carry a
    partial environment dictionary plus a separate horizon. A policy without
    either form remains playable with an explicit unknown-distribution warning,
    but cannot be used to promise a matched session.
    """
    checkpoint = training['result']['checkpoint']
    effective = checkpoint.get('environment_config')
    if effective is not None:
        if not isinstance(effective, dict):
            raise ValueError('Policy effective environment metadata must be an object')
        normalized = normalize_environment_config(effective)
        if 'max_steps' in checkpoint and normalized.get('max_steps') != checkpoint['max_steps']:
            raise ValueError('Policy horizon disagrees with its effective environment metadata')
        return normalized
    partial = checkpoint.get('environment')
    horizon = checkpoint.get('max_steps')
    if partial is None and horizon is None:
        return None
    if not isinstance(partial, dict):
        raise ValueError('Policy environment metadata must be an object')
    return normalize_environment_config({**partial, **({'max_steps': horizon} if horizon is not None else {})})


def policy_context(training: dict, record: dict) -> dict:
    """Report a configuration comparison, never a generalization claim."""
    trained = training_environment_config(training)
    current = normalize_environment_config(record['config'])
    reasons, differences = [], []
    if trained is None:
        reasons.append('Training environment metadata is unavailable for this checkpoint.')
    else:
        for key in sorted(set(trained) | set(current)):
            if trained.get(key) != current.get(key):
                differences.append({'parameter': key, 'training': trained.get(key), 'session': current.get(key)})
                reasons.append(f'Environment parameter {key} differs from training.')
    patient_override = record.get('patient') is not None
    if patient_override:
        reasons.append('This episode uses an explicit patient override outside the recorded training setup.')
    training_seeds = training['result'].get('metadata', {}).get('training_seeds')
    return {
        'version': 'policy-context-1', 'training_run_id': str(training['id']),
        'out_of_distribution': bool(reasons), 'comparison_available': trained is not None,
        'reasons': reasons, 'differing_parameters': differences,
        'training_environment': trained, 'session_environment': current,
        'patient_override': patient_override,
        'session_seed_in_training': record.get('seed') in training_seeds if isinstance(training_seeds, list) else None,
        'clinical_use': False,
        'interpretation': 'Matching simulator settings is not evidence of policy generalization or clinical validity.',
    }


class EpisodeConflict(RuntimeError):
    pass


class EpisodeStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'episodes.sqlite3'
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                seed INTEGER NOT NULL, config TEXT NOT NULL, patient TEXT,
                engine_version TEXT NOT NULL, status TEXT NOT NULL,
                revision INTEGER NOT NULL, snapshot TEXT NOT NULL, reset_from_id TEXT,
                policy_context TEXT, physics_fingerprint TEXT, protocol TEXT
            )''')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(sessions)')}
            if 'policy_context' not in columns:
                db.execute('ALTER TABLE sessions ADD COLUMN policy_context TEXT')
            if 'protocol' not in columns:
                db.execute('ALTER TABLE sessions ADD COLUMN protocol TEXT')
            if 'physics_fingerprint' not in columns:
                db.execute('ALTER TABLE sessions ADD COLUMN physics_fingerprint TEXT')
                # Preserve a fingerprint only when the historical snapshot
                # actually recorded one. Never infer old physics from the
                # currently installed model or a matching version string.
                for row in db.execute('SELECT id,snapshot FROM sessions').fetchall():
                    recorded = json.loads(row['snapshot']).get('info', {}).get('physics_fingerprint')
                    if isinstance(recorded, str) and recorded:
                        db.execute('UPDATE sessions SET physics_fingerprint=? WHERE id=?', (recorded, row['id']))
            db.execute('''CREATE TABLE IF NOT EXISTS transitions (
                session_id TEXT NOT NULL, revision INTEGER NOT NULL,
                action TEXT NOT NULL, transition TEXT NOT NULL,
                PRIMARY KEY (session_id, revision),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        for field in ('config', 'patient', 'snapshot', 'policy_context', 'protocol'):
            result[field] = json.loads(result[field]) if result[field] is not None else None
        return result

    def get(self, session_id: str):
        with self.connect() as db:
            row = db.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
            record = self.decode(row)
            if record is not None:
                transition = db.execute('SELECT transition FROM transitions WHERE session_id=? AND revision=?',
                                        (session_id, record['revision'])).fetchone()
                record['source'] = json.loads(transition['transition']).get('source') if transition else None
        return record

    def list(self, limit: int, offset: int):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
        return [self.decode(row) for row in rows]

    def active_count(self):
        with self.connect() as db:
            return db.execute("""SELECT COUNT(*) FROM sessions
                WHERE status='active' AND engine_version=? AND physics_fingerprint=?""",
                (current_engine_version(), current_physics_fingerprint())).fetchone()[0]

    def create(self, seed: int, config: dict, patient: dict | None, snapshot: dict, version: str,
               reset_from_id=None, context=None, physics_fingerprint=None, protocol=None):
        session_id, now = str(uuid4()), timestamp()
        with self.connect() as db:
            db.execute('''INSERT INTO sessions
                       (id,created_at,updated_at,seed,config,patient,engine_version,status,revision,snapshot,reset_from_id,policy_context,physics_fingerprint,protocol)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (session_id, now, now, seed, encoded(config), encoded(patient) if patient is not None else None,
                        version, 'active', 0, encoded(snapshot), reset_from_id, encoded(context) if context else None,
                        physics_fingerprint, encoded(protocol) if protocol else None))
        return self.get(session_id)

    def append(self, record: dict, action: list, snapshot: dict, status: str, source: dict, *, protocol=None):
        revision = record['revision'] + 1
        transition = {'revision': revision, 'action': action, 'source': source, **snapshot}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision,status FROM sessions WHERE id=?', (record['id'],)).fetchone()
            if row is None or row['revision'] != record['revision'] or row['status'] != 'active':
                raise EpisodeConflict('Episode changed during this operation; fetch its latest state.')
            db.execute('INSERT INTO transitions VALUES (?,?,?,?)',
                       (record['id'], revision, encoded(action), encoded(transition)))
            db.execute('UPDATE sessions SET revision=?,snapshot=?,status=?,updated_at=? WHERE id=?',
                       (revision, encoded(snapshot), status, timestamp(), record['id']))
            if source.get('policy_context') is not None:
                db.execute('UPDATE sessions SET policy_context=? WHERE id=?',
                           (encoded(source['policy_context']), record['id']))
            if protocol is not None:
                db.execute('UPDATE sessions SET protocol=? WHERE id=?', (encoded(protocol), record['id']))
        return self.get(record['id'])

    def update_protocol(self, record: dict, protocol: dict):
        """Complete controller bookkeeping without inventing a tissue action."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            changed = db.execute('UPDATE sessions SET protocol=?,updated_at=? WHERE id=? AND revision=? AND status=?',
                                 (encoded(protocol), timestamp(), record['id'], record['revision'], record['status'])).rowcount
            if changed != 1:
                raise EpisodeConflict('Episode changed during this operation; fetch its latest state.')
        return self.get(record['id'])

    def close(self, record: dict):
        with self.connect() as db:
            updated = db.execute("UPDATE sessions SET status='closed',updated_at=? WHERE id=? AND revision=? AND status=?",
                                 (timestamp(), record['id'], record['revision'], record['status'])).rowcount
            if updated != 1:
                raise EpisodeConflict('Episode changed during this operation; fetch its latest state.')
            protocol = record.get('protocol')
            if protocol and protocol['status'] == 'active':
                interrupted = {**protocol, 'status': 'interrupted', 'completed': True,
                               'completion_reason': 'Episode closed before the controlled response protocol completed.'}
                db.execute('UPDATE sessions SET protocol=? WHERE id=?', (encoded(interrupted), record['id']))
        return self.get(record['id'])

    def transitions(self, session_id: str, *, after_revision: int = 0, limit: int | None = None):
        with self.connect() as db:
            query = 'SELECT transition FROM transitions WHERE session_id=? AND revision>? ORDER BY revision'
            parameters: list = [session_id, after_revision]
            if limit is not None:
                query += ' LIMIT ?'
                parameters.append(limit)
            rows = db.execute(query, parameters).fetchall()
        return [json.loads(row['transition']) for row in rows]


def snapshot(observation, info, *, reward=0.0, episode_reward=0.0, terminated=False, truncated=False):
    info = json_safe(info)
    return {'observation': json_safe(observation), 'info': info, 'state': info.get('state', {}),
            'reward': float(reward), 'episode_reward': float(episode_reward),
            'reward_components': info.get('reward_components', {}), 'events': info.get('events', []),
            'terminated': bool(terminated), 'truncated': bool(truncated)}


def public_record(record: dict, *, include_state=True):
    metadata = {key: value for key, value in record.items() if key != 'snapshot'}
    metadata.update(api_version=API_VERSION, mode='synthetic-tissue-research', clinical_use=False)
    reason = engine_incompatibility(record)
    metadata.update(engine_compatible=reason is None, read_only_reason=reason)
    if include_state:
        metadata.update(record['snapshot'])
    else:
        metadata['episode_reward'] = record['snapshot']['episode_reward']
        metadata['termination_reason'] = record['snapshot']['info'].get('termination_reason')
    return metadata


class ProcedureService:
    def __init__(self, directory: Path):
        self.store = EpisodeStore(directory)
        self.lock = threading.RLock()
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_revisions: dict[str, int] = {}
        self.closed = False

    def discard(self, session_id):
        environment = self.cache.pop(session_id, None)
        self.cache_revisions.pop(session_id, None)
        if environment is not None and hasattr(environment, 'close'):
            environment.close()

    def cache_environment(self, session_id, environment, revision=0):
        self.cache[session_id] = environment
        self.cache_revisions[session_id] = revision
        self.cache.move_to_end(session_id)
        while len(self.cache) > MAX_CACHED_ENVIRONMENTS:
            oldest = next(iter(self.cache))
            self.discard(oldest)

    def require(self, session_id: str, expected_revision: int | None = None):
        record = self.store.get(session_id)
        if record is None:
            raise HTTPException(404, 'Procedure session not found')
        if expected_revision is not None and expected_revision != record['revision']:
            raise HTTPException(409, {'message': 'Revision conflict; fetch the current state before another action.',
                                      'current_revision': record['revision']})
        return record

    def create(self, config: SessionConfig, *, reset_from_id=None, context=None, protocol=None):
        with self.lock:
            if self.closed:
                raise HTTPException(503, 'Procedure service is stopping')
            if self.store.active_count() >= MAX_ACTIVE_SESSIONS:
                raise HTTPException(429, 'Close an active procedure before creating another session (limit 32).')
            environment = None
            try:
                normalized = normalize_environment_config(config.config)
                environment = make_environment(normalized)
                patient = config.patient.model_dump() if config.patient else None
                observation, info = environment.reset(seed=config.seed, options={'patient': patient} if patient else None)
                initial = snapshot(observation, info)
                encoded(initial)
                record = self.store.create(config.seed, normalized, patient, initial, current_engine_version(),
                                           reset_from_id, context=context, physics_fingerprint=current_physics_fingerprint(),
                                           protocol=protocol)
            except Exception as error:
                if environment is not None and hasattr(environment, 'close'):
                    environment.close()
                if isinstance(error, (ValueError, TypeError)):
                    raise HTTPException(422, str(error)) from error
                raise
            self.cache_environment(record['id'], environment)
            return public_record(record)

    def restore(self, record):
        incompatibility = engine_incompatibility(record)
        if incompatibility:
            raise HTTPException(409, incompatibility)
        if record['id'] in self.cache:
            if self.cache_revisions[record['id']] == record['revision']:
                environment = self.cache[record['id']]
                self.cache.move_to_end(record['id'])
                return environment
            self.discard(record['id'])
        environment = make_environment(record['config'])
        try:
            observation, info = environment.reset(seed=record['seed'], options={'patient': record['patient']} if record['patient'] else None)
            restored = snapshot(observation, info)
            for transition in self.store.transitions(record['id']):
                observation, reward, terminated, truncated, info = environment.step(transition['action'])
                restored = snapshot(observation, info, reward=reward, episode_reward=restored['episode_reward'] + float(reward),
                                    terminated=terminated, truncated=truncated)
            if encoded(restored) != encoded(record['snapshot']):
                raise HTTPException(409, 'Saved episode could not be replayed exactly. Its original records remain available for export.')
        except Exception:
            if hasattr(environment, 'close'):
                environment.close()
            raise
        self.cache_environment(record['id'], environment, record['revision'])
        return environment

    def step(self, session_id: str, action: list[float], expected_revision=None, source=None, *, protocol_controller=None):
        with self.lock:
            record = self.require(session_id, expected_revision)
            if record['status'] != 'active':
                raise HTTPException(409, 'This episode is finished or closed. Reset it to create a new episode.')
            if record['revision'] >= MAX_EPISODE_TRANSITIONS:
                raise HTTPException(409, 'Episode transition limit reached; close or reset this session.')
            environment = self.restore(record)
            try:
                observation, reward, terminated, truncated, info = environment.step(action)
                current = snapshot(observation, info, reward=reward,
                                   episode_reward=record['snapshot']['episode_reward'] + float(reward),
                                   terminated=terminated, truncated=truncated)
                status = 'terminated' if terminated else 'truncated' if truncated else 'active'
                source = dict(source or {'type': 'manual'})
                protocol = None
                if protocol_controller is not None:
                    if terminated or truncated:
                        stopped = protocol_controller.next_action(observation, info)
                        if stopped is not None:
                            raise ValueError('Response protocol attempted another action after episode termination')
                    protocol = protocol_progress(protocol_controller, record.get('protocol'),
                                                 episode_finished=terminated or truncated)
                    source['controller_state'] = protocol['controller_state']
                elif record.get('protocol') and record['protocol']['status'] == 'active':
                    protocol = {**record['protocol'], 'status': 'interrupted', 'completed': True,
                                'completion_reason': 'Episode advanced outside the controlled response protocol.'}
                updated = self.store.append(record, action, current, status, source, protocol=protocol)
                self.cache_revisions[session_id] = updated['revision']
            except Exception:
                # The simulation may already have advanced. Evict it and replay
                # the committed log before the next attempt instead of silently
                # diverging from the durable episode.
                self.discard(session_id)
                raise
            return public_record(updated)

    def close(self):
        with self.lock:
            self.closed = True
            for session_id in list(self.cache):
                self.discard(session_id)


class TrainingCancelled(RuntimeError):
    pass


class LearningService:
    def __init__(self, directory: Path):
        self.store = Store(directory)
        self.store.recover_interrupted()
        from .reference import seed_reference
        self.reference = seed_reference(self.store)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='learning')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.cancellations: dict[str, threading.Event] = {}
        self.futures: dict[str, Any] = {}

    def work(self, run_id, config, cancelled):
        try:
            def progress(value, *args, **kwargs):
                if cancelled.is_set() or self.stop.is_set():
                    raise TrainingCancelled('Training cancelled; completed episodes and existing models are unchanged.')
                self.store.update(run_id, progress=max(0.01, min(0.99, float(value))))
            progress(0)
            self.store.update(run_id, status='running')
            result = json_safe(train_policy(config, progress))
            if cancelled.is_set() or self.stop.is_set():
                raise TrainingCancelled('Training cancelled before its result was committed.')
            if not isinstance(result.get('checkpoint'), dict):
                raise ValueError('Trainer did not return a JSON checkpoint')
            summary = {'algorithm': result.get('algorithm'), 'model_version': result.get('model_version'),
                       'episodes_completed': len(result.get('learning_curve', [])), 'clinical_use': False}
            self.store.update(run_id, status='completed', progress=1.0, summary=summary, result=result)
        except Exception as error:
            logger.exception('Local policy training %s failed', run_id)
            message = str(error) if isinstance(error, TrainingCancelled) else 'Policy training failed. Inspect the server log for details.'
            self.store.update(run_id, status='failed', error=message)
        finally:
            with self.lock:
                self.cancellations.pop(run_id, None)
                self.futures.pop(run_id, None)

    def create(self, config):
        try:
            normalized = normalize_training_config(config)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        with self.lock:
            if self.stop.is_set():
                raise HTTPException(503, 'Learning service is stopping')
            if self.store.active_count() >= 2:
                raise HTTPException(429, 'Two learning runs are already active. Wait for a run to finish or cancel one.')
            record = self.store.create(normalized)
            cancelled = threading.Event()
            self.cancellations[record['id']] = cancelled
            self.futures[record['id']] = self.executor.submit(self.work, record['id'], normalized, cancelled)
        return record

    def require(self, run_id, *, result=False):
        record = self.store.get(str(run_id), include_result=result)
        if record is None:
            raise HTTPException(404, 'Learning run not found')
        if result and record['status'] != 'completed':
            raise HTTPException(409, 'The policy is available after training completes.')
        return record

    def cancel(self, run_id):
        with self.lock:
            record = self.require(run_id)
            if record['status'] not in ('queued', 'running'):
                raise HTTPException(409, 'This learning run is already finished')
            cancel = self.cancellations.get(str(run_id))
            if cancel:
                cancel.set()
            future = self.futures.get(str(run_id))
            if future and future.cancel():
                self.store.update(str(run_id), status='failed', error='Training cancelled before it started.')
                self.futures.pop(str(run_id), None)
                self.cancellations.pop(str(run_id), None)
        return self.require(run_id)

    def close(self):
        self.stop.set()
        with self.lock:
            for event in self.cancellations.values():
                event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.store.recover_interrupted()


def initialize_procedure(app, data_dir: Path | str):
    directory = Path(data_dir)
    app.state.procedure = ProcedureService(directory / 'procedure')
    app.state.learning = LearningService(directory / 'learning')


def shutdown_procedure(app):
    app.state.learning.close()
    app.state.procedure.close()


router = APIRouter(prefix='/api/procedure', tags=['live synthetic tissue procedure'])
learning_router = APIRouter(prefix='/api/learning', tags=['local research policy learning'])


@learning_router.get('/reference')
def read_bundled_reference(request: Request):
    return request.app.state.learning.reference


@router.get('/spec')
def describe():
    environment = make_environment({})
    try:
        spec = json_safe(environment.describe())
        return {**spec, 'api_version': API_VERSION, 'engine_version': current_engine_version(),
                'mode': 'synthetic-tissue-research', 'clinical_use': False,
                'limits': {'max_active_sessions': MAX_ACTIVE_SESSIONS, 'max_transitions': MAX_EPISODE_TRANSITIONS}}
    finally:
        if hasattr(environment, 'close'):
            environment.close()


@router.post('/sessions', status_code=201)
def create_session(config: SessionConfig, request: Request):
    return request.app.state.procedure.create(config)


@router.get('/scenarios')
def list_response_protocols():
    from .scenarios import scenario_catalog
    return {'scenarios': json_safe(scenario_catalog())}


@router.post('/scenarios/{scenario_id}/session', status_code=201)
def create_response_protocol_session(scenario_id: str, config: ScenarioSessionConfig, request: Request):
    return create_protocol_session(request.app.state.procedure, scenario_id, config)


@router.get('/sessions')
def list_sessions(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {'sessions': [public_record(record, include_state=False)
                         for record in request.app.state.procedure.store.list(limit, offset)]}


@router.get('/sessions/{session_id}')
def read_session(session_id: UUID, request: Request):
    return public_record(request.app.state.procedure.require(str(session_id)))


@router.post('/sessions/{session_id}/step')
def step_session(session_id: UUID, config: StepConfig, request: Request):
    try:
        return request.app.state.procedure.step(str(session_id), config.action, config.expected_revision)
    except EpisodeConflict as error:
        raise HTTPException(409, str(error)) from error


@router.post('/sessions/{session_id}/policy-step')
def policy_step(session_id: UUID, config: PolicyStepConfig, request: Request):
    procedure = request.app.state.procedure
    learning = request.app.state.learning
    with procedure.lock:
        record = procedure.require(str(session_id), config.expected_revision)
        training = learning.require(config.training_run_id, result=True)
        try:
            environment = procedure.restore(record)
            validate_policy_compatibility(training['result']['checkpoint'], environment.describe())
            context = policy_context(training, record)
            action = trained_action(training['result']['checkpoint'], record['snapshot']['observation'])
            action = StepConfig(action=action).action
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, f'The policy cannot act on this environment: {error}') from error
        return procedure.step(str(session_id), action, config.expected_revision,
                              source={'type': 'trained-policy', 'training_run_id': str(config.training_run_id),
                                      'policy_context': context})


@router.get('/sessions/{session_id}/policy-context')
def read_policy_context(session_id: UUID, training_run_id: UUID, request: Request):
    """Preview distribution differences without advancing or altering an episode."""
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id))
        training = request.app.state.learning.require(training_run_id, result=True)
        try:
            return policy_context(training, record)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, f'The policy environment metadata is invalid: {error}') from error


@router.post('/sessions/{session_id}/scripted-step')
def scripted_step(session_id: UUID, config: RevisionConfig, request: Request):
    """One transparent feedback-baseline action; this is not a trained policy."""
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id), config.expected_revision)
        if record['status'] != 'active':
            raise HTTPException(409, 'This episode is finished or closed. Reset it to create a new episode.')
        environment = procedure.restore(record)
        action = StepConfig(action=json_safe(environment.scripted_action(record['snapshot']['observation']))).action
        return procedure.step(str(session_id), action, config.expected_revision,
                              source={'type': 'scripted-feedback-baseline'})


@router.post('/sessions/{session_id}/protocol-step')
def response_protocol_step(session_id: UUID, config: RevisionConfig, request: Request):
    from .scenarios import ScenarioController
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id), config.expected_revision)
        incompatibility = engine_incompatibility(record)
        if incompatibility:
            raise HTTPException(409, incompatibility)
        protocol = record.get('protocol')
        if not protocol:
            raise HTTPException(409, 'Select a controlled response protocol by creating a new protocol episode.')
        if protocol['status'] == 'interrupted':
            raise HTTPException(409, 'This response protocol was interrupted by another controller. Reset to create a fresh protocol episode.')
        if protocol.get('completed'):
            return public_record(record)
        if record['status'] != 'active':
            raise HTTPException(409, 'This episode is finished or closed. Reset it to create a new protocol episode.')
        try:
            controller = ScenarioController.from_state(protocol['controller_state'])
            if controller.scenario_id != protocol['scenario_id'] or controller.state_dict()['protocol_version'] != protocol['version']:
                raise ValueError('Saved response protocol identity does not match its controller state')
            action = controller.next_action(record['snapshot']['observation'], record['snapshot']['info'])
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(409, f'The saved response protocol cannot resume: {error}. Reset to create a fresh protocol episode.') from error
        try:
            if action is None:
                progress = protocol_progress(controller, protocol)
                return public_record(procedure.store.update_protocol(record, progress))
            action = StepConfig(action=json_safe(action)).action
            source = {'type': 'response-protocol', 'scenario_id': controller.scenario_id,
                      'version': protocol['version'], 'phase': controller.phase, 'protocol_step': controller.steps}
            return procedure.step(str(session_id), action, config.expected_revision, source=source,
                                  protocol_controller=controller)
        except EpisodeConflict as error:
            raise HTTPException(409, str(error)) from error


@router.post('/sessions/{session_id}/reset', status_code=201)
def reset_session(session_id: UUID, config: ResetConfig, request: Request):
    procedure = request.app.state.procedure
    with procedure.lock:
        previous = procedure.require(str(session_id), config.expected_revision)
        if previous['status'] != 'closed':
            procedure.store.close(previous)
        procedure.discard(str(session_id))
        if previous.get('protocol'):
            return create_protocol_session(procedure, previous['protocol']['scenario_id'],
                                           ScenarioSessionConfig(seed=config.seed if config.seed is not None else previous['seed'],
                                                                 config=previous['config'], patient=previous['patient']),
                                           reset_from_id=str(session_id))
        return procedure.create(SessionConfig(seed=config.seed if config.seed is not None else previous['seed'],
                                              config=previous['config'], patient=previous['patient']),
                                reset_from_id=str(session_id))


@router.delete('/sessions/{session_id}')
def close_session(session_id: UUID, request: Request, expected_revision: int | None = Query(None, ge=0)):
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id), expected_revision)
        if record['status'] != 'closed':
            record = procedure.store.close(record)
        procedure.discard(str(session_id))
        return public_record(record)


@router.get('/sessions/{session_id}/events')
def read_events(session_id: UUID, request: Request,
                after_revision: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id))
        transitions = procedure.store.transitions(str(session_id), after_revision=after_revision, limit=limit)
    next_revision = transitions[-1]['revision'] if transitions else after_revision
    events = [{'revision': transition['revision'], **event}
              for transition in transitions for event in transition.get('events', [])]
    return {'events': events, 'next_revision': next_revision,
            'current_revision': record['revision'], 'has_more': next_revision < record['revision']}


@router.get('/sessions/{session_id}/transitions')
def read_transitions(session_id: UUID, request: Request,
                     after_revision: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id))
        transitions = procedure.store.transitions(str(session_id), after_revision=after_revision, limit=limit)
    next_revision = transitions[-1]['revision'] if transitions else after_revision
    return {'transitions': transitions, 'next_revision': next_revision,
            'current_revision': record['revision'], 'has_more': next_revision < record['revision']}


@router.get('/sessions/{session_id}/export')
def export_session(session_id: UUID, request: Request):
    procedure = request.app.state.procedure
    with procedure.lock:
        record = procedure.require(str(session_id))
        payload = {'format': 'retraction-procedure-episode-v1', 'session': public_record(record),
                   'transitions': procedure.store.transitions(str(session_id)),
                   'replay': {'engine_version': record['engine_version'], 'seed': record['seed'],
                              'physics_fingerprint': record.get('physics_fingerprint'),
                              'config': record['config'], 'patient': record['patient'],
                              'method': 'Reset the same engine version and seed, then apply recorded actions in revision order.'},
                   'limitations': ['Synthetic tissue research only; no patient data or clinical safety validation.',
                                   'Replay is checked against the saved snapshot before another action is accepted.']}
    return Response(encoded(payload), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="procedure-{session_id}.json"'})


@learning_router.post('/runs', status_code=202)
def create_learning_run(config: dict[str, Any], request: Request):
    return request.app.state.learning.create(config)


@learning_router.get('/runs')
def list_learning_runs(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {'runs': request.app.state.learning.store.list(limit, offset)}


@learning_router.get('/runs/{run_id}')
def read_learning_run(run_id: UUID, request: Request):
    return request.app.state.learning.require(run_id)


@learning_router.get('/runs/{run_id}/results')
def learning_result(run_id: UUID, request: Request):
    return request.app.state.learning.require(run_id, result=True)['result']


@learning_router.post('/runs/{run_id}/session', status_code=201)
def create_matched_policy_session(run_id: UUID, config: MatchedPolicySessionConfig, request: Request):
    """Create an episode with the checkpoint's recorded settings and horizon."""
    training = request.app.state.learning.require(run_id, result=True)
    environment = None
    try:
        settings = training_environment_config(training)
        if settings is None:
            raise ValueError('Training environment metadata is unavailable; a matched episode cannot be created')
        environment = make_environment(settings)
        validate_policy_compatibility(training['result']['checkpoint'], environment.describe())
        context = policy_context(training, {'config': settings, 'patient': None, 'seed': config.seed})
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, f'A matched policy episode cannot be created: {error}') from error
    finally:
        if environment is not None and hasattr(environment, 'close'):
            environment.close()
    return request.app.state.procedure.create(SessionConfig(seed=config.seed, config=settings), context=context)


@learning_router.get('/runs/{run_id}/export')
def learning_export(run_id: UUID, request: Request):
    record = request.app.state.learning.require(run_id, result=True)
    return Response(encoded({'format': 'retraction-policy-training-v1', 'run': record}),
                    media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="policy-training-{run_id}.json"'})


@learning_router.post('/runs/{run_id}/cancel')
def cancel_learning_run(run_id: UUID, request: Request):
    return request.app.state.learning.cancel(run_id)
