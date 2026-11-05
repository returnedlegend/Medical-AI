"""Load one sealed, inspectable research policy from a fixed bundled resource.

The reference is optional during development. Startup never downloads, trains,
executes a checkpoint, or substitutes invented results when it is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid5

from .storage import Store, timestamp

logger = logging.getLogger(__name__)
REFERENCE_RESOURCE = 'stateful-policy-1.1.json'
REFERENCE_FORMAT = 'retraction-bundled-reference-v1'
# Set only after the frozen validation protocol has selected and serialized the
# real reference result. The seal is code-owned, never read from the same file.
REFERENCE_SHA256: str | None = None
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


def reference_path() -> Path:
    bundle_root = getattr(sys, '_MEIPASS', None)
    root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[2]
    return root / 'examples' / REFERENCE_RESOURCE


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Reference JSON contains duplicate object keys')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Reference JSON contains a nonfinite value: {value}')


def validate_reference(payload: dict) -> tuple[dict, dict]:
    from . import environment as environment_module
    from .environment import TissueRetractionEnv, normalize_env_config
    from .learning import normalize_training_config, validate_checkpoint_compatibility

    if not isinstance(payload, dict) or payload.get('format') != REFERENCE_FORMAT:
        raise ValueError('Unsupported bundled reference format')
    if set(payload) != {'format', 'result', 'provenance'}:
        raise ValueError('Bundled reference fields are missing or unknown')
    result, provenance = payload['result'], payload['provenance']
    if not isinstance(result, dict) or not isinstance(provenance, dict):
        raise ValueError('Bundled reference result and provenance must be objects')
    checkpoint = result.get('checkpoint')
    if not isinstance(checkpoint, dict):
        raise ValueError('Bundled reference has no declarative checkpoint')
    for key, value in (('checkpoint_sha256', checkpoint), ('result_sha256', result)):
        if provenance.get(key) != canonical_sha256(value):
            raise ValueError(f'Bundled reference {key} does not match its recorded data')
    if provenance.get('validation_only') is not True:
        raise ValueError('Bundled reference must disclose validation-only selection')
    if not isinstance(provenance.get('selection_rule'), str) or not provenance['selection_rule'].strip():
        raise ValueError('Bundled reference selection rule is missing')
    if not re.fullmatch(r'[0-9a-f]{64}', str(provenance.get('study_protocol_sha256', ''))):
        raise ValueError('Bundled reference frozen protocol hash is missing')
    sources = provenance.get('source_sha256')
    source_names = {'learning.py', 'tabular.py', 'environment.py', 'tissue.py'}
    if (not isinstance(sources, dict) or set(sources) != source_names
            or any(not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in sources.values())):
        raise ValueError('Bundled reference training source provenance is incomplete')
    source_directory = Path(environment_module.__file__).resolve().parent
    for name in ('environment.py', 'tissue.py'):
        if hashlib.sha256((source_directory / name).read_bytes()).hexdigest() != sources[name]:
            raise ValueError('Bundled reference physics source provenance differs from the installed model')
    if not isinstance(result.get('config'), dict):
        raise ValueError('Bundled reference training configuration is missing')
    config = normalize_training_config(result['config'])
    environment_config = normalize_env_config({**config['environment'], 'max_steps': config['max_steps']})
    if checkpoint.get('environment_config') != environment_config:
        raise ValueError('Bundled checkpoint environment differs from its training result')
    if checkpoint.get('max_steps') != config['max_steps']:
        raise ValueError('Bundled checkpoint horizon differs from its training result')
    env = TissueRetractionEnv(environment_config)
    try:
        description = env.describe()
        validate_checkpoint_compatibility(checkpoint, description)
        if result.get('model_version') != description['model_version']:
            raise ValueError('Bundled result model version differs from the running environment')
    finally:
        env.close()
    curve, metadata = result.get('learning_curve'), result.get('metadata')
    if not isinstance(curve, list) or len(curve) != config['episodes']:
        raise ValueError('Bundled reference does not contain the full recorded learning curve')
    if not isinstance(metadata, dict):
        raise ValueError('Bundled reference training metadata is missing')
    training, testing = metadata.get('training_seeds'), metadata.get('test_seeds')
    if (not isinstance(training, list) or not training or not isinstance(testing, list)
            or len(testing) != config['eval_episodes']):
        raise ValueError('Bundled reference seed provenance is incomplete')
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in training + testing):
        raise ValueError('Bundled reference seed provenance is invalid')
    if set(training).intersection(testing):
        raise ValueError('Bundled training and evaluation seeds overlap')
    evaluation = result.get('evaluation')
    if not isinstance(evaluation, dict) or any(mode not in evaluation for mode in ('learned', 'random', 'random_no_handoff', 'scripted')):
        raise ValueError('Bundled reference controller comparisons are incomplete')
    for mode in ('learned', 'random', 'random_no_handoff', 'scripted'):
        metrics = evaluation[mode]
        episodes = metrics.get('episodes') if isinstance(metrics, dict) else None
        if (not isinstance(episodes, list) or len(episodes) != len(testing)
                or any(not isinstance(episode, dict) for episode in episodes)
                or [episode.get('seed') for episode in episodes] != testing or metrics.get('n') != len(testing)):
            raise ValueError('Bundled reference comparisons do not use the recorded evaluation seeds')
        calculated = {'mean_return': math.fsum(episode['return'] for episode in episodes) / len(episodes)}
        for field in ('success', 'severe_harm', 'handoff'):
            if any(not isinstance(episode.get(field), bool) for episode in episodes):
                raise ValueError('Bundled reference episode outcomes must be boolean')
            calculated[f'{field}_rate'] = sum(bool(episode[field]) for episode in episodes) / len(episodes)
        if any(not math.isclose(metrics.get(key, math.nan), value, rel_tol=1e-7, abs_tol=1e-9)
               for key, value in calculated.items()):
            raise ValueError('Bundled reference aggregate metrics disagree with its recorded episodes')
    return result, provenance


def seed_reference(store: Store) -> dict:
    """Insert a content-addressed completed run once; never overwrite a study."""
    unavailable = {'available': False, 'label': 'Bundled reference', 'clinical_use': False,
                   'resource': REFERENCE_RESOURCE}
    path = reference_path()
    if not path.is_file():
        return {**unavailable, 'reason': 'The optional bundled reference policy is not installed.'}
    if REFERENCE_SHA256 is None:
        return {**unavailable, 'reason': 'The bundled reference policy has not been sealed for this application build.'}
    try:
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError('Bundled reference exceeds the supported size')
        with path.open('rb') as handle:
            contents = handle.read(MAX_REFERENCE_BYTES + 1)
        if len(contents) > MAX_REFERENCE_BYTES:
            raise ValueError('Bundled reference exceeds the supported size')
        digest = hashlib.sha256(contents).hexdigest()
        if digest != REFERENCE_SHA256:
            raise ValueError('Bundled reference file SHA-256 does not match this application build')
        payload = json.loads(contents, object_pairs_hook=_json_object, parse_constant=_reject_constant)
        result, provenance = validate_reference(payload)
        run_id = str(uuid5(NAMESPACE_URL, f'medical-ai:bundled-reference:{digest}'))
        origin = {'kind': 'bundled-reference', 'resource': REFERENCE_RESOURCE,
                  'file_sha256': digest, 'provenance': provenance}
        summary = {'label': 'Bundled reference', 'origin': origin, 'algorithm': result['algorithm'],
                   'model_version': result['model_version'], 'episodes_completed': len(result['learning_curve']),
                   'clinical_use': False}
        now = timestamp()
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',
                           (run_id, now, now, 'completed', 1.0, canonical_json(result['config']), None,
                            canonical_json(summary), canonical_json(result)))
            else:
                existing = Store.decode(row, include_result=True)
                existing_summary = existing.get('summary')
                existing_origin = existing_summary.get('origin') if isinstance(existing_summary, dict) else None
                if (existing['status'] != 'completed' or canonical_sha256(existing.get('result')) != provenance['result_sha256']
                        or not isinstance(existing_origin, dict) or existing_origin.get('file_sha256') != digest):
                    raise ValueError('A different saved record already uses the bundled reference identifier; no record was overwritten')
        return {'available': True, 'label': 'Bundled reference', 'clinical_use': False,
                'resource': REFERENCE_RESOURCE, 'run_id': run_id, 'origin': origin}
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        logger.warning('Bundled reference is unavailable: %s', error)
        return {**unavailable, 'reason': str(error)}
