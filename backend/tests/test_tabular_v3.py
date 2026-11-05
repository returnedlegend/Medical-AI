"""Versioned boundary fixes and bounded sparse weights for reactive policies."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from retraction.environment import TissueRetractionEnv
from retraction.learning import (RAW_FEATURE_TRANSFORM, _encode, evaluate_policy,
                                 load_checkpoint, policy_action, train_policy)
from retraction.tabular import (TabularPolicy, V2_BIN_VERSION, V2_STATE_SHAPE,
                               V3_BIN_VERSION, V3_STATE_SHAPE)


def encoders():
    env = TissueRetractionEnv()
    observation, _ = env.reset(seed=42)
    low, high = env.observation_space.low, env.observation_space.high
    legacy = TabularPolicy(np.zeros((int(np.prod(V2_STATE_SHAPE)), 1)), low, high,
                           env.observation_names, V2_BIN_VERSION)
    current = TabularPolicy(np.zeros((int(np.prod(V3_STATE_SHAPE)), 1)), low, high,
                            env.observation_names, V3_BIN_VERSION)
    return env, observation, legacy, current


def coordinates(policy, observation):
    transform = RAW_FEATURE_TRANSFORM if policy.encoding_version == V3_BIN_VERSION else None
    encoded = _encode(observation, policy.low, policy.high, transform)
    state = policy.state_indices(encoded)
    return np.asarray(np.unravel_index(state, policy.shape))


def test_v3_raw_public_measurements_roundtrip_without_box_precision_loss():
    env, observation, _, current = encoders()
    np.testing.assert_array_equal(_encode(observation, current.low, current.high, RAW_FEATURE_TRANSFORM), observation)
    expected = coordinates(current, observation)
    env.reset(seed=918)
    np.testing.assert_array_equal(coordinates(current, observation), expected)
    env.close()


def test_v3_horizon_boundary_is_exact_for_every_supported_training_horizon():
    env, observation, legacy, current = encoders()
    rows, expected = [], []
    for horizon in range(20, 301):
        boundary = (4 * horizon + 4) // 5
        for step in (boundary - 1, boundary, boundary + 1):
            row = observation.copy()
            row[env.observation_names.index('max_steps')] = horizon
            row[env.observation_names.index('step_count')] = step
            rows.append(row)
            expected.append(5 * (horizon - step) <= horizon)
    result = current.state_indices(np.asarray(rows))
    np.testing.assert_array_equal(np.unravel_index(result, current.shape)[-1], expected)
    # Existing V2 checkpoints retain the old float32 boundary convention.
    for horizon in (25, 50, 100):
        row = observation.copy()
        row[env.observation_names.index('max_steps')] = horizon
        row[env.observation_names.index('step_count')] = 4 * horizon // 5
        assert coordinates(legacy, row)[-1, 0] == 0
        assert coordinates(current, row)[-1, 0] == 1
    env.close()


def test_v3_ratio_bins_include_exact_acceptance_limits_on_the_lower_side():
    env, observation, _, current = encoders()
    for edge_index, edge in enumerate((.5, .85, .95, 1., 1.05)):
        for delta, expected in ((-1e-5, edge_index), (0, edge_index), (1e-5, edge_index + 1)):
            for measurement, limit, coordinate in (('contact_force_n', 'force_limit_n', 2),
                                                   ('n03_strain', 'strain_limit', 3)):
                row = observation.copy()
                row[env.observation_names.index(measurement)] = float(row[env.observation_names.index(limit)]) * (edge + delta)
                assert coordinates(current, row)[coordinate, 0] == expected
    env.close()


def test_v3_critical_injury_is_distinct_and_exact_damage_threshold_is_high_side():
    env, observation, legacy, current = encoders()
    body, artery = observation.copy(), observation.copy()
    body[env.observation_names.index('n03_mechanical_damage')] = .1
    artery[env.observation_names.index('n16_mechanical_damage')] = .1
    np.testing.assert_array_equal(coordinates(legacy, body), coordinates(legacy, artery))
    assert coordinates(current, body)[5, 0] == 0
    assert coordinates(current, artery)[5, 0] == 1
    for damage, expected in ((.07999, 0), (.08, 1), (.08001, 1)):
        row = observation.copy()
        row[env.observation_names.index('n16_mechanical_damage')] = damage
        assert coordinates(current, row)[5, 0] == expected
    env.close()


def test_v3_acquisition_matches_nearest_node_rejection_and_exact_three_mm():
    env, observation, legacy, current = encoders()
    for node in range(23):
        observation[env.observation_names.index(f'n{node:02d}_x')] = 25
        observation[env.observation_names.index(f'n{node:02d}_y')] = 35
    observation[env.observation_names.index('grasper_x')] = 0
    observation[env.observation_names.index('grasper_y')] = 0
    observation[env.observation_names.index('n00_x')] = 0
    observation[env.observation_names.index('n00_y')] = 0
    observation[env.observation_names.index('n00_mechanical_damage')] = .95
    observation[env.observation_names.index('n03_x')] = 2
    observation[env.observation_names.index('n03_y')] = 0
    assert coordinates(legacy, observation)[0, 0] == 1
    assert coordinates(current, observation)[0, 0] == 0  # Closest torn node blocks acquisition.
    observation[env.observation_names.index('n00_x')] = 25
    observation[env.observation_names.index('n00_y')] = 35
    observation[env.observation_names.index('n03_x')] = -9.25
    observation[env.observation_names.index('grasper_x')] = -6.25
    assert coordinates(legacy, observation)[0, 0] == 0  # Preserve historical rounding.
    assert coordinates(current, observation)[0, 0] == 1
    env.close()


@pytest.fixture(scope='module')
def v3_result():
    return train_policy({'algorithm': 'tabular_q', 'tabular_state_encoding': 'v3',
                         'episodes': 20, 'demonstration_episodes': 5, 'max_steps': 40,
                         'eval_episodes': 5, 'training_seed': 998})


def test_real_v3_training_uses_safe_sparse_json_and_reconstructs_exact_values(v3_result):
    checkpoint = v3_result['checkpoint']
    assert v3_result['metadata']['tabular_states'] == 34560
    assert checkpoint['feature_transform'] == RAW_FEATURE_TRANSFORM
    assert 'q_values' not in checkpoint
    assert 0 < len(checkpoint['q_sparse']['indices']) < 34560
    restored = json.loads(json.dumps(checkpoint, allow_nan=False))
    first, *_ = load_checkpoint(checkpoint)
    second, *_ = load_checkpoint(restored)
    np.testing.assert_array_equal(first.values, second.values)
    env = TissueRetractionEnv({'max_steps': 40})
    observation, _ = env.reset(seed=9)
    assert policy_action(checkpoint, observation) == policy_action(restored, observation)
    env.close()


@pytest.mark.parametrize('corrupt', [
    lambda cp: cp['q_sparse']['indices'].__setitem__(0, -1),
    lambda cp: cp['q_sparse']['indices'].__setitem__(0, True),
    lambda cp: cp['q_sparse']['indices'].__setitem__(1, cp['q_sparse']['indices'][0]),
    lambda cp: cp['q_sparse']['values'][0].__setitem__(0, float('nan')),
    lambda cp: cp['q_sparse']['values'].pop(),
    lambda cp: cp.update(q_values=[]),
    lambda cp: cp.update(feature_transform=None),
])
def test_v3_sparse_checkpoint_rejects_ambiguous_or_unbounded_data(v3_result, corrupt):
    checkpoint = copy.deepcopy(v3_result['checkpoint'])
    corrupt(checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint)


def test_v2_published_candidate_replays_exactly_after_v3_extension():
    path = Path(__file__).resolve().parents[2] / 'examples/stateful-policy-1.1-study-002.json'
    result = json.loads(path.read_text())['result']
    expected = result['evaluation']['learned']['episodes'][0]
    actual = evaluate_policy(result['checkpoint'], result['config'], [expected['seed']])['learned']['episodes'][0]
    assert actual == expected
