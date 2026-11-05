"""Meaningful algorithm/checkpoint tests independent of optional renderers."""
import copy
import json

import numpy as np
import pytest

from retraction.learning import (
    ALGORITHM, CHECKPOINT_FORMAT, Adam, Network, Replay, action_primitives,
    load_checkpoint, normalize_training_config, optimize, policy_action,
)


def checkpoint(network, low, high):
    names, actions = action_primitives()
    return {'format': CHECKPOINT_FORMAT, 'algorithm': ALGORITHM,
            'architecture': {'input': len(low), 'hidden': network.w1.shape[1],
                             'output': len(names), 'activation': 'tanh'},
            'observation_low': list(low), 'observation_high': list(high),
            'action_primitives': actions.tolist(),
            'weights': {key: value.tolist() for key, value in network.arrays().items()}}


def test_td_learning_changes_behavior_from_actual_reward_sign():
    """A two-state terminal MDP has opposite optimal controls at each state."""
    rng = np.random.default_rng(16)
    network = Network.initialize(2, 2, rng, hidden=16)
    target = network.copy()
    optimizer = Adam(network, .005)
    states = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    actions = np.array([0, 1, 0, 1])
    rewards = np.array([1, -1, -1, 1], dtype=np.float32)
    batch = (states, actions, rewards, states, np.ones(4, dtype=np.float32))
    initial = network.w1.copy()
    losses = [optimize(network, target, optimizer, batch, .98) for _ in range(400)]
    values, _ = network.forward(states[[0, 2]])
    assert np.argmax(values, axis=1).tolist() == [0, 1]
    assert losses[-1] < losses[0] / 100
    assert not np.array_equal(initial, network.w1)
    # Swapping rewards must reverse the learned action, not preserve a hidden
    # heuristic independent of the transition reward.
    inverse = Network.initialize(2, 2, np.random.default_rng(16), hidden=16)
    inverse_optimizer = Adam(inverse, .005)
    for _ in range(400):
        optimize(inverse, inverse.copy(), inverse_optimizer,
                 (states, actions, -rewards, states, np.ones(4, dtype=np.float32)), .98)
    inverse_values, _ = inverse.forward(states[[0, 2]])
    assert np.argmax(inverse_values, axis=1).tolist() == [1, 0]


def test_terminal_transition_does_not_bootstrap_future_reward():
    rng = np.random.default_rng(3)
    first = Network.initialize(2, 2, rng, hidden=8)
    second = first.copy()
    target1, target2 = first.copy(), first.copy()
    target2.b2[:] = 10_000
    batch = (np.ones((4, 2), dtype=np.float32), np.zeros(4, dtype=int),
             np.ones(4, dtype=np.float32), np.ones((4, 2), dtype=np.float32),
             np.ones(4, dtype=np.float32))
    optimize(first, target1, Adam(first, .001), batch, .99)
    optimize(second, target2, Adam(second, .001), batch, .99)
    for key in first.arrays():
        np.testing.assert_array_equal(first.arrays()[key], second.arrays()[key])


def test_checkpoint_roundtrip_preserves_action_without_pickle():
    names, actions = action_primitives()
    network = Network.initialize(3, len(names), np.random.default_rng(4))
    source = checkpoint(network, [-1]*3, [1]*3)
    serialized = json.dumps(source, allow_nan=False)
    restored = json.loads(serialized)
    assert policy_action(restored, [.1, -.2, .4]) == policy_action(source, [.1, -.2, .4])
    assert len(policy_action(restored, [.1, -.2, .4])) == 7
    loaded, *_ = load_checkpoint(restored)
    np.testing.assert_array_equal(loaded.w1, network.w1)


@pytest.mark.parametrize('change', [
    lambda c: c.update(format='pickle'),
    lambda c: c['architecture'].update(input=10_000_000),
    lambda c: c['weights']['w1'][0].__setitem__(0, float('nan')),
    lambda c: c['action_primitives'][0].__setitem__(0, 2),
    lambda c: c['observation_high'].__setitem__(0, -2),
    lambda c: c.update(observation_low=[-3e38] * 2, observation_high=[3e38] * 2),
    lambda c: c.update(observation_low=[0] * 2, observation_high=[1e-9] * 2),
])
def test_checkpoint_rejects_unsafe_or_incompatible_arrays(change):
    names, _ = action_primitives()
    source = checkpoint(Network.initialize(2, len(names), np.random.default_rng(1)), [-1, -1], [1, 1])
    change(source)
    with pytest.raises(ValueError):
        load_checkpoint(source)


def test_wrong_or_nonfinite_observation_is_rejected():
    names, _ = action_primitives()
    source = checkpoint(Network.initialize(2, len(names), np.random.default_rng(1)), [-1, -1], [1, 1])
    for observation in ([1], [1, 2, 3], [float('inf'), 0]):
        with pytest.raises(ValueError):
            policy_action(source, observation)


def test_replay_wraps_but_keeps_transition_components_paired():
    replay = Replay(3, 2)
    for number in range(7):
        replay.append([number, number], number, number / 2, [number + 1, number + 1], number % 2)
    assert replay.count == 3
    observations, actions, rewards, following, terminals = replay.sample(np.random.default_rng(7), 100)
    np.testing.assert_array_equal(observations[:, 0], actions)
    np.testing.assert_array_equal(rewards * 2, actions)
    np.testing.assert_array_equal(following[:, 0], actions + 1)
    np.testing.assert_array_equal(terminals, actions % 2)
    assert set(actions) <= {4, 5, 6}


@pytest.mark.parametrize('bad', [
    {'episodes': True}, {'episodes': 3001}, {'eval_episodes': 1},
    {'learning_rate': float('nan')}, {'gamma': 1.1}, {'unknown': 1},
    {'environment': []}, {'environment': {'max_steps': 10}},
    {'tabular_state_encoding': 'v4'},
])
def test_training_configuration_has_bounded_cost_and_no_unknown_fields(bad):
    with pytest.raises(ValueError):
        normalize_training_config(bad)


def test_real_training_uses_disjoint_patients_and_produces_runnable_policy():
    from retraction.environment import TissueRetractionEnv
    from retraction.learning import train_policy, validate_checkpoint_compatibility
    result = train_policy({'episodes': 20, 'eval_episodes': 5, 'max_steps': 40, 'training_seed': 19})
    assert result['metadata']['gradient_updates'] > 0
    assert not set(result['metadata']['training_seeds']) & set(result['metadata']['test_seeds'])
    assert len(result['learning_curve']) == 20
    assert len(result['evaluation']['learned']['episodes']) == 5
    assert [r['seed'] for r in result['evaluation']['learned']['episodes']] == result['metadata']['test_seeds']
    assert [r['seed'] for r in result['evaluation']['random']['episodes']] == result['metadata']['test_seeds']
    assert [r['seed'] for r in result['evaluation']['scripted']['episodes']] == result['metadata']['test_seeds']
    assert json.dumps(result, allow_nan=False)
    env = TissueRetractionEnv({'max_steps': 40})
    observation, _ = env.reset(seed=123)
    validate_checkpoint_compatibility(result['checkpoint'], env.describe())
    action = policy_action(result['checkpoint'], observation)
    assert env.action_space.contains(np.asarray(action, dtype=np.float32))
    observation, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(observation).all() and np.isfinite(reward)
    broken = copy.deepcopy(result['checkpoint'])
    broken['model_version'] = 'different-tissue-laws'
    with pytest.raises(ValueError, match='version'):
        validate_checkpoint_compatibility(broken, env.describe())
    broken = copy.deepcopy(result['checkpoint'])
    broken['physics_fingerprint'] = '0' * 64
    with pytest.raises(ValueError, match='fingerprint'):
        validate_checkpoint_compatibility(broken, env.describe())
    env.close()


def test_episode_metrics_do_not_confuse_no_injury_with_success():
    from retraction.learning import _episode_summary
    healthy_timeout = _episode_summary(-2, 120, {'termination_reason': 'time_limit', 'success': False,
                                                'state': {'max_damage': .05}}, False, True)
    harmed = _episode_summary(-40, 5, {'termination_reason': 'severe_injury', 'success': False,
                                     'state': {'max_damage': .7}}, True, False)
    assert not healthy_timeout['success'] and not healthy_timeout['severe_harm']
    assert harmed['severe_harm'] and harmed['max_damage'] == .7
    assert not harmed['success']


def test_demonstration_margin_and_balancing_do_not_erase_transition_reward():
    from retraction.learning import demonstration_margin_update
    replay = Replay(100, 2)
    for index in range(90):
        replay.append([1, 0], 0, 1, [1, 0], True)
    replay.append([0, 1], 1, 1, [0, 1], True)
    rng = np.random.default_rng(8)
    batch = replay.sample_balanced(rng, 1000)
    assert 400 < np.sum(batch[1] == 1) < 600
    network = Network.initialize(2, 2, np.random.default_rng(3), hidden=16)
    optimizer = Adam(network, .003)
    for _ in range(200):
        example = replay.sample_balanced(rng, 32)
        optimize(network, network.copy(), optimizer, example, .98)
        demonstration_margin_update(network, optimizer, example)
    values, _ = network.forward(np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert np.argmax(values, axis=1).tolist() == [0, 1]
    # New data invalidates the sampler's action-group cache.
    replay.append([.5, .5], 2, 0, [.5, .5], True)
    assert 2 in replay.sample_balanced(rng, 100)[1]


def test_evaluation_progress_can_cancel_without_mutating_checkpoint():
    from retraction.learning import train_policy, evaluate_policy
    result = train_policy({'episodes': 20, 'demonstration_episodes': 0, 'eval_episodes': 5,
                           'max_steps': 20, 'training_seed': 7})
    original = json.dumps(result['checkpoint'], sort_keys=True)
    class Cancelled(Exception):
        pass
    def stop(fraction, message):
        raise Cancelled()
    with pytest.raises(Cancelled):
        evaluate_policy(result['checkpoint'], result['config'], [1, 2, 3], progress_callback=stop)
    assert json.dumps(result['checkpoint'], sort_keys=True) == original


def test_public_feature_scales_preserve_early_harm_signals_and_use_no_future_oracle():
    from retraction.environment import TissueRetractionEnv
    from retraction.learning import feature_specification, _encode
    env = TissueRetractionEnv()
    observation, _ = env.reset(seed=11)
    low, high = env.observation_space.low, env.observation_space.high
    spec = feature_specification(env.observation_names, low, high)
    initial = _encode(observation, low, high, spec)
    changed = observation.copy()
    strain_index = env.observation_names.index('n03_strain')
    force_index = env.observation_names.index('contact_force_n')
    changed[strain_index] = .6
    changed[force_index] = 2.5
    injured = _encode(changed, low, high, spec)
    assert injured[strain_index] - initial[strain_index] == pytest.approx(1)
    assert injured[force_index] - initial[force_index] == pytest.approx(1)
    # Derived risk ratios depend on supplied current measurements/limits only.
    assert injured[spec['feature_names'].index('max_strain_limit_ratio')] == pytest.approx(1)
    assert injured[spec['feature_names'].index('force_limit_ratio')] == pytest.approx(1)
    env.reset(seed=999)
    np.testing.assert_array_equal(injured, _encode(changed, low, high, spec))
    assert all('optimal' not in name and 'oracle' not in name and 'future' not in name
               for name in spec['feature_names'])
    env.close()


def test_checkpoint_rejects_modified_feature_transform():
    from retraction.environment import TissueRetractionEnv
    from retraction.learning import feature_specification, _checkpoint
    env = TissueRetractionEnv()
    low, high = env.observation_space.low, env.observation_space.high
    spec = feature_specification(env.observation_names, low, high)
    network = Network.initialize(len(spec['feature_names']), len(action_primitives()[0]), np.random.default_rng(3))
    source = _checkpoint(network, env, normalize_training_config(), low, high, env.describe(), spec)
    loaded = json.loads(json.dumps(source))
    observation, _ = env.reset(seed=5)
    assert policy_action(source, observation) == policy_action(loaded, observation)
    loaded['feature_transform']['scale'][0] = 0
    with pytest.raises(ValueError, match='feature transform'):
        load_checkpoint(loaded)
    env.close()


def test_zero_observed_harm_is_not_reported_as_zero_uncertainty():
    from retraction.learning import _aggregate
    rows = [{'return': 1, 'success': False, 'severe_harm': False, 'handoff': False,
             'max_damage': 0, 'steps': 10}] * 30
    metrics = _aggregate(rows, 1)
    assert metrics['severe_harm_rate'] == 0
    assert metrics['severe_harm_rate_ci'][0] == pytest.approx(0)
    assert .10 < metrics['severe_harm_rate_ci'][1] < .12


@pytest.mark.parametrize('seeds', [[], [1, 1], [True], [-1], [2**32], [1.5]])
def test_evaluation_rejects_invalid_or_repeated_seeds(seeds):
    from retraction.learning import evaluate_policy
    with pytest.raises(ValueError, match='seeds'):
        evaluate_policy({}, {}, seeds)


def test_seeded_training_is_deterministic_without_demonstrations():
    from retraction.learning import train_policy
    config = {'training_seed': 29, 'episodes': 20, 'eval_episodes': 5,
              'demonstration_episodes': 0, 'max_steps': 20}
    first, second = train_policy(config), train_policy(config)
    assert first['checkpoint'] == second['checkpoint']
    assert first['learning_curve'] == second['learning_curve']
    assert first['evaluation'] == second['evaluation']
