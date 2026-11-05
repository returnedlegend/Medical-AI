"""Reward dependence, bounded serialization, and partial-state policy tests."""
import copy
import json

import numpy as np
import pytest

from retraction.learning import (action_primitives, policy_action, train_policy,
                                 validate_checkpoint_compatibility)
from retraction.tabular import double_q_update, load_tabular_checkpoint


def test_double_q_learns_terminal_rewards_and_reverses_when_reward_reverses():
    def trained(sign):
        first = np.zeros((2, 2), dtype=np.float32)
        second = first.copy()
        rng = np.random.default_rng(3)
        for _ in range(100):
            for state in range(2):
                for action in range(2):
                    reward = sign * (1 if state == action else -1)
                    double_q_update(first, second, state, action, reward, 1, True, .2, .99, rng)
        return np.argmax(first + second, axis=1).tolist()
    assert trained(1) == [0, 1]
    assert trained(-1) == [1, 0]


def test_tabular_terminal_update_ignores_other_table_future_values():
    first = np.zeros((2, 2), dtype=np.float32)
    second = np.full((2, 2), 10000, dtype=np.float32)
    # Select the first table deterministically with a tiny test RNG.
    class ChooseFirst:
        def random(self):
            return 0.
    double_q_update(first, second, 0, 1, -2, 1, True, 1, .99, ChooseFirst())
    assert first[0, 1] == -2


@pytest.fixture(scope='module')
def tabular_result():
    return train_policy({'algorithm': 'tabular_q', 'training_seed': 79, 'episodes': 20,
                         'max_steps': 40, 'eval_episodes': 5, 'demonstration_episodes': 5})


def test_actual_tabular_training_exports_a_valid_runnable_deterministic_table(tabular_result):
    from retraction.environment import TissueRetractionEnv
    result = tabular_result
    assert result['metadata']['td_updates'] > 0
    assert result['metadata']['gradient_updates'] == 0
    assert result['metadata']['visited_training_states'] > 0
    assert not set(result['metadata']['training_seeds']) & set(result['metadata']['test_seeds'])
    checkpoint = json.loads(json.dumps(result['checkpoint'], allow_nan=False))
    policy, *_ = load_tabular_checkpoint(checkpoint)
    assert np.any(policy.values != 0)
    env = TissueRetractionEnv({'max_steps': 40})
    observation, _ = env.reset(seed=13)
    validate_checkpoint_compatibility(checkpoint, env.describe())
    assert policy_action(checkpoint, observation) == policy_action(result['checkpoint'], observation)
    assert env.action_space.contains(np.asarray(policy_action(checkpoint, observation), dtype=np.float32))
    env.close()


@pytest.mark.parametrize('change', [
    lambda cp: cp['state_encoding']['edges']['exposure_error'].append(50),
    lambda cp: cp['q_values'][0].__setitem__(0, float('nan')),
    lambda cp: cp['q_values'].append([0] * len(action_primitives()[0])),
    lambda cp: cp['action_primitives'][0].__setitem__(0, 1.1),
    lambda cp: cp.update(feature_transform={'version': 'arbitrary-code'}),
])
def test_tabular_checkpoint_tampering_is_rejected(tabular_result, change):
    broken = copy.deepcopy(tabular_result['checkpoint'])
    change(broken)
    with pytest.raises(ValueError):
        load_tabular_checkpoint(broken)


def test_partial_state_aliasing_is_explicit_and_uses_only_supplied_observation(tabular_result):
    from retraction.environment import TissueRetractionEnv
    checkpoint = tabular_result['checkpoint']
    env = TissueRetractionEnv({'max_steps': 40})
    observation, _ = env.reset(seed=21)
    changed = observation.copy()
    changed[env.observation_names.index('attachment_length_mm')] += 1
    assert policy_action(checkpoint, observation) == policy_action(checkpoint, changed)
    env.reset(seed=321)
    assert policy_action(checkpoint, observation) == policy_action(checkpoint, changed)
    assert 'not a Markov' in tabular_result['metadata']['observation_encoding']
    env.close()


def test_actual_tissue_harm_penalties_change_a_learned_return_preference():
    """Stress fixture: two fixed macro trajectories, not a surgical-policy claim.

    The real tissue simulator supplies every reward and injury measurement. A
    terminal-return Q update should reject the injurious trajectory because of
    its observed harm costs; removing those costs reverses that preference.
    """
    from retraction.environment import TissueRetractionEnv
    totals, ablated_totals, damages = [], [], []
    for pulling in (True, False):
        env = TissueRetractionEnv()
        env.reset(seed=42)
        total, ablated = 0., 0.
        for _ in range(200):
            action = [1, 0, .55, 0, 0, 0, 0] if pulling else [0] * 7
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            components = info['reward_components']
            removed = sum(components.get(key, 0) for key in
                          ('tissue_damage', 'critical_damage', 'bleeding', 'bile_leak', 'force_excess'))
            if info['severe_harm']:
                removed += components['terminal']
            ablated += reward - removed
            if terminated or truncated:
                break
        totals.append(total)
        ablated_totals.append(ablated)
        damages.append(info['state']['max_damage'])
        env.close()
    assert damages[0] > .6 and damages[1] == 0
    assert totals[0] < totals[1] and ablated_totals[0] > ablated_totals[1]
    def preference(returns):
        first = np.zeros((1, 2), dtype=np.float32)
        second = first.copy()
        rng = np.random.default_rng(5)
        for _ in range(100):
            for action, reward in enumerate(returns):
                double_q_update(first, second, 0, action, reward, 0, True, .2, .99, rng)
        return int(np.argmax(first + second))
    assert preference(totals) == 1
    assert preference(ablated_totals) == 0
