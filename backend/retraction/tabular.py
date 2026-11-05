"""Inspectable reactive Double-Q baseline for the synthetic tissue environment.

The bins use current public measurements. This intentionally reduced state is
not Markov: different anatomy/deformation histories can share a bin. It is a
small comparison to the full-observation neural baseline, not a clinical policy.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

TABULAR_ALGORITHM = "numpy-tabular-double-q-1.0"
BIN_VERSION = "public-reactive-bins-1"
BIN_EDGES = {
    "exposure_error": [-.02, 0., .01, .03, .08, .2, .4],
    "force_limit_ratio": [.5, .85, .95, 1.05],
    "max_strain_limit_ratio": [.5, .85, .95, 1.05],
    "max_mechanical_damage": [.05, .15],
}
STATE_SHAPE = (2, *(len(edges) + 1 for edges in BIN_EDGES.values()))
STATE_COUNT = int(np.prod(STATE_SHAPE))


class TabularPolicy:
    def __init__(self, values, low, high, observation_names):
        self.values = values
        self.low, self.high = low, high
        self.index = {name: i for i, name in enumerate(observation_names)}
        self.strains = [i for i, name in enumerate(observation_names) if name.endswith('_strain')]
        self.damage = [i for i, name in enumerate(observation_names) if name.endswith('_mechanical_damage')]
        required = ('contact_node_index', 'exposure_target', 'exposure', 'contact_force_n',
                    'force_limit_n', 'strain_limit')
        if any(name not in self.index for name in required) or not self.strains or not self.damage:
            raise ValueError('Tabular policy requires the declared public tissue measurements')

    def state_indices(self, normalized):
        raw = (np.atleast_2d(normalized) + 1) * .5 * (self.high - self.low) + self.low
        measurements = [raw[:, self.index['exposure_target']] - raw[:, self.index['exposure']],
                        raw[:, self.index['contact_force_n']] / np.maximum(raw[:, self.index['force_limit_n']], 1e-6),
                        raw[:, self.strains].max(axis=1) / np.maximum(raw[:, self.index['strain_limit']], 1e-6),
                        raw[:, self.damage].max(axis=1)]
        coordinates = [(raw[:, self.index['contact_node_index']] >= 0).astype(int)]
        # Round only at the bin boundary comparison to avoid float32 affine
        # reconstruction moving an exactly zero exposure error across a bin.
        coordinates.extend(np.searchsorted(edges, np.round(values, 6), side='right')
                           for edges, values in zip(BIN_EDGES.values(), measurements))
        return np.ravel_multi_index(tuple(coordinates), STATE_SHAPE)

    def forward(self, normalized):
        selected = self.values[self.state_indices(normalized)]
        return (selected[0] if np.asarray(normalized).ndim == 1 else selected), None


def double_q_update(first, second, state, action, reward, following, terminal, rate, gamma, rng):
    """One real transition updates one table; the other evaluates its argmax."""
    updated, evaluator = (first, second) if rng.random() < .5 else (second, first)
    target = float(reward)
    if not terminal:
        target += gamma * evaluator[following, int(np.argmax(updated[following]))]
    error = target - updated[state, action]
    updated[state, action] += rate * error
    return float(abs(error))


def load_tabular_checkpoint(checkpoint):
    from .learning import CHECKPOINT_FORMAT
    if checkpoint.get('format') != CHECKPOINT_FORMAT or checkpoint.get('algorithm') != TABULAR_ALGORITHM:
        raise ValueError('Unsupported tabular checkpoint format')
    if checkpoint.get('state_encoding') != {'version': BIN_VERSION, 'edges': BIN_EDGES, 'shape': list(STATE_SHAPE)}:
        raise ValueError('Unsupported tabular state encoding')
    if checkpoint.get('feature_transform') is not None:
        raise ValueError('Tabular policy must use its declared Box reconstruction')
    try:
        low = np.asarray(checkpoint['observation_low'], dtype=np.float32)
        high = np.asarray(checkpoint['observation_high'], dtype=np.float32)
        actions = np.asarray(checkpoint['action_primitives'], dtype=np.float32)
        values = np.asarray(checkpoint['q_values'], dtype=np.float32)
        names = checkpoint['observation_names']
        if low.ndim != 1 or not 1 <= len(low) <= 2048 or high.shape != low.shape:
            raise ValueError('Invalid tabular observation bounds')
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
            raise ValueError('Invalid tabular observation bounds')
        if not isinstance(names, list) or len(names) != len(low) or any(not isinstance(name, str) for name in names):
            raise ValueError('Invalid tabular observation names')
        if actions.ndim != 2 or not 1 <= len(actions) <= 128 or actions.shape[1] != 7:
            raise ValueError('Invalid tabular action dimensions')
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError('Invalid tabular actions')
        if values.shape != (STATE_COUNT, len(actions)) or not np.isfinite(values).all() or np.any(np.abs(values) > 1e6):
            raise ValueError('Invalid tabular Q values')
        return TabularPolicy(values, low, high, names), low, high, actions
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError('Invalid tabular checkpoint arrays') from error


def train_tabular_policy(config, progress_callback: Callable | None = None):
    from .learning import (_aggregate, _bounds, _describe, _episode_summary, _make_env,
                           _normalize, action_primitives, CHECKPOINT_FORMAT, evaluate_policy)
    cfg = dict(config)
    env = _make_env(cfg)
    description = _describe(env)
    low, high = _bounds(env)
    names, primitives = action_primitives()
    rng = np.random.default_rng(np.random.SeedSequence([cfg['training_seed'], 71]))
    patient_rng = np.random.default_rng(np.random.SeedSequence([cfg['training_seed'], 72]))
    training_seeds = [int(v) for v in patient_rng.integers(0, 2**32, cfg['episodes'], dtype=np.uint64)]
    used = set(training_seeds)
    def distinct_seeds(stream, count):
        source = np.random.default_rng(np.random.SeedSequence([cfg['training_seed'], stream]))
        collected = []
        while len(collected) < count:
            seed = int(source.integers(0, 2**32, dtype=np.uint64))
            if seed not in used:
                collected.append(seed)
                used.add(seed)
        return collected
    demonstration_seeds = distinct_seeds(74, cfg['demonstration_episodes'])
    test_seeds = distinct_seeds(73, cfg['eval_episodes'])
    first = np.zeros((STATE_COUNT, len(primitives)), dtype=np.float32)
    second = first.copy()
    policy = TabularPolicy(first, low, high, env.observation_names)
    sums, counts = np.zeros_like(first), np.zeros_like(first, dtype=np.int32)
    demo_transitions, demo_episodes, curve = [], [], []
    visits = np.zeros(STATE_COUNT, dtype=np.int32)
    updates, environment_steps, reward_scale = 0, 0, 10.
    def state(observation):
        return int(policy.state_indices(_normalize(observation, low, high))[0])
    try:
        for number, seed in enumerate(demonstration_seeds):
            observation, info = env.reset(seed=seed)
            trajectory, total = [], 0.
            for step in range(cfg['max_steps']):
                proposed = np.asarray(env.scripted_action(observation))
                action = int(np.argmin(np.sum((primitives - proposed)**2, axis=1)))
                following, reward, terminated, truncated, info = env.step(primitives[action])
                transition = (state(observation), action, float(reward) / reward_scale,
                              state(following), bool(terminated or truncated))
                trajectory.append(transition)
                total += float(reward)
                environment_steps += 1
                observation = following
                if terminated or truncated:
                    break
            demo_transitions.extend(trajectory)
            discounted_return = 0.
            for index, action, reward, _, _ in reversed(trajectory):
                discounted_return = reward + cfg['gamma'] * discounted_return
                sums[index, action] += discounted_return
                counts[index, action] += 1
            demo_episodes.append({'seed': seed, **_episode_summary(total, step + 1, info, terminated, truncated)})
            if progress_callback:
                progress_callback(.1 * (number + 1) / len(demonstration_seeds),
                                  f'Collecting training-only tabular demonstrations: {number + 1}/{len(demonstration_seeds)}')
        observed = counts > 0
        first[observed] = sums[observed] / counts[observed]
        second[:] = first
        # The initialization is a sample mean of observed discounted returns,
        # not a synthetic expert bonus or a hidden inference-time controller.
        initial_observed_actions = int(observed.sum())
        for episode, seed in enumerate(training_seeds):
            observation, info = env.reset(seed=seed)
            epsilon = max(.05, 1 - .95 * episode / max(1, cfg['episodes'] * .7))
            total, errors = 0., []
            for step in range(cfg['max_steps']):
                current = state(observation)
                visits[current] += 1
                if rng.random() < epsilon:
                    action = int(rng.integers(len(primitives)))
                else:
                    values = first[current] + second[current]
                    best = np.flatnonzero(values == values.max())
                    action = int(rng.choice(best))
                following, reward, terminated, truncated, info = env.step(primitives[action])
                next_state = state(following)
                errors.append(double_q_update(first, second, current, action, float(reward) / reward_scale,
                                               next_state, terminated or truncated,
                                               cfg['tabular_learning_rate'], cfg['gamma'], rng))
                updates += 1
                if demo_transitions:
                    example = demo_transitions[int(rng.integers(len(demo_transitions)))]
                    errors.append(double_q_update(first, second, *example,
                                                   cfg['tabular_learning_rate'], cfg['gamma'], rng))
                    updates += 1
                total += float(reward)
                environment_steps += 1
                observation = following
                if terminated or truncated:
                    break
            curve.append({'episode': episode + 1, 'seed': seed,
                          **_episode_summary(total, step + 1, info, terminated, truncated),
                          'epsilon': float(epsilon), 'mean_td_loss': float(np.mean(errors)),
                          'td_metric': 'mean absolute TD error'})
            if progress_callback and (episode + 1) % max(1, cfg['episodes'] // 100) == 0:
                progress_callback(.1 + .75 * (episode + 1) / cfg['episodes'],
                                  f'Tabular training episode {episode + 1}/{cfg["episodes"]}')
        checkpoint = {'format': CHECKPOINT_FORMAT, 'algorithm': TABULAR_ALGORITHM,
                      'model_version': description.get('model_version', description.get('version')),
                      'physics_fingerprint': description.get('physics_fingerprint'),
                      'environment': cfg['environment'], 'environment_config': description['config'],
                      'max_steps': cfg['max_steps'], 'training_config': cfg,
                      'training_method': 'demonstration-initialized Double-Q learning' if demo_transitions else 'Double-Q learning from uniform exploration',
                      'warning': 'Unvalidated synthetic research policy; do not use for clinical decisions or surgery.',
                      'observation_names': list(env.observation_names), 'observation_low': low.tolist(),
                      'observation_high': high.tolist(), 'feature_transform': None,
                      'action_names': list(env.action_names), 'primitive_names': names,
                      'action_primitives': primitives.tolist(),
                      'state_encoding': {'version': BIN_VERSION, 'edges': BIN_EDGES, 'shape': list(STATE_SHAPE)},
                      'q_values': ((first + second) / 2).tolist()}
    finally:
        env.close()
    if progress_callback:
        progress_callback(.9, 'Evaluating tabular policy on disjoint patient seeds')
    def evaluation_progress(fraction, message):
        if progress_callback:
            progress_callback(.9 + .1 * fraction, message)
    evaluation = evaluate_policy(checkpoint, cfg, test_seeds, evaluation_progress)
    return {'algorithm': TABULAR_ALGORITHM, 'model_version': checkpoint['model_version'],
            'config': cfg, 'checkpoint': checkpoint, 'learning_curve': curve, 'evaluation': evaluation,
            'metadata': {'training_seeds': training_seeds + demonstration_seeds,
                         'training_episode_seeds': training_seeds, 'demonstration_seeds': demonstration_seeds,
                         'test_seeds': test_seeds, 'training_method': checkpoint['training_method'],
                         'demonstration_transitions': len(demo_transitions),
                         'demonstration_initialization': 'Mean observed discounted return for each visited state/action; zero otherwise',
                         'demonstration_initial_state_actions': initial_observed_actions,
                         'demonstration_replay_updates_per_online_step': 1 if demo_transitions else 0,
                         'demonstration_summary': _aggregate(demo_episodes, cfg['training_seed']) if demo_episodes else None,
                         'environment_steps': environment_steps, 'gradient_updates': 0,
                         'td_updates': updates, 'tabular_states': STATE_COUNT,
                         'visited_training_states': int(np.count_nonzero(visits)),
                         'reward_scale': reward_scale, 'environment_description': description,
                         'observation_encoding': 'Reactive partial-state bins, not a Markov state representation',
                         'unused_dqn_settings': ['learning_rate', 'observation_encoding', 'demonstration_margin', 'demonstration_weight'],
                         'evaluation': 'Per-run held-out seeds are not used for fitting; reused engineering cohorts remain development data.',
                         'limitations': ['Illustrative synthetic dynamics without clinical validation.',
                                         'Reactive bins alias distinct anatomy and deformation histories.',
                                         'A discrete primitive wrapper restricts continuous tool controls.',
                                         'Demonstration initialization/replay is additional assistance when enabled.',
                                         'Confidence intervals exclude tissue-model and training-seed uncertainty.']}}
