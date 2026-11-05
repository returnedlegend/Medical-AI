"""Inspectable reactive Double-Q baseline for the synthetic tissue environment.

The bins use current public measurements. This intentionally reduced state is
not Markov: different anatomy/deformation histories can share a bin. It is a
small comparison to the full-observation neural baseline, not a clinical policy.
"""
from __future__ import annotations

from copy import deepcopy
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
V2_BIN_VERSION = "public-reactive-bins-2"
V2_BIN_EDGES = {**{name: edges for name, edges in BIN_EDGES.items() if name != 'max_mechanical_damage'},
                'max_combined_damage': [.05, .15]}
V2_STATE_SHAPE = (5, *(len(edges) + 1 for edges in V2_BIN_EDGES.values()), 2, 2)
V3_BIN_VERSION = "public-reactive-bins-3"
V3_BIN_EDGES = {**V2_BIN_EDGES, 'force_limit_ratio': [.5, .85, .95, 1., 1.05],
                'max_strain_limit_ratio': [.5, .85, .95, 1., 1.05], 'max_critical_damage': [.08]}
V3_COMPARISON_SIDES = {name: ('right' if name in ('max_combined_damage', 'max_critical_damage') else 'left')
                       for name in V3_BIN_EDGES}
V3_STATE_SHAPE = (5, *(len(edges) + 1 for edges in V3_BIN_EDGES.values()), 2, 2)
STATE_ENCODINGS = {
    BIN_VERSION: {'version': BIN_VERSION, 'edges': BIN_EDGES, 'shape': list(STATE_SHAPE)},
    V2_BIN_VERSION: {'version': V2_BIN_VERSION, 'edges': V2_BIN_EDGES, 'shape': list(V2_STATE_SHAPE),
                     'acquisition_contexts': ['ungrasped_far', 'ungrasped_near_node03', 'ungrasped_near_other',
                                              'grasped_node03', 'grasped_other'],
                     'acquisition_distance_mm': 3., 'intact_combined_damage_below': .9,
                     'exposure_marker_node_index': 3, 'temperature_threshold_c': 43.,
                     'late_remaining_fraction': .2},
    V3_BIN_VERSION: {'version': V3_BIN_VERSION, 'edges': V3_BIN_EDGES, 'shape': list(V3_STATE_SHAPE),
                     'acquisition_contexts': ['ungrasped_no_acquirable_nearest', 'ungrasped_near_node03', 'ungrasped_near_other',
                                              'grasped_node03', 'grasped_other'],
                     'acquisition_distance_mm': 3., 'intact_combined_damage_below': .9,
                     'nearest_node_rule': 'nearest_all_then_reject_damage',
                     'exposure_marker_node_index': 3, 'critical_node_indices': list(range(16, 23)),
                     'temperature_threshold_c': 43., 'late_remaining_rule': '5*(max_steps-step_count)<=max_steps',
                     'comparison_scale': 1_000_000, 'comparison_rounding': 'nearest_even_integer',
                     'comparison_sides': V3_COMPARISON_SIDES},
}


class TabularPolicy:
    def __init__(self, values, low, high, observation_names, encoding_version=BIN_VERSION):
        self.values = values
        self.low, self.high = low, high
        self.encoding_version = encoding_version
        self.shape = tuple(STATE_ENCODINGS[encoding_version]['shape'])
        self.index = {name: i for i, name in enumerate(observation_names)}
        self.strains = [i for i, name in enumerate(observation_names) if name.endswith('_strain')]
        self.damage = [i for i, name in enumerate(observation_names) if name.endswith('_mechanical_damage')]
        required = ('contact_node_index', 'exposure_target', 'exposure', 'contact_force_n',
                    'force_limit_n', 'strain_limit')
        if any(name not in self.index for name in required) or not self.strains or not self.damage:
            raise ValueError('Tabular policy requires the declared public tissue measurements')
        if encoding_version in (V2_BIN_VERSION, V3_BIN_VERSION):
            extra = ('grasper_x', 'grasper_y', 'step_count', 'max_steps')
            prefixes = [name[:-len('_mechanical_damage')] for name in observation_names if name.endswith('_mechanical_damage')]
            self.node_x = [self.index[f'{prefix}_x'] for prefix in prefixes]
            self.node_y = [self.index[f'{prefix}_y'] for prefix in prefixes]
            self.doses = [self.index[f'{prefix}_thermal_dose'] for prefix in prefixes]
            self.temperatures = [self.index[f'{prefix}_temperature_c'] for prefix in prefixes]
            self.critical_nodes = [prefixes.index(f'n{node:02d}') for node in range(16, 23)]
            if any(name not in self.index for name in extra) or len(prefixes) != 23:
                raise ValueError('V2 reactive encoding requires the full public tissue/tool measurements')

    def state_indices(self, normalized):
        if self.encoding_version == V3_BIN_VERSION:
            return self._v3_indices(np.atleast_2d(normalized).astype(np.float64))
        raw = (np.atleast_2d(normalized) + 1) * .5 * (self.high - self.low) + self.low
        if self.encoding_version == V2_BIN_VERSION:
            return self._v2_indices(raw)
        # Preserve the original v1 arithmetic and bin order for old policies.
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

    def _v2_indices(self, raw):
        combined = 1 - (1 - raw[:, self.damage]) * np.exp(-np.maximum(raw[:, self.doses], 0) / .8)
        dx = raw[:, self.node_x] - raw[:, self.index['grasper_x'], None]
        dy = raw[:, self.node_y] - raw[:, self.index['grasper_y'], None]
        distances = np.where(combined < .9, np.sqrt(dx * dx + dy * dy), np.inf)
        nearest = np.argmin(distances, axis=1)
        near = distances[np.arange(len(raw)), nearest] <= 3.
        contact = np.rint(raw[:, self.index['contact_node_index']]).astype(int)
        context = np.where(contact >= 0, np.where(contact == 3, 3, 4),
                           np.where(near, np.where(nearest == 3, 1, 2), 0))
        measurements = [raw[:, self.index['exposure_target']] - raw[:, self.index['exposure']],
                        raw[:, self.index['contact_force_n']] / np.maximum(raw[:, self.index['force_limit_n']], 1e-6),
                        raw[:, self.strains].max(axis=1) / np.maximum(raw[:, self.index['strain_limit']], 1e-6),
                        combined.max(axis=1)]
        coordinates = [context]
        coordinates.extend(np.searchsorted(edges, np.round(values, 6), side='right')
                           for edges, values in zip(V2_BIN_EDGES.values(), measurements))
        pending_heat = raw[:, self.temperatures].max(axis=1) > 43.
        remaining = (raw[:, self.index['max_steps']] - raw[:, self.index['step_count']]) / np.maximum(raw[:, self.index['max_steps']], 1)
        coordinates.extend((pending_heat.astype(int), (remaining <= .2).astype(int)))
        return np.ravel_multi_index(tuple(coordinates), self.shape)

    def _v3_indices(self, raw):
        combined = 1 - (1 - raw[:, self.damage]) * np.exp(-np.maximum(raw[:, self.doses], 0) / .8)
        dx = raw[:, self.node_x] - raw[:, self.index['grasper_x'], None]
        dy = raw[:, self.node_y] - raw[:, self.index['grasper_y'], None]
        distances = np.sqrt(dx * dx + dy * dy)
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(len(raw))
        near = (distances[rows, nearest] <= 3.) & (combined[rows, nearest] < .9)
        contact = np.rint(raw[:, self.index['contact_node_index']]).astype(int)
        context = np.where(contact >= 0, np.where(contact == 3, 3, 4),
                           np.where(near, np.where(nearest == 3, 1, 2), 0))
        measurements = [raw[:, self.index['exposure_target']] - raw[:, self.index['exposure']],
                        raw[:, self.index['contact_force_n']] / np.maximum(raw[:, self.index['force_limit_n']], 1e-6),
                        raw[:, self.strains].max(axis=1) / np.maximum(raw[:, self.index['strain_limit']], 1e-6),
                        combined.max(axis=1), combined[:, self.critical_nodes].max(axis=1)]
        coordinates = [context]
        for (name, edges), values in zip(V3_BIN_EDGES.items(), measurements):
            quantized = np.rint(values * 1_000_000).astype(np.int64)
            thresholds = np.rint(np.asarray(edges) * 1_000_000).astype(np.int64)
            coordinates.append(np.searchsorted(thresholds, quantized, side=V3_COMPARISON_SIDES[name]))
        pending_heat = raw[:, self.temperatures].max(axis=1) > 43.
        late = 5 * (raw[:, self.index['max_steps']] - raw[:, self.index['step_count']]) <= raw[:, self.index['max_steps']]
        coordinates.extend((pending_heat.astype(int), late.astype(int)))
        return np.ravel_multi_index(tuple(coordinates), self.shape)

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
    from .learning import CHECKPOINT_FORMAT, RAW_FEATURE_TRANSFORM
    if checkpoint.get('format') != CHECKPOINT_FORMAT or checkpoint.get('algorithm') != TABULAR_ALGORITHM:
        raise ValueError('Unsupported tabular checkpoint format')
    specification = checkpoint.get('state_encoding')
    version = specification.get('version') if isinstance(specification, dict) else None
    if not isinstance(version, str) or version not in STATE_ENCODINGS or specification != STATE_ENCODINGS[version]:
        raise ValueError('Unsupported tabular state encoding')
    state_count = int(np.prod(STATE_ENCODINGS[version]['shape']))
    expected_transform = RAW_FEATURE_TRANSFORM if version == V3_BIN_VERSION else None
    if checkpoint.get('feature_transform') != expected_transform:
        raise ValueError('Tabular policy preprocessing does not match its encoder version')
    try:
        low = np.asarray(checkpoint['observation_low'], dtype=np.float32)
        high = np.asarray(checkpoint['observation_high'], dtype=np.float32)
        actions = np.asarray(checkpoint['action_primitives'], dtype=np.float32)
        names = checkpoint['observation_names']
        if low.ndim != 1 or not 1 <= len(low) <= 2048 or high.shape != low.shape:
            raise ValueError('Invalid tabular observation bounds')
        if (not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low)
                or np.any(np.abs(low) > 1e6) or np.any(np.abs(high) > 1e6)
                or np.any(high.astype(np.float64) - low.astype(np.float64) < 1e-6)):
            raise ValueError('Invalid tabular observation bounds')
        if not isinstance(names, list) or len(names) != len(low) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError('Invalid tabular observation names')
        if actions.ndim != 2 or not 1 <= len(actions) <= 128 or actions.shape[1] != 7:
            raise ValueError('Invalid tabular action dimensions')
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError('Invalid tabular actions')
        if version == V3_BIN_VERSION:
            sparse = checkpoint.get('q_sparse')
            if 'q_values' in checkpoint or not isinstance(sparse, dict) or set(sparse) != {'indices', 'values'}:
                raise ValueError('V3 requires declarative sparse Q rows')
            indices = sparse['indices']
            if (not isinstance(indices, list) or len(indices) > state_count
                    or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < state_count for index in indices)
                    or any(left >= right for left, right in zip(indices, indices[1:]))):
                raise ValueError('Invalid sparse Q row indices')
            supplied = sparse['values']
            if not isinstance(supplied, list) or len(supplied) != len(indices):
                raise ValueError('Invalid sparse Q row values')
            supplied = np.asarray(supplied, dtype=np.float32) if supplied else np.zeros((0, len(actions)), dtype=np.float32)
            if supplied.shape != (len(indices), len(actions)) or not np.isfinite(supplied).all() or np.any(np.abs(supplied) > 1e6):
                raise ValueError('Invalid sparse Q row values')
            values = np.zeros((state_count, len(actions)), dtype=np.float32)
            values[np.asarray(indices, dtype=np.int64)] = supplied
        else:
            supplied_values = checkpoint['q_values']
            if 'q_sparse' in checkpoint or not isinstance(supplied_values, list) or len(supplied_values) != state_count:
                raise ValueError('Invalid tabular Q dimensions')
            values = np.asarray(supplied_values, dtype=np.float32)
        if values.shape != (state_count, len(actions)) or not np.isfinite(values).all() or np.any(np.abs(values) > 1e6):
            raise ValueError('Invalid tabular Q values')
        return TabularPolicy(values, low, high, names, version), low, high, actions
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError('Invalid tabular checkpoint arrays') from error


def train_tabular_policy(config, progress_callback: Callable | None = None):
    from .learning import (_aggregate, _bounds, _describe, _encode, _episode_summary, _make_env,
                           RAW_FEATURE_TRANSFORM, action_primitives, CHECKPOINT_FORMAT, evaluate_policy)
    cfg = dict(config)
    env = _make_env(cfg)
    description = _describe(env)
    encoding_version = {'v1': BIN_VERSION, 'v2': V2_BIN_VERSION, 'v3': V3_BIN_VERSION}[cfg['tabular_state_encoding']]
    feature_transform = RAW_FEATURE_TRANSFORM if encoding_version == V3_BIN_VERSION else None
    state_count = int(np.prod(STATE_ENCODINGS[encoding_version]['shape']))
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
    first = np.zeros((state_count, len(primitives)), dtype=np.float32)
    second = first.copy()
    policy = TabularPolicy(first, low, high, env.observation_names, encoding_version)
    sums, counts = np.zeros_like(first), np.zeros_like(first, dtype=np.int32)
    demo_transitions, demo_episodes, curve = [], [], []
    visits = np.zeros(state_count, dtype=np.int32)
    updates, environment_steps, reward_scale = 0, 0, 10.
    def state(observation):
        return int(policy.state_indices(_encode(observation, low, high, feature_transform))[0])
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
                      'observation_high': high.tolist(), 'feature_transform': deepcopy(feature_transform),
                      'action_names': list(env.action_names), 'primitive_names': names,
                      'action_primitives': primitives.tolist(),
                      'state_encoding': deepcopy(STATE_ENCODINGS[encoding_version])}
        averaged = (first + second) / 2
        if encoding_version == V3_BIN_VERSION:
            nonzero = np.flatnonzero(np.any(averaged != 0, axis=1))
            checkpoint['q_sparse'] = {'indices': nonzero.tolist(), 'values': averaged[nonzero].tolist()}
        else:
            checkpoint['q_values'] = averaged.tolist()
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
                         'td_updates': updates, 'tabular_states': state_count,
                         'tabular_state_encoding': encoding_version,
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
