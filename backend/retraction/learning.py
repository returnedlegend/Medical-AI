"""Small, inspectable NumPy DQN for the stateful tissue-response environment.

The policy chooses from declared tool-control primitives; the underlying Gym
interface remains continuous. Every update uses observed transitions and the
actual environment reward. Checkpoints contain JSON arrays, never pickle code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

ALGORITHM = "numpy-dqn-1.0"
CHECKPOINT_FORMAT = "retraction-dqn-json-1"
FEATURE_VERSION = "public-tissue-features-1"
DEFAULT_TRAINING_CONFIG = {
    "algorithm": "dqn",
    "training_seed": 42, "episodes": 300, "eval_episodes": 20,
    "max_steps": 200, "learning_rate": 0.0007, "gamma": 0.98,
    "demonstration_episodes": 40,
    "demonstration_margin": .1, "demonstration_weight": .5,
    "observation_encoding": "characteristic",
    "tabular_learning_rate": .15,
    "tabular_state_encoding": "v1",
    "environment": {},
}


def normalize_training_config(config: dict | None = None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise ValueError("Training configuration must be an object")
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_TRAINING_CONFIG)
    if unknown:
        raise ValueError(f"Unknown training settings: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_TRAINING_CONFIG, **supplied}
    if result["algorithm"] not in ("dqn", "tabular_q"):
        raise ValueError("algorithm must be dqn or tabular_q")
    if result["observation_encoding"] not in ("box", "characteristic"):
        raise ValueError("observation_encoding must be box or characteristic")
    if result["tabular_state_encoding"] not in ("v1", "v2"):
        raise ValueError("tabular_state_encoding must be v1 or v2")
    for key, lo, hi in (("training_seed", 0, 2**32 - 1), ("episodes", 20, 3000),
                        ("eval_episodes", 5, 100), ("max_steps", 20, 300),
                        ("demonstration_episodes", 0, 300)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    for key, lo, hi in (("learning_rate", 0.00001, 0.01), ("gamma", 0.8, 0.999),
                        ("demonstration_margin", 0, 1), ("demonstration_weight", 0, 5)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} must be finite and in [{lo}, {hi}]")
        result[key] = float(value)
    value = result["tabular_learning_rate"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .001 <= value <= 1:
        raise ValueError("tabular_learning_rate must be finite and in [.001, 1]")
    result["tabular_learning_rate"] = float(value)
    if not isinstance(result["environment"], dict):
        raise ValueError("environment must be an object")
    if "max_steps" in result["environment"]:
        raise ValueError("Set max_steps at the top level, not inside environment")
    # Validate real environment options when available, including its own bounds.
    try:
        from .environment import TissueRetractionEnv
    except ImportError:
        # The dependency or engine absence is surfaced at training start, not
        # silently replaced with another simulator.
        return result
    env = TissueRetractionEnv({**result["environment"], "max_steps": result["max_steps"]})
    env.close()
    return result


def action_primitives() -> tuple[list[str], np.ndarray]:
    names, actions = [], []

    def add(name: str, values: list[float]):
        names.append(name)
        actions.append(values)

    add("hold", [0] * 7)
    add("grasp_gently", [0, 0, 0.55, 0, 0, 0, 0])
    add("release", [0, 0, -1, 0, 0, 0, 0])
    for label, x, y in (("east", 1, 0), ("north_east", .707, .707),
                        ("north", 0, 1), ("north_west", -.707, .707),
                        ("west", -1, 0), ("south_west", -.707, -.707),
                        ("south", 0, -1), ("south_east", .707, -.707)):
        add(f"grasper_{label}", [x, y, 0, 0, 0, 0, 0])
    for label, x, y in (("east", 1, 0), ("north", 0, 1), ("west", -1, 0), ("south", 0, -1)):
        add(f"grasper_fine_{label}", [.35 * x, .35 * y, 0, 0, 0, 0, 0])
    for label, x, y in (("east", 1, 0), ("north", 0, 1), ("west", -1, 0), ("south", 0, -1)):
        add(f"cautery_{label}", [0, 0, 0, x, y, 0, 0])
    add("grasper_micro_east", [.10, 0, 0, 0, 0, 0, 0])
    add("grasper_slow_east", [.20, 0, 0, 0, 0, 0, 0])
    for name, speed in (("medium", .30), ("gentle", .15), ("creep", .05), ("ultrafine", .025)):
        add(f"grasper_{name}_east", [speed, 0, 0, 0, 0, 0, 0])
    add("grasper_micro_west", [-.03, 0, 0, 0, 0, 0, 0])
    add("cautery_pulse", [0, 0, 0, 0, 0, 0.35, 0])
    add("request_handoff", [0, 0, 0, 0, 0, 0, 1])
    return names, np.asarray(actions, dtype=np.float32)


@dataclass
class Network:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def initialize(cls, inputs: int, outputs: int, rng: np.random.Generator, hidden: int = 64):
        return cls(rng.normal(0, np.sqrt(1 / inputs), (inputs, hidden)).astype(np.float32),
                   np.zeros(hidden, dtype=np.float32),
                   rng.normal(0, .05, (hidden, outputs)).astype(np.float32),
                   np.zeros(outputs, dtype=np.float32))

    def forward(self, observations: np.ndarray):
        hidden = np.tanh(observations @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2, hidden

    def copy(self):
        return Network(*(value.copy() for value in (self.w1, self.b1, self.w2, self.b2)))

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: getattr(self, key) for key in ("w1", "b1", "w2", "b2")}


class Adam:
    def __init__(self, network: Network, rate: float):
        self.rate, self.t = rate, 0
        self.m = {key: np.zeros_like(value) for key, value in network.arrays().items()}
        self.v = {key: np.zeros_like(value) for key, value in network.arrays().items()}

    def update(self, network: Network, gradients: dict[str, np.ndarray]):
        self.t += 1
        norm = math.sqrt(sum(float(np.sum(value**2)) for value in gradients.values()))
        multiplier = min(1.0, 10 / max(norm, 1e-9))
        for key, parameter in network.arrays().items():
            gradient = gradients[key] * multiplier
            self.m[key] = .9 * self.m[key] + .1 * gradient
            self.v[key] = .999 * self.v[key] + .001 * gradient**2
            parameter -= self.rate * (self.m[key] / (1 - .9**self.t)) / (np.sqrt(self.v[key] / (1 - .999**self.t)) + 1e-8)


class Replay:
    def __init__(self, capacity: int, dimension: int):
        self.observations = np.zeros((capacity, dimension), dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.count, self.position = capacity, 0, 0
        self._balanced_indices_cache = None

    def append(self, observation, action, reward, next_observation, terminal):
        self._balanced_indices_cache = None
        index = self.position
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.position = (index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_balanced(self, rng: np.random.Generator, batch: int):
        """Balance demonstrated controls so a one-step initial grasp is represented."""
        if self._balanced_indices_cache is None:
            self._balanced_indices_cache = {int(action): np.flatnonzero(self.actions[:self.count] == action)
                                            for action in np.unique(self.actions[:self.count])}
        present = np.asarray(list(self._balanced_indices_cache))
        selected_actions = rng.choice(present, size=batch)
        idx = np.empty(batch, dtype=np.int64)
        for action in present:
            rows = np.flatnonzero(selected_actions == action)
            idx[rows] = rng.choice(self._balanced_indices_cache[int(action)], size=len(rows))
        return tuple(value[idx] for value in (self.observations, self.actions, self.rewards,
                                              self.next_observations, self.terminals))

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.count, batch)
        return tuple(value[idx] for value in (self.observations, self.actions, self.rewards,
                                              self.next_observations, self.terminals))


def optimize(network: Network, target: Network, optimizer: Adam, batch: tuple, gamma: float) -> float:
    observations, actions, rewards, next_observations, terminals = batch
    # Double-DQN action selection: online network selects, target evaluates.
    next_online, _ = network.forward(next_observations)
    next_target, _ = target.forward(next_observations)
    greedy = np.argmax(next_online, axis=1)
    targets = rewards + gamma * (1 - terminals) * next_target[np.arange(len(actions)), greedy]
    qvalues, hidden = network.forward(observations)
    delta = qvalues[np.arange(len(actions)), actions] - targets
    absolute = np.abs(delta)
    loss = np.mean(np.where(absolute <= 1, .5 * delta**2, absolute - .5))
    output_gradient = np.zeros_like(qvalues)
    output_gradient[np.arange(len(actions)), actions] = np.clip(delta, -1, 1) / len(actions)
    hidden_gradient = (output_gradient @ network.w2.T) * (1 - hidden**2)
    gradients = {"w2": hidden.T @ output_gradient, "b2": output_gradient.sum(axis=0),
                 "w1": observations.T @ hidden_gradient, "b1": hidden_gradient.sum(axis=0)}
    optimizer.update(network, gradients)
    return float(loss)


def demonstration_margin_update(network: Network, optimizer: Adam, batch: tuple,
                                margin: float = .1, coefficient: float = .5) -> float:
    """DQfD-style large-margin action guidance, separate from TD feedback."""
    observations, actions, *_ = batch
    qvalues, hidden = network.forward(observations)
    augmented = qvalues + margin
    augmented[np.arange(len(actions)), actions] -= margin
    competitor = np.argmax(augmented, axis=1)
    active = competitor != actions
    loss = coefficient * np.mean(augmented[np.arange(len(actions)), competitor] -
                                 qvalues[np.arange(len(actions)), actions])
    output_gradient = np.zeros_like(qvalues)
    rows = np.flatnonzero(active)
    output_gradient[rows, competitor[rows]] += coefficient / len(actions)
    output_gradient[rows, actions[rows]] -= coefficient / len(actions)
    hidden_gradient = (output_gradient @ network.w2.T) * (1 - hidden**2)
    optimizer.update(network, {"w2": hidden.T @ output_gradient, "b2": output_gradient.sum(axis=0),
                               "w1": observations.T @ hidden_gradient, "b1": hidden_gradient.sum(axis=0)})
    return float(loss)


def _bounds(env) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.asarray(env.observation_space.low), np.asarray(env.observation_space.high)
    if low.ndim != 1 or low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Learning requires a finite, nondegenerate flat observation Box")
    return low.astype(np.float32), high.astype(np.float32)


def _normalize(observation, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != low.shape or not np.isfinite(values).all():
        raise ValueError("Observation dimensions/values do not match the policy specification")
    return np.clip(2 * (values - low) / (high - low) - 1, -1, 1)


def feature_specification(observation_names: list[str], low, high) -> dict:
    """Declare public measurements and characteristic scales, never future outcomes.

    Gym bounds are deliberately broad enough for catastrophic injury. They are
    poor neural-network scales for early injury (strain .6 versus a bound of
    30, for example). Explicit scales preserve that early feedback signal.
    """
    center = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float)) / 2
    scale = (np.asarray(high, dtype=float) - np.asarray(low, dtype=float)) / 2
    special = {"contact_force_n": (0, 2.5), "elapsed_s": (0, 20),
               "step_count": (0, 200), "stable_exposure_steps": (0, 5),
               "blood_loss_ml": (0, 100), "bile_leak_ml": (0, 20),
               "bleeding_rate_ml_per_s": (0, 10), "bile_leak_rate_ml_per_s": (0, 5)}
    for index, name in enumerate(observation_names):
        if name in special:
            center[index], scale[index] = special[name]
        elif name.endswith("_strain"):
            center[index], scale[index] = 0, .6
        elif name.endswith("_thermal_dose"):
            center[index], scale[index] = 0, .8
        elif name.endswith("_temperature_c"):
            center[index], scale[index] = 37, 6
        elif name.endswith("_mechanical_damage") or name.endswith("_coagulation"):
            center[index], scale[index] = 0, .15
    derived = ["has_contact", "exposure_error", "force_limit_ratio", "max_strain_limit_ratio",
               "max_mechanical_damage", "max_thermal_damage", "max_temperature_rise",
               "remaining_steps_fraction", "stable_hold_fraction"]
    return {"version": FEATURE_VERSION, "observation_names": list(observation_names),
            "center": center.tolist(), "scale": scale.tolist(),
            "feature_names": list(observation_names) + derived}


def _encode(observation, low, high, specification: dict | None = None) -> np.ndarray:
    if specification is None:  # Earlier JSON checkpoints retain their exact transform.
        return _normalize(observation, low, high)
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != low.shape or not np.isfinite(values).all():
        raise ValueError("Observation dimensions/values do not match the policy specification")
    names = specification["observation_names"]
    index = {name: position for position, name in enumerate(names)}
    def get(name):
        return float(values[index[name]])
    strains = [values[position] for position, name in enumerate(names) if name.endswith("_strain")]
    mechanical = [values[position] for position, name in enumerate(names) if name.endswith("_mechanical_damage")]
    dose = [values[position] for position, name in enumerate(names) if name.endswith("_thermal_dose")]
    temperatures = [values[position] for position, name in enumerate(names) if name.endswith("_temperature_c")]
    derived = [float(get("contact_node_index") >= 0),
               5 * (get("exposure_target") - get("exposure")),
               get("contact_force_n") / max(get("force_limit_n"), 1e-6),
               max(strains) / max(get("strain_limit"), 1e-6),
               max(mechanical) / .15, (1 - math.exp(-max(dose) / .8)) / .15,
               (max(temperatures) - 37) / 6,
               (get("max_steps") - get("step_count")) / max(get("max_steps"), 1),
               get("stable_exposure_steps") / max(get("success_hold_steps"), 1)]
    centered = (values - np.asarray(specification["center"])) / np.asarray(specification["scale"])
    return np.clip(np.concatenate((centered, derived)), -5, 5).astype(np.float32)


def _checkpoint(network: Network, env, config: dict, low, high, description: dict, feature_spec: dict | None) -> dict:
    names, actions = action_primitives()
    return {"format": CHECKPOINT_FORMAT, "algorithm": ALGORITHM,
            "model_version": description.get("model_version", description.get("version", "unknown")),
            "physics_fingerprint": description.get("physics_fingerprint"),
            "environment": config["environment"], "max_steps": config["max_steps"],
            "environment_config": description.get("config"),
            "training_config": dict(config),
            "training_method": "demonstration-assisted Double-DQN" if config["demonstration_episodes"] else "Double-DQN from uniform exploration",
            "warning": "Unvalidated synthetic research policy; do not use for clinical decisions or surgery.",
            "observation_names": list(env.observation_names),
            "observation_low": low.tolist(), "observation_high": high.tolist(),
            "feature_transform": feature_spec,
            "action_names": list(env.action_names), "primitive_names": names,
            "action_primitives": actions.tolist(),
            "architecture": {"input": network.w1.shape[0], "hidden": network.w1.shape[1], "output": len(names), "activation": "tanh"},
            "weights": {key: value.tolist() for key, value in network.arrays().items()}}


def load_checkpoint(checkpoint: dict) -> tuple[Network, np.ndarray, np.ndarray, np.ndarray]:
    """Validate declarative weights strictly before allocating or evaluating them."""
    if isinstance(checkpoint, dict) and checkpoint.get("algorithm") == "numpy-tabular-double-q-1.0":
        from .tabular import load_tabular_checkpoint
        return load_tabular_checkpoint(checkpoint)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("algorithm") != ALGORITHM:
        raise ValueError("Unsupported policy checkpoint format or algorithm")
    architecture = checkpoint.get("architecture", {})
    dimensions = [architecture.get(key) for key in ("input", "hidden", "output")]
    if any(isinstance(n, bool) or not isinstance(n, int) for n in dimensions):
        raise ValueError("Invalid policy architecture")
    inputs, hidden, outputs = dimensions
    if not (1 <= inputs <= 2048 and 1 <= hidden <= 256 and 1 <= outputs <= 128):
        raise ValueError("Policy architecture exceeds supported bounds")
    if architecture.get("activation") != "tanh":
        raise ValueError("Unsupported policy activation")
    expected = {"w1": (inputs, hidden), "b1": (hidden,), "w2": (hidden, outputs), "b2": (outputs,)}
    weights = checkpoint.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("Checkpoint weights are missing or unknown")
    arrays = {}
    try:
        for key, shape in expected.items():
            value = np.asarray(weights[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all() or np.any(np.abs(value) > 1e6):
                raise ValueError(f"Invalid policy weights: {key}")
            arrays[key] = value
        low = np.asarray(checkpoint["observation_low"], dtype=np.float32)
        high = np.asarray(checkpoint["observation_high"], dtype=np.float32)
        actions = np.asarray(checkpoint["action_primitives"], dtype=np.float32)
        if low.ndim != 1 or not 1 <= len(low) <= 2048 or high.shape != low.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
            raise ValueError("Invalid checkpoint observation bounds")
        feature_spec = checkpoint.get("feature_transform")
        if feature_spec is None:
            if len(low) != inputs:
                raise ValueError("Policy input dimensions do not match observation bounds")
        else:
            if not isinstance(feature_spec, dict) or feature_spec.get("version") != FEATURE_VERSION:
                raise ValueError("Unsupported policy feature transform")
            names = checkpoint.get("observation_names")
            if not isinstance(names, list) or len(names) != len(low) or any(not isinstance(name, str) for name in names):
                raise ValueError("Invalid policy observation names")
            # Exact derived definitions/scales are versioned, not executable data
            # or arbitrary index expressions from a downloaded checkpoint.
            expected_spec = feature_specification(names, low, high)
            if feature_spec != expected_spec or len(expected_spec["feature_names"]) != inputs:
                raise ValueError("Policy feature transform does not match the declared specification")
            _encode(low, low, high, feature_spec)
        if actions.shape != (outputs, 7) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("Invalid checkpoint action primitives")
    except (KeyError, TypeError, OverflowError, IndexError) as error:
        raise ValueError("Invalid checkpoint arrays") from error
    return Network(**arrays), low, high, actions


def validate_checkpoint_compatibility(checkpoint: dict, description: dict) -> None:
    """Reject a valid JSON policy when its environment signature differs."""
    load_checkpoint(checkpoint)
    version = description.get("model_version", description.get("version"))
    if checkpoint.get("model_version") != version:
        raise ValueError("Policy tissue-model version does not match this environment")
    fingerprint = description.get("physics_fingerprint")
    if fingerprint and checkpoint.get("physics_fingerprint") != fingerprint:
        raise ValueError("Policy physics fingerprint does not match this environment")
    if checkpoint.get("observation_names") != description.get("observation_names"):
        raise ValueError("Policy observation specification does not match this environment")
    if checkpoint.get("action_names") != description.get("action_names"):
        raise ValueError("Policy action specification does not match this environment")


def policy_action(checkpoint: dict, observation) -> list[float]:
    network, low, high, actions = load_checkpoint(checkpoint)
    normalized = _encode(observation, low, high, checkpoint.get("feature_transform"))
    qvalues, _ = network.forward(normalized)
    return actions[int(np.argmax(qvalues))].astype(float).tolist()


def _describe(env) -> dict:
    description = env.describe()
    if not isinstance(description, dict):
        raise ValueError("Environment description must be a JSON object")
    return description


def _make_env(config: dict):
    from .environment import TissueRetractionEnv
    return TissueRetractionEnv({**config["environment"], "max_steps": config["max_steps"]})


def _episode_summary(total: float, steps: int, info: dict, terminated: bool, truncated: bool) -> dict:
    state = info.get("state", {})
    status = info.get("termination_reason") or state.get("termination_reason") or info.get("outcome") or "unknown"
    damage = info.get("damage", state.get("damage", {}))
    maximum = state.get("max_damage", damage.get("max_damage", 0) if isinstance(damage, dict) else 0)
    # Final exact field adapters are kept in one place and tested against the
    # environment, rather than inferring a successful policy from return alone.
    return {"return": float(total), "steps": int(steps), "terminated": bool(terminated),
            "truncated": bool(truncated), "outcome": str(status),
            "success": bool(info.get("success", status == "success")),
            "max_damage": float(maximum), "severe_harm": bool(info.get("severe_harm", status in ("severe_harm", "critical_injury", "severe_injury", "blood_loss"))),
            "handoff": status == "handoff"}


def _aggregate(episodes: list[dict], seed: int) -> dict:
    returns = np.array([row["return"] for row in episodes])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6001]))
    means = returns[rng.integers(0, len(returns), (1000, len(returns)))].mean(axis=1)
    def binary_interval(key):
        proportion = float(np.mean([row[key] for row in episodes]))
        z, n = 1.959963984540054, len(episodes)
        denominator = 1 + z * z / n
        center = (proportion + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
        return [max(0., center - radius), min(1., center + radius)]
    return {"n": len(episodes), "mean_return": float(returns.mean()),
            "mean_return_ci": np.quantile(means, [.025, .975]).tolist(),
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "success_rate_ci": binary_interval("success"),
            "severe_harm_rate": float(np.mean([row["severe_harm"] for row in episodes])),
            "severe_harm_rate_ci": binary_interval("severe_harm"),
            "handoff_rate": float(np.mean([row["handoff"] for row in episodes])),
            "handoff_rate_ci": binary_interval("handoff"),
            "mean_max_damage": float(np.mean([row["max_damage"] for row in episodes])),
            "mean_steps": float(np.mean([row["steps"] for row in episodes]))}


def evaluate_policy(checkpoint: dict, config: dict, seeds: list[int], progress_callback: Callable | None = None) -> dict:
    from threadpoolctl import threadpool_limits
    if not isinstance(seeds, list) or not seeds or len(seeds) > 10_000 or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Evaluation requires 1–10,000 integer patient seeds in [0, 2**32)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation patient seeds must be distinct")
    network, low, high, primitives = load_checkpoint(checkpoint)
    evaluated = {}
    with threadpool_limits(limits=1):
        modes = ("learned", "random", "random_no_handoff", "scripted")
        for mode_index, mode in enumerate(modes):
            episodes = []
            for seed_index, seed in enumerate(seeds):
                env = _make_env(config)
                observation, info = env.reset(seed=seed)
                rng = np.random.default_rng(np.random.SeedSequence([seed, 333]))
                total = 0.0
                try:
                    for step in range(config["max_steps"]):
                        if mode == "learned":
                            values, _ = network.forward(_encode(observation, low, high, checkpoint.get("feature_transform")))
                            action = primitives[int(np.argmax(values))]
                        elif mode in ("random", "random_no_handoff"):
                            available = len(primitives) - (1 if mode == "random_no_handoff" else 0)
                            action = primitives[int(rng.integers(available))]
                        else:
                            action = env.scripted_action(observation)
                        observation, reward, terminated, truncated, info = env.step(action)
                        total += float(reward)
                        if terminated or truncated:
                            break
                    episodes.append({"seed": seed, **_episode_summary(total, step + 1, info, terminated, truncated)})
                finally:
                    env.close()
                if progress_callback:
                    progress_callback((mode_index * len(seeds) + seed_index + 1) / (len(modes) * len(seeds)),
                                      f"Evaluating {mode}: {seed_index + 1}/{len(seeds)}")
            evaluated[mode] = {**_aggregate(episodes, config["training_seed"]), "episodes": episodes}
    # A paired bootstrap uses the same held-out patients for each comparator.
    rng = np.random.default_rng(np.random.SeedSequence([config["training_seed"], 6002]))
    indices = rng.integers(0, len(seeds), (1000, len(seeds)))
    for comparator in ("random", "random_no_handoff", "scripted"):
        contrast = {}
        for metric, label in (("return", "mean_return"), ("success", "success_rate"),
                              ("severe_harm", "severe_harm_rate"), ("max_damage", "mean_max_damage")):
            difference = np.asarray([row[metric] for row in evaluated["learned"]["episodes"]], dtype=float) - np.asarray(
                [row[metric] for row in evaluated[comparator]["episodes"]], dtype=float)
            sampled = difference[indices].mean(axis=1)
            contrast[f"{label}_difference"] = float(difference.mean())
            contrast[f"{label}_paired_bootstrap_ci"] = np.quantile(sampled, [.025, .975]).tolist()
        # Preserve the original public field for older clients.
        contrast["paired_bootstrap_ci"] = contrast["mean_return_paired_bootstrap_ci"]
        evaluated[f"learned_minus_{comparator}"] = contrast
    return evaluated


def train_policy(config: dict | None = None, progress_callback: Callable | None = None) -> dict:
    """Train from actual state/action/reward transitions, then evaluate untouched seeds."""
    from threadpoolctl import threadpool_limits
    cfg = normalize_training_config(config)
    if cfg["algorithm"] == "tabular_q":
        from .tabular import train_tabular_policy
        return train_tabular_policy(cfg, progress_callback)
    env = _make_env(cfg)
    description = _describe(env)
    names, primitives = action_primitives()
    low, high = _bounds(env)
    feature_spec = feature_specification(list(env.observation_names), low, high) if cfg["observation_encoding"] == "characteristic" else None
    feature_dimension = len(feature_spec["feature_names"]) if feature_spec else len(low)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["training_seed"], 71]))
    patient_rng = np.random.default_rng(np.random.SeedSequence([cfg["training_seed"], 72]))
    training_seeds = [int(value) for value in patient_rng.integers(0, 2**32, cfg["episodes"], dtype=np.uint64)]
    used = set(training_seeds)
    demo_rng = np.random.default_rng(np.random.SeedSequence([cfg["training_seed"], 74]))
    demonstration_seeds = []
    while len(demonstration_seeds) < cfg["demonstration_episodes"]:
        candidate = int(demo_rng.integers(0, 2**32, dtype=np.uint64))
        if candidate not in used:
            demonstration_seeds.append(candidate)
            used.add(candidate)
    test_rng = np.random.default_rng(np.random.SeedSequence([cfg["training_seed"], 73]))
    test_seeds = []
    while len(test_seeds) < cfg["eval_episodes"]:
        candidate = int(test_rng.integers(0, 2**32, dtype=np.uint64))
        if candidate not in used:
            test_seeds.append(candidate)
            used.add(candidate)
    network = Network.initialize(feature_dimension, len(primitives), rng)
    target, optimizer = network.copy(), Adam(network, cfg["learning_rate"])
    replay = Replay(20_000, feature_dimension)
    demonstration_replay = Replay(max(1, cfg["demonstration_episodes"] * cfg["max_steps"]), feature_dimension)
    curve, environment_steps, updates = [], 0, 0
    # Scaling stabilizes numerical optimization; raw rewards/returns remain
    # untouched in learning curves and evaluation.
    reward_scale = 10.0
    try:
        with threadpool_limits(limits=1):
            for demo_index, patient_seed in enumerate(demonstration_seeds):
                observation, info = env.reset(seed=patient_seed)
                for step in range(cfg["max_steps"]):
                    proposed = np.asarray(env.scripted_action(observation))
                    chosen = int(np.argmin(np.sum((primitives - proposed)**2, axis=1)))
                    next_observation, reward, terminated, truncated, info = env.step(primitives[chosen])
                    demonstration_replay.append(_encode(observation, low, high, feature_spec), chosen, float(reward) / reward_scale,
                                                _encode(next_observation, low, high, feature_spec), terminated or truncated)
                    observation = next_observation
                    environment_steps += 1
                    if terminated or truncated:
                        break
                if progress_callback:
                    progress_callback(.08 * (demo_index + 1) / len(demonstration_seeds),
                                      f"Collecting training-only demonstrations: {demo_index + 1}/{len(demonstration_seeds)}")
            if demonstration_replay.count >= 64:
                for warmup in range(1000):
                    demo_batch = demonstration_replay.sample_balanced(rng, 64)
                    optimize(network, target, optimizer, demo_batch, cfg["gamma"])
                    demonstration_margin_update(network, optimizer, demo_batch, cfg["demonstration_margin"], cfg["demonstration_weight"])
                    updates += 1
                    if updates % 100 == 0:
                        target = network.copy()
                    if progress_callback and warmup % 100 == 0:
                        progress_callback(.08 + .02 * warmup / 1000, "Warm-starting Q-values from observed demonstration transitions")
            for episode, patient_seed in enumerate(training_seeds):
                observation, info = env.reset(seed=patient_seed)
                epsilon = max(.05, 1 - .95 * episode / max(1, cfg["episodes"] * .7))
                total, losses = 0.0, []
                for step in range(cfg["max_steps"]):
                    normalized = _encode(observation, low, high, feature_spec)
                    if rng.random() < epsilon:
                        chosen = int(rng.integers(len(primitives)))
                    else:
                        values, _ = network.forward(normalized)
                        chosen = int(np.argmax(values))
                    next_observation, reward, terminated, truncated, info = env.step(primitives[chosen])
                    normalized_next = _encode(next_observation, low, high, feature_spec)
                    replay.append(normalized, chosen, float(reward) / reward_scale, normalized_next, terminated or truncated)
                    total += float(reward)
                    observation = next_observation
                    environment_steps += 1
                    if replay.count >= 128 and environment_steps % 4 == 0:
                        if demonstration_replay.count:
                            online_batch = replay.sample(rng, 32)
                            demo_batch = demonstration_replay.sample_balanced(rng, 32)
                            batch = tuple(np.concatenate((left, right)) for left, right in zip(online_batch, demo_batch))
                        else:
                            batch = replay.sample(rng, 64)
                        losses.append(optimize(network, target, optimizer, batch, cfg["gamma"]))
                        if demonstration_replay.count:
                            demonstration_margin_update(network, optimizer, demo_batch, cfg["demonstration_margin"], cfg["demonstration_weight"])
                        updates += 1
                        if updates % 100 == 0:
                            target = network.copy()
                    if terminated or truncated:
                        break
                curve.append({"episode": episode + 1, "seed": patient_seed,
                              **_episode_summary(total, step + 1, info, terminated, truncated),
                              "epsilon": float(epsilon), "mean_td_loss": float(np.mean(losses)) if losses else None})
                if progress_callback and ((episode + 1) % max(1, cfg["episodes"] // 100) == 0):
                    progress_callback(.1 + .75 * (episode + 1) / cfg["episodes"], f"Training episode {episode + 1}/{cfg['episodes']}")
        checkpoint = _checkpoint(network, env, cfg, low, high, description, feature_spec)
    finally:
        env.close()
    if progress_callback:
        progress_callback(.9, "Evaluating disjoint held-out seeds")
    def evaluation_progress(fraction, message):
        if progress_callback:
            progress_callback(.9 + .1 * fraction, message)
    evaluation = evaluate_policy(checkpoint, cfg, test_seeds, progress_callback=evaluation_progress)
    result = {"algorithm": ALGORITHM, "model_version": description.get("model_version", description.get("version")),
              "config": cfg, "checkpoint": checkpoint, "learning_curve": curve,
              "evaluation": evaluation,
              "metadata": {"training_seeds": training_seeds + demonstration_seeds, "training_episode_seeds": training_seeds,
                           "demonstration_seeds": demonstration_seeds, "test_seeds": test_seeds,
                           "training_method": "demonstration-assisted Double-DQN" if demonstration_seeds else "Double-DQN from uniform exploration",
                           "demonstration_transitions": demonstration_replay.count, "demonstration_batch_fraction": .5 if demonstration_seeds else 0,
                           "demonstration_action_balancing": bool(demonstration_seeds),
                           "demonstration_margin": cfg["demonstration_margin"] if demonstration_seeds else 0,
                           "demonstration_margin_weight": cfg["demonstration_weight"] if demonstration_seeds else 0,
                           "environment_steps": environment_steps, "gradient_updates": updates,
                           "replay_capacity": 20_000, "batch_size": 64, "reward_scale": reward_scale,
                           "target_update_interval": 100, "training_frequency_steps": 4,
                           "observation_names": list(env.observation_names), "primitive_names": names,
                           "feature_transform": feature_spec,
                           "environment_description": description,
                           "unused_tabular_settings": ["tabular_learning_rate", "tabular_state_encoding"],
                           "evaluation": "Per-run holdout is excluded from fitting and exploration. This function returns the final checkpoint; inspecting or selecting with these results makes the cohort validation data.",
                           "limitations": ["Synthetic, illustrative tissue dynamics; not clinical training or a validated surgical policy.",
                                           "A discrete primitive wrapper limits the continuous environment action space.",
                                           "One training seed per run; use independent runs before judging learning stability.",
                                           "Return intervals reflect held-out synthetic patient sampling, not tissue-model uncertainty.",
                                           "No policy is promoted as safer because of reward alone; harm and success rates are reported separately."]}}
    if progress_callback:
        progress_callback(1.0, "Training and held-out evaluation complete")
    return result
