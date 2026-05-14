"""
Custom Gymnasium wrapper for LunarLander-v3.

Adds:
- Observation normalization (zero-mean, unit-variance running stats)
- Environment variants for generalisation evaluation
- Deterministic seeding for reproducibility

Original environment: https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Pre-defined variants used for generalisation testing (SRPs 4/4 requirement)
ENV_VARIANTS = {
    "standard": dict(enable_wind=False),
    "wind":      dict(enable_wind=True, wind_power=15.0, turbulence_power=0.0),
    "turbulent": dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5),
    "heavy":     dict(gravity=-11.5),   # stronger gravity (capped at -12 by env)
}


class RunningNormalizer:
    """Online running mean/std normalizer (Welford's algorithm)."""

    def __init__(self, shape: tuple, clip: float = 5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = 1e-8
        self.clip = clip

    def update(self, x: np.ndarray):
        self.count += 1
        delta  = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var / self.count) + 1e-8
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class LunarLanderWrapper(gym.Wrapper):
    """
    Wrapper around LunarLander-v3 that supports:
    - Online observation normalisation (toggled via `normalize`)
    - Environment variant injection (wind, turbulence, gravity)
    - Fixed seeding
    """

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        normalize: bool = True,
        variant: str = "standard",
        seed: int = 42,
        **variant_kwargs,
    ):
        # Build the env with the chosen variant parameters
        kwargs = ENV_VARIANTS.get(variant, {})
        kwargs.update(variant_kwargs)
        base_env = gym.make(env_name, **kwargs)
        super().__init__(base_env)

        self.normalize = normalize
        self.seed_val  = seed
        self.variant   = variant

        obs_shape = self.observation_space.shape
        self._normalizer = RunningNormalizer(obs_shape)
        self._first_reset = True   # seed only on first call

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        # Seed only on first reset so subsequent episodes vary naturally.
        # Callers can still pass an explicit seed= to override.
        if self._first_reset and "seed" not in kwargs:
            kwargs["seed"] = self.seed_val
            self._first_reset = False
        obs, info = self.env.reset(**kwargs)
        if self.normalize:
            self._normalizer.update(obs)
            obs = self._normalizer.normalize(obs)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.normalize:
            self._normalizer.update(obs)
            obs = self._normalizer.normalize(obs)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    # Expose raw normaliser so agents can persist it
    @property
    def normalizer(self):
        return self._normalizer


def make_env(
    env_name: str = "LunarLander-v3",
    normalize: bool = True,
    variant: str = "standard",
    seed: int = 42,
) -> LunarLanderWrapper:
    """Factory function — preferred entry point for creating environments."""
    return LunarLanderWrapper(
        env_name=env_name,
        normalize=normalize,
        variant=variant,
        seed=seed,
    )
