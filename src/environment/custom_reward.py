"""
custom_reward.py — CustomLunarLander with configurable reward coefficients.

Subclasses gymnasium's LunarLander and overrides step() to expose all
internal reward coefficients as constructor parameters.  The formula
is identical to the original — only the weights change.

Original reward (LunarLander-v3 source, gymnasium):

    shaping_t = - pos_coef   * sqrt(x² + y²)       # distance to pad
                - vel_coef   * sqrt(vx² + vy²)      # total speed
                - angle_coef * |angle|               # tilt
                + leg_reward * (leg_L + leg_R)       # ground contact

    r_t = shaping_t - shaping_{t-1}                 # dense shaping Δ
        - main_engine_coef * m_power                 # fuel (main)
        - side_engine_coef * s_power                 # fuel (side)

    terminal:
        crash or out-of-bounds  →  r = -crash_penalty
        safe landing            →  r = +land_bonus

Default values reproduce the original environment exactly.

Usage
-----
    from src.environment.custom_reward import CustomLunarLander
    env = CustomLunarLander(
        reward_coefs={"angle_coef": 200.0, "vel_coef": 50.0},
        enable_wind=True, wind_power=15.0,
    )
"""

from __future__ import annotations

import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander


# Default coefficients — identical to the gymnasium source
DEFAULT_COEFS: dict = {
    "pos_coef":         100.0,   # distance from landing pad
    "vel_coef":         100.0,   # total velocity magnitude
    "angle_coef":       100.0,   # absolute tilt angle
    "leg_reward":        10.0,   # reward per leg in ground contact
    "main_engine_coef":   0.30,  # penalty per frame with main engine on
    "side_engine_coef":   0.03,  # penalty per frame with side engines on
    "crash_penalty":    100.0,   # terminal penalty for crash / out-of-bounds
    "land_bonus":       100.0,   # terminal bonus for safe landing
}


class CustomLunarLander(LunarLander):
    """
    LunarLander-v3 with configurable reward coefficients.

    Parameters
    ----------
    reward_coefs : dict, optional
        Subset of DEFAULT_COEFS to override.  Unspecified keys keep
        their default value.  Pass an empty dict or None for vanilla env.
    **kwargs
        Forwarded verbatim to LunarLander (enable_wind, gravity,
        render_mode, …).
    """

    def __init__(self, reward_coefs: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        c = {**DEFAULT_COEFS, **(reward_coefs or {})}
        self._c_pos   = float(c["pos_coef"])
        self._c_vel   = float(c["vel_coef"])
        self._c_angle = float(c["angle_coef"])
        self._c_leg   = float(c["leg_reward"])
        self._c_main  = float(c["main_engine_coef"])
        self._c_side  = float(c["side_engine_coef"])
        self._c_crash = float(c["crash_penalty"])
        self._c_land  = float(c["land_bonus"])
        self._custom_prev_shaping: float | None = None

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        self._custom_prev_shaping = None
        return super().reset(**kwargs)

    def step(self, action):
        # Run the original step — we discard its reward and recompute ours
        obs, _raw_reward, terminated, truncated, info = super().step(action)

        # obs == state vector:
        #   [x, y, vx, vy, angle, angular_vel, leg_L, leg_R]
        x, y, vx, vy, angle, _angvel, leg_l, leg_r = obs

        # Custom potential-based shaping
        shaping = (
            - self._c_pos   * np.sqrt(x * x + y * y)
            - self._c_vel   * np.sqrt(vx * vx + vy * vy)
            - self._c_angle * abs(angle)
            + self._c_leg   * leg_l
            + self._c_leg   * leg_r
        )

        if self._custom_prev_shaping is not None:
            reward = float(shaping - self._custom_prev_shaping)
        else:
            reward = 0.0
        self._custom_prev_shaping = float(shaping)

        # Engine penalties — discrete action space
        #   action 2 → main engine full power  (m_power = 1.0)
        #   action 1, 3 → side engine           (s_power = 1.0)
        reward -= self._c_main * (1.0 if action == 2 else 0.0)
        reward -= self._c_side * (1.0 if action in (1, 3) else 0.0)

        # Terminal override
        if terminated:
            if self.game_over or abs(x) >= 1.0:   # crash / out-of-bounds
                reward = -self._c_crash
            elif not self.lander.awake:            # safe landing
                reward = +self._c_land

        return obs, reward, terminated, truncated, info
