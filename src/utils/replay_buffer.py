"""
Replay buffer for Deep SARSA.

Note on on-policy vs off-policy
--------------------------------
Classic tabular SARSA is strictly on-policy (it uses the *next* action
chosen by the *current* policy in the update target). When approximating
with a neural network we face the moving-target problem, and a small
*short-horizon* replay buffer (storing only recent transitions) is used
to improve sample efficiency while staying approximately on-policy.

This buffer stores (s, a, r, s', a', done) tuples — the extra 'a''
field is what distinguishes a SARSA buffer from a standard DQN buffer.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size circular replay buffer for (s, a, r, s', a', done).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to keep.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        next_action: int,
        done:       bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.LongTensor(next_actions).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """True once the buffer holds at least one full batch (set externally)."""
        return len(self.buffer) > 0
