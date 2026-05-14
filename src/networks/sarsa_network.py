"""
Neural network for Deep SARSA.

Approximates Q(s, a) for all discrete actions simultaneously
(one forward pass → Q-values for all actions).

Architecture: MLP with configurable hidden layers and activation.
A target network (identical structure, lagged parameters) is used
to stabilise the Bellman target — a technique borrowed from DQN
(Mnih et al., 2015) that is equally beneficial for Deep SARSA.

References:
  - Mnih et al. (2015), "Human-level control through deep reinforcement learning"
  - JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3 (base structure)
"""

import torch
import torch.nn as nn
from typing import List


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SARSANetwork(nn.Module):
    """
    Q-network for Deep SARSA.

    Parameters
    ----------
    state_dim  : dimension of the observation vector (8 for LunarLander)
    action_dim : number of discrete actions (4 for LunarLander)
    hidden_layers : list of hidden-layer widths, e.g. [256, 256]
    activation : nonlinearity name ('relu', 'tanh', 'elu')
    dropout : dropout probability (0 = disabled)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = _build_mlp(
            state_dim, action_dim, list(hidden_layers), activation, dropout
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Tensor of shape (batch, state_dim) or (state_dim,)

        Returns
        -------
        q_values : Tensor of shape (batch, action_dim)
        """
        return self.net(state)
