"""
Actor and Critic networks for Advantage Actor-Critic (A2C).

Design choices
--------------
* Separate actor and critic networks (separate weights) to allow
  independent learning rates, as recommended for continuous-action or
  high-variance environments (Sutton & Barto, 2018, Ch.13).
* Shared feature extraction is common in A3C/A2C (Mnih et al., 2016),
  but separate networks give more stable updates on LunarLander.
* The actor outputs a **Categorical** distribution over the 4 discrete
  actions; this enables entropy regularisation naturally via
  dist.entropy().

References:
  - Mnih et al. (2016), "Asynchronous Methods for Deep RL" (A3C paper)
  - nikhilbarhate99/Actor-Critic-PyTorch (base structure, extended here)
  - Sutton & Barto (2018), "RL: An Introduction", 2nd ed., Ch.13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple


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


class ActorNetwork(nn.Module):
    """
    Stochastic policy π(a | s; θ).

    Outputs a Categorical distribution (softmax over logits) rather
    than raw action indices, enabling clean log-probability and
    entropy computations for the policy gradient loss.
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
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions.

        Parameters
        ----------
        state : Tensor (batch, state_dim)

        Returns
        -------
        dist : torch.distributions.Categorical
        """
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given state-action pairs."""
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    State-value function V(s; w).

    The critic outputs a scalar estimate of V(s), used to compute the
    advantage A(s, a) = G_t - V(s), which reduces variance in the
    policy gradient without introducing bias (baseline theorem).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_layers: List[int] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = _build_mlp(
            state_dim, 1, list(hidden_layers), activation, dropout
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Tensor (batch, state_dim)

        Returns
        -------
        value : Tensor (batch, 1)
        """
        return self.net(state)
