"""
Deep SARSA Agent for LunarLander-v3.

Algorithm
---------
SARSA (State-Action-Reward-State-Action) is a model-free, on-policy
TD control method.  The update rule is:

    Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]

where a' is sampled from the *current policy* (ε-greedy), making the
method fundamentally on-policy — unlike Q-Learning / DQN which use
max_a' Q(s', a').

We extend tabular SARSA with:
  1. A neural network for function approximation (Deep SARSA)
  2. A target network (synced every `target_update_freq` episodes)
     to stabilise the bootstrapped target, as in DQN (Mnih et al., 2015)
  3. A short-horizon experience replay buffer (semi-on-policy) to
     de-correlate consecutive updates — transitions are stored and
     replayed within the same behavioural episode

References:
  - Sutton & Barto (2018), "RL: An Introduction", §6.4 (SARSA)
  - Mnih et al. (2015), "Human-level control through deep RL" (DQN)
  - JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3 (base,
    significantly extended: target network, config-driven arch, stats)
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.networks.sarsa_network import SARSANetwork
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import TrainingLogger
from src.environment.lunar_lander_wrapper import make_env


class DeepSARSAAgent:
    """
    Deep SARSA agent.

    Parameters
    ----------
    state_dim      : observation dimension (8 for LunarLander)
    action_dim     : number of discrete actions (4)
    config         : dict loaded from sarsa_config.yaml
    device         : torch device
    seed           : RNG seed (for reproducibility)
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        config:     Dict,
        device:     torch.device = torch.device("cpu"),
        seed:       int = 42,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config
        self.device     = device
        self.seed       = seed

        # Hyperparameters
        agent_cfg  = config.get("agent", {})
        net_cfg    = config.get("network", {})

        self.alpha              = agent_cfg.get("alpha", 5e-4)
        self.gamma              = agent_cfg.get("gamma", 0.99)
        self.epsilon            = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_end        = agent_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay      = agent_cfg.get("epsilon_decay", 0.995)
        self.target_update_freq = agent_cfg.get("target_update_freq", 10)
        self.batch_size         = agent_cfg.get("batch_size", 64)

        hidden     = net_cfg.get("hidden_layers", [256, 256])
        activation = net_cfg.get("activation", "relu")
        dropout    = net_cfg.get("dropout", 0.0)

        # Networks
        self.q_network = SARSANetwork(
            state_dim, action_dim, hidden, activation, dropout
        ).to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.target_network.eval()

        self.alpha_end = agent_cfg.get("alpha_end", 1e-5)  # lr floor for cosine schedule
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        # Cosine annealing: lr decays smoothly from alpha to alpha_end over n_episodes
        _n_episodes = config.get("training", {}).get("n_episodes", 3000)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=_n_episodes, eta_min=self.alpha_end
        )

        # Replay buffer (short-horizon, semi-on-policy)
        self.replay_buffer = ReplayBuffer(capacity=agent_cfg.get("buffer_capacity", 50_000))

        # Tracking
        self._episode = 0
        self._inference_times: List[float] = []

        # RNG
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        t0 = time.perf_counter()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        action = int(q_values.argmax(dim=1).item())
        self._inference_times.append(time.perf_counter() - t0)
        return action

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state:       np.ndarray,
        action:      int,
        reward:      float,
        next_state:  np.ndarray,
        next_action: int,
        done:        bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, next_action, done)

    def update(self) -> Optional[float]:
        """
        Sample a batch and perform one SARSA gradient step.

        SARSA target: r + γ Q_target(s', a')     [on-policy: a' from ε-greedy]
        Loss:         MSE( Q(s,a), target )
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, next_actions, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)

        # Current Q-values Q(s, a)
        q_values = self.q_network(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA target: r + γ Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_network(next_states)
            q_sa_prime = q_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * q_sa_prime * (1 - dones)

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        log_dir:  str = "results/logs",
        save_dir: str = "models",
        verbose:  bool = True,
    ) -> List[float]:
        """
        Full training loop for one seed.

        Returns
        -------
        episode_rewards : list of undiscounted returns per episode
        """
        train_cfg = self.config.get("training", {})
        n_episodes  = train_cfg.get("n_episodes", 2000)
        max_steps   = train_cfg.get("max_steps", 1000)
        save_every  = train_cfg.get("save_every", 200)
        log_every   = train_cfg.get("log_every", 10)

        env_cfg  = self.config.get("environment", {})
        env = make_env(
            env_name  = env_cfg.get("name", "LunarLander-v3"),
            normalize = env_cfg.get("normalize_obs", True),
            seed      = self.seed,
        )

        logger = TrainingLogger(log_dir, agent_name="sarsa", seed=self.seed)
        episode_rewards: List[float] = []
        best_mean   = -float("inf")
        best_path   = os.path.join(save_dir, f"sarsa_seed{self.seed}_best.pt")

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            action    = self.select_action(state)
            ep_reward = 0.0
            ep_loss   = []

            for _ in range(max_steps):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done        = terminated or truncated
                next_action = self.select_action(next_state)

                self.store_transition(state, action, reward, next_state, next_action, float(done))
                loss = self.update()
                if loss is not None:
                    ep_loss.append(loss)

                state      = next_state
                action     = next_action
                ep_reward += reward

                if done:
                    break

            self.decay_epsilon()
            if ep_loss:  # step scheduler only after at least one optimizer step
                self.scheduler.step()
            if ep % self.target_update_freq == 0:
                self.sync_target_network()

            episode_rewards.append(ep_reward)
            self._episode = ep

            if ep % log_every == 0:
                last_n = episode_rewards[-log_every:]
                logger.log(ep, {
                    "episode_reward": ep_reward,
                    "mean_reward_last_n": float(np.mean(last_n)),
                    "epsilon": round(self.epsilon, 4),
                    "mean_loss": float(np.mean(ep_loss)) if ep_loss else 0.0,
                    "buffer_size": len(self.replay_buffer),
                    "lr": round(self.optimizer.param_groups[0]["lr"], 6),
                }, verbose=verbose and (ep % 100 == 0))

            if ep % save_every == 0:
                self.save(os.path.join(save_dir, f"sarsa_seed{self.seed}_ep{ep}.pt"), env=env)

            # ── Best checkpoint ───────────────────────────────────────────
            if len(episode_rewards) >= 100:
                current_mean = float(np.mean(episode_rewards[-100:]))
                if current_mean > best_mean:
                    best_mean = current_mean
                    self.save(best_path, env=env)
                    if verbose:
                        print(f"  [best] ep={ep}  mean100={best_mean:.2f}  → saved")

        logger.close()
        env.close()
        return episode_rewards

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        env_variant: str = "standard",
        n_episodes:  int = 20,
        seed:        int = 42,
        warmup_episodes: int = 200,
    ) -> Dict:
        """
        Greedy evaluation on a given environment variant.
        Returns mean, std, and per-episode rewards.
        
        warmup_episodes: episodes run first (not recorded) to warm up the normalizer.
        """
        env_cfg = self.config.get("environment", {})
        normalize = env_cfg.get("normalize_obs", True)
        env = make_env(
            env_name  = env_cfg.get("name", "LunarLander-v3"),
            normalize = normalize,
            variant   = env_variant,
            seed      = seed,
        )
        # Restore normalizer state if available (saved from training)
        if normalize and hasattr(self, "_norm_mean") and self._norm_mean is not None:
            env._normalizer.mean  = self._norm_mean.copy()
            env._normalizer.var   = self._norm_var.copy()
            env._normalizer.count = self._norm_count
            env.freeze_normalizer()   # lock stats — prevent drift during evaluation
        elif normalize and warmup_episodes > 0:
            # Warm up normalizer for warmup_episodes to approximate training stats
            for _ in range(warmup_episodes):
                s, _ = env.reset()
                done = False
                while not done:
                    a = self.select_action(s, greedy=True)
                    s, _, term, trunc, _ = env.step(a)
                    done = term or trunc
            env.freeze_normalizer()   # lock warmed-up stats for the actual eval episodes
        rewards = []
        inf_times = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                t0 = time.perf_counter()
                action = self.select_action(state, greedy=True)
                inf_times.append(time.perf_counter() - t0)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
            rewards.append(ep_reward)

        env.close()
        return {
            "variant":        env_variant,
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "min_reward":     float(np.min(rewards)),
            "max_reward":     float(np.max(rewards)),
            "rewards":        rewards,
            "mean_inf_time":  float(np.mean(inf_times)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, env=None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        norm_mean = env._normalizer.mean.copy() if env is not None else getattr(self, "_norm_mean", None)
        norm_var  = env._normalizer.var.copy()  if env is not None else getattr(self, "_norm_var",  None)
        norm_count = env._normalizer.count      if env is not None else getattr(self, "_norm_count", None)
        torch.save({
            "q_network":     self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "episode":       self._episode,
            "config":        self.config,
            "seed":          self.seed,
            "norm_mean":     norm_mean,
            "norm_var":      norm_var,
            "norm_count":    norm_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon  = ckpt["epsilon"]
        self._episode = ckpt["episode"]
        self._norm_mean  = ckpt.get("norm_mean")
        self._norm_var   = ckpt.get("norm_var")
        self._norm_count = ckpt.get("norm_count")

    @property
    def inference_times(self) -> List[float]:
        return self._inference_times
