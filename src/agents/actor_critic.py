"""
Advantage Actor-Critic (A2C) Agent for LunarLander-v3.

Algorithm
---------
A2C is a synchronous policy-gradient method.  At each step:

  1. Actor selects action a ~ π(· | s; θ)
  2. Critic estimates V(s; w)
  3. After n steps (n-step return), compute:
        G_t = r_t + γ r_{t+1} + … + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n}; w)
  4. Advantage:  A_t = G_t - V(s_t; w)
  5. Actor  loss: -log π(a_t | s_t; θ) · A_t - β H[π(· | s_t; θ)]
  6. Critic loss: (G_t - V(s_t; w))²

The entropy bonus β H[π] prevents premature convergence to deterministic
policies, which is crucial in the stochastic Lunar Lander environment.

Design vs Deep SARSA
--------------------
* On-policy (no replay buffer needed)
* Policy gradient vs value-based — fundamentally different credit assignment
* Handles exploration implicitly via entropy, rather than ε-greedy

References:
  - Mnih et al. (2016), "Asynchronous Methods for Deep RL" (A3C)
  - Sutton & Barto (2018), "RL: An Introduction", §13.5
  - nikhilbarhate99/Actor-Critic-PyTorch (base, extended: n-step returns,
    entropy scheduling, multi-seed, generalisation evaluation)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks.actor_critic_network import ActorNetwork, CriticNetwork
from src.utils.logger import TrainingLogger
from src.environment.lunar_lander_wrapper import make_env


class ActorCriticAgent:
    """
    Advantage Actor-Critic (A2C) agent.

    Parameters
    ----------
    state_dim  : observation dimension (8 for LunarLander)
    action_dim : number of discrete actions (4)
    config     : dict loaded from actor_critic_config.yaml
    device     : torch device
    seed       : RNG seed
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

        agent_cfg = config.get("agent", {})
        net_cfg   = config.get("network", {})

        self.actor_lr      = agent_cfg.get("actor_lr", 3e-4)
        self.critic_lr     = agent_cfg.get("critic_lr", 1e-3)
        self.gamma         = agent_cfg.get("gamma", 0.99)
        self.entropy_coef  = agent_cfg.get("entropy_coef", 0.01)
        self.n_steps       = agent_cfg.get("n_steps", 5)

        hidden     = net_cfg.get("hidden_layers", [256, 256])
        activation = net_cfg.get("activation", "relu")
        dropout    = net_cfg.get("dropout", 0.0)

        self.actor  = ActorNetwork(state_dim, action_dim, hidden, activation, dropout).to(device)
        self.critic = CriticNetwork(state_dim, hidden, activation, dropout).to(device)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self._episode = 0
        self._inference_times: List[float] = []

        torch.manual_seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, state: np.ndarray, greedy: bool = False
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Sample from π(·|s) or return argmax for greedy evaluation.

        Returns
        -------
        action   : int
        log_prob : Tensor (None in greedy mode)
        """
        t0 = time.perf_counter()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if greedy:
            with torch.no_grad():
                dist   = self.actor(state_t)
                action = int(dist.probs.argmax(dim=1).item())
            self._inference_times.append(time.perf_counter() - t0)
            return action, None

        dist     = self.actor(state_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        self._inference_times.append(time.perf_counter() - t0)
        return int(action_t.item()), log_prob

    # ------------------------------------------------------------------
    # n-step return computation
    # ------------------------------------------------------------------

    def _compute_returns(
        self,
        rewards:    List[float],
        values:     List[torch.Tensor],
        dones:      List[bool],
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute n-step discounted returns G_t for a trajectory segment.

        G_t = r_t + γ r_{t+1} + … + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n})
        """
        returns = []
        G = last_value.detach()
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1.0 - float(done))
            returns.insert(0, G)
        return torch.stack(returns).squeeze(-1)  # (n_steps, 1)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        states:     List[np.ndarray],
        actions:    List[int],
        rewards:    List[float],
        dones:      List[bool],
        last_state: np.ndarray,
        last_done:  bool,
    ) -> Dict[str, float]:
        """
        One gradient update over a trajectory segment of length n_steps.

        Returns dict with actor_loss, critic_loss, entropy.
        """
        states_t  = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)

        # Critic values for current states
        values = self.critic(states_t)          # (n, 1)

        # Bootstrap value at last state
        with torch.no_grad():
            if last_done:
                last_value = torch.zeros(1, 1, device=self.device)
            else:
                last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                last_value   = self.critic(last_state_t)

        returns = self._compute_returns(rewards, values, dones, last_value)  # (n, 1)

        # Detach returns once; normalise for stable critic targets
        returns_det = returns.detach()
        if returns_det.numel() > 1:
            returns_norm = (returns_det - returns_det.mean()) / (returns_det.std() + 1e-8)
        else:
            returns_norm = returns_det

        # Advantages
        advantages = (returns_det - values.detach())
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss: REINFORCE with baseline (advantage)
        log_probs, entropy = self.actor.evaluate(states_t, actions_t)
        actor_loss = -(log_probs * advantages.squeeze()).mean() \
                     - self.entropy_coef * entropy.mean()

        # Critic loss: MSE vs normalised n-step returns
        critic_loss = nn.MSELoss()(values, returns_norm)

        # Optimise actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Optimise critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropy.mean().item(),
        }

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
        n_episodes = train_cfg.get("n_episodes", 2000)
        max_steps  = train_cfg.get("max_steps", 1000)
        save_every = train_cfg.get("save_every", 200)
        log_every  = train_cfg.get("log_every", 10)

        env_cfg = self.config.get("environment", {})
        env = make_env(
            env_name  = env_cfg.get("name", "LunarLander-v3"),
            normalize = env_cfg.get("normalize_obs", True),
            seed      = self.seed,
        )

        logger = TrainingLogger(log_dir, agent_name="ac", seed=self.seed)
        episode_rewards: List[float] = []

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            ep_reward  = 0.0
            ep_metrics = {"actor_loss": [], "critic_loss": [], "entropy": []}

            # Collect n-step segments within the episode
            seg_states, seg_actions, seg_rewards, seg_dones = [], [], [], []
            done = False

            for step in range(max_steps):
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                seg_states.append(state)
                seg_actions.append(action)
                seg_rewards.append(reward)
                seg_dones.append(done)
                ep_reward += reward
                state = next_state

                # Update every n_steps OR at episode end
                if len(seg_states) == self.n_steps or done:
                    metrics = self.update(
                        seg_states, seg_actions, seg_rewards, seg_dones,
                        last_state=next_state, last_done=done
                    )
                    for k in ep_metrics:
                        ep_metrics[k].append(metrics[k])
                    seg_states, seg_actions, seg_rewards, seg_dones = [], [], [], []

                if done:
                    break

            episode_rewards.append(ep_reward)
            self._episode = ep

            if ep % log_every == 0:
                last_n = episode_rewards[-log_every:]
                logger.log(ep, {
                    "episode_reward":    ep_reward,
                    "mean_reward_last_n": float(np.mean(last_n)),
                    "actor_loss":  float(np.mean(ep_metrics["actor_loss"])) if ep_metrics["actor_loss"] else 0.0,
                    "critic_loss": float(np.mean(ep_metrics["critic_loss"])) if ep_metrics["critic_loss"] else 0.0,
                    "entropy":     float(np.mean(ep_metrics["entropy"])) if ep_metrics["entropy"] else 0.0,
                }, verbose=verbose and (ep % 100 == 0))

            if ep % save_every == 0:
                self.save(os.path.join(save_dir, f"ac_seed{self.seed}_ep{ep}.pt"), env=env)

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
        """Greedy evaluation on a given environment variant."""
        env_cfg = self.config.get("environment", {})
        normalize = env_cfg.get("normalize_obs", True)
        env = make_env(
            env_name  = env_cfg.get("name", "LunarLander-v3"),
            normalize = normalize,
            variant   = env_variant,
            seed      = seed,
        )
        # Restore normalizer state if available
        if normalize and hasattr(self, "_norm_mean") and self._norm_mean is not None:
            env._normalizer.mean  = self._norm_mean.copy()
            env._normalizer.var   = self._norm_var.copy()
            env._normalizer.count = self._norm_count
        elif normalize and warmup_episodes > 0:
            for _ in range(warmup_episodes):
                s, _ = env.reset()
                done = False
                while not done:
                    a, _ = self.select_action(s, greedy=True)
                    s, _, term, trunc, _ = env.step(a)
                    done = term or trunc
        rewards   = []
        inf_times = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                t0 = time.perf_counter()
                action, _ = self.select_action(state, greedy=True)
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
        norm_mean  = env._normalizer.mean.copy() if env is not None else getattr(self, "_norm_mean",  None)
        norm_var   = env._normalizer.var.copy()  if env is not None else getattr(self, "_norm_var",   None)
        norm_count = env._normalizer.count       if env is not None else getattr(self, "_norm_count", None)
        torch.save({
            "actor":           self.actor.state_dict(),
            "critic":          self.critic.state_dict(),
            "actor_optimizer":  self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "episode":         self._episode,
            "config":          self.config,
            "seed":            self.seed,
            "norm_mean":       norm_mean,
            "norm_var":        norm_var,
            "norm_count":      norm_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._episode = ckpt["episode"]
        self._norm_mean  = ckpt.get("norm_mean")
        self._norm_var   = ckpt.get("norm_var")
        self._norm_count = ckpt.get("norm_count")

    @property
    def inference_times(self) -> List[float]:
        return self._inference_times
