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

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.networks.actor_critic_network import ActorNetwork, CriticNetwork
from src.utils.logger import TrainingLogger
from src.environment.lunar_lander_wrapper import make_env, make_vec_env, VecRunningNormalizer


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
        self.num_envs      = agent_cfg.get("num_envs", 1)   # >1 = vectorised training

        hidden     = net_cfg.get("hidden_layers", [256, 256])
        activation = net_cfg.get("activation", "relu")
        dropout    = net_cfg.get("dropout", 0.0)

        self.actor  = ActorNetwork(state_dim, action_dim, hidden, activation, dropout).to(device)
        self.critic = CriticNetwork(state_dim, hidden, activation, dropout).to(device)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Cosine annealing: lr decade da actor_lr/critic_lr a lr_end in n_episodes
        _n_ep = config.get("training", {}).get("n_episodes", 3000)
        _lr_end = agent_cfg.get("lr_end", 1e-5)
        self.actor_scheduler  = lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,  T_max=_n_ep, eta_min=_lr_end
        )
        self.critic_scheduler = lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=_n_ep, eta_min=_lr_end
        )

        self._episode = 0
        self._inference_times: List[float] = []

        # Reward coefficient overrides for CustomLunarLander (None = use gymnasium defaults)
        coef_cfg = config.get("reward_coefficients", {})
        if coef_cfg.get("enabled", False):
            self.reward_coefs = {k: v for k, v in coef_cfg.items() if k != "enabled"}
        else:
            self.reward_coefs = None

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

    def _step_schedulers(self) -> None:
        """Advance cosine LR schedulers — call once per virtual episode."""
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Vectorised helpers (used when num_envs > 1)
    # ------------------------------------------------------------------

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Sample one action per environment.

        Parameters
        ----------
        states : (N, state_dim)

        Returns
        -------
        actions : np.ndarray (N,)
        """
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            dist = self.actor(states_t)
            actions = dist.sample()
        return actions.cpu().numpy()

    def _compute_returns_vec(
        self,
        rewards:     np.ndarray,   # (T, N)
        dones:       np.ndarray,   # (T, N)  float, 1.0 = terminal
        last_values: np.ndarray,   # (N,)
    ) -> np.ndarray:
        """
        Vectorised n-step discounted returns.

        Episode boundaries (done=1) correctly zero out cross-episode
        bootstrap: G[t,i] = r[t,i] + γ·G[t+1,i]·(1−done[t,i])
        This means a terminal step at t contributes only its own reward
        to the return of steps before t — no contamination from the
        next episode's rewards collected in the same segment.

        Returns
        -------
        returns : np.ndarray (T, N)
        """
        T, N = rewards.shape
        returns = np.zeros((T, N), dtype=np.float32)
        G = last_values.copy()                        # (N,)
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G * (1.0 - dones[t])
            returns[t] = G
        return returns

    def update_vec(
        self,
        states:      np.ndarray,   # (T, N, state_dim)
        actions:     np.ndarray,   # (T, N)
        rewards:     np.ndarray,   # (T, N)
        dones:       np.ndarray,   # (T, N)  float
        last_states: np.ndarray,   # (N, state_dim)
        last_dones:  np.ndarray,   # (N,)    float
    ) -> Dict[str, float]:
        """
        One gradient update over T×N transitions from N parallel envs.
        """
        T, N, state_dim = states.shape

        # Bootstrap value at the end of each env's segment
        with torch.no_grad():
            last_t = torch.FloatTensor(last_states).to(self.device)
            last_values = self.critic(last_t).squeeze(-1).cpu().numpy()  # (N,)
            last_values *= (1.0 - last_dones)   # zero out terminated envs

        # Returns (T, N)
        returns = self._compute_returns_vec(rewards, dones, last_values)

        # Flatten → (T*N, ...)
        states_flat  = states.reshape(-1, state_dim)
        actions_flat = actions.reshape(-1)
        returns_flat = returns.reshape(-1)

        states_t  = torch.FloatTensor(states_flat).to(self.device)
        actions_t = torch.LongTensor(actions_flat).to(self.device)
        returns_t = torch.FloatTensor(returns_flat).to(self.device)

        # Critic values V(s) for all (T*N) states
        values = self.critic(states_t).squeeze(-1)          # (T*N,)

        # Normalise returns for stable critic targets
        returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Advantages A = G - V (normalised for stable actor gradient)
        advantages = returns_t - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss: REINFORCE with advantage baseline + entropy bonus
        log_probs, entropy = self.actor.evaluate(states_t, actions_t)
        actor_loss  = -(log_probs * advantages).mean() - self.entropy_coef * entropy.mean()

        # Critic loss: MSE vs normalised returns
        critic_loss = nn.MSELoss()(values, returns_norm)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropy.mean().item(),
        }

    def train(
        self,
        log_dir:  str = "results/logs",
        save_dir: str = "models",
        verbose:  bool = True,
        tag:      str = "",
    ) -> List[float]:
        """
        Full training loop for one seed.

        Returns
        -------
        episode_rewards : list of undiscounted returns per episode
        """
        train_cfg  = self.config.get("training", {})
        n_episodes = train_cfg.get("n_episodes", 2000)
        max_steps  = train_cfg.get("max_steps", 1000)
        save_every = train_cfg.get("save_every", 200)
        log_every  = train_cfg.get("log_every", 10)
        env_cfg    = self.config.get("environment", {})
        env_name   = env_cfg.get("name", "LunarLander-v3")
        normalize  = env_cfg.get("normalize_obs", True)

        if self.num_envs > 1:
            return self._train_vec(
                env_name, normalize, n_episodes, max_steps,
                save_every, log_every, log_dir, save_dir, verbose, tag,
            )

        # ── Single-environment path (original) ────────────────────────
        env = make_env(env_name=env_name, normalize=normalize, seed=self.seed, reward_coefs=self.reward_coefs)

        run_name    = f"{tag}ac_seed{self.seed}"
        run_log_dir = os.path.join(log_dir, run_name)
        logger = TrainingLogger(
            run_log_dir, agent_name="ac", seed=self.seed,
            run_info={
                "run_name":   run_name,
                "agent":      "ac",
                "tag":        tag,
                "n_steps":    self.n_steps,
                "num_envs":   self.num_envs,
                "config":     self.config,
                "start_time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        episode_rewards: List[float] = []
        total_steps = 0
        best_mean = -float("inf")
        best_path = os.path.join(save_dir, f"{tag}ac_seed{self.seed}_best.pt")

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            ep_reward  = 0.0
            ep_metrics = {"actor_loss": [], "critic_loss": [], "entropy": []}

            seg_states, seg_actions, seg_rewards, seg_dones = [], [], [], []
            done = False

            for step in range(max_steps):
                action, _ = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_steps += 1
                done = terminated or truncated

                seg_states.append(state)
                seg_actions.append(action)
                seg_rewards.append(reward)
                seg_dones.append(done)
                ep_reward += reward
                state = next_state

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
            self._step_schedulers()  # cosine lr decay — un passo per episodio

            if ep % log_every == 0:
                last_n = episode_rewards[-log_every:]
                logger.log(ep, {
                    "episode_reward":     ep_reward,
                    "mean_reward_last_n": float(np.mean(last_n)),
                    "actor_loss":  float(np.mean(ep_metrics["actor_loss"]))  if ep_metrics["actor_loss"]  else 0.0,
                    "critic_loss": float(np.mean(ep_metrics["critic_loss"])) if ep_metrics["critic_loss"] else 0.0,
                    "entropy":     float(np.mean(ep_metrics["entropy"]))     if ep_metrics["entropy"]     else 0.0,
                }, verbose=verbose and (ep % 100 == 0))

            if ep % save_every == 0:
                self.save(os.path.join(save_dir, f"{tag}ac_seed{self.seed}_ep{ep}.pt"), env=env)

            # ── Best checkpoint ───────────────────────────────────────────
            if len(episode_rewards) >= 100:
                current_mean = float(np.mean(episode_rewards[-100:]))
                if current_mean > best_mean:
                    best_mean = current_mean
                    self.save(best_path, env=env)
                    if verbose:
                        print(f"  [best] ep={ep}  mean100={best_mean:.2f}  → saved")

        count_step_path = os.path.join(run_log_dir, "count_step.json")
        with open(count_step_path, "w") as f:
            json.dump({
                "agent":          "ac",
                "seed":           self.seed,
                "tag":            tag,
                "run_name":       run_name,
                "total_episodes": len(episode_rewards),
                "count_step":     total_steps,
            }, f, indent=2)

        logger.close()
        env.close()
        return episode_rewards

    def _train_vec(
        self,
        env_name:   str,
        normalize:  bool,
        n_episodes: int,
        max_steps:  int,
        save_every: int,
        log_every:  int,
        log_dir:    str,
        save_dir:   str,
        verbose:    bool,
        tag:        str = "",
    ) -> List[float]:
        """
        Vectorised training loop for num_envs > 1.

        Runs self.num_envs environments in parallel; each outer iteration
        collects n_steps transitions from every env (total: n_steps × N
        per gradient update). Episodes complete asynchronously via the
        gym.vector auto-reset mechanism.
        """
        N = self.num_envs
        vec_env = make_vec_env(env_name=env_name, num_envs=N, seed=self.seed)
        normalizer = VecRunningNormalizer(
            shape=(vec_env.single_observation_space.shape[0],)
        ) if normalize else None

        run_name    = f"{tag}ac_e{N}s{self.n_steps}_seed{self.seed}"
        run_log_dir = os.path.join(log_dir, run_name)
        logger = TrainingLogger(
            run_log_dir, agent_name="ac", seed=self.seed,
            run_info={
                "run_name":   run_name,
                "agent":      "ac",
                "tag":        tag,
                "n_steps":    self.n_steps,
                "num_envs":   N,
                "config":     self.config,
                "start_time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        episode_rewards: List[float] = []
        total_steps = 0

        # Per-env running reward accumulators
        ep_rewards_running = np.zeros(N, dtype=np.float32)
        ep_metrics_buf     = {"actor_loss": [], "critic_loss": [], "entropy": []}
        best_mean_vec = -float("inf")
        best_path_vec = os.path.join(save_dir, f"{tag}ac_seed{self.seed}_best.pt")

        # Initialise environments
        obs_batch, _ = vec_env.reset()
        if normalizer is not None:
            states = normalizer.update_and_normalize(obs_batch)
        else:
            states = obs_batch.astype(np.float32)

        ep_count = 0  # completed episodes so far

        while ep_count < n_episodes * N:  # n_episodes = target per singolo env
            # ── Collect n_steps from all N envs ───────────────────────
            seg_states  = np.zeros((self.n_steps, N, states.shape[1]), dtype=np.float32)
            seg_actions = np.zeros((self.n_steps, N), dtype=np.int64)
            seg_rewards = np.zeros((self.n_steps, N), dtype=np.float32)
            seg_dones   = np.zeros((self.n_steps, N), dtype=np.float32)

            for t in range(self.n_steps):
                actions = self.select_action_batch(states)       # (N,)
                obs_batch, rewards, terminated, truncated, _ = vec_env.step(actions)
                total_steps += N   # N parallel env interactions per vec_env.step()
                dones = (terminated | truncated).astype(np.float32)

                seg_states[t]  = states
                seg_actions[t] = actions
                seg_rewards[t] = rewards
                seg_dones[t]   = dones

                ep_rewards_running += rewards

                # Record completed episodes
                for i in range(N):
                    if dones[i]:
                        episode_rewards.append(float(ep_rewards_running[i]))
                        ep_rewards_running[i] = 0.0
                        ep_count += 1

                if normalizer is not None:
                    states = normalizer.update_and_normalize(obs_batch)
                else:
                    states = obs_batch.astype(np.float32)

            # ── Gradient update ────────────────────────────────────────
            last_dones = seg_dones[-1]   # (N,) — done flags at end of segment
            metrics = self.update_vec(
                seg_states, seg_actions, seg_rewards, seg_dones,
                last_states=states, last_dones=last_dones,
            )
            for k in ep_metrics_buf:
                ep_metrics_buf[k].append(metrics[k])

            self._episode = ep_count
            virtual_ep = ep_count // N  # episodio equivalente per singolo env
            self._step_schedulers()      # cosine lr decay — un passo per virtual episode

            # ── Logging ────────────────────────────────────────────────
            if len(episode_rewards) >= log_every and len(episode_rewards) % log_every == 0:
                last_n = episode_rewards[-log_every:]
                logger.log(virtual_ep, {
                    "episode_reward":     episode_rewards[-1],
                    "mean_reward_last_n": float(np.mean(last_n)),
                    "actor_loss":  float(np.mean(ep_metrics_buf["actor_loss"]))  if ep_metrics_buf["actor_loss"]  else 0.0,
                    "critic_loss": float(np.mean(ep_metrics_buf["critic_loss"])) if ep_metrics_buf["critic_loss"] else 0.0,
                    "entropy":     float(np.mean(ep_metrics_buf["entropy"]))     if ep_metrics_buf["entropy"]     else 0.0,
                }, verbose=verbose and (virtual_ep % 100 == 0))
                ep_metrics_buf = {"actor_loss": [], "critic_loss": [], "entropy": []}

            # ── Checkpoint ─────────────────────────────────────────────
            if virtual_ep > 0 and virtual_ep % save_every == 0:
                self._vec_normalizer = normalizer
                self.save(os.path.join(save_dir, f"{tag}ac_seed{self.seed}_ep{virtual_ep}.pt"))

            # ── Best checkpoint ───────────────────────────────────────────
            if len(episode_rewards) >= 100:
                current_mean = float(np.mean(episode_rewards[-100:]))
                if current_mean > best_mean_vec:
                    best_mean_vec = current_mean
                    self._vec_normalizer = normalizer
                    self.save(best_path_vec)
                    if verbose:
                        print(f"  [best] ep~{virtual_ep}  mean100={best_mean_vec:.2f}  → saved")

        count_step_path = os.path.join(run_log_dir, "count_step.json")
        with open(count_step_path, "w") as f:
            json.dump({
                "agent":          "ac",
                "seed":           self.seed,
                "tag":            tag,
                "run_name":       run_name,
                "num_envs":       N,
                "n_steps":        self.n_steps,
                "total_episodes": len(episode_rewards),
                "count_step":     total_steps,
            }, f, indent=2)

        logger.close()
        vec_env.close()
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
            env_name       = env_cfg.get("name", "LunarLander-v3"),
            normalize      = normalize,
            variant        = env_variant,
            seed           = seed,
            reward_coefs   = self.reward_coefs,
        )
        # Restore normalizer state if available
        if normalize and hasattr(self, "_norm_mean") and self._norm_mean is not None:
            env._normalizer.mean  = self._norm_mean.copy()
            env._normalizer.var   = self._norm_var.copy()
            env._normalizer.count = self._norm_count
            env.freeze_normalizer()   # lock stats — prevent drift during evaluation
        elif normalize and warmup_episodes > 0:
            for _ in range(warmup_episodes):
                s, _ = env.reset()
                done = False
                while not done:
                    a, _ = self.select_action(s, greedy=True)
                    s, _, term, trunc, _ = env.step(a)
                    done = term or trunc
            env.freeze_normalizer()   # lock warmed-up stats for the actual eval episodes
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
        # Priority: single env > vec normalizer > cached stats from load()
        if env is not None:
            norm_mean  = env._normalizer.mean.copy()
            norm_var   = env._normalizer.var.copy()
            norm_count = env._normalizer.count
        elif hasattr(self, "_vec_normalizer") and self._vec_normalizer is not None:
            norm_mean  = self._vec_normalizer.mean   # VecRunningNormalizer returns copies
            norm_var   = self._vec_normalizer.var
            norm_count = self._vec_normalizer.count
        else:
            norm_mean  = getattr(self, "_norm_mean",  None)
            norm_var   = getattr(self, "_norm_var",   None)
            norm_count = getattr(self, "_norm_count", None)
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
