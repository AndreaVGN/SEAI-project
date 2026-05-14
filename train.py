"""
train.py — Multi-seed training script for Deep SARSA and A2C.

Usage
-----
# Train both agents with all seeds defined in the config files:
    python train.py

# Train only one agent:
    python train.py --agent sarsa
    python train.py --agent ac

# Override number of episodes (quick smoke-test):
    python train.py --episodes 100 --seeds 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
import torch
import numpy as np

from src.agents.deep_sarsa   import DeepSARSAAgent
from src.agents.actor_critic  import ActorCriticAgent
from src.environment.lunar_lander_wrapper import make_env


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_env_dims(env_name: str = "LunarLander-v3"):
    """Instantiate a temporary env to read state/action dims."""
    import gymnasium as gym
    env = gym.make(env_name)
    s   = env.observation_space.shape[0]
    a   = env.action_space.n
    env.close()
    return s, a


def train_agent(agent_type: str, config: dict, seeds: list[int], device: torch.device):
    env_name  = config["environment"]["name"]
    state_dim, action_dim = get_env_dims(env_name)

    all_rewards = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Training {agent_type.upper()} | seed={seed}")
        print(f"{'='*60}")

        if agent_type == "sarsa":
            agent = DeepSARSAAgent(state_dim, action_dim, config, device, seed=seed)
        else:
            agent = ActorCriticAgent(state_dim, action_dim, config, device, seed=seed)

        t0      = time.time()
        rewards = agent.train(
            log_dir  = "results/logs",
            save_dir = "models",
            verbose  = True,
        )
        elapsed = time.time() - t0

        print(f"\n  Done in {elapsed:.1f}s | "
              f"Final mean (last 100 ep): {np.mean(rewards[-100:]):.2f}")
        all_rewards.append(rewards)

    return all_rewards


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on LunarLander-v3")
    parser.add_argument("--agent",    choices=["sarsa", "ac", "both"], default="both")
    parser.add_argument("--episodes", type=int,  default=None, help="Override n_episodes")
    parser.add_argument("--seeds",    type=int,  nargs="+",    default=None)
    parser.add_argument("--device",   default="auto")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    sarsa_cfg = load_config("config/sarsa_config.yaml")
    ac_cfg    = load_config("config/actor_critic_config.yaml")

    # CLI overrides
    if args.episodes:
        sarsa_cfg["training"]["n_episodes"] = args.episodes
        ac_cfg["training"]["n_episodes"]    = args.episodes
    seeds_sarsa = args.seeds or sarsa_cfg["training"]["seeds"]
    seeds_ac    = args.seeds or ac_cfg["training"]["seeds"]

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("models",       exist_ok=True)

    if args.agent in ("sarsa", "both"):
        train_agent("sarsa", sarsa_cfg, seeds_sarsa, device)

    if args.agent in ("ac", "both"):
        train_agent("ac", ac_cfg, seeds_ac, device)

    print("\n[train.py] All training runs complete.")
    print("Run python compare.py to generate comparison plots and statistics.")


if __name__ == "__main__":
    main()
