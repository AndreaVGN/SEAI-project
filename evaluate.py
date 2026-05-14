"""
evaluate.py — Load trained models and evaluate on all environment variants.

Usage
-----
    python evaluate.py --sarsa_ckpt models/sarsa_seed42_ep2000.pt \
                       --ac_ckpt    models/ac_seed42_ep2000.pt
"""

from __future__ import annotations

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path

from src.agents.deep_sarsa   import DeepSARSAAgent
from src.agents.actor_critic  import ActorCriticAgent
from src.environment.lunar_lander_wrapper import ENV_VARIANTS, make_env


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_env_dims(env_name: str = "LunarLander-v3"):
    import gymnasium as gym
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def evaluate_agent(agent, variants: list[str], n_episodes: int = 20) -> dict:
    results = {}
    for variant in variants:
        res = agent.evaluate(env_variant=variant, n_episodes=n_episodes)
        results[variant] = res
        print(f"  [{variant:12s}]  mean={res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarsa_ckpt", type=str, default=None)
    parser.add_argument("--ac_ckpt",    type=str, default=None)
    parser.add_argument("--n_eval",     type=int, default=20)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    sarsa_cfg = load_config("config/sarsa_config.yaml")
    ac_cfg    = load_config("config/actor_critic_config.yaml")
    state_dim, action_dim = get_env_dims()

    variants = list(ENV_VARIANTS.keys())
    all_results = {}

    if args.sarsa_ckpt:
        print(f"\n--- Deep SARSA evaluation [{args.sarsa_ckpt}] ---")
        sarsa = DeepSARSAAgent(state_dim, action_dim, sarsa_cfg, device)
        sarsa.load(args.sarsa_ckpt)
        sarsa.q_network.eval()
        all_results["sarsa"] = evaluate_agent(sarsa, variants, args.n_eval)

    if args.ac_ckpt:
        print(f"\n--- Actor-Critic evaluation [{args.ac_ckpt}] ---")
        ac = ActorCriticAgent(state_dim, action_dim, ac_cfg, device)
        ac.load(args.ac_ckpt)
        ac.actor.eval()
        ac.critic.eval()
        all_results["ac"] = evaluate_agent(ac, variants, args.n_eval)

    # Save results
    out = Path("results/evaluation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        # Remove non-serialisable reward lists for brevity
        serialisable = {}
        for agent_name, vres in all_results.items():
            serialisable[agent_name] = {}
            for v, r in vres.items():
                serialisable[agent_name][v] = {k: val for k, val in r.items() if k != "rewards"}
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
