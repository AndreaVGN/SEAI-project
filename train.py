"""
train.py — Multi-seed training script for Deep SARSA and A2C.

Usage
-----
# Train both agents with all seeds defined in the config files:
    python train.py

# Train only one agent:
    python train.py --agent sarsa
    python train.py --agent ac

# Override number of episodes or seeds:
    python train.py --episodes 100 --seeds 42

# Grid search — specify param=val1,val2 for any agent-section parameter:
    python train.py --agent sarsa --seeds 42 --grid alpha=0.0003,0.0005 epsilon_decay=0.995,0.997
    python train.py --agent ac   --seeds 42 --grid entropy_coef=0.001,0.003,0.005
    python train.py --agent ac   --seeds 42 --grid entropy_coef=0.001,0.003 n_steps=100,200

  Grid search runs all combinations (cartesian product), suppresses per-step
  logging, and prints a ranked summary at the end.
"""

from __future__ import annotations

import argparse
import ast
import copy
import os
import time
from itertools import product

import numpy as np
import torch
import yaml

from src.agents.deep_sarsa   import DeepSARSAAgent
from src.agents.actor_critic  import ActorCriticAgent
from src.environment.lunar_lander_wrapper import make_env


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_env_dims(env_name: str = "LunarLander-v3"):
    import gymnasium as gym
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def make_agent(agent_type: str, config: dict, device: torch.device, seed: int):
    env_name = config["environment"]["name"]
    state_dim, action_dim = get_env_dims(env_name)
    if agent_type == "sarsa":
        return DeepSARSAAgent(state_dim, action_dim, config, device, seed=seed)
    return ActorCriticAgent(state_dim, action_dim, config, device, seed=seed)


def parse_grid(grid_args: list[str]) -> dict[str, list]:
    """
    Parse grid search CLI args.

    Each element has the form  "param=val1,val2,..."
    Values are converted to int/float/bool where possible.

    Returns
    -------
    {"param": [val1, val2, ...], ...}
    """
    grid: dict[str, list] = {}
    for arg in grid_args:
        if "=" not in arg:
            raise ValueError(f"Grid arg must be 'param=val1,val2,...', got: {arg!r}")
        param, values_str = arg.split("=", 1)
        values = []
        for v in values_str.split(","):
            v = v.strip()
            try:
                values.append(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                values.append(v)   # keep as string
        grid[param.strip()] = values
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Standard training
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(
    agent_type: str,
    config: dict,
    seeds: list[int],
    device: torch.device,
    log_dir: str = "results/logs",
    save_dir: str = "models",
    verbose: bool = True,
) -> list[list[float]]:
    all_rewards = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Training {agent_type.upper()} | seed={seed}")
        print(f"{'='*60}")
        agent = make_agent(agent_type, config, device, seed)
        t0 = time.time()
        rewards = agent.train(log_dir=log_dir, save_dir=save_dir, verbose=verbose)
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s | "
              f"Final mean (last 100 ep): {np.mean(rewards[-100:]):.2f}")
        all_rewards.append(rewards)
    return all_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    agent_type: str,
    base_config: dict,
    grid: dict[str, list],
    seeds: list[int],
    device: torch.device,
) -> list[dict]:
    """
    Run cartesian product of grid values, one combination at a time.
    Returns results sorted best → worst.
    """
    param_names  = list(grid.keys())
    combinations = list(product(*[grid[p] for p in param_names]))
    n_total      = len(combinations) * len(seeds)

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH — {agent_type.upper()}")
    print(f"  Parameters : {param_names}")
    print(f"  Combinations: {len(combinations)}  ×  seeds: {len(seeds)}  =  {n_total} runs")
    print(f"{'='*60}")

    results = []

    for combo_idx, combo in enumerate(combinations, 1):
        cfg = copy.deepcopy(base_config)
        for param, val in zip(param_names, combo):
            cfg["agent"][param] = val

        combo_label = "  ".join(f"{p}={v}" for p, v in zip(param_names, combo))
        print(f"\n[{combo_idx}/{len(combinations)}]  {combo_label}")

        seed_finals: list[float] = []
        for seed in seeds:
            agent = make_agent(agent_type, cfg, device, seed)
            t0 = time.time()
            rewards = agent.train(
                log_dir  = "results/logs/grid",
                save_dir = "models/grid",
                verbose  = False,        # silenzioso durante la grid search
            )
            elapsed = time.time() - t0
            final = float(np.mean(rewards[-100:]))
            seed_finals.append(final)
            print(f"    seed={seed}  →  {final:7.2f}  ({elapsed:.0f}s)")

        results.append({
            "params":   dict(zip(param_names, combo)),
            "mean":     float(np.mean(seed_finals)),
            "std":      float(np.std(seed_finals, ddof=1)) if len(seed_finals) > 1 else 0.0,
            "per_seed": seed_finals,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["mean"], reverse=True)

    col_w = max(len("  ".join(f"{p}={v}" for p, v in r["params"].items())) for r in results) + 2
    hdr   = f"  {'Parameters':<{col_w}}  {'Mean':>8}  {'Std':>7}  Seeds"
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH RESULTS — {agent_type.upper()}")
    print(f"{'='*60}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for i, r in enumerate(results):
        label  = "  ".join(f"{p}={v}" for p, v in r["params"].items())
        seeds_str = "  ".join(f"{s:.1f}" for s in r["per_seed"])
        marker = "  ← BEST" if i == 0 else ""
        print(f"  {label:<{col_w}}  {r['mean']:8.2f}  {r['std']:7.2f}  [{seeds_str}]{marker}")

    print(f"\n  Best config: {results[0]['params']}")
    print(f"{'='*60}\n")

    # ── Save to JSON ──────────────────────────────────────────────────────
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"grid_{agent_type}_{ts}.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "agent":     agent_type,
            "grid":      {k: [str(v) for v in vs] for k, vs in grid.items()},
            "seeds":     seeds,
            "results":   results,
            "best":      results[0],
            "timestamp": ts,
        }, f, indent=2)
    print(f"  Results saved → {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RL agents on LunarLander-v3")
    parser.add_argument("--agent",    choices=["sarsa", "ac", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override n_episodes in config")
    parser.add_argument("--seeds",    type=int, nargs="+", default=None)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--grid",     nargs="+", default=None, metavar="PARAM=V1,V2",
                        help="Grid search over agent params, e.g. --grid alpha=0.0003,0.0005 epsilon_decay=0.995,0.997")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # ── Configs ───────────────────────────────────────────────────────────────
    sarsa_cfg = load_config("config/sarsa_config.yaml")
    ac_cfg    = load_config("config/actor_critic_config.yaml")

    if args.episodes:
        sarsa_cfg["training"]["n_episodes"] = args.episodes
        ac_cfg["training"]["n_episodes"]    = args.episodes

    seeds_sarsa = args.seeds or sarsa_cfg["training"]["seeds"]
    seeds_ac    = args.seeds or ac_cfg["training"]["seeds"]

    os.makedirs("results/logs",      exist_ok=True)
    os.makedirs("results/logs/grid", exist_ok=True)
    os.makedirs("models",            exist_ok=True)
    os.makedirs("models/grid",       exist_ok=True)

    # ── Grid search mode ──────────────────────────────────────────────────────
    if args.grid:
        grid = parse_grid(args.grid)
        if args.agent == "both":
            parser.error("--grid requires --agent sarsa or --agent ac (not 'both')")
        cfg   = sarsa_cfg if args.agent == "sarsa" else ac_cfg
        seeds = seeds_sarsa if args.agent == "sarsa" else seeds_ac
        run_grid_search(args.agent, cfg, grid, seeds, device)
        return

    # ── Standard training mode ────────────────────────────────────────────────
    if args.agent in ("sarsa", "both"):
        train_agent("sarsa", sarsa_cfg, seeds_sarsa, device)

    if args.agent in ("ac", "both"):
        train_agent("ac", ac_cfg, seeds_ac, device)

    print("\n[train.py] All training runs complete.")
    print("Run python compare.py to generate comparison plots and statistics.")


if __name__ == "__main__":
    main()
