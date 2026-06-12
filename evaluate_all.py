"""
evaluate_all.py — Aggregate evaluation across all seeds for both agents.

Loads best (or last) checkpoint for each seed, evaluates on all environment
variants, and reports per-seed results + aggregated mean ± std.

Usage
-----
    # Best checkpoints (default)
    python evaluate_all.py --n_eval 100

    # Last checkpoints
    python evaluate_all.py --ckpt_type last --n_eval 100

    # Custom episodes per training run
    python evaluate_all.py --sarsa_episodes 3000 --ac_episodes 5000 --n_eval 100

    # Ablation runs (tagged checkpoints, e.g. noreplay_sarsa_seed42_best.pt)
    python evaluate_all.py --tag noreplay_ --agents sarsa --n_eval 100
    python evaluate_all.py --tag parallel8_ --agents sarsa --n_eval 100
    python evaluate_all.py --tag nsteps50_ --agents ac --n_eval 100
    python evaluate_all.py --tag reward_coef_ --agents sarsa ac --n_eval 100
"""

from __future__ import annotations

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path

from src.agents.deep_sarsa    import DeepSARSAAgent
from src.agents.actor_critic  import ActorCriticAgent
from src.environment.lunar_lander_wrapper import ENV_VARIANTS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_env_dims(env_name: str = "LunarLander-v3") -> tuple[int, int]:
    import gymnasium as gym
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def ckpt_path(agent: str, seed: int, ckpt_type: str,
              sarsa_ep: int, ac_ep: int, tag: str = "") -> Path:
    """Build checkpoint path based on agent, seed, type (best/last), and optional tag prefix."""
    ep = sarsa_ep if agent == "sarsa" else ac_ep
    if ckpt_type == "best":
        return Path(f"models/{tag}{agent}_seed{seed}_best.pt")
    return Path(f"models/{tag}{agent}_seed{seed}_ep{ep}.pt")


def evaluate_seed(agent_name: str, ckpt: Path, config: dict,
                  state_dim: int, action_dim: int,
                  device: torch.device, variants: list,
                  n_eval: int):
    """Load checkpoint and evaluate on all variants. Returns None if missing."""
    if not ckpt.exists():
        print(f"    ⚠  checkpoint not found: {ckpt} — skipping")
        return None

    if agent_name == "sarsa":
        agent = DeepSARSAAgent(state_dim, action_dim, config, device)
        agent.load(str(ckpt))
        agent.q_network.eval()
    else:
        agent = ActorCriticAgent(state_dim, action_dim, config, device)
        agent.load(str(ckpt))
        agent.actor.eval()
        agent.critic.eval()

    results = {}
    for variant in variants:
        res = agent.evaluate(env_variant=variant, n_episodes=n_eval)
        results[variant] = {
            "mean": res["mean_reward"],
            "std":  res["std_reward"],
            "min":  res["min_reward"],
            "max":  res["max_reward"],
        }
        print(f"    [{variant:12s}]  mean={res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")
    return results


def aggregate(per_seed: list, variants: list) -> dict:
    """Average per-seed results across seeds for each variant."""
    agg = {}
    for v in variants:
        means = [s[v]["mean"] for s in per_seed if s is not None]
        stds  = [s[v]["std"]  for s in per_seed if s is not None]
        if not means:
            continue
        agg[v] = {
            "mean_of_means": float(np.mean(means)),
            "std_of_means":  float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "mean_of_stds":  float(np.mean(stds)),
            "n_seeds":       len(means),
        }
    return agg


def print_summary(agent_label: str, agg: dict, variants: list) -> None:
    print(f"\n  ┌─ {agent_label} — aggregated over {list(agg.values())[0]['n_seeds']} seeds")
    for v in variants:
        r = agg[v]
        print(f"  │  [{v:12s}]  "
              f"mean={r['mean_of_means']:8.2f} ± {r['std_of_means']:.2f}  "
              f"(within-seed std ≈ {r['mean_of_stds']:.2f})")
    print("  └" + "─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation across all seeds for SARSA and A2C."
    )
    parser.add_argument("--seeds",           type=int, nargs="+",
                        default=[42, 123, 456, 789, 1234])
    parser.add_argument("--n_eval",          type=int, default=100,
                        help="Episodes per variant per seed")
    parser.add_argument("--ckpt_type",       choices=["best", "last"], default="best",
                        help="'best' = *_best.pt  |  'last' = *_ep{N}.pt")
    parser.add_argument("--sarsa_episodes",  type=int, default=3000,
                        help="Used to build last-checkpoint filename for SARSA")
    parser.add_argument("--ac_episodes",     type=int, default=5000,
                        help="Used to build last-checkpoint filename for A2C")
    parser.add_argument("--agents",          nargs="+",
                        choices=["sarsa", "ac"], default=["sarsa", "ac"])
    parser.add_argument("--tag",             type=str, default="",
                        help="Tag prefix for checkpoint filenames (e.g. 'noreplay_', 'parallel8_'). "
                             "Looks for models/{tag}{agent}_seed{seed}_best.pt")
    parser.add_argument("--device",          default="auto")
    args = parser.parse_args()

    # Ensure tag ends with _ if non-empty and doesn't already
    tag = args.tag
    if tag and not tag.endswith("_"):
        tag = tag + "_"

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
    print(f"Checkpoint type: {args.ckpt_type}")
    if tag:
        print(f"Tag prefix: '{tag}'")
    print(f"Seeds: {args.seeds}  |  n_eval: {args.n_eval} episodes/variant\n")

    sarsa_cfg = load_config("config/sarsa_config.yaml")
    ac_cfg    = load_config("config/actor_critic_config.yaml")
    state_dim, action_dim = get_env_dims()
    variants  = list(ENV_VARIANTS.keys())

    configs  = {"sarsa": sarsa_cfg, "ac": ac_cfg}
    labels   = {"sarsa": "Deep SARSA", "ac": "Actor-Critic (A2C)"}
    all_data = {}   # {agent: {seed: {variant: {...}}}}
    all_agg  = {}   # {agent: {variant: aggregated}}

    for agent_name in args.agents:
        cfg   = configs[agent_name]
        label = labels[agent_name]
        print(f"{'='*60}")
        print(f"  {label}" + (f"  [tag={tag}]" if tag else ""))
        print(f"{'='*60}")

        per_seed_results = []
        seed_data        = {}

        for seed in args.seeds:
            ckpt = ckpt_path(agent_name, seed, args.ckpt_type,
                             args.sarsa_episodes, args.ac_episodes, tag=tag)
            print(f"\n  seed={seed}  [{ckpt.name}]")
            res = evaluate_seed(agent_name, ckpt, cfg, state_dim, action_dim,
                                device, variants, args.n_eval)
            per_seed_results.append(res)
            seed_data[seed] = res

        agg = aggregate(per_seed_results, variants)
        all_agg[agent_name]  = agg
        all_data[agent_name] = seed_data
        print_summary(label, agg, variants)

    # ── Comparison table (only when both agents evaluated) ───────────────────
    if len(args.agents) == 2 and "sarsa" in all_agg and "ac" in all_agg:
        print(f"\n{'='*60}")
        print("  GENERALISATION COMPARISON  (mean of means across seeds)")
        print(f"{'='*60}")
        print(f"  {'Variant':<14}  {'SARSA':>10}  {'A2C':>10}  {'Winner'}")
        print(f"  {'-'*50}")
        for v in variants:
            s_mean = all_agg["sarsa"][v]["mean_of_means"]
            a_mean = all_agg["ac"][v]["mean_of_means"]
            winner = "SARSA" if s_mean >= a_mean else "A2C"
            print(f"  {v:<14}  {s_mean:10.2f}  {a_mean:10.2f}  {winner}")
        print(f"{'='*60}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    import datetime
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_str = tag.rstrip("_") if tag else "baseline"
    out     = Path(f"results/evaluation_{tag_str}_{args.ckpt_type}_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "tag":       tag,
            "ckpt_type": args.ckpt_type,
            "seeds":     args.seeds,
            "n_eval":    args.n_eval,
            "per_seed":  {
                agent: {
                    str(seed): res
                    for seed, res in seed_data.items()
                    if res is not None
                }
                for agent, seed_data in all_data.items()
            },
            "aggregated": all_agg,
        }, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
