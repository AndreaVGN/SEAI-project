"""
compare.py — Statistical comparison of Deep SARSA vs Actor-Critic.

Usage
-----
    python compare.py
    python compare.py --window 100 --last_n 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.metrics import (
    compute_statistics,
    welch_t_test,
    plot_comparison,
    load_seed_rewards,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _safe(v):
    """Convert numpy scalars and nan to JSON-serializable Python types."""
    if v is None:
        return None
    v = float(v) if hasattr(v, "item") else v
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def print_summary(name: str, rewards_per_seed: list, last_n: int = 100):
    finals = [np.mean(r[-last_n:]) for r in rewards_per_seed]
    print(f"\n  {name}  (n={len(finals)} seed{'s' if len(finals)!=1 else ''})")
    print(f"    Mean final return : {np.mean(finals):.2f}")
    if len(finals) > 1:
        print(f"    Std               : {np.std(finals, ddof=1):.2f}")
        print(f"    Min / Max (seeds) : {np.min(finals):.2f} / {np.max(finals):.2f}")
    else:
        print(f"    (only 1 seed — run all 5 seeds for full statistics)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="results/logs")
    parser.add_argument("--window",  type=int, default=50)
    parser.add_argument("--last_n",  type=int, default=100)
    args = parser.parse_args()

    sarsa_cfg = load_config("config/sarsa_config.yaml")
    ac_cfg    = load_config("config/actor_critic_config.yaml")

    seeds_sarsa = sarsa_cfg["training"]["seeds"]
    seeds_ac    = ac_cfg["training"]["seeds"]

    print("\n=== Loading training logs ===")
    sarsa_rewards = load_seed_rewards(args.log_dir, "sarsa", seeds_sarsa)
    ac_rewards    = load_seed_rewards(args.log_dir, "ac",    seeds_ac)

    if not sarsa_rewards:
        print("[compare.py] No SARSA logs found. Run: python train.py --agent sarsa")
        return
    if not ac_rewards:
        print("[compare.py] No Actor-Critic logs found. Run: python train.py --agent ac")
        return

    print("\n=== Final Performance Summary ===")
    print_summary("Deep SARSA",         sarsa_rewards, args.last_n)
    print_summary("Actor-Critic (A2C)", ac_rewards,    args.last_n)

    # --- Welch's t-test ---
    finals_sarsa = [float(np.mean(r[-args.last_n:])) for r in sarsa_rewards]
    finals_ac    = [float(np.mean(r[-args.last_n:])) for r in ac_rewards]
    t_result     = welch_t_test(finals_sarsa, finals_ac)

    print("\n=== Welch's t-test (final performance) ===")
    print(f"  t-statistic : {t_result['t_statistic']}")
    print(f"  p-value     : {t_result['p_value']}")
    print(f"  Verdict     : {t_result['verdict']}")

    # --- Plots ---
    Path("results").mkdir(exist_ok=True)
    plot_comparison(
        sarsa_rewards, ac_rewards,
        save_path=f"results/comparison_w{args.window}.png",
        window=args.window,
    )

    # --- Save statistics to JSON (all values made JSON-safe) ---
    def safe_std(lst):
        return float(np.std(lst, ddof=1)) if len(lst) > 1 else None

    stats_out = {
        "deep_sarsa": {
            "mean_final": _safe(round(float(np.mean(finals_sarsa)), 3)),
            "std_final":  _safe(safe_std(finals_sarsa)),
            "n_seeds":    len(finals_sarsa),
            "per_seed":   [_safe(round(x, 3)) for x in finals_sarsa],
        },
        "actor_critic": {
            "mean_final": _safe(round(float(np.mean(finals_ac)), 3)),
            "std_final":  _safe(safe_std(finals_ac)),
            "n_seeds":    len(finals_ac),
            "per_seed":   [_safe(round(x, 3)) for x in finals_ac],
        },
        "welch_t_test": t_result,
    }

    out_path = Path("results/statistics.json")
    with open(out_path, "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"\nStatistics saved → {out_path}")
    print(f"Plot saved       → results/comparison_w{args.window}.png")
    if len(finals_sarsa) < 5 or len(finals_ac) < 5:
        print("\n[NOTE] Only partial seeds available. For full statistical analysis run:")
        print("       python train.py  (trains all 5 seeds, ~2h on CPU)")


if __name__ == "__main__":
    main()
