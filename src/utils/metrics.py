"""
Statistical evaluation utilities.

Implements the metrics required by SRPs (4/4):
- Mean / std / 95 % CI over multiple seeds
- Welch's t-test for statistical significance
- Learning curve and comparison plots
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import scipy.stats as stats
import seaborn as sns


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_statistics(
    seed_data: List[Tuple[np.ndarray, np.ndarray]],
    window: int = 50,
    step: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of per-seed (episode_numbers, episode_rewards) arrays,
    compute the smoothed mean and std across seeds, indexed by the REAL
    episode number.

    NOTE on a previous bug: this function used to take plain reward lists
    and assume row index == episode number. That breaks for Deep SARSA
    (which logs only every `log_every`-th episode, so 300 logged rows
    actually span 3 000 real episodes) and for A2C (whose vectorised
    training logs at irregular cadences). Both agents' logged "episode"
    column already carries the correct real episode number — we use it
    directly here instead of `np.arange(len(...))`.

    Because seeds/agents log at different, non-uniform cadences, each
    seed's curve is linearly interpolated onto a common episode grid
    (resolution = `step` episodes) before averaging across seeds. A
    rolling mean equivalent to `window` real episodes is then applied on
    that grid (so "window=50" means ~50 episodes regardless of how
    sparsely a given seed logged).

    Parameters
    ----------
    seed_data : list of (episode_array, reward_array) per seed
    window    : rolling window, in real episodes, for smoothing
    step      : resolution (in real episodes) of the common grid

    Returns
    -------
    episodes : 1-D array (real episode numbers, common grid)
    mean     : smoothed mean across seeds
    std      : smoothed std across seeds (ddof=1; zeros if only 1 seed)
    """
    if not seed_data:
        raise ValueError("compute_statistics: no seed data provided")

    # Common grid: only cover the range every seed actually reached, to
    # avoid extrapolating past where a seed stopped training.
    first_ep = max(ep.min() for ep, _ in seed_data)
    last_ep  = min(ep.max() for ep, _ in seed_data)
    grid = np.arange(first_ep, last_ep + 1, step)

    win_pts = max(1, round(window / step))
    curves = []
    for ep, rew in seed_data:
        interp = np.interp(grid, ep, rew)
        smooth = pd.Series(interp).rolling(win_pts, min_periods=1).mean().to_numpy()
        curves.append(smooth)

    mat  = np.array(curves)           # (n_seeds, n_grid_points)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

    return grid, mean, std


def welch_t_test(
    sarsa_rewards: List[float],
    ac_rewards:    List[float],
    alpha:         float = 0.05,
) -> Dict:
    """
    Welch's t-test comparing mean final episode rewards of two agents.

    Requires at least 2 samples per group. Returns a dict with t-stat,
    p-value, and a plain-English verdict — all values are Python native
    types (JSON-serializable).
    """
    if len(sarsa_rewards) < 2 or len(ac_rewards) < 2:
        return {
            "t_statistic": None,
            "p_value":     None,
            "alpha":       float(alpha),
            "significant": False,
            "verdict": (
                f"Cannot perform t-test: need ≥2 seeds per agent "
                f"(got {len(sarsa_rewards)} SARSA, {len(ac_rewards)} A2C). "
                "Run train.py with all 5 seeds first."
            ),
        }

    t_stat, p_value = stats.ttest_ind(sarsa_rewards, ac_rewards, equal_var=False)

    # Convert explicitly to Python native types — scipy may return numpy scalars
    # whose __class__.__name__ is 'bool' / 'float64' but are NOT JSON-serializable
    t_stat  = float(t_stat)
    p_value = float(p_value)
    significant = bool(p_value < alpha)  # numpy.bool_ → Python bool

    return {
        "t_statistic": round(t_stat, 4)  if t_stat == t_stat else None,   # nan check
        "p_value":     round(p_value, 6) if p_value == p_value else None,
        "alpha":       float(alpha),
        "significant": significant,
        "verdict": (
            f"Significant difference (p={p_value:.4f} < {alpha})"
            if significant
            else f"No significant difference (p={p_value:.4f} ≥ {alpha})"
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def plot_learning_curves(
    episodes:   np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    agent_name: str,
    color:      str,
    ax:         plt.Axes,
    label:      str | None = None,
) -> None:
    """Plot mean ± 1 std for one agent on a given Axes."""
    lbl = label or agent_name
    ax.plot(episodes, mean, label=lbl, color=color, linewidth=1.5)
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)


def plot_comparison(
    sarsa_seed_data: List[Tuple[np.ndarray, np.ndarray]],
    ac_seed_data:    List[Tuple[np.ndarray, np.ndarray]],
    save_path:       str = "results/comparison.png",
    window:          int = 50,
    step:            int = 10,
) -> None:
    """
    Full comparison figure:
      1. Learning curves with mean ± 1 std (real episode numbers on x)
      2. Box-plot of final-100-episode returns per seed

    `sarsa_seed_data` / `ac_seed_data` are lists of (episode_array,
    reward_array) per seed, as returned by `load_seed_episode_rewards`.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deep SARSA vs Advantage Actor-Critic — LunarLander-v3", fontsize=13)

    colors = {"sarsa": "#2196F3", "ac": "#FF5722"}

    # --- subplot 1: learning curves ---
    ax = axes[0]
    for name, seed_data, color in [
        ("Deep SARSA", sarsa_seed_data, colors["sarsa"]),
        ("Actor-Critic (A2C)", ac_seed_data, colors["ac"]),
    ]:
        ep, mean, std = compute_statistics(seed_data, window, step)
        max_ep = max(e.max() for e, _ in seed_data)
        ep_label = f"{max_ep/1000:.0f}k ep" if max_ep >= 1000 else f"{int(max_ep)} ep"
        plot_learning_curves(ep, mean, std, name, color, ax, label=f"{name} ({ep_label})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (rolling mean, ~{window}-ep window)")
    ax.set_title("Learning Curves (mean ± 1 std across seeds)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1, label="Solved (200)")
    ax.legend(fontsize=9)

    # --- subplot 2: box-plot of final returns ---
    ax = axes[1]
    sarsa_rewards = [rew for _, rew in sarsa_seed_data]
    ac_rewards    = [rew for _, rew in ac_seed_data]
    final_sarsa = [np.mean(r[-100:]) for r in sarsa_rewards]
    final_ac    = [np.mean(r[-100:]) for r in ac_rewards]
    df = pd.DataFrame(
        [(k, v) for k, vals in [
            ("Deep SARSA", final_sarsa),
            ("Actor-Critic (A2C)", final_ac),
        ] for v in vals],
        columns=["Agent", "Mean Return (last 100 logged points)"],
    )
    # seaborn ≥0.14: assign x to hue to avoid FutureWarning
    palette = {"Deep SARSA": colors["sarsa"], "Actor-Critic (A2C)": colors["ac"]}
    sns.boxplot(
        data=df, x="Agent", y="Mean Return (last 100 logged points)", ax=ax,
        hue="Agent", palette=palette, legend=False,
    )
    sns.stripplot(
        data=df, x="Agent", y="Mean Return (last 100 logged points)", ax=ax,
        hue="Agent", palette={"Deep SARSA": "black", "Actor-Critic (A2C)": "black"},
        size=4, jitter=True, alpha=0.6, legend=False,
    )
    ax.set_title("Final Performance Distribution (per seed)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Comparison plot saved → {save_path}")


def plot_inference_time(
    sarsa_times: List[float],
    ac_times:    List[float],
    save_path:   str = "results/inference_time.png",
) -> None:
    """Bar chart comparing mean inference latency (ms) per action."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    agents = ["Deep SARSA", "Actor-Critic (A2C)"]
    means  = [np.mean(sarsa_times) * 1e3, np.mean(ac_times) * 1e3]
    stds   = [np.std(sarsa_times)  * 1e3, np.std(ac_times)  * 1e3]

    bars = ax.bar(agents, means, yerr=stds, capsize=6,
                  color=["#2196F3", "#FF5722"], alpha=0.8)
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Mean Inference Time per Action")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                f"{m:.3f} ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Inference-time plot saved → {save_path}")


def load_seed_rewards(csv_dir: str, agent_name: str, seeds: List[int]) -> List[List[float]]:
    """Load per-episode rewards from CSV files produced by TrainingLogger."""
    all_rewards = []
    for seed in seeds:
        path = Path(csv_dir) / f"{agent_name}_seed{seed}.csv"
        if not path.exists():
            print(f"[metrics] WARNING: {path} not found — skipping seed {seed}")
            continue
        df = pd.read_csv(path)
        if "episode_reward" in df.columns:
            all_rewards.append(df["episode_reward"].tolist())
        else:
            print(f"[metrics] WARNING: 'episode_reward' column missing in {path}")
    return all_rewards


def load_seed_episode_rewards(
    csv_dir: str, agent_name: str, seeds: List[int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load (episode_number, episode_reward) arrays per seed from CSV files
    produced by TrainingLogger, using the REAL 'episode' column (not the
    CSV row index — see compute_statistics docstring for why that matters).
    """
    all_data = []
    for seed in seeds:
        path = Path(csv_dir) / f"{agent_name}_seed{seed}.csv"
        if not path.exists():
            print(f"[metrics] WARNING: {path} not found — skipping seed {seed}")
            continue
        df = pd.read_csv(path)
        if "episode" not in df.columns or "episode_reward" not in df.columns:
            print(f"[metrics] WARNING: 'episode'/'episode_reward' columns missing in {path}")
            continue
        df = df.sort_values("episode")
        all_data.append((
            df["episode"].to_numpy(dtype=float),
            df["episode_reward"].to_numpy(dtype=float),
        ))
    return all_data
