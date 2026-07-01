"""
Standalone regeneration of the (previously hand-drawn / buggy) learning
curve chart, using the real training logs in results/logs and the fixed
compute_statistics() in src/utils/metrics.py.

Fixes applied vs. the original chart:
  1. x-axis uses the REAL 'episode' column from each CSV (not row index),
     so SARSA correctly spans ~3000 episodes (not ~300, and the legend's
     stale "2k ep" is now derived from the data: current config trains
     3000 episodes) and A2C correctly spans ~5000 episodes.
  2. Seeds are aligned via interpolation onto a common episode grid before
     averaging, since SARSA logs every 10 episodes while A2C logs at
     irregular cadences (vectorised training).
  3. Band is mean ± 1 std across the 5 seeds (matching the intended
     caption), not a 95% CI as the code previously computed.
"""
import sys
import importlib.util

import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics.py directly, bypassing src/utils/__init__.py (which also
# imports replay_buffer.py -> torch; not needed just to make this plot).
_spec = importlib.util.spec_from_file_location("metrics", "src/utils/metrics.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_statistics = _metrics.compute_statistics
load_seed_episode_rewards = _metrics.load_seed_episode_rewards

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

with open("config/sarsa_config.yaml") as f:
    sarsa_cfg = yaml.safe_load(f)
with open("config/actor_critic_config.yaml") as f:
    ac_cfg = yaml.safe_load(f)

seeds_sarsa = sarsa_cfg["training"]["seeds"]
seeds_ac    = ac_cfg["training"]["seeds"]

sarsa_data = load_seed_episode_rewards("results/logs", "sarsa", seeds_sarsa)
ac_data    = load_seed_episode_rewards("results/logs", "ac",    seeds_ac)

print(f"SARSA seeds loaded: {len(sarsa_data)} (max episode per seed: "
      f"{[int(ep.max()) for ep, _ in sarsa_data]})")
print(f"A2C   seeds loaded: {len(ac_data)} (max episode per seed: "
      f"{[int(ep.max()) for ep, _ in ac_data]})")

WINDOW = 50
colors = {"sarsa": "#2196F3", "ac": "#FF5722"}

fig, ax = plt.subplots(figsize=(9, 5.5))

for name, data, color in [
    ("Deep SARSA", sarsa_data, colors["sarsa"]),
    ("Actor-Critic (A2C)", ac_data, colors["ac"]),
]:
    ep, mean, std = compute_statistics(data, window=WINDOW, step=10)
    max_ep = max(e.max() for e, _ in data)
    ep_label = f"{max_ep/1000:.0f}k ep" if max_ep >= 1000 else f"{int(max_ep)} ep"
    ax.plot(ep, mean, label=f"{name} ({ep_label})", color=color, linewidth=1.6)
    ax.fill_between(ep, mean - std, mean + std, alpha=0.2, color=color)

ax.axhline(200, color="green", linestyle="--", linewidth=1.3, label="Solved (200)")
ax.set_xlabel("Episode")
ax.set_ylabel(f"Return ({WINDOW}-ep rolling mean)")
ax.set_title("Learning Curves — LunarLander-v3")
ax.legend(loc="lower right", fontsize=10)
ax.text(0.5, -0.14, "Mean ± 1 std across 5 seeds", transform=ax.transAxes,
        ha="center", fontsize=9, color="dimgray")

plt.tight_layout()
plt.savefig("results/learning_curves_correct.png", bbox_inches="tight")
print("Saved -> results/learning_curves_correct.png")
