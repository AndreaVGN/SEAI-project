"""
stat_test.py — Welch's t-test between two evaluate_all.py JSON outputs.

Generic comparator: takes any two JSON files produced by `evaluate_all.py`
(e.g. baseline vs an ablation) and runs Welch's t-test variant-by-variant,
agent-by-agent, using the per-seed mean reward as the sample (n = n_seeds).

This is the same test/methodology as `compare.py`, just applied to
evaluation JSONs (which cover all environment variants) instead of
training logs (which only cover the standard env).

Usage
-----
    # Compare AC with entropy bonus (baseline) vs without (ablation),
    # one row per environment variant:
    python stat_test.py results/evaluation_baseline_best_*.json \
                         results/evaluation_noentropy_best_*.json

    # Custom display labels (default: inferred from the JSON's "tag" field)
    python stat_test.py file_a.json file_b.json \
        --labels "With entropy" "No entropy"

    # Restrict to one agent when a file contains more than one
    python stat_test.py file_a.json file_b.json --agent ac

    # Compare two DIFFERENT agents, e.g. sarsa vs ac inside the SAME file
    # (pass the same file twice, pick one agent per side):
    python stat_test.py results/evaluation_all_best_*.json \
                         results/evaluation_all_best_*.json \
        --agent_a sarsa --agent_b ac

    # Different significance level / custom output path
    python stat_test.py file_a.json file_b.json --alpha 0.01 \
        --out results/stat_test_entropy_ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.metrics import welch_t_test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def per_seed_means(data: dict, agent: str, variant: str) -> list[float]:
    """Per-seed mean reward for one agent/variant, as logged by evaluate_all.py."""
    seeds = data.get("per_seed", {}).get(agent, {})
    out = []
    for seed_results in seeds.values():
        if seed_results and variant in seed_results:
            out.append(float(seed_results[variant]["mean"]))
    return out


def variant_set(data: dict, agent: str) -> set[str]:
    """All variants found for an agent, from per-seed data (fallback: aggregated)."""
    variants: set[str] = set()
    for seed_results in data.get("per_seed", {}).get(agent, {}).values():
        if seed_results:
            variants.update(seed_results.keys())
    if not variants:
        variants = set(data.get("aggregated", {}).get(agent, {}).keys())
    return variants


def resolve_agent_pairs(
    data_a: dict, data_b: dict,
    requested: str | None,
    agent_a_override: str | None = None,
    agent_b_override: str | None = None,
) -> list[tuple[str, str]]:
    """Decide which agent(s) to compare between the two files."""
    agents_a = set(data_a.get("per_seed", {}).keys())
    agents_b = set(data_b.get("per_seed", {}).keys())

    # Explicit per-side agent selection — lets you compare two DIFFERENT
    # agents (e.g. sarsa vs ac), including from the same file passed twice.
    if agent_a_override or agent_b_override:
        a = agent_a_override or requested
        b = agent_b_override or requested
        if not a or not b:
            raise SystemExit(
                "[stat_test] When using --agent_a/--agent_b you must specify "
                "both sides (the missing side falls back to --agent if given)."
            )
        if a not in agents_a:
            raise SystemExit(f"[stat_test] Agent '{a}' not found in file A ({sorted(agents_a)})")
        if b not in agents_b:
            raise SystemExit(f"[stat_test] Agent '{b}' not found in file B ({sorted(agents_b)})")
        return [(a, b)]

    if requested:
        if requested not in agents_a or requested not in agents_b:
            raise SystemExit(
                f"[stat_test] Agent '{requested}' not found in both files "
                f"(file A: {sorted(agents_a)}, file B: {sorted(agents_b)})"
            )
        return [(requested, requested)]

    common = sorted(agents_a & agents_b)
    if common:
        return [(a, a) for a in common]

    # No agent name in common — if each file has exactly one agent, pair
    # them anyway (e.g. comparing a sarsa-only file against an ac-only file).
    if len(agents_a) == 1 and len(agents_b) == 1:
        return [(next(iter(agents_a)), next(iter(agents_b)))]

    raise SystemExit(
        f"[stat_test] No common agent between the two files "
        f"(file A: {sorted(agents_a)}, file B: {sorted(agents_b)}). "
        "Use --agent (same agent both sides) or --agent_a/--agent_b "
        "(different agent per side) to force a comparison."
    )


def label_for(data: dict, fallback: str) -> str:
    tag = data.get("tag", "")
    return tag.rstrip("_") if tag else fallback


def safe_round(v) -> float | None:
    v = float(v)
    return None if v != v else round(v, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Welch's t-test between two evaluate_all.py JSON outputs."
    )
    parser.add_argument("file_a", help="First evaluate_all.py JSON output")
    parser.add_argument("file_b", help="Second evaluate_all.py JSON output")
    parser.add_argument("--agent", default=None,
                         help="Restrict comparison to one agent, same on both sides "
                              "(e.g. 'ac' or 'sarsa'). Default: compare every agent "
                              "common to both files.")
    parser.add_argument("--agent_a", default=None,
                         help="Agent to use from file A (use with --agent_b to compare "
                              "two different agents, e.g. sarsa vs ac).")
    parser.add_argument("--agent_b", default=None,
                         help="Agent to use from file B (use with --agent_a).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--labels", nargs=2, default=None,
                         metavar=("LABEL_A", "LABEL_B"),
                         help="Display labels for the two runs (default: inferred from 'tag').")
    parser.add_argument("--out", default=None,
                         help="Output JSON path (default: results/stat_test_<A>_vs_<B>.json)")
    args = parser.parse_args()

    data_a = load_json(args.file_a)
    data_b = load_json(args.file_b)

    pairs = resolve_agent_pairs(data_a, data_b, args.agent, args.agent_a, args.agent_b)

    if args.labels:
        label_a, label_b = args.labels
    elif len(pairs) == 1 and pairs[0][0] != pairs[0][1]:
        # Single cross-agent comparison (e.g. sarsa vs ac, possibly same file)
        # — default to the agent names, since "tag" is identical on both sides.
        label_a, label_b = pairs[0]
    else:
        label_a = label_for(data_a, "A")
        label_b = label_for(data_b, "B")

    print(f"\nComparing:  A = '{label_a}'  ({args.file_a})")
    print(f"            B = '{label_b}'  ({args.file_b})")

    all_results = {}
    for agent_a, agent_b in pairs:
        agent_label = agent_a if agent_a == agent_b else f"{agent_a}/{agent_b}"
        variants = sorted(variant_set(data_a, agent_a) & variant_set(data_b, agent_b))
        if not variants:
            print(f"\n[stat_test] No common variants for agent '{agent_label}' — skipping.")
            continue

        print(f"\n{'='*78}")
        print(f"  Agent: {agent_label}   (Welch's t-test, alpha={args.alpha})")
        print(f"{'='*78}")
        print(f"  {'Variant':<14}{'n_A':>5}{'mean_A':>11}{'n_B':>5}{'mean_B':>11}{'t':>10}{'p':>10}   Verdict")
        print(f"  {'-'*92}")

        agent_results = {}
        for variant in variants:
            means_a = per_seed_means(data_a, agent_a, variant)
            means_b = per_seed_means(data_b, agent_b, variant)
            result = welch_t_test(means_a, means_b, alpha=args.alpha)

            mean_a_str = f"{np.mean(means_a):.2f}" if means_a else "—"
            mean_b_str = f"{np.mean(means_b):.2f}" if means_b else "—"
            t_str = f"{result['t_statistic']:.3f}" if result["t_statistic"] is not None else "—"
            p_str = f"{result['p_value']:.4f}" if result["p_value"] is not None else "—"
            sig = ("significant" if result["significant"]
                   else "n.s." if result["t_statistic"] is not None else "n/a")

            print(f"  {variant:<14}{len(means_a):>5}{mean_a_str:>11}"
                  f"{len(means_b):>5}{mean_b_str:>11}{t_str:>10}{p_str:>10}   {sig}")

            agent_results[variant] = {
                "n_a": len(means_a),
                "mean_a": safe_round(np.mean(means_a)) if means_a else None,
                "per_seed_a": [round(v, 3) for v in means_a],
                "n_b": len(means_b),
                "mean_b": safe_round(np.mean(means_b)) if means_b else None,
                "per_seed_b": [round(v, 3) for v in means_b],
                **result,
            }
        all_results[agent_label] = agent_results

    print()
    out_path = Path(args.out) if args.out else Path(
        f"results/stat_test_{label_a}_vs_{label_b}.json".replace(" ", "_")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "file_a": args.file_a, "label_a": label_a,
            "file_b": args.file_b, "label_b": label_b,
            "alpha": args.alpha,
            "results": all_results,
        }, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
