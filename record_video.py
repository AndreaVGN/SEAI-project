"""
record_video.py — Registra video delle simulazioni di LunarLander.

Carica un checkpoint addestrato e salva un video per ogni episodio richiesto
in results/videos/<agent>/<variant>/.

Uso
---
    # SARSA, variante standard, 3 episodi (seed 42, best checkpoint):
    python record_video.py --agent sarsa --seed 42

    # A2C, tutte le varianti, 1 episodio ciascuna:
    python record_video.py --agent ac --seed 42 --variants standard wind turbulent heavy

    # Checkpoint specifico:
    python record_video.py --agent sarsa --ckpt models/sarsa_seed42_best.pt

Dipendenze aggiuntive:
    pip install moviepy          # per la codifica MP4
    # oppure:
    pip install gymnasium[box2d] # include swig/pyglet necessari
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_env_dims(env_name: str = "LunarLander-v3") -> tuple[int, int]:
    import gymnasium as gym
    env = gym.make(env_name)
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def resolve_checkpoint(agent: str, seed: int, ckpt_type: str) -> Path:
    """Trova il checkpoint corrispondente ad agent/seed/type."""
    model_dir = Path("models")
    pattern = f"{agent}_seed{seed}_{ckpt_type}.pt"
    ckpt = model_dir / pattern
    if not ckpt.exists():
        candidates = sorted(model_dir.glob(f"{agent}_seed{seed}_*.pt"))
        if not candidates:
            sys.exit(f"[ERROR] Nessun checkpoint trovato per {agent} seed={seed} in {model_dir}/")
        ckpt = candidates[-1]
        print(f"[WARN] {pattern} non trovato — uso {ckpt.name}")
    return ckpt


def restore_normalizer(env, agent) -> None:
    """Ripristina le statistiche del normalizer salvate nel checkpoint."""
    from src.environment.lunar_lander_wrapper import LunarLanderWrapper
    if not isinstance(env, LunarLanderWrapper):
        return
    norm_mean  = getattr(agent, "_norm_mean", None)
    norm_var   = getattr(agent, "_norm_var",  None)
    norm_count = getattr(agent, "_norm_count", None)
    if norm_mean is not None:
        env._normalizer.mean  = norm_mean.copy()
        env._normalizer.var   = norm_var.copy()
        env._normalizer.count = norm_count
        env.freeze_normalizer()
    else:
        # Warm-up veloce se le statistiche non sono salvate
        print("  [INFO] Statistiche normalizer non trovate — warm-up con 200 episodi random")
        _warmup_normalizer(env, n_episodes=200)
        env.freeze_normalizer()


def _warmup_normalizer(env, n_episodes: int = 200) -> None:
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


# ── recording ────────────────────────────────────────────────────────────────

def record_episodes(
    agent_type: str,
    ckpt_path: Path,
    variants: list[str],
    n_episodes: int,
    seed: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    from gymnasium.wrappers import RecordVideo
    from src.environment.lunar_lander_wrapper import make_env, ENV_VARIANTS
    from src.agents.deep_sarsa  import DeepSARSAAgent
    from src.agents.actor_critic import ActorCriticAgent

    state_dim, action_dim = get_env_dims()

    if agent_type == "sarsa":
        cfg   = load_config("config/sarsa_config.yaml")
        agent = DeepSARSAAgent(state_dim, action_dim, cfg, device)
        agent.load(str(ckpt_path))
        agent.q_network.eval()
        def select_action(obs):
            return agent.select_action(obs, greedy=True)
    else:
        cfg   = load_config("config/actor_critic_config.yaml")
        agent = ActorCriticAgent(state_dim, action_dim, cfg, device)
        agent.load(str(ckpt_path))
        agent.actor.eval()
        agent.critic.eval()
        def select_action(obs):
            action, _ = agent.select_action(obs, greedy=True)
            return action

    for variant in variants:
        if variant not in ENV_VARIANTS:
            print(f"[WARN] Variante '{variant}' non riconosciuta — skip")
            continue

        video_dir = out_dir / agent_type / variant
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{agent_type.upper()} | {variant}]  → {video_dir}")

        # Crea env con render_mode=rgb_array e RecordVideo wrapper
        base_env = make_env(
            variant=variant,
            seed=seed,
            render_mode="rgb_array",
        )
        env = RecordVideo(
            base_env,
            video_folder=str(video_dir),
            name_prefix=f"{agent_type}_{variant}",
            episode_trigger=lambda ep: True,   # registra tutti gli episodi
        )

        # Ripristina il normalizer dall'agente
        restore_normalizer(base_env, agent)

        total_reward = 0.0
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward
            print(f"  episodio {ep + 1}/{n_episodes}  reward={ep_reward:.1f}")

        env.close()
        mean_r = total_reward / n_episodes
        print(f"  media reward: {mean_r:.2f}")
        print(f"  video salvati in: {video_dir}/")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Registra video delle simulazioni di LunarLander."
    )
    parser.add_argument(
        "--agent", choices=["sarsa", "ac"], required=True,
        help="Agente da usare (sarsa o ac)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed del checkpoint (default: 42)"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Percorso checkpoint esplicito (sovrascrive --seed e --ckpt_type)"
    )
    parser.add_argument(
        "--ckpt_type", default="best",
        help="Tipo checkpoint: 'best' o 'epXXXX' (default: best)"
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=["standard"],
        choices=["standard", "wind", "turbulent", "heavy"],
        help="Varianti ambiente da registrare (default: standard)"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=3,
        help="Episodi da registrare per variante (default: 3)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/videos",
        help="Cartella di output per i video (default: results/videos)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Dispositivo PyTorch: cpu, cuda, mps, auto (default: auto)"
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else resolve_checkpoint(
        args.agent, args.seed, args.ckpt_type
    )
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] Checkpoint non trovato: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}")

    record_episodes(
        agent_type=args.agent,
        ckpt_path=ckpt_path,
        variants=args.variants,
        n_episodes=args.n_episodes,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        device=device,
    )

    print("\nFatto! Video salvati in:", args.out_dir)


if __name__ == "__main__":
    main()
