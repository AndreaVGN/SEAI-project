"""
make_progress_video.py — Video unico che mostra l'apprendimento nel tempo.

Carica i checkpoint in ordine (ep200, ep400, …, best), registra un episodio
per ciascuno e incolla tutto in un unico MP4 con overlay testuale.

Dipendenze:
    pip install moviepy Pillow

Uso
---
    # SARSA seed 42, variante standard:
    python make_progress_video.py --agent sarsa --seed 42

    # A2C, variante wind, checkpoint ogni 1000 ep:
    python make_progress_video.py --agent ac --seed 42 --variant wind --step 1000

    # Con tag (ablation run):
    python make_progress_video.py --agent sarsa --seed 42 --tag noreplay_

    # Scegli i checkpoint manualmente:
    python make_progress_video.py --agent sarsa --seed 42 --episodes 200 600 1000 2000 best
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(agent: str) -> dict:
    path = "config/sarsa_config.yaml" if agent == "sarsa" else "config/actor_critic_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_env_dims():
    import gymnasium as gym
    env = gym.make("LunarLander-v3")
    s, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return s, a


def find_checkpoints(agent: str, seed: int, tag: str, step: int) -> list[Path]:
    """Return sorted ep checkpoints + best."""
    model_dir = Path("models")
    prefix = f"{tag}{agent}_seed{seed}_ep"
    eps = []
    for p in model_dir.glob(f"{prefix}*.pt"):
        m = re.search(r"_ep(\d+)\.pt$", p.name)
        if m:
            eps.append((int(m.group(1)), p))
    eps.sort(key=lambda x: x[0])
    # Keep only multiples of step
    eps = [(ep, p) for ep, p in eps if ep % step == 0]
    # Add best at the end
    best = model_dir / f"{tag}{agent}_seed{seed}_best.pt"
    if best.exists():
        eps.append((-1, best))
    return eps


def load_agent(agent_type: str, ckpt_path: Path, config: dict, device: torch.device):
    from src.agents.deep_sarsa import DeepSARSAAgent
    from src.agents.actor_critic import ActorCriticAgent
    state_dim, action_dim = get_env_dims()
    if agent_type == "sarsa":
        ag = DeepSARSAAgent(state_dim, action_dim, config, device)
        ag.load(str(ckpt_path))
        ag.q_network.eval()
        def act(obs): return ag.select_action(obs, greedy=True)
    else:
        ag = ActorCriticAgent(state_dim, action_dim, config, device)
        ag.load(str(ckpt_path))
        ag.actor.eval()
        def act(obs):
            a, _ = ag.select_action(obs, greedy=True)
            return a
    return ag, act


def restore_norm(env, agent):
    from src.environment.lunar_lander_wrapper import LunarLanderWrapper
    if not isinstance(env, LunarLanderWrapper):
        return
    nm = getattr(agent, "_norm_mean", None)
    if nm is not None:
        env._normalizer.mean  = nm.copy()
        env._normalizer.var   = getattr(agent, "_norm_var").copy()
        env._normalizer.count = getattr(agent, "_norm_count")
        env.freeze_normalizer()
    else:
        # Quick random warmup to approximate normalizer
        for _ in range(100):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _, t, tr, _ = env.step(env.action_space.sample())
                done = t or tr
        env.freeze_normalizer()


def record_episode(act_fn, env) -> tuple[list, float]:
    """Run one greedy episode, return (frames, total_reward)."""
    frames = []
    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = act_fn(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        done = terminated or truncated
    # Capture last frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    return frames, total


def add_overlay(frame: np.ndarray, label: str, reward: float | None = None) -> np.ndarray:
    """Burn text overlay onto a frame using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Semi-transparent black bar at top
    bar_h = 36
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bar = Image.new("RGBA", (img.width, bar_h), (0, 0, 0, 160))
    overlay.paste(bar, (0, 0))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a decent font, fall back to default
    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except Exception:
            font = ImageFont.load_default()

    text = label
    if reward is not None:
        text += f"   reward: {reward:+.1f}"
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def make_title_card(text: str, size: tuple[int, int], n_frames: int,
                    color=(20, 20, 40)) -> list[np.ndarray]:
    """Return n_frames identical title-card frames."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - tw) // 2
    y = (size[1] - th) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    arr = np.array(img)
    return [arr] * n_frames


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crea un video unico che mostra l'apprendimento per checkpoint."
    )
    parser.add_argument("--agent",    choices=["sarsa", "ac"], required=True)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--tag",      type=str, default="",
                        help="Prefisso checkpoint (es. 'noreplay_')")
    parser.add_argument("--variant",  default="standard",
                        choices=["standard", "wind", "turbulent", "heavy"])
    parser.add_argument("--step",     type=int, default=200,
                        help="Includi un checkpoint ogni STEP episodi (default: 200)")
    parser.add_argument("--episodes", nargs="+", default=None,
                        help="Seleziona checkpoint specifici: es. 200 600 1000 best")
    parser.add_argument("--fps",      type=int, default=30)
    parser.add_argument("--title_sec", type=float, default=1.5,
                        help="Secondi di title card tra checkpoint (default: 1.5)")
    parser.add_argument("--out",      type=str, default=None,
                        help="File di output (default: results/videos/progress_{agent}_{variant}_seed{seed}.mp4)")
    parser.add_argument("--device",   default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(args.device)

    config = load_config(args.agent)

    # Resolve checkpoints
    if args.episodes:
        model_dir = Path("models")
        ckpts = []
        for ep in args.episodes:
            if ep == "best":
                p = model_dir / f"{args.tag}{args.agent}_seed{args.seed}_best.pt"
                ckpts.append((-1, p))
            else:
                p = model_dir / f"{args.tag}{args.agent}_seed{args.seed}_ep{ep}.pt"
                ckpts.append((int(ep), p))
    else:
        ckpts = [(ep, p) for ep, p in find_checkpoints(args.agent, args.seed, args.tag, args.step)
                 if ep != -1]  # escludi best

    if not ckpts:
        sys.exit(f"[ERROR] Nessun checkpoint trovato. Controlla agent/seed/tag.")

    missing = [str(p) for _, p in ckpts if not p.exists()]
    if missing:
        sys.exit(f"[ERROR] Checkpoint mancanti:\n" + "\n".join(missing))

    print(f"Checkpoint trovati: {len(ckpts)}")
    for ep, p in ckpts:
        label = "best" if ep == -1 else f"ep{ep}"
        print(f"  {label:10s}  {p.name}")

    out_path = args.out or f"results/videos/progress_{args.tag}{args.agent}_{args.variant}_seed{args.seed}.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect all frames
    all_frames: list[np.ndarray] = []
    title_frames = int(args.fps * args.title_sec)
    frame_size = None

    from src.environment.lunar_lander_wrapper import make_env
    import gymnasium as gym

    for idx, (ep_num, ckpt_path) in enumerate(ckpts):
        label = "BEST" if ep_num == -1 else f"Episode {ep_num}"
        print(f"\n[{idx+1}/{len(ckpts)}] {label}  ({ckpt_path.name})")

        agent, act_fn = load_agent(args.agent, ckpt_path, config, device)

        env = make_env(variant=args.variant, seed=args.seed, render_mode="rgb_array")
        restore_norm(env, agent)

        frames, reward = record_episode(act_fn, env)
        env.close()

        print(f"  {len(frames)} frames  reward={reward:+.1f}")

        if not frames:
            continue

        if frame_size is None:
            frame_size = (frames[0].shape[1], frames[0].shape[0])  # (W, H)

        # Title card before each checkpoint
        card_text = f"{args.agent.upper()}  —  {label}"
        title_cards = make_title_card(card_text, frame_size, title_frames)
        all_frames.extend(title_cards)

        # Episode frames with overlay
        for f in frames:
            all_frames.append(add_overlay(f, label, reward))

    if not all_frames:
        sys.exit("[ERROR] Nessun frame registrato.")

    # Write video
    print(f"\nAssemblaggio video: {len(all_frames)} frame totali @ {args.fps} fps")
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clip = ImageSequenceClip(all_frames, fps=args.fps)
        clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    except Exception as e:
        # Fallback: imageio
        print(f"  moviepy fallito ({e}), provo imageio…")
        import imageio
        with imageio.get_writer(out_path, fps=args.fps, codec="libx264") as writer:
            for f in all_frames:
                writer.append_data(f)

    print(f"\nVideo salvato → {out_path}")
    duration = len(all_frames) / args.fps
    print(f"Durata: {duration:.1f}s  ({len(ckpts)} checkpoint)")


if __name__ == "__main__":
    main()
