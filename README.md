# Deep SARSA vs Advantage Actor-Critic on LunarLander-v3

**SEAI Project — Francesco Galardi, Andrea Vagnoli**  
*Symbolic and Evolutionary Artificial Intelligence — University of Pisa, 2026*

## Project Summary

Standard RL Project (SRLP) comparing two fundamentally different RL paradigms:

| Algorithm | Type | Exploration | Training |
|-----------|------|-------------|----------|
| **Deep SARSA** | On-policy, value-based (TD) | ε-greedy | 3 000 episodes, 1 env |
| **A2C (Actor-Critic)** | On-policy, policy-gradient | Entropy regularisation | 5 000 episodes, 8 parallel envs |

Environment: `LunarLander-v3` (Gymnasium) + 3 custom variants for generalisation tests.

### Key results (5 seeds, best checkpoint, 100 eval episodes)

| Variant | Deep SARSA | A2C | Winner |
|---------|-----------|-----|--------|
| Standard | **202.8 ± 22.2** | 180.0 ± 39.3 | SARSA ✓ (solved) |
| Wind | 112.3 ± 39.1 | **136.3 ± 51.6** | A2C |
| Turbulent | 47.2 ± 36.8 | **126.1 ± 32.0** | A2C |
| Heavy gravity | 145.8 ± 22.7 | **167.9 ± 46.1** | A2C |

Welch's t-test on training finals: p = 0.039 (SARSA > A2C, significant at α = 0.05).

## Repository Structure

```
project/
├── config/
│   ├── sarsa_config.yaml          # SARSA hyperparameters
│   └── actor_critic_config.yaml   # A2C hyperparameters
├── src/
│   ├── environment/
│   │   └── lunar_lander_wrapper.py  # Custom wrapper + variants + VecRunningNormalizer
│   ├── networks/
│   │   ├── sarsa_network.py         # Q-network (MLP)
│   │   └── actor_critic_network.py  # Actor + Critic networks
│   ├── agents/
│   │   ├── deep_sarsa.py            # Deep SARSA agent
│   │   └── actor_critic.py          # A2C agent (vectorised training)
│   └── utils/
│       ├── replay_buffer.py         # SARSA replay buffer
│       ├── logger.py                # CSV logger
│       └── metrics.py               # Statistics + plots
├── train.py                        # Multi-seed training + grid search
├── evaluate.py                     # Single-seed evaluation on all variants
├── evaluate_all.py                 # Aggregate evaluation across all seeds
├── compare.py                      # Learning curves + Welch's t-test
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train both agents — all 5 seeds:
python train.py --device cpu

# Train a single agent:
python train.py --agent sarsa --device cpu
python train.py --agent ac    --device cpu

# Quick smoke-test (1 seed, 100 episodes):
python train.py --agent sarsa --episodes 100 --seeds 42 --device cpu
```

### Grid search

```bash
# SARSA — search learning rate × epsilon decay:
python train.py --agent sarsa --seeds 42 --device cpu \
    --grid alpha=0.0003,0.0005 epsilon_decay=0.995,0.997

# A2C — search entropy coefficient:
python train.py --agent ac --seeds 42 --device cpu \
    --grid entropy_coef=0.001,0.003,0.005,0.01

# Results are printed ranked and saved to results/grid_{agent}_{timestamp}.json
```

### Evaluation

```bash
# Single seed — evaluate best checkpoint on all variants:
python evaluate.py --agent sarsa --seed 42 --ckpt_type best --device cpu
python evaluate.py --agent ac    --seed 42 --ckpt_type best --device cpu

# All seeds — aggregate evaluation (mean ± std across 5 seeds):
python evaluate_all.py --n_eval 100 --ckpt_type best --device cpu
```

### Analysis

```bash
# Learning curves + Welch's t-test on training finals:
python compare.py
```

## Hyperparameters

Both agents use **cosine annealing** learning-rate scheduling and **best-checkpoint saving** (rolling mean of last 100 episodes). Final hyperparameters were selected via grid search.

| Parameter | Deep SARSA | A2C |
|-----------|-----------|-----|
| LR (start → end) | 5×10⁻⁴ → 5×10⁻⁵ | actor 3×10⁻⁴ → 10⁻⁵ / critic 10⁻⁴ → 10⁻⁵ |
| γ | 0.99 | 0.99 |
| ε decay | 0.995 (→ 0.01 at ep ~920) | — |
| Entropy coef β | — | 0.003 (grid search) |
| n-step horizon | — | 200 (near-Monte Carlo) |
| Parallel envs | 1 | 8 (SyncVectorEnv) |
| Batch size | 128 | — |
| Buffer | 50 000 | — |
| Episodes | 3 000 | 5 000 |
| Seeds | 42, 123, 456, 789, 1234 | 42, 123, 456, 789, 1234 |

## Referenced Repositories

- Deep SARSA base: [JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3](https://github.com/JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3)
- Actor-Critic base: [nikhilbarhate99/Actor-Critic-PyTorch](https://github.com/nikhilbarhate99/Actor-Critic-PyTorch)

Both were significantly extended with: unified training framework, custom environment wrapper with observation normalisation, vectorised environments (A2C), cosine annealing LR scheduling, grid search CLI, best-checkpoint auto-saving, multi-seed aggregate evaluation, and statistical analysis.

---

## Ablation Studies

All ablations use the same CLI as standard training — pass flags to override config values.  
Checkpoints are saved with the `--tag` prefix so they never overwrite the baseline.

### 1 · Deep SARSA — con e senza Replay Buffer

**Baseline (con buffer)** — comportamento di default, nessun flag necessario:
```bash
python train.py --agent sarsa --seeds 42 --episodes 3000 --device cpu
```

**Senza replay buffer** (online TD, correlazione massima tra osservazioni):
```bash
python train.py --agent sarsa --seeds 42 --episodes 3000 \
    --no_replay_buffer --tag noreplay_
```

Confronta le learning curve nei log in `results/logs/` per vedere l'impatto
della decorrelazione tramite buffer.

---

### 2 · Deep SARSA — Ambienti Paralleli senza Replay Buffer

Decorrelazione via diversità degli ambienti invece del buffer.  
`--parallel_envs N` implica automaticamente `--no_replay_buffer`.

```bash
# 4 ambienti paralleli
python train.py --agent sarsa --seeds 42 --episodes 3000 \
    --parallel_envs 4 --tag parallel4_

# 8 ambienti paralleli
python train.py --agent sarsa --seeds 42 --episodes 3000 \
    --parallel_envs 8 --tag parallel8_
```

Confronto a tre vie: baseline (buffer) · no buffer (1 env) · no buffer (N env).

---

### 3 · A2C — Sweep di n\_steps

`n_steps` controlla il trade-off bias–varianza del return:
- valori piccoli → TD puro (alto bias, bassa varianza)
- valori grandi → near-Monte Carlo (basso bias, alta varianza)

```bash
# n_steps = 5 (TD breve)
python train.py --agent ac --seeds 42 --n_steps 5 --tag nstep5_

# n_steps = 20
python train.py --agent ac --seeds 42 --n_steps 20 --tag nstep20_

# n_steps = 50
python train.py --agent ac --seeds 42 --n_steps 50 --tag nstep50_

# n_steps = 200 (baseline)
python train.py --agent ac --seeds 42
```

---

### 4 · A2C — Sweep di num\_envs (a n\_steps fisso)

Fissa `n_steps` piccolo e varia il numero di ambienti paralleli per vedere
se la diversità compensa la finestra corta.

```bash
# 1 env, n_steps 20
python train.py --agent ac --seeds 42 --n_steps 20 --num_envs 1 --tag e1s20_

# 4 env, n_steps 20
python train.py --agent ac --seeds 42 --n_steps 20 --num_envs 4 --tag e4s20_

# 8 env, n_steps 20  (stesse transizioni totali del baseline e8s200 ma orizz. più corto)
python train.py --agent ac --seeds 42 --n_steps 20 --num_envs 8 --tag e8s20_

# 16 env, n_steps 20
python train.py --agent ac --seeds 42 --n_steps 20 --num_envs 16 --tag e16s20_
```

---

### 5 · Reward Shaping (entrambi gli agenti)

Aggiunge penalità configurabili sopra al reward originale dell'ambiente.  
I coefficienti positivi producono penalità (reward sottratto).

Componenti disponibili:

| Chiave | Formula | Significato |
|--------|---------|-------------|
| `angle_penalty` | `−c·\|obs[4]\|` | penalizza l'inclinazione |
| `angular_vel_penalty` | `−c·\|obs[5]\|` | penalizza la rotazione |
| `vel_penalty` | `−c·√(vx²+vy²)` | penalizza la velocità totale |
| `x_penalty` | `−c·\|obs[0]\|` | penalizza l'offset orizzontale dal pad |
| `fuel_penalty_main` | `−c` se `action==2` | penalizza il motore principale |
| `fuel_penalty_side` | `−c` se `action∈{1,3}` | penalizza i motori laterali |

**Esempi:**

```bash
# SARSA — penalizza inclinazione e rotazione
python train.py --agent sarsa --seeds 42 \
    --reward_shaping angle_penalty=0.5 angular_vel_penalty=0.1 \
    --tag shape_angle_

# A2C — penalizza consumo carburante
python train.py --agent ac --seeds 42 \
    --reward_shaping fuel_penalty_main=0.3 fuel_penalty_side=0.1 \
    --tag shape_fuel_

# A2C — shaping aggressivo su velocità
python train.py --agent ac --seeds 42 \
    --reward_shaping vel_penalty=0.5 angle_penalty=0.3 \
    --tag shape_vel_
```

In alternativa, modifica direttamente la sezione `reward_shaping:` nei file
`config/sarsa_config.yaml` o `config/actor_critic_config.yaml` e imposta
`enabled: true`.

---

### Riepilogo flag CLI per ablation

| Flag | Agente | Descrizione |
|------|--------|-------------|
| `--no_replay_buffer` | SARSA | Disabilita il replay buffer |
| `--parallel_envs N` | SARSA | N ambienti paralleli, nessun buffer |
| `--n_steps N` | A2C | Override di `n_step_horizon` |
| `--num_envs N` | A2C | Override di `num_envs` |
| `--reward_shaping K=V ...` | entrambi | Pesi di reward shaping |
| `--tag PREFIX` | entrambi | Prefisso per i checkpoint (evita sovrascritture) |

I checkpoint ablation vengono salvati come `models/{tag}sarsa_seed{seed}_best.pt`.  
Per valutare un checkpoint ablation usa `evaluate.py --sarsa_ckpt models/{tag}...`.
