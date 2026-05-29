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
# Train both agents — all 5 seeds (CPU recommended, ~1-2h):
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
