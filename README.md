# Deep SARSA vs Advantage Actor-Critic on LunarLander-v3

**SEAI Project — Francesco Galardi, Andrea Vagnoli**  
*Symbolic and Evolutionary Artificial Intelligence — University of Pisa, 2026*

## Project Summary

Standard RL Project (SRLP) comparing two fundamentally different RL paradigms:

| Algorithm | Type | Exploration |
|-----------|------|-------------|
| **Deep SARSA** | On-policy, value-based (TD) | ε-greedy |
| **A2C (Actor-Critic)** | On-policy, policy-gradient | Entropy regularisation |

Environment: `LunarLander-v3` (Gymnasium) + 3 custom variants for generalisation.

## Repository Structure

```
project/
├── config/
│   ├── sarsa_config.yaml          # SARSA hyperparameters
│   └── actor_critic_config.yaml   # A2C hyperparameters
├── src/
│   ├── environment/
│   │   └── lunar_lander_wrapper.py  # Custom wrapper + variants
│   ├── networks/
│   │   ├── sarsa_network.py         # Q-network (MLP)
│   │   └── actor_critic_network.py  # Actor + Critic networks
│   ├── agents/
│   │   ├── deep_sarsa.py            # Deep SARSA agent
│   │   └── actor_critic.py          # A2C agent
│   └── utils/
│       ├── replay_buffer.py         # SARSA replay buffer
│       ├── logger.py                # CSV + TensorBoard logger
│       └── metrics.py               # Statistics + plots
├── notebooks/
│   └── analysis.ipynb              # Interactive analysis
├── train.py                        # Multi-seed training script
├── evaluate.py                     # Evaluation on all variants
├── compare.py                      # Statistical comparison
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Train both agents (all 5 seeds, ~40 min on CPU):
python train.py

# Quick smoke-test (1 seed, 100 episodes):
python train.py --episodes 100 --seeds 42

# Evaluate on all environment variants:
python evaluate.py --sarsa_ckpt models/sarsa_seed42_ep2000.pt \
                   --ac_ckpt    models/ac_seed42_ep2000.pt

# Generate comparison plots + Welch's t-test:
python compare.py
```

## Referenced Repositories

- Deep SARSA base: [JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3](https://github.com/JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v3)
- Actor-Critic base: [nikhilbarhate99/Actor-Critic-PyTorch](https://github.com/nikhilbarhate99/Actor-Critic-PyTorch)

Both were significantly extended with: unified framework, custom environment wrapper,
multi-seed training, statistical analysis, generalisation evaluation, and inference
latency benchmarking.
