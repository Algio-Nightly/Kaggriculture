# Kaggriculture Project Completion Checklist

This document tracks the implementation status of all components for the Kaggriculture JAX/Flax Reinforcement Learning & Supervised Training pipeline.

---

## 1. Data Pipeline & Representation
- [x] **Parser**: Raw Kaggle JSON observation dictionary traversal and inspection.
- [x] **Encoder**: Dual-Tower spatial grid tensor generation (`10x10x21` for own and opponent farms) and global scalar vector generation (`47` normalized features including derived town shop demand rates).
- [x] **Offline Replay Dataset Loader & Preprocessing**: Parse replay JSON files into cached binary JAX datasets for Phase 1 Behavioral Cloning (`training/dataset.py`).

---

## 2. Core Model Architecture & Brain
- [x] **The Actual Model In Between**: Dual-Tower Siamese 2D-CNN + Global MLP Late Fusion Network (`ActorCriticNet`).
- [x] **Factored Multi-Discrete Action Space (The Brain)**: Parallel discrete output heads branching from latent vector $\mathbf{z}$:
  - **Farming Strategy Head**: Selects core macro-protocol (`Sustain`, `Hoard`, `Expand`, etc.).
  - **Selling Strategy Head**: Selects market posture (`Hold`, `Drip-Feed`, `Dump`, etc.).
  - **Target Parameter Heads**: Discrete argument selections (`Crop_Type`, `Labor_Scale`, `Target_Quadrant`).
- [x] **Adaptive Logit & Gradient Masking**:
  - **Forward Pass Logit Masking**: Suppresses illegal combinations via $-10^9$ logit penalty.
  - **Backward Pass Gradient Masking**: Masks out loss on inactive parameter heads to prevent noisy weight updates.
- [x] **Daily Decision Cadence**: Evaluates the board once per in-game day (every 24 turns) to generate macro-plans for Action Scripts.
- [x] **The Actor Critic Cycle**: Single-step and multi-step rollout prediction for Policy ($\pi_\theta$) and Value ($V_\phi$).
- [x] **Inference Architecture & Kaggle Entrypoint**: Production `agent(obs)` function for live Kaggle competition submission (`main.py`).

---

## 3. Macro Action Scripts & Strategy Engine
- [x] **Action Scripts (Abstraction Level 1)**: Parameterized 1-turn action primitives (`farming_ops`, `livestock_ops`, `inventory_ops`, `market_ops`, `pathfinding`) for 1-step RL execution.
- [x] **Role of each of the Action Scripts**: 1-turn parameterized operations executing direct moves, tile operations, and market queues cross-validated against official game rules.
- [ ] `# Abstraction Level 2 Multi-Turn Macro Scripts`: High-level multi-step sweep and goal routines (Reserved for Phase 2).

---

## 4. Rewards & Reward Shaping
- [x] **Delta Net Worth ($\Delta\text{NW}$) Engine**: Real-time monetary evaluation $NW_t = \text{Cash}_t + \text{ShedValue}_t + \text{FieldValue}_t + \text{SeedValue}_t$.
- [x] **Step Reward & Living Penalty**: Dense turn reward $R_t = (NW_t - NW_{t-1}) - \lambda$ with penalty $\lambda = 0.05$.
- [x] **Terminal Win Bonus**: Sparse outcome delta bonus at turn 720.

---

## 5. 3-Phase Training Progression
- [x] **Phase 1: Pure Supervised Learning (Behavioral Cloning)**: Offline pre-training on 224 replay JSONs using Masked Categorical Cross-Entropy (Actor) and MSE vs. Final Coins (Critic) (`training/train_bc.py`).
- [x] **Phase 2: Ghost-Play Reinforcement Learning**: Online PPO optimization against scripted historical replay trajectories on matching episode seeds (`training/ghost_env.py`).
- [ ] **Phase 3: True Self-Play PPO League**: Evolutionary arms race training on randomized seeds against past self-play snapshots pool ($\pi_{t-50}$).

---

## 6. Verification, Tooling & Submission
- [ ] `# Model Checkpoint Serialization & Weight Bundler`: Save/load JAX NNX weights via `orbax` / `msgpack` and bundle into `submission.tar.gz` for Kaggle CLI.
- [ ] `# Local Tournament & Evaluation Harness`: Benchmarking harness evaluating new checkpoints against built-in agents (`starter`, `random`, `pass`) and past self-play checkpoints.
- [ ] `# Training Metrics & Logging`: TensorBoard / Weights & Biases logging for loss curves, reward pacing, win rates, and KL divergence.
