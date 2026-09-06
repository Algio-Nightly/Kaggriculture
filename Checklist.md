# Kaggriculture Project Completion Checklist

This document tracks the implementation status of all components for the Kaggriculture JAX/Flax Reinforcement Learning & Supervised Training pipeline.

---

## 1. Data Pipeline & Representation
- [x] **Parser**: Raw Kaggle JSON observation dictionary traversal and inspection.
- [x] **Encoder**: Dual-Tower spatial grid tensor generation (`10x10x21` for own and opponent farms) and global scalar vector generation (`47` normalized features including derived town shop demand rates).
- [ ] `# Offline Replay Dataset Loader & Preprocessing`: Parse 224 replay JSON files into cached binary JAX datasets for fast pre-training.

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
- [ ] **Daily Decision Cadence**: Evaluates the board once per in-game day (every 24 turns) to generate macro-plans for Action Scripts.
- [ ] **The Actor Critic Cycle**: Single-step and multi-step rollout prediction for Policy ($\pi_\theta$) and Value ($V_\phi$).
- [ ] **Inference Architecture & Kaggle Entrypoint**: Production `agent(obs)` function for live Kaggle competition submission (`main.py`).

---

## 3. Macro Action Scripts & Strategy Engine
- [ ] **Action Scripts**: Deterministic pathfinding and tile-action execution subroutines (Pathfinding to shed, auto-watering loops, seed planting, animal feeding).
- [ ] **Role of each of the Action Scripts**: High-level macro strategy executor fulfilling the Brain's daily factored decision.

---

## 4. Rewards & Reward Shaping
- [ ] **The Rewards**: Sparse season win/loss reward based on final bank money delta at turn 720 ($R_{720} = \text{money}_{\text{final}} - \text{opp\_money}_{\text{final}}$).
- [ ] **Reward Pacing**: Dense turn-by-turn Net Worth delta shaping ($\Delta \text{Money} + \Delta \text{Shed Stock} + \Delta \text{Crop Assets}$) with penalties for unwatered crop decay or lost animals.

---

## 5. Distributed Training Infrastructure
- [ ] **The Learner Network**: Central JAX optimizer (Optax) maintaining master parameters $\theta_{\text{learner}}$ and computing PPO gradients.
- [ ] **The Player Network**: Parallel rollout worker instances executing fast environment inference during data collection.
- [ ] **The Supervised Training Loop and Scripts**: Behavioral Cloning pre-training on 224 offline replay files using Cross-Entropy loss before RL self-play.
- [ ] **PPO Training Loop**: Proximal Policy Optimization with GAE-$\lambda$ advantage estimation, clipped surrogate loss, value MSE loss, and entropy bonus.
- [ ] **Self Play Loop and Optimizations**: Historical opponent policy pool (league training) to prevent strategy cycling, with JAX vectorization optimizations (`jax.vmap` / `jax.lax.scan`).

---

## 6. Verification, Tooling & Submission
- [ ] `# Model Checkpoint Serialization & Weight Bundler`: Save/load JAX NNX weights via `orbax` / `msgpack` and bundle into `submission.tar.gz` for Kaggle CLI.
- [ ] `# Local Tournament & Evaluation Harness`: Benchmarking harness evaluating new checkpoints against built-in agents (`starter`, `random`, `pass`) and past self-play checkpoints.
- [ ] `# Training Metrics & Logging`: TensorBoard / Weights & Biases logging for loss curves, reward pacing, win rates, and KL divergence.
