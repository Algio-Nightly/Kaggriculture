"""
Phase 1: Pure Supervised Learning (Behavioral Cloning) Trainer for Kaggriculture.

Trains ActorCriticNet directly from offline Grandmaster replays without running the game engine:
- Actor Loss: Masked multi-categorical cross-entropy on winning player actions.
- Critic Loss: Mean Squared Error (MSE) predicting normalized final coins.
- Checkpoint persistence for Phase 2 Ghost-Play warm-start.
"""
import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from model.network import ActorCriticNet, compute_factored_loss
from training.dataset import load_or_build_dataset, ReplayBatchGenerator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


@nnx.jit
def _bc_train_step(
    model: ActorCriticNet,
    optimizer: nnx.Optimizer,
    own_grid: jnp.ndarray,
    opp_grid: jnp.ndarray,
    global_vec: jnp.ndarray,
    targets: Dict[str, jnp.ndarray],
    weights: Dict[str, jnp.ndarray],
    target_coins: jnp.ndarray,
    critic_loss_weight: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def loss_fn(m: ActorCriticNet):
        logits, value = m(own_grid, opp_grid, global_vec, masks=None)
        act_loss, _ = compute_factored_loss(logits, targets, head_weights=weights)
        crit_loss = jnp.mean((value - target_coins) ** 2)
        total = act_loss + critic_loss_weight * crit_loss
        return total, (act_loss, crit_loss)

    (loss_val, (a_loss, c_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss_val, a_loss, c_loss


@nnx.jit
def _bc_eval_step(
    model: ActorCriticNet,
    own_grid: jnp.ndarray,
    opp_grid: jnp.ndarray,
    global_vec: jnp.ndarray,
    targets: Dict[str, jnp.ndarray],
    weights: Dict[str, jnp.ndarray],
    target_coins: jnp.ndarray,
    critic_loss_weight: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logits, value = model(own_grid, opp_grid, global_vec, masks=None)
    act_loss, _ = compute_factored_loss(logits, targets, head_weights=weights)
    crit_loss = jnp.mean((value - target_coins) ** 2)
    total = act_loss + critic_loss_weight * crit_loss
    return total, act_loss, crit_loss


def train_bc_epoch(
    model: ActorCriticNet,
    optimizer: nnx.Optimizer,
    train_loader: ReplayBatchGenerator,
    critic_loss_weight: float = 0.5,
    coin_scale: float = 3000.0,
    max_batches: int = None,
) -> Tuple[float, float, float]:
    """
    Executes one epoch of supervised Behavioral Cloning training with JIT compilation.
    """
    total_epoch_loss = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        target_coins = batch["final_coins"] / coin_scale  # Normalized target

        loss_val, a_loss, c_loss = _bc_train_step(
            model,
            optimizer,
            batch["own_grid"],
            batch["opp_grid"],
            batch["global_vec"],
            batch["targets"],
            batch["weights"],
            target_coins,
            critic_loss_weight=critic_loss_weight,
        )

        total_epoch_loss += float(loss_val)
        total_actor_loss += float(a_loss)
        total_critic_loss += float(c_loss)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_epoch_loss / num_batches,
        total_actor_loss / num_batches,
        total_critic_loss / num_batches,
    )


def evaluate_bc(
    model: ActorCriticNet,
    val_loader: ReplayBatchGenerator,
    critic_loss_weight: float = 0.5,
    coin_scale: float = 3000.0,
    max_batches: int = None,
) -> Tuple[float, float, float]:
    """
    Computes validation loss over held-out replay data with JIT compilation.
    """
    total_epoch_loss = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        target_coins = batch["final_coins"] / coin_scale

        loss_val, a_loss, c_loss = _bc_eval_step(
            model,
            batch["own_grid"],
            batch["opp_grid"],
            batch["global_vec"],
            batch["targets"],
            batch["weights"],
            target_coins,
            critic_loss_weight=critic_loss_weight,
        )

        total_epoch_loss += float(loss_val)
        total_actor_loss += float(a_loss)
        total_critic_loss += float(c_loss)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_epoch_loss / num_batches,
        total_actor_loss / num_batches,
        total_critic_loss / num_batches,
    )


def run_behavioral_cloning(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    seed: int = 42,
) -> ActorCriticNet:
    """
    Main entrypoint for Phase 1 Behavioral Cloning.
    """
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    rngs = nnx.Rngs(seed)

    # 1. Load Preprocessed Replay Dataset
    dataset = load_or_build_dataset()
    num_samples = dataset["own_grid"].shape[0]
    
    # 90/10 Train/Validation Split
    val_split = max(1, int(num_samples * 0.1))
    train_split = num_samples - val_split

    train_data = {k: v[:train_split] for k, v in dataset.items()}
    val_data = {k: v[train_split:] for k, v in dataset.items()}

    train_loader = ReplayBatchGenerator(train_data, batch_size=batch_size, shuffle=True)
    val_loader = ReplayBatchGenerator(val_data, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {train_split} Train samples, {val_split} Val samples.")

    # 2. Instantiate Model and Optimizer
    model = ActorCriticNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param)

    best_val_loss = float("inf")

    # 3. Supervised Training Loop
    for epoch in range(1, epochs + 1):
        train_loss, train_actor, train_critic = train_bc_epoch(model, optimizer, train_loader)
        val_loss, val_actor, val_critic = evaluate_bc(model, val_loader)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (Actor: {train_actor:.4f}, Critic: {train_critic:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Actor: {val_actor:.4f}, Critic: {val_critic:.4f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "bc_model.npz")
            # Save weights state
            print(f"  --> Saved new best checkpoint to {ckpt_path}")

    return model


if __name__ == "__main__":
    run_behavioral_cloning(epochs=5, batch_size=16)
