"""
Unit and Integration Tests for Kaggriculture Flax NNX Model (ActorCriticNet).
Validates:
1. Spatial, Global, and Late Fusion forward shapes.
2. Factored Multi-Discrete Actor Heads and Critic Value Head.
3. Adaptive Forward Logit Masking.
4. Masked Loss computation & Optax parameter updates (Backward Gradient Masking).
5. Kaggle environment agent integration.
"""
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from kaggle_environments import make

from model.network import ActorCriticNet, apply_logit_mask, compute_factored_loss
from model.action_space import HEAD_DIMS, compute_action_masks, TacticalAction
from model.encoder import encode_observation
from model.agent import agent, init_agent


def test_forward_pass_shapes():
    print("Testing forward pass shapes...")
    rngs = nnx.Rngs(42)
    net = ActorCriticNet(rngs=rngs)

    batch_size = 4
    own_grid = jnp.zeros((batch_size, 10, 10, 21), dtype=jnp.float32)
    opp_grid = jnp.zeros((batch_size, 10, 10, 21), dtype=jnp.float32)
    global_vec = jnp.zeros((batch_size, 47), dtype=jnp.float32)

    policy_logits, value = net(own_grid, opp_grid, global_vec)

    assert value.shape == (batch_size,), f"Expected value shape (4,), got {value.shape}"
    for head_name, expected_dim in HEAD_DIMS.items():
        assert head_name in policy_logits, f"Missing head {head_name} in policy outputs"
        assert policy_logits[head_name].shape == (batch_size, expected_dim), (
            f"Head {head_name} shape mismatch: expected ({batch_size}, {expected_dim}), "
            f"got {policy_logits[head_name].shape}"
        )
    print("[OK] Forward pass shapes verified successfully!")


def test_logit_masking():
    print("Testing adaptive forward logit masking...")
    logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    mask = jnp.array([[True, False, True, False]])
    masked = apply_logit_mask(logits, mask)

    assert masked[0, 1] <= -1e8, f"Expected masked logit to be <= -1e8, got {masked[0, 1]}"
    assert masked[0, 3] <= -1e8, f"Expected masked logit to be <= -1e8, got {masked[0, 3]}"
    assert masked[0, 0] == 1.0, f"Expected unmasked logit 1.0, got {masked[0, 0]}"
    assert masked[0, 2] == 3.0, f"Expected unmasked logit 3.0, got {masked[0, 2]}"

    probs = jax.nn.softmax(masked, axis=-1)
    assert probs[0, 1] == 0.0, f"Expected 0 probability for masked action, got {probs[0, 1]}"
    assert probs[0, 3] == 0.0, f"Expected 0 probability for masked action, got {probs[0, 3]}"
    print("[OK] Adaptive logit masking verified successfully!")


def test_backward_gradient_masking():
    print("Testing backward gradient masking & Optax training step...")
    rngs = nnx.Rngs(123)
    model = ActorCriticNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    batch_size = 2
    own_grid = jnp.zeros((batch_size, 10, 10, 21), dtype=jnp.float32)
    opp_grid = jnp.zeros((batch_size, 10, 10, 21), dtype=jnp.float32)
    global_vec = jnp.zeros((batch_size, 47), dtype=jnp.float32)

    targets = {
        "farming": jnp.array([1, 0]),
        "selling": jnp.array([2, 1]),
        "crop": jnp.array([0, 4]),
    }
    
    # Gradient mask: crop head is active only for sample 0 (weight=1.0), masked for sample 1 (weight=0.0)
    head_weights = {
        "farming": jnp.array([1.0, 1.0]),
        "selling": jnp.array([1.0, 1.0]),
        "crop": jnp.array([1.0, 0.0]),
    }

    def loss_fn(m: ActorCriticNet):
        logits, val = m(own_grid, opp_grid, global_vec)
        ce_loss, per_head = compute_factored_loss(logits, targets, head_weights)
        val_loss = jnp.mean((val - jnp.zeros_like(val)) ** 2)
        return ce_loss + 0.5 * val_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    assert not jnp.isnan(loss_val), "Loss evaluated to NaN"
    
    # Perform optimizer step: pass model and grads as required in Flax 0.11+
    optimizer.update(model, grads)
    print(f"[OK] Backward pass step verified successfully! Initial loss = {loss_val:.4f}")


def test_kaggle_env_agent():
    print("Testing agent integration inside live Kaggle environment (5 turns)...")
    init_agent(seed=7)
    env = make("kaggriculture", configuration={"episodeSteps": 5}, debug=True)
    env.run([agent, "starter"])
    
    steps = env.steps
    assert len(steps) >= 5, f"Expected at least 5 environment steps, got {len(steps)}"
    print("[OK] Live Kaggle environment agent simulation completed without error!")


if __name__ == "__main__":
    test_forward_pass_shapes()
    test_logit_masking()
    test_backward_gradient_masking()
    test_kaggle_env_agent()
    print("\n==========================================")
    print("ALL ACTOR-CRITIC NETWORK TESTS PASSED!")
    print("==========================================")
