"""
Actor-Critic Rollout Cycle and Generalized Advantage Estimation (GAE) for Kaggriculture.

Implements:
1. Stochastic action sampling across factored multi-discrete heads with log-prob tracking.
2. Transition data structure (PyTree compatible).
3. Vectorized GAE-lambda advantage and return computation in JAX.
4. RolloutBuffer for trajectory storage and PPO batch preparation.
"""
from typing import Dict, Tuple, NamedTuple, Optional, Any, List
import jax
import jax.numpy as jnp
from model.network import apply_logit_mask


class Transition(NamedTuple):
    """Single-step transition tuple for reinforcement learning rollouts."""
    own_grid: jnp.ndarray       # (..., 10, 10, 21)
    opp_grid: jnp.ndarray       # (..., 10, 10, 21)
    global_vec: jnp.ndarray     # (..., 47)
    actions: Dict[str, jnp.ndarray]  # Dict of head name -> action integer
    log_prob: jnp.ndarray       # Joint log probability sum_h log pi(a^h | s)
    value: jnp.ndarray          # Critic value estimate V(s)
    reward: jnp.ndarray         # Scalar transition reward
    done: jnp.ndarray           # Terminal/done boolean flag
    masks: Dict[str, jnp.ndarray]  # Head validity masks


def sample_factored_action(
    key: jax.Array,
    policy_logits: Dict[str, jnp.ndarray],
    masks: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Samples actions across all factored multi-discrete heads and computes joint log-probability.
    
    Args:
        key: JAX PRNGKey.
        policy_logits: Dict mapping head name to logits tensor (..., num_actions).
        masks: Optional Dict mapping head name to boolean mask (..., num_actions).
        
    Returns:
        actions: Dict mapping head name to integer action tensors (...,).
        head_log_probs: Dict mapping head name to individual log probabilities (...,).
        total_log_prob: Joint log probability tensor (...,) summed over active heads.
    """
    actions = {}
    head_log_probs = {}
    total_log_prob = None

    for head_name, logits in policy_logits.items():
        key, subkey = jax.random.split(key)
        head_mask = masks.get(head_name, None) if masks is not None else None
        
        # Apply validity mask before sampling
        masked_logits = apply_logit_mask(logits, head_mask)
        
        # Sample discrete categorical action
        sample = jax.random.categorical(subkey, masked_logits, axis=-1)
        actions[head_name] = sample
        
        # Compute log probability: log_softmax(logits)[sample]
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        one_hot = jax.nn.one_hot(sample, num_classes=masked_logits.shape[-1])
        action_log_prob = jnp.sum(one_hot * log_probs, axis=-1)
        head_log_probs[head_name] = action_log_prob
        
        # Sum joint log probabilities
        if total_log_prob is None:
            total_log_prob = action_log_prob
        else:
            total_log_prob = total_log_prob + action_log_prob

    return actions, head_log_probs, total_log_prob


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: jnp.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Generalized Advantage Estimation (GAE-lambda) using jax.lax.scan.
    
    Args:
        rewards: Tensor of shape (T, ...), step rewards.
        values: Tensor of shape (T, ...), value estimates V(s_t).
        dones: Tensor of shape (T, ...), done flags.
        next_value: Tensor of shape (...), estimated value of the state after step T.
        gamma: Discount factor.
        lambda_: GAE trace decay parameter.
        
    Returns:
        advantages: Tensor of shape (T, ...), computed advantages.
        returns: Tensor of shape (T, ...), empirical targets (advantages + values).
    """
    # Append next_value for terminal bootstrap
    all_values = jnp.concatenate([values, jnp.expand_dims(next_value, axis=0)], axis=0)

    def _gae_step(carried_advantage, transition_tuple):
        r, v, v_next, d = transition_tuple
        # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        delta = r + gamma * v_next * (1.0 - d) - v
        advantage = delta + gamma * lambda_ * (1.0 - d) * carried_advantage
        return advantage, advantage

    # Scan backward from step T-1 down to 0
    tuples = (rewards, all_values[:-1], all_values[1:], dones)
    initial_adv = jnp.zeros_like(next_value)
    _, advantages = jax.lax.scan(_gae_step, initial_adv, tuples, reverse=True)

    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """In-memory rollout trajectory accumulator for PPO training."""

    def __init__(self, capacity: int = 720):
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add(self, transition: Transition):
        """Append a single-step transition."""
        self.transitions.append(transition)

    def is_full(self) -> bool:
        return len(self.transitions) >= self.capacity

    def clear(self):
        self.transitions.clear()

    def get_trajectory_pytree(self) -> Transition:
        """Stacks list of Transitions into a batched PyTree of shape (T, ...)."""
        if not self.transitions:
            raise ValueError("Buffer is empty!")
            
        return jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0),
            *self.transitions
        )
