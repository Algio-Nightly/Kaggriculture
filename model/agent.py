"""
Model Agent Entrypoint for Kaggriculture using Dual-Tower Flax NNX.

Interfaces between Kaggle environment observations and the Dual-Tower Flax NNX model
(ActorCriticNet), utilizing forward logit masking and tactical action decoding.
"""
from typing import Dict, Any
from flax import nnx
import jax
import jax.numpy as jnp
from model.encoder import encode_observation
from model.network import ActorCriticNet
from model.action_space import compute_action_masks, tactical_index_to_dict

# Global NNX model instance
_model: ActorCriticNet = None


def init_agent(seed: int = 0) -> ActorCriticNet:
    """Initialize NNX model weights."""
    global _model
    rngs = nnx.Rngs(seed)
    _model = ActorCriticNet(rngs=rngs)
    return _model


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kaggle environment agent interface function.
    
    Args:
        obs: Environment observation dictionary.
        
    Returns:
        Action dictionary: {"farmer": [...], "hands": [...], "market": [...]}
    """
    global _model
    if _model is None:
        init_agent()

    # 1. Encode spatial and scalar state representations
    own_grid_t, opp_grid_t, global_t = encode_observation(obs)

    # 2. Compute state-dependent validity masks
    raw_masks = compute_action_masks(obs)
    jax_masks = {k: jnp.asarray(v) for k, v in raw_masks.items()}

    # 3. Model forward pass with logit masking
    policy_outputs, value_est = _model(own_grid_t, opp_grid_t, global_t, masks=jax_masks)
    
    # 4. Select tactical action for turn execution (highest masked logit)
    tactical_logits = policy_outputs["tactical"]
    action_idx = int(jnp.argmax(tactical_logits))

    return tactical_index_to_dict(action_idx)
