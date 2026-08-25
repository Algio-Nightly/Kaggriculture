"""
Model Agent Entrypoint for Kaggriculture using Dual-Tower Flax NNX.
Interfaces between environment observations and the Dual-Tower Flax NNX model.
"""
from typing import Dict, Any
from flax import nnx
import jax.numpy as jnp
from model.encoder import encode_observation
from model.network import ActorCriticNet
from model.action_space import action_index_to_dict

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

    own_grid_t, opp_grid_t, global_t = encode_observation(obs)
    policy_outputs, _ = _model(own_grid_t, opp_grid_t, global_t)
    
    farmer_logits = policy_outputs["farmer"]
    action_idx = int(jnp.argmax(farmer_logits))

    return action_index_to_dict(action_idx)
