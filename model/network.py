"""
Flax NNX Neural Network Modules for Kaggriculture.

Implements Dual-Tower Siamese 2D-CNN + Global MLP Late Fusion Architecture (ActorCriticNet)
with Factored Multi-Discrete Policy Heads, Critic Value Head, Forward Logit Masking,
and Backward Active-Head Gradient Masking.
"""
from typing import Dict, Tuple, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from model.action_space import HEAD_DIMS


def apply_logit_mask(logits: jnp.ndarray, mask: Optional[jnp.ndarray], penalty: float = -1e9) -> jnp.ndarray:
    """
    Applies additive penalty to invalid action indices in policy logits.
    
    Args:
        logits: Unnormalized log probabilities (..., num_actions).
        mask: Boolean mask where True = valid, False = invalid (..., num_actions).
        penalty: Large negative constant to zero out probability post-softmax.
        
    Returns:
        Masked logits array of same shape.
    """
    if mask is None:
        return logits
    return jnp.where(mask, logits, penalty)


class SpatialEncoder(nnx.Module):
    """
    2D Convolutional backbone to process a 10x10 spatial farm grid feature plane.
    Used as a Siamese shared-weights encoder for both Own Farm and Opponent Farm.
    """

    def __init__(self, in_channels: int = 21, features: int = 64, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=in_channels, out_features=32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=features, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        # Flatten spatial dimensions: batch_shape + (10 * 10 * features = 6400,)
        return x.reshape((*x.shape[:-3], -1))


class GlobalEncoder(nnx.Module):
    """
    MLP backbone to process global scalar features (market prices, inventory, time, money, town demand).
    """

    def __init__(self, in_features: int = 47, hidden_dim: int = 128, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(in_features=in_features, out_features=hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return x


class ActorCriticNet(nnx.Module):
    """
    Dual-Tower Siamese Actor-Critic Network in Flax NNX with Factored Multi-Discrete Brain.
    
    Data Flow:
    1. Spatial Branch A (Own Farm): 2D-CNN extracts 10x10x21 grid features -> Own Spatial Embedding (6400)
    2. Spatial Branch B (Opp Farm): Siamese 2D-CNN extracts 10x10x21 grid features -> Opp Spatial Embedding (6400)
    3. Scalar Branch: GlobalEncoder MLP extracts 47 state features -> Global Embedding (128)
    4. Late Fusion: Concatenates (6400 + 6400 + 128 = 12928) -> Dense(256) -> Latent vector z (256)
    5. Factored Actor Heads:
       - farming_head: 256 -> 6
       - selling_head: 256 -> 5
       - crop_head: 256 -> 5
       - labor_head: 256 -> 6
       - quadrant_head: 256 -> 4
       - tactical_head: 256 -> 15
    6. Critic Value Head:
       - 256 -> 64 -> 1 scalar baseline estimate V(s)
    """

    def __init__(
        self,
        grid_channels: int = 21,
        global_features: int = 47,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        # 1. Siamese Shared Spatial Encoder
        self.spatial_encoder = SpatialEncoder(in_channels=grid_channels, features=64, rngs=rngs)
        
        # 2. Global Scalar Encoder
        self.global_encoder = GlobalEncoder(in_features=global_features, hidden_dim=128, rngs=rngs)

        # 3. Late Fusion Layer
        fused_dim = (10 * 10 * 64) + (10 * 10 * 64) + 128  # 12928
        self.fused_dense = nnx.Linear(in_features=fused_dim, out_features=hidden_dim, rngs=rngs)

        # 4. Factored Multi-Discrete Actor Heads
        self.farming_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["farming"], rngs=rngs)
        self.selling_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["selling"], rngs=rngs)
        self.crop_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["crop"], rngs=rngs)
        self.labor_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["labor"], rngs=rngs)
        self.quadrant_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["quadrant"], rngs=rngs)
        self.tactical_head = nnx.Linear(in_features=hidden_dim, out_features=HEAD_DIMS["tactical"], rngs=rngs)

        # 5. Critic Value Head
        self.value_dense = nnx.Linear(in_features=hidden_dim, out_features=64, rngs=rngs)
        self.value_out = nnx.Linear(in_features=64, out_features=1, rngs=rngs)

    def extract_latent(self, own_grid_x: jnp.ndarray, opp_grid_x: jnp.ndarray, global_x: jnp.ndarray) -> jnp.ndarray:
        """Extracts the 256-dimensional fused latent state representation vector z."""
        own_emb = self.spatial_encoder(own_grid_x)
        opp_emb = self.spatial_encoder(opp_grid_x)
        global_emb = self.global_encoder(global_x)

        fused = jnp.concatenate([own_emb, opp_emb, global_emb], axis=-1)
        return nnx.relu(self.fused_dense(fused))

    def __call__(
        self,
        own_grid_x: jnp.ndarray,
        opp_grid_x: jnp.ndarray,
        global_x: jnp.ndarray,
        masks: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Forward pass predicting factored policy distributions and state value.
        
        Args:
            own_grid_x: (..., 10, 10, 21) Own farm tensor.
            opp_grid_x: (..., 10, 10, 21) Opponent farm tensor.
            global_x: (..., 47) Global scalar vector.
            masks: Optional dictionary of action validity masks per head.
            
        Returns:
            Tuple of:
              - Dict[str, jnp.ndarray] of masked policy logits per head.
              - jnp.ndarray of estimated scalar values V(s).
        """
        # 1. Fuse Representations
        latent = self.extract_latent(own_grid_x, opp_grid_x, global_x)

        # 2. Factored Actor Logits
        raw_logits = {
            "farming": self.farming_head(latent),
            "selling": self.selling_head(latent),
            "crop": self.crop_head(latent),
            "labor": self.labor_head(latent),
            "quadrant": self.quadrant_head(latent),
            "tactical": self.tactical_head(latent),
        }

        # 3. Apply Adaptive Forward Logit Masks
        policy_logits = {}
        for head_name, logits in raw_logits.items():
            head_mask = masks.get(head_name, None) if masks is not None else None
            policy_logits[head_name] = apply_logit_mask(logits, head_mask)

        # 4. Critic Value Prediction
        val_hidden = nnx.relu(self.value_dense(latent))
        value = self.value_out(val_hidden).squeeze(-1)

        return policy_logits, value


# =============================================================================
# Loss Functions with Adaptive Gradient Masking
# =============================================================================

def compute_factored_loss(
    policy_logits: Dict[str, jnp.ndarray],
    targets: Dict[str, jnp.ndarray],
    head_weights: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Computes masked multi-categorical cross-entropy loss across all factored heads.
    
    Applies gradient masking via head_weights: when an argument head is inactive
    (e.g., crop_param is inactive when farming_strategy != EXPAND_CROPS),
    its head_weight is set to 0.0, blocking noisy gradients to that head.
    
    Args:
        policy_logits: Dict mapping head name to (batch_size, num_actions) logits.
        targets: Dict mapping head name to (batch_size,) integer target labels.
        head_weights: Dict mapping head name to (batch_size,) float weights (1.0 = active, 0.0 = masked).
        
    Returns:
        total_loss: Scalar weighted cross-entropy loss.
        per_head_losses: Dict mapping head names to scalar mean losses.
    """
    total_loss = 0.0
    per_head_losses = {}

    for head_name, logits in policy_logits.items():
        if head_name not in targets:
            continue
            
        target = targets[head_name]
        # Standard Cross-Entropy: -log(softmax(logits)[target])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(target, num_classes=logits.shape[-1])
        nll = -jnp.sum(one_hot * log_probs, axis=-1)

        # Apply active-head gradient mask
        if head_weights is not None and head_name in head_weights:
            weight = head_weights[head_name]
            weighted_nll = nll * weight
            head_loss = jnp.sum(weighted_nll) / jnp.maximum(jnp.sum(weight), 1.0)
        else:
            head_loss = jnp.mean(nll)

        per_head_losses[head_name] = head_loss
        total_loss = total_loss + head_loss

    return total_loss, per_head_losses
