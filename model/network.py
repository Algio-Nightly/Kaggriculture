"""
Flax NNX Neural Network Modules for Kaggriculture.
Implements Dual-Tower Siamese 2D-CNN + Global MLP Late Fusion Architecture.
"""
from typing import Dict, Tuple
from flax import nnx
import jax.numpy as jnp


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
        # Flatten spatial dimensions (10 * 10 * features = 6400)
        return x.reshape((*x.shape[:-3], -1))


class GlobalEncoder(nnx.Module):
    """
    MLP backbone to process global scalar features (market prices, inventory, time, money, town product demand).
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
    Dual-Tower Siamese Actor-Critic Network in Flax NNX for Kaggriculture.
    
    Architecture:
    1. Spatial Branch A (Own Farm): 2D-CNN extracts 10x10x21 grid features -> Own Spatial Embedding (6400)
    2. Spatial Branch B (Opponent Farm): Siamese 2D-CNN extracts 10x10x21 grid features -> Opponent Spatial Embedding (6400)
    3. Scalar Branch: MLP extracts Global Market/State features -> Global Embedding (128)
    4. Late Fusion: Concatenates (Own Embedding + Opponent Embedding + Global Embedding) = 12928 -> Latent (256)
    5. Heads: Policy Network (Actor) & State Value Network (Critic)
    """

    def __init__(
        self,
        grid_channels: int = 21,
        global_features: int = 47,
        num_farmer_actions: int = 15,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        # Siamese shared spatial encoder for both farms
        self.spatial_encoder = SpatialEncoder(in_channels=grid_channels, features=64, rngs=rngs)
        self.global_encoder = GlobalEncoder(in_features=global_features, hidden_dim=128, rngs=rngs)

        # Dual-Tower Late Fusion dimension: (10*10*64) + (10*10*64) + 128 = 12928
        fused_dim = (10 * 10 * 64) + (10 * 10 * 64) + 128
        self.fused_dense = nnx.Linear(in_features=fused_dim, out_features=hidden_dim, rngs=rngs)

        # Policy Head (Actor) & Value Head (Critic)
        self.farmer_head = nnx.Linear(in_features=hidden_dim, out_features=num_farmer_actions, rngs=rngs)
        self.value_head = nnx.Linear(in_features=hidden_dim, out_features=1, rngs=rngs)

    def __call__(
        self, own_grid_x: jnp.ndarray, opp_grid_x: jnp.ndarray, global_x: jnp.ndarray
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        # 1. Siamese Feature Extraction for Own and Opponent Farms
        own_emb = self.spatial_encoder(own_grid_x)
        opp_emb = self.spatial_encoder(opp_grid_x)
        global_emb = self.global_encoder(global_x)

        # 2. Dual-Tower Late Fusion
        fused = jnp.concatenate([own_emb, opp_emb, global_emb], axis=-1)
        latent = nnx.relu(self.fused_dense(fused))

        # 3. Predict Policy Logits & State Value
        farmer_logits = self.farmer_head(latent)
        value = self.value_head(latent).squeeze(-1)

        policy_outputs = {
            "farmer": farmer_logits,
        }

        return policy_outputs, value
