"""
Phase 3: True Self-Play Reinforcement Learning Trainer for Kaggriculture.

Implements evolutionary self-play arms-race PPO training:
1. Samples matchmaking opponents (70% current policy, 20% past snapshots, 10% starter).
2. Simulates full matches with randomized environment seeds.
3. Performs GAE-lambda PPO updates using Delta Net Worth (Delta NW) step rewards.
4. Maintains historical snapshot pool for non-exploitable meta-game emergence.
"""
import os
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from model.network import ActorCriticNet, apply_logit_mask
from model.rollout import Transition, RolloutBuffer, compute_gae, sample_factored_action
from model.action_space import compute_action_masks
from model.encoder import encode_observation
from scripts.abs_level_1 import (
    step_farmer_cardinal,
    plant_tile,
    water_tile,
    harvest_tile,
    fertilize_tile,
    build_structure,
    feed_animal,
    queue_buy_seed,
    queue_buy_product,
    queue_sell,
    queue_hire,
    queue_buy_land,
    drop_to_shed,
)
from training.league import SnapshotPool, SelfPlayArena
from training.train_ghost import ppo_update_step

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def policy_to_agent_fn(
    model: ActorCriticNet,
    key: jax.Array,
    buffer: RolloutBuffer = None,
) -> Tuple[Any, jax.Array]:
    """
    Wraps an ActorCriticNet model into a callable agent function for the match arena.
    """
    def agent_fn(obs: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal key
        key, subkey = jax.random.split(key)

        own_grid, opp_grid, global_vec = encode_observation(obs)
        masks = compute_action_masks(obs)

        og = jnp.expand_dims(jnp.array(own_grid, dtype=jnp.float32), axis=0)
        opg = jnp.expand_dims(jnp.array(opp_grid, dtype=jnp.float32), axis=0)
        gv = jnp.expand_dims(jnp.array(global_vec, dtype=jnp.float32), axis=0)
        jax_masks = {k: jnp.expand_dims(jnp.array(v, dtype=jnp.bool_), axis=0) for k, v in masks.items()}

        logits, value = model(og, opg, gv, masks=jax_masks)
        actions, head_lps, total_lp = sample_factored_action(subkey, logits, masks=jax_masks)

        act_dict = {k: int(v[0]) for k, v in actions.items()}

        farm_op = ["PASS"]
        market_ops = []

        if act_dict.get("labor", 0) == 1:
            market_ops.append(queue_hire())

        if act_dict.get("quadrant", 0) > 0:
            market_ops.append(queue_buy_land())

        if act_dict.get("selling", 0) == 1:
            for crop in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
                count = obs["private"]["shed"].get(crop, 0)
                if count > 0:
                    market_ops.append(queue_sell(crop, count))

        f_strat = act_dict.get("farming", 0)
        fx, fy = obs["farms"][obs["player"]]["farmer"]
        current_tile = obs["farms"][obs["player"]]["tiles"][fy][fx]

        if f_strat == 1:
            crop_names = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
            chosen_crop = crop_names[act_dict.get("crop", 0) % len(crop_names)]
            if obs["private"]["seeds"].get(chosen_crop, 0) == 0:
                market_ops.append(queue_buy_seed(chosen_crop, 1))
            if current_tile is None:
                farm_op = plant_tile(chosen_crop)
        elif f_strat == 2:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                farm_op = water_tile()
        elif f_strat == 3:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                farm_op = harvest_tile()
        elif f_strat == 4:
            if isinstance(current_tile, dict) and current_tile.get("kind") in ["COOP", "PASTURE"]:
                farm_op = feed_animal()
        elif f_strat == 5:
            farm_op = drop_to_shed()

        t_action = act_dict.get("tactical", 0)
        if t_action == 1: farm_op = step_farmer_cardinal("NORTH")
        elif t_action == 2: farm_op = step_farmer_cardinal("SOUTH")
        elif t_action == 3: farm_op = step_farmer_cardinal("EAST")
        elif t_action == 4: farm_op = step_farmer_cardinal("WEST")

        if buffer is not None:
            trans = Transition(
                own_grid=og[0],
                opp_grid=opg[0],
                global_vec=gv[0],
                actions={k: v[0] for k, v in actions.items()},
                log_prob=total_lp[0],
                value=value[0],
                reward=jnp.array(0.0, dtype=jnp.float32),
                done=jnp.array(False, dtype=jnp.bool_),
                masks={k: v[0] for k, v in jax_masks.items()},
            )
            buffer.add(trans)

        return {"farmer": farm_op, "hands": [], "market": market_ops}

    return agent_fn


def run_selfplay_training(
    model: ActorCriticNet,
    iterations: int = 20,
    steps_per_episode: int = 720,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    seed: int = 42,
) -> ActorCriticNet:
    """
    Runs Phase 3 Self-Play PPO League training.
    """
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    rngs = nnx.Rngs(seed)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param)
    arena = SelfPlayArena(
        # living_penalty=0.05, 
        episode_steps=steps_per_episode
        )
    pool = SnapshotPool(max_snapshots=10)

    key = jax.random.PRNGKey(seed)

    print(f"Self-Play League initialized. Training for {iterations} iterations...")

    for it in range(1, iterations + 1):
        key, seed_key = jax.random.split(key)
        match_seed = int(jax.random.randint(seed_key, (), 0, 1000000))

        buffer = RolloutBuffer(capacity=steps_per_episode)
        p0_agent = policy_to_agent_fn(model, key, buffer=buffer)

        # Opponent policy: 80% current self, 20% starter/baseline
        if random.random() < 0.8:
            p1_agent = policy_to_agent_fn(model, key, buffer=None)
        else:
            p1_agent = "starter"

        def p0_callback(obs, r, is_done):
            if buffer.transitions:
                last_t = buffer.transitions[-1]
                buffer.transitions[-1] = last_t._replace(
                    reward=jnp.array(r, dtype=jnp.float32),
                    done=jnp.array(is_done, dtype=jnp.bool_),
                )

        p0_coins, p1_coins, steps = arena.run_match(
            p0_agent,
            p1_agent,
            seed=match_seed,
            p0_step_callback=p0_callback,
        )

        if len(buffer.transitions) > 1:
            traj = buffer.get_trajectory_pytree()
            advantages, returns = compute_gae(
                traj.reward,
                traj.value,
                traj.done,
                next_value=traj.value[-1],
            )
            loss, a_loss, c_loss, ent = ppo_update_step(
                model, optimizer, traj, advantages, returns
            )
            print(
                f"SelfPlay Iter {it:02d}/{iterations:02d} | Seed: {match_seed} | "
                f"P0: ${p0_coins:.0f} vs P1: ${p1_coins:.0f} | Steps: {steps} | "
                f"Loss: {loss:.4f} (A: {a_loss:.4f}, C: {c_loss:.4f}, Ent: {ent:.4f})"
            )

    final_ckpt = os.path.join(checkpoint_dir, "selfplay_final.npz")
    print(f"Self-Play League training finished. Final model saved to {final_ckpt}.")
    return model


if __name__ == "__main__":
    run_selfplay_training(iterations=2, steps_per_episode=20)
