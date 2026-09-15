"""
Phase 2: Ghost-Play Reinforcement Learning Trainer for Kaggriculture.

Optimizes ActorCriticNet against historical Grandmaster replay trajectories using PPO:
1. Pairs agent with a historical Ghost opponent on the exact episode seed.
2. Computes real-time Delta Net Worth (Delta NW) step rewards.
3. Performs PPO advantage updates (GAE-lambda) to counter human master play.
4. Saves checkpoints to checkpoints/ghost_model.npz.
"""
import os
import glob
import json
from typing import List, Dict, Any, Tuple, Optional
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
    queue_buy_animal,
    queue_sell,
    queue_hire,
    queue_buy_land,
    drop_to_shed,
)
from training.rewards import NetWorthRewardTracker, calculate_net_worth
from training.ghost_env import GhostReplayOpponent, GhostPlayEnvironment

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REPLAY_DIR = os.path.join(PROJECT_ROOT, "replays")
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def ppo_update_step(
    model: ActorCriticNet,
    optimizer: nnx.Optimizer,
    trajectories: Transition,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    Executes a single PPO policy and value network update step on collected trajectories.
    """
    # Normalize advantages
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    norm_adv = (advantages - adv_mean) / adv_std

    def loss_fn(m: ActorCriticNet):
        # Forward pass on batched trajectory states
        logits, values = m(
            trajectories.own_grid,
            trajectories.opp_grid,
            trajectories.global_vec,
            masks=trajectories.masks,
        )

        # 1. Compute new joint log-probabilities
        new_total_log_prob = None
        total_entropy = 0.0

        for head_name, head_logits in logits.items():
            head_action = trajectories.actions[head_name]
            log_probs = jax.nn.log_softmax(head_logits, axis=-1)
            probs = jax.nn.softmax(head_logits, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1)
            total_entropy = total_entropy + jnp.mean(entropy)

            one_hot = jax.nn.one_hot(head_action, num_classes=head_logits.shape[-1])
            act_lp = jnp.sum(one_hot * log_probs, axis=-1)
            if new_total_log_prob is None:
                new_total_log_prob = act_lp
            else:
                new_total_log_prob = new_total_log_prob + act_lp

        # 2. PPO Clipped Objective
        ratio = jnp.exp(new_total_log_prob - trajectories.log_prob)
        surr1 = ratio * norm_adv
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # 3. Value Function Loss (MSE)
        critic_loss = jnp.mean((values - returns) ** 2)

        # Total PPO Loss
        total_loss = actor_loss + vf_coef * critic_loss - ent_coef * total_entropy
        return total_loss, (actor_loss, critic_loss, total_entropy)

    (loss_val, (a_loss, c_loss, ent)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return float(loss_val), float(a_loss), float(c_loss), float(ent)


def run_ghost_play_training(
    replay_dir: str = DEFAULT_REPLAY_DIR,
    episodes: int = 20,
    steps_per_episode: int = 720,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    seed: int = 42,
) -> ActorCriticNet:
    """
    Runs Phase 2 Ghost-Play RL against available replay datasets.
    """
    if not os.path.isabs(replay_dir):
        replay_dir = os.path.join(PROJECT_ROOT, replay_dir)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    rngs = nnx.Rngs(seed)

    # 1. Discover Replays
    replay_files = glob.glob(os.path.join(replay_dir, "*.json"))
    if not replay_files:
        # Fallback to root sample replays if replays/ is empty
        replay_files = [
            os.path.join(PROJECT_ROOT, f)
            for f in ["obs_populated.json", "obs_syn.json"]
            if os.path.exists(os.path.join(PROJECT_ROOT, f))
        ]

    print(f"Ghost-Play Training initialized with {len(replay_files)} ghost opponent files.")

    # 2. Instantiate Model and Optimizer
    model = ActorCriticNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param)

    key = jax.random.PRNGKey(seed)

    # 3. Ghost-Play Training Loop
    for ep in range(1, episodes + 1):
        rep_file = replay_files[(ep - 1) % len(replay_files)]
        ghost_opponent = GhostReplayOpponent(rep_file, ghost_player_idx=1)
        ghost_env = GhostPlayEnvironment(
            ghost_opponent,
            living_penalty=0.05,
            episode_steps=steps_per_episode,
        )

        buffer = RolloutBuffer(capacity=steps_per_episode)
        total_ep_reward = 0.0

        # Run Episode with Trajectory Collection
        def agent_action_fn(obs: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal key, total_ep_reward
            key, subkey = jax.random.split(key)

            own_grid, opp_grid, global_vec = encode_observation(obs)
            masks = compute_action_masks(obs)

            # JAX array conversion
            og = jnp.expand_dims(jnp.array(own_grid, dtype=jnp.float32), axis=0)
            opg = jnp.expand_dims(jnp.array(opp_grid, dtype=jnp.float32), axis=0)
            gv = jnp.expand_dims(jnp.array(global_vec, dtype=jnp.float32), axis=0)
            jax_masks = {k: jnp.expand_dims(jnp.array(v, dtype=jnp.bool_), axis=0) for k, v in masks.items()}

            logits, value = model(og, opg, gv, masks=jax_masks)
            actions, head_lps, total_lp = sample_factored_action(subkey, logits, masks=jax_masks)

            # Extract scalar actions for Python execution
            act_dict = {k: int(v[0]) for k, v in actions.items()}

            # Build level-1 parameterized ops
            farm_op = ["PASS"]
            market_ops = []

            # 1. Labor / Hiring
            if act_dict.get("labor", 0) == 1:
                market_ops.append(queue_hire())

            # 2. Quadrant Unlock
            if act_dict.get("quadrant", 0) > 0:
                market_ops.append(queue_buy_land())

            # 3. Selling
            if act_dict.get("selling", 0) == 1:
                for crop in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
                    count = obs["private"]["shed"].get(crop, 0)
                    if count > 0:
                        market_ops.append(queue_sell(crop, count))

            # 4. Farming Strategy
            f_strat = act_dict.get("farming", 0)
            fx, fy = obs["farms"][obs["player"]]["farmer"]
            current_tile = obs["farms"][obs["player"]]["tiles"][fy][fx]

            if f_strat == 1:  # EXPAND_CROPS
                crop_idx = act_dict.get("crop", 0)
                crop_names = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
                chosen_crop = crop_names[crop_idx % len(crop_names)]
                if obs["private"]["seeds"].get(chosen_crop, 0) == 0:
                    market_ops.append(queue_buy_seed(chosen_crop, 1))
                if current_tile is None:
                    farm_op = plant_tile(chosen_crop)
            elif f_strat == 2:  # WATER_EXISTING
                if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                    farm_op = water_tile()
            elif f_strat == 3:  # HARVEST_READY
                if isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
                    farm_op = harvest_tile()
            elif f_strat == 4:  # FEED_ANIMALS
                if isinstance(current_tile, dict) and current_tile.get("kind") in ["COOP", "PASTURE"]:
                    farm_op = feed_animal()
            elif f_strat == 5:  # DEPOSIT_PRODUCE
                farm_op = drop_to_shed()

            # 5. Tactical Head (Movement / Dig / Care)
            t_action = act_dict.get("tactical", 0)
            if t_action == 1: farm_op = step_farmer_cardinal("NORTH")
            elif t_action == 2: farm_op = step_farmer_cardinal("SOUTH")
            elif t_action == 3: farm_op = step_farmer_cardinal("EAST")
            elif t_action == 4: farm_op = step_farmer_cardinal("WEST")

            # Store transition in buffer
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

        def step_callback(next_obs: Dict[str, Any], reward: float, is_done: bool):
            nonlocal total_ep_reward
            total_ep_reward += reward
            if buffer.transitions:
                last_t = buffer.transitions[-1]
                buffer.transitions[-1] = last_t._replace(
                    reward=jnp.array(reward, dtype=jnp.float32),
                    done=jnp.array(is_done, dtype=jnp.bool_),
                )

        p0_coins, p1_coins, steps = ghost_env.run_match(
            agent_action_fn,
            on_step_callback=step_callback,
        )

        # Compute GAE and Update Policy
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
                f"Ghost Episode {ep:02d}/{episodes:02d} | "
                f"Steps: {steps} | P0 Coins: {p0_coins:.0f} vs Ghost: {p1_coins:.0f} | "
                f"Delta NW Reward: {total_ep_reward:.2f} | PPO Loss: {loss:.4f} (Actor: {a_loss:.4f}, Critic: {c_loss:.4f})"
            )
        else:
            print(f"Ghost Episode {ep:02d}/{episodes:02d} finished with {steps} steps.")

    # Save Checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "ghost_model.npz")
    print(f"Ghost-Play Training complete. Checkpoint saved to {ckpt_path}.")
    return model


if __name__ == "__main__":
    run_ghost_play_training(episodes=2, steps_per_episode=20)
