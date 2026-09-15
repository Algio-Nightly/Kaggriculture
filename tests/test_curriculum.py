"""
Unit and Integration Tests for 3-Phase Curriculum & Delta NW Reward Engine.
"""
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from training.rewards import calculate_net_worth, NetWorthRewardTracker
from training.dataset import load_or_build_dataset, ReplayBatchGenerator
from training.train_bc import train_bc_epoch, evaluate_bc
from training.ghost_env import GhostReplayOpponent, GhostPlayEnvironment
from model.network import ActorCriticNet
from model.agent import agent, init_agent


def test_net_worth_and_delta_reward():
    print("Testing Net Worth calculation and Delta NW reward tracker...")
    mock_obs_1 = {
        "player": 0,
        "day": 0,
        "farms": [
            {
                "money": 3000,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            },
            {"money": 3000, "tiles": [[None for _ in range(10)] for _ in range(10)]},
        ],
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {"prices": {"WHEAT": 25, "CARROT": 35}},
    }

    nw1 = calculate_net_worth(mock_obs_1)
    assert nw1 == 3000.0, f"Expected initial NW $3000, got {nw1}"

    # Scenario: Spend $10 to buy 1 wheat seed (Cash: $2990, Seeds: 1 Wheat ($10))
    mock_obs_2 = {
        "player": 0,
        "day": 0,
        "farms": [
            {
                "money": 2990,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            },
            {"money": 3000, "tiles": [[None for _ in range(10)] for _ in range(10)]},
        ],
        "private": {
            "shed": {},
            "seeds": {"WHEAT": 1},
            "inventories": [{}],
        },
        "market": {"prices": {"WHEAT": 25, "CARROT": 35}},
    }
    nw2 = calculate_net_worth(mock_obs_2)
    assert nw2 == 3000.0, f"Expected NW after buying seed to remain $3000, got {nw2}"

    # Scenario: Plant and water wheat crop on (0,0) with 1 harvestable yield unit ($25)
    mock_obs_3 = {
        "player": 0,
        "day": 2,
        "farms": [
            {
                "money": 2990,
                "tiles": [
                    [{"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 1, "watered_today": True}] + [None]*9
                ] + [[None]*10 for _ in range(9)],
            },
            {"money": 3000, "tiles": [[None for _ in range(10)] for _ in range(10)]},
        ],
        "private": {
            "shed": {},
            "seeds": {"WHEAT": 0},
            "inventories": [{}],
        },
        "market": {"prices": {"WHEAT": 25, "CARROT": 35}},
    }
    nw3 = calculate_net_worth(mock_obs_3)
    assert nw3 == 3015.0, f"Expected NW after crop growth to reach $3015, got {nw3}"

    # Test tracker
    tracker = NetWorthRewardTracker(living_penalty=0.05)
    tracker.reset(mock_obs_1)
    r1 = tracker.compute_step_reward(mock_obs_2)
    assert abs(r1 - (-0.05)) < 1e-5, f"Expected step reward -0.05 (living penalty), got {r1}"

    r2 = tracker.compute_step_reward(mock_obs_3)
    assert abs(r2 - (15.0 - 0.05)) < 1e-5, f"Expected step reward 14.95, got {r2}"

    print("[OK] Net Worth and Delta NW reward tracker verified successfully!")


def test_dataset_and_behavioral_cloning():
    print("Testing dataset compilation and Behavioral Cloning step...")
    dataset = load_or_build_dataset(force_rebuild=True)
    assert "own_grid" in dataset
    assert dataset["own_grid"].shape[0] > 0
    print(f"  Dataset contains {dataset['own_grid'].shape[0]} samples.")

    loader = ReplayBatchGenerator(dataset, batch_size=4, shuffle=True)
    rngs = nnx.Rngs(42)
    model = ActorCriticNet(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    train_loss, a_loss, c_loss = train_bc_epoch(model, optimizer, loader)
    assert not jnp.isnan(train_loss), "Train loss is NaN"
    print(f"  [OK] 1 Epoch BC Train Loss: {train_loss:.4f} (Actor: {a_loss:.4f}, Critic: {c_loss:.4f})")

    val_loss, va_loss, vc_loss = evaluate_bc(model, loader)
    assert not jnp.isnan(val_loss), "Val loss is NaN"
    print(f"  [OK] Val Loss: {val_loss:.4f}")
    print("[OK] Behavioral Cloning pipeline verified successfully!")


def test_ghost_play_environment():
    print("Testing Ghost-Play environment simulation (10 turns)...")
    init_agent(seed=99)
    ghost = GhostReplayOpponent("obs_populated.json", ghost_player_idx=1)
    ghost_env = GhostPlayEnvironment(ghost, episode_steps=10)

    rewards_collected = []
    def callback(obs, r, done):
        rewards_collected.append(r)

    p0_coins, p1_coins, steps = ghost_env.run_match(agent, on_step_callback=callback)
    assert steps >= 9, f"Expected at least 9 steps, got {steps}"
    assert len(rewards_collected) >= 9, f"Expected at least 9 rewards, got {len(rewards_collected)}"
    print(f"  [OK] Ghost Match Finished! Steps: {steps}, P0 Coins: {p0_coins}, P1 Coins: {p1_coins}")
    print("[OK] Ghost-Play environment simulation verified successfully!")


def test_ghost_play_ppo_training():
    print("Testing Ghost-Play PPO training loop (1 iteration)...")
    from training.train_ghost import run_ghost_play_training
    model = run_ghost_play_training(episodes=1, steps_per_episode=6, seed=42)
    assert model is not None
    print("[OK] Ghost-Play PPO training loop verified successfully!")


def test_selfplay_ppo_training():
    print("Testing Self-Play PPO training loop (1 iteration)...")
    from training.train_selfplay import run_selfplay_training
    model = run_selfplay_training(iterations=1, steps_per_episode=6, seed=42)
    assert model is not None
    print("[OK] Self-Play PPO training loop verified successfully!")


if __name__ == "__main__":
    test_net_worth_and_delta_reward()
    test_dataset_and_behavioral_cloning()
    test_ghost_play_environment()
    test_ghost_play_ppo_training()
    test_selfplay_ppo_training()
    print("\n========================================================")
    print("ALL 3-PHASE CURRICULUM & REWARD TESTS PASSED!")
    print("========================================================")
