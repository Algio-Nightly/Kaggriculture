"""
Unit and Integration Tests for Daily Decision Cadence, Rollout Cycle, and Production Inference.
"""
import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from kaggle_environments import make

from model.rollout import sample_factored_action, compute_gae, RolloutBuffer, Transition
from model.cadence import DailyPlanState, resolve_daily_market_orders
from model.action_space import FarmingStrategy, SellingStrategy
from model.agent import init_agent, agent


def test_stochastic_sampling_and_masking():
    print("Testing stochastic factored action sampling & logit masking...")
    key = jax.random.PRNGKey(0)

    # 4 classes for head A, 3 classes for head B
    logits = {
        "head_a": jnp.array([1.0, 2.0, 3.0, 4.0]),
        "head_b": jnp.array([0.5, 1.5, 2.5]),
    }
    # Mask out index 3 in head_a and index 1 in head_b
    masks = {
        "head_a": jnp.array([True, True, True, False]),
        "head_b": jnp.array([True, False, True]),
    }

    # Draw 500 samples and verify masked actions are NEVER selected
    for _ in range(500):
        key, subkey = jax.random.split(key)
        actions, head_log_probs, total_log_prob = sample_factored_action(subkey, logits, masks)
        assert actions["head_a"] != 3, "Masked action index 3 in head_a was sampled!"
        assert actions["head_b"] != 1, "Masked action index 1 in head_b was sampled!"
        assert not jnp.isnan(total_log_prob), "Total log prob is NaN"

    print("[OK] Stochastic factored sampling & masking verified successfully!")


def test_gae_computation():
    print("Testing GAE-lambda advantage and return computation...")
    # T = 4 steps
    rewards = jnp.array([1.0, 2.0, 0.5, 3.0])
    values = jnp.array([1.5, 2.0, 1.0, 2.5])
    dones = jnp.array([0.0, 0.0, 0.0, 1.0])
    next_value = jnp.array(0.0)

    gamma = 0.99
    lambda_ = 0.95
    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma, lambda_)

    assert advantages.shape == (4,), f"Expected advantages shape (4,), got {advantages.shape}"
    assert returns.shape == (4,), f"Expected returns shape (4,), got {returns.shape}"
    # Target return = advantages + values
    assert jnp.allclose(returns, advantages + values), "Returns do not equal advantages + values!"
    print("[OK] GAE-lambda advantages & returns verified successfully!")


def test_daily_market_orders_resolution():
    print("Testing DailyPlanState and market orders resolution...")
    plan = DailyPlanState(
        day=0,
        farming_strategy=FarmingStrategy.EXPAND_CROPS,
        selling_strategy=SellingStrategy.DRIP_FEED_TOWN,
        target_crop="CARROT",
        hires_count=2,
        target_quadrant="NE",
        market_orders_dispatched=False,
    )

    mock_obs = {
        "player": 0,
        "farms": [
            {
                "money": 3000,
                "unlocked_quadrants": ["NW"],
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            },
            {},
        ],
        "private": {
            "shed": {"WHEAT": 5, "CARROT": 3},
            "seeds": {"CARROT": 0},
        },
        "market": {"prices": {"WHEAT": 10, "CARROT": 25}},
        "derived_town_demand": {"WHEAT": 2.0, "CARROT": 1.0},
    }

    orders = resolve_daily_market_orders(mock_obs, plan)
    # Check HIRE orders
    hire_orders = [o for o in orders if o[0] == "HIRE"]
    assert len(hire_orders) == 2, f"Expected 2 HIRE orders, got {len(hire_orders)}"

    # Check BUY_LAND order (NE costs 1000, we have 3000)
    land_orders = [o for o in orders if o[0] == "BUY_LAND"]
    assert len(land_orders) == 1, f"Expected 1 BUY_LAND order, got {len(land_orders)}"

    # Check BUY_SEED order for CARROT
    seed_orders = [o for o in orders if o[0] == "BUY_SEED" and o[1] == "CARROT"]
    assert len(seed_orders) == 1, f"Expected BUY_SEED for CARROT, got {seed_orders}"

    # Check SELL orders
    sell_orders = [o for o in orders if o[0] == "SELL"]
    assert len(sell_orders) >= 1, "Expected SELL orders under DRIP_FEED_TOWN"

    print("[OK] Daily market orders resolution verified successfully!")


def test_production_agent_simulation():
    print("Testing production agent in live Kaggle environment (48 turns / 2 full days)...")
    init_agent(seed=101)
    env = make("kaggriculture", configuration={"episodeSteps": 48}, debug=True)
    env.run([agent, "starter"])

    steps = env.steps
    assert len(steps) >= 48, f"Expected at least 48 steps, got {len(steps)}"
    p0_reward = steps[-1][0]["reward"]
    p1_reward = steps[-1][1]["reward"]
    print(f"Match Completed! Steps: {len(steps)}")
    print(f"Our Agent Reward: {p0_reward}, Starter Agent Reward: {p1_reward}")
    print("[OK] Production agent 48-turn simulation completed without error!")


if __name__ == "__main__":
    test_stochastic_sampling_and_masking()
    test_gae_computation()
    test_daily_market_orders_resolution()
    test_production_agent_simulation()
    print("\n========================================================")
    print("ALL DAILY CADENCE, ROLLOUT & INFERENCE TESTS PASSED!")
    print("========================================================")
