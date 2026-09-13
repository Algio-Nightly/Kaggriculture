"""
Production Model Agent for Kaggriculture with Daily Decision Cadence.

Coordinates:
1. Daily Decision Cadence: Evaluates board at hour == 0 to set DailyPlanState and emit day-start market orders.
2. Intra-Day Tactical Execution: Controls Farmer and Hired Hands actions throughout the in-game day.
3. Exception-safe fallback to prevent game forfeiture.
"""
from typing import Dict, Any, List, Optional
from flax import nnx
import jax
import jax.numpy as jnp

from model.encoder import encode_observation
from model.network import ActorCriticNet
from model.action_space import (
    compute_action_masks,
    tactical_index_to_dict,
    FarmingStrategy,
)
from model.cadence import (
    DailyPlanState,
    evaluate_daily_plan,
    resolve_daily_market_orders,
)

# Global Agent State
_model: Optional[ActorCriticNet] = None
_current_plan: Optional[DailyPlanState] = None
_last_day_evaluated: int = -1


def init_agent(seed: int = 42) -> ActorCriticNet:
    """Initialize NNX model weights and warm up JIT execution."""
    global _model, _current_plan, _last_day_evaluated
    rngs = nnx.Rngs(seed)
    _model = ActorCriticNet(rngs=rngs)
    _current_plan = None
    _last_day_evaluated = -1
    
    # Warmup forward pass with dummy inputs
    dummy_own = jnp.zeros((1, 10, 10, 21), dtype=jnp.float32)
    dummy_opp = jnp.zeros((1, 10, 10, 21), dtype=jnp.float32)
    dummy_glob = jnp.zeros((1, 47), dtype=jnp.float32)
    _ = _model(dummy_own, dummy_opp, dummy_glob)
    
    return _model


def get_current_plan() -> Optional[DailyPlanState]:
    """Returns currently active DailyPlanState."""
    return _current_plan


def execute_hands_actions(obs: Dict[str, Any], plan: Optional[DailyPlanState]) -> List[List[str]]:
    """
    Coordinates work routines for all hired hands based on the active daily macro plan.
    
    Hands prioritize:
    1. Harvest mature crops
    2. Water unwatered crops
    3. Plant target crop seeds
    4. Dig weeds
    5. Explore/Pass
    """
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    hands_pos = me.get("hands", [])
    target_crop = plan.target_crop if plan is not None else "WHEAT"
    target_seeds = private.get("seeds", {}).get(target_crop, 0)

    hands_actions = []
    for hx, hy in hands_pos:
        tile = me["tiles"][hy][hx] if (0 <= hy < 10 and 0 <= hx < 10) else None

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                hands_actions.append(["HARVEST"])
            elif not tile.get("watered_today"):
                hands_actions.append(["WATER"])
            else:
                hands_actions.append(["SOUTH" if hy < 4 else "WEST"])
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            hands_actions.append(["DIG"])
        elif tile is None:
            if target_seeds > 0:
                hands_actions.append(["PLANT", target_crop])
                target_seeds -= 1
            else:
                hands_actions.append(["EAST" if hx < 4 else "SOUTH"])
        else:
            hands_actions.append(["PASS"])

    return hands_actions


def agent(obs: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Production Kaggle competition entrypoint function.
    
    Args:
        obs: Raw environment observation dictionary.
        config: Optional Kaggle environment configuration dictionary.
        
    Returns:
        Action dictionary: {"farmer": [...], "hands": [...], "market": [...]}
    """
    global _model, _current_plan, _last_day_evaluated

    try:
        if _model is None:
            init_agent()

        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        market_orders: List[List[Any]] = []

        # 1. Day Start Cadence (Hour 0): Evaluate board & dispatch macro market orders
        if hour == 0 or _current_plan is None or day != _last_day_evaluated:
            _current_plan = evaluate_daily_plan(obs, _model, deterministic=True)
            _last_day_evaluated = day
            market_orders = resolve_daily_market_orders(obs, _current_plan)
            _current_plan.market_orders_dispatched = True

        # 2. Turn-Level Tactical Execution
        own_grid_t, opp_grid_t, global_t = encode_observation(obs)
        raw_masks = compute_action_masks(obs)
        jax_masks = {k: jnp.asarray(v) for k, v in raw_masks.items()}

        policy_outputs, _ = _model(own_grid_t, opp_grid_t, global_t, masks=jax_masks)
        tactical_logits = policy_outputs["tactical"]
        action_idx = int(jnp.argmax(tactical_logits))
        farmer_action_dict = tactical_index_to_dict(action_idx)
        farmer_op = farmer_action_dict["farmer"]

        # Check if farmer is adjacent to shed carrying non-animal inventory -> auto-deposit
        player = obs["player"]
        fx, fy = obs["farms"][player]["farmer"]
        if (fx, fy) in [(4, 4), (5, 4), (4, 5), (5, 5)]:
            farmer_inv = obs["private"]["inventories"][0] if len(obs["private"]["inventories"]) > 0 else {}
            if any(cnt > 0 for itm, cnt in farmer_inv.items() if itm not in ["GOOSE", "COW", "SHEEP"]):
                farmer_op = ["DROP"]

        # 3. Hired Hands Work Execution
        hands_ops = execute_hands_actions(obs, _current_plan)

        return {
            "farmer": farmer_op,
            "hands": hands_ops,
            "market": market_orders,
        }

    except Exception as e:
        # Resilient fallback to avoid forfeiture
        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }
