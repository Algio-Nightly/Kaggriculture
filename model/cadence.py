"""
Daily Decision Cadence and Macro Strategy Resolution Engine for Kaggriculture.

This module handles:
1. DailyPlanState container tracking the active daily macro-strategy.
2. Macro-strategy evaluation via ActorCriticNet at day boundaries (hour == 0).
3. Day-start market orders dispatch (Hires, Land expansion, Seeds, Produce sales).
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from model.action_space import (
    FarmingStrategy,
    SellingStrategy,
    CropParam,
    LaborParam,
    QuadrantParam,
    CROP_NAMES,
    CROP_SEED_PRICES,
    QUADRANT_COSTS,
    QUADRANT_NAMES,
    FIB_HIRE_COSTS,
    compute_action_masks,
)
from model.encoder import encode_observation


@dataclass
class DailyPlanState:
    """Tracks the active macro-plan decisions for the current in-game day."""
    day: int
    farming_strategy: FarmingStrategy
    selling_strategy: SellingStrategy
    target_crop: str
    hires_count: int
    target_quadrant: str
    market_orders_dispatched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "farming_strategy": self.farming_strategy.name,
            "selling_strategy": self.selling_strategy.name,
            "target_crop": self.target_crop,
            "hires_count": self.hires_count,
            "target_quadrant": self.target_quadrant,
            "market_orders_dispatched": self.market_orders_dispatched,
        }


def resolve_daily_market_orders(obs: Dict[str, Any], plan: DailyPlanState) -> List[List[Any]]:
    """
    Translates macro-plan decisions into an ordered queue of market orders at day start (hour == 0).
    
    Order of operations:
    1. Labor Hires (immediate hands spawn for the day)
    2. Land Expansion (BUY_LAND if targeted and budget allows)
    3. Seed / Input purchases (if EXPAND_CROPS)
    4. Produce Sales (according to selling_strategy posture)
    
    Capped at maxMarketOrdersPerTurn (default 10).
    """
    orders: List[List[Any]] = []
    player = obs["player"]
    me = obs["farms"][player]
    money = me.get("money", 0)
    unlocked_quads = me.get("unlocked_quadrants", ["NW"])
    shed = obs["private"].get("shed", {})
    market_prices = obs["market"].get("prices", {})
    derived_demand = obs.get("derived_town_demand", {})

    remaining_money = money

    # 1. HIRE Orders
    hires_to_dispatch = min(plan.hires_count, 5)
    cumulative_hire_cost = 0
    actual_hires = 0
    for h in range(hires_to_dispatch):
        cost = FIB_HIRE_COSTS[h]
        if remaining_money >= cost:
            orders.append(["HIRE"])
            remaining_money -= cost
            actual_hires += 1
        else:
            break

    # 2. BUY_LAND Order
    if (
        plan.target_quadrant not in unlocked_quads
        and plan.target_quadrant in QUADRANT_COSTS
    ):
        quad_cost = QUADRANT_COSTS[plan.target_quadrant]
        if remaining_money >= quad_cost:
            orders.append(["BUY_LAND"])
            remaining_money -= quad_cost

    # 3. Seed Purchases (if EXPAND_CROPS)
    if plan.farming_strategy == FarmingStrategy.EXPAND_CROPS:
        crop_name = plan.target_crop
        seed_price = CROP_SEED_PRICES.get(crop_name, 10)
        
        # Count empty unlocked tiles on our farm
        empty_tiles = 0
        for row in me["tiles"]:
            for tile in row:
                if tile is None:
                    empty_tiles += 1

        current_seeds = obs["private"].get("seeds", {}).get(crop_name, 0)
        seeds_needed = max(0, min(empty_tiles, 10) - current_seeds)
        affordable_seeds = remaining_money // seed_price
        buy_count = min(seeds_needed, affordable_seeds, 5)
        
        if buy_count > 0:
            orders.append(["BUY_SEED", crop_name, int(buy_count)])
            remaining_money -= buy_count * seed_price

    # 4. Animal Purchases (if EXPAND_LIVESTOCK)
    elif plan.farming_strategy == FarmingStrategy.EXPAND_LIVESTOCK:
        if remaining_money >= 80 and shed.get("GOOSE", 0) == 0:
            # Buy GOOSE seed animal
            orders.append(["BUY_ANIMAL", "GOOSE", 1])
            remaining_money -= 80

    # 5. Produce Sales (Selling Strategy)
    if plan.selling_strategy == SellingStrategy.MARKET_DUMP:
        # Sell entire shed produce inventory
        for product, count in shed.items():
            if count > 0 and product not in ["GOOSE", "COW", "SHEEP"]:
                orders.append(["SELL", product, int(count)])

    elif plan.selling_strategy == SellingStrategy.DRIP_FEED_TOWN:
        # Sell only up to daily town demand
        for product, count in shed.items():
            if count > 0 and product not in ["GOOSE", "COW", "SHEEP"]:
                demand = derived_demand.get(product, 1.0)
                sell_amt = int(min(count, max(1, round(demand))))
                if sell_amt > 0:
                    orders.append(["SELL", product, sell_amt])

    elif plan.selling_strategy == SellingStrategy.SELL_EXCESS:
        # Keep safety stock of 10 wheat and 5 fertilizer, sell remainder
        for product, count in shed.items():
            if product == "WHEAT":
                excess = count - 10
            elif product == "FERTILIZER":
                excess = count - 5
            else:
                excess = count
            if excess > 0:
                orders.append(["SELL", product, int(excess)])

    elif plan.selling_strategy == SellingStrategy.EMERGENCY_CASH:
        # If money < 100, liquidate highest price goods
        if money < 100:
            sorted_shed = sorted(
                [(prod, cnt) for prod, cnt in shed.items() if prod not in ["GOOSE", "COW", "SHEEP"] and cnt > 0],
                key=lambda x: market_prices.get(x[0], 0),
                reverse=True
            )
            for product, count in sorted_shed:
                orders.append(["SELL", product, int(count)])

    # Cap at 10 market orders per turn (game rule)
    return orders[:10]


def evaluate_daily_plan(
    obs: Dict[str, Any],
    model: Any,
    rng: Optional[jax.Array] = None,
    deterministic: bool = False,
) -> DailyPlanState:
    """
    Evaluates the board at day start using ActorCriticNet to establish the DailyPlanState.
    
    Args:
        obs: Raw environment observation dictionary.
        model: ActorCriticNet instance.
        rng: Optional JAX PRNGKey for stochastic sampling.
        deterministic: If True, selects argmax action for each head.
        
    Returns:
        DailyPlanState instance.
    """
    # 1. Encode state representations
    own_grid_t, opp_grid_t, global_t = encode_observation(obs)

    # 2. Compute state-dependent validity masks
    raw_masks = compute_action_masks(obs)
    jax_masks = {k: jnp.asarray(v) for k, v in raw_masks.items()}

    # 3. Model forward pass
    policy_logits, _ = model(own_grid_t, opp_grid_t, global_t, masks=jax_masks)

    # 4. Action selection per macro head
    selected_indices = {}
    for head in ["farming", "selling", "crop", "labor", "quadrant"]:
        logits = policy_logits[head]
        if deterministic or rng is None:
            selected_indices[head] = int(jnp.argmax(logits))
        else:
            rng, subkey = jax.random.split(rng)
            selected_indices[head] = int(jax.random.categorical(subkey, logits))

    farming_strat = FarmingStrategy(selected_indices["farming"])
    selling_strat = SellingStrategy(selected_indices["selling"])
    crop_name = CROP_NAMES[selected_indices["crop"]]
    hires_count = int(selected_indices["labor"])
    quad_name = QUADRANT_NAMES[selected_indices["quadrant"]]

    return DailyPlanState(
        day=obs.get("day", 0),
        farming_strategy=farming_strat,
        selling_strategy=selling_strat,
        target_crop=crop_name,
        hires_count=hires_count,
        target_quadrant=quad_name,
        market_orders_dispatched=False,
    )
