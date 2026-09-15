"""
Delta Net Worth (Delta NW) Reward Engine for Kaggriculture.

Implements:
1. Real-time farm equity evaluation: NW_t = Cash_t + ShedValue_t + FieldValue_t + SeedValue_t
2. Step reward computation: R_t = (NW_t - NW_{t-1}) - lambda (with living penalty)
3. Terminal game outcome delta evaluation.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Economic Base Price Constants
CROP_SEED_COSTS = {
    "WHEAT": 10.0,
    "CARROT": 20.0,
    "TOMATO": 50.0,
    "STRAWBERRY": 100.0,
    "MELON": 80.0,
}

ANIMAL_COSTS = {
    "GOOSE": 300.0,
    "COW": 400.0,
    "SHEEP": 500.0,
}

CROP_MAX_YIELD_AGE = {
    "WHEAT": 4,
    "CARROT": 3,
    "TOMATO": 11,
    "STRAWBERRY": 16,
    "MELON": 10,
}

CROP_BASE_PRICES = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
}

ANIMAL_PRODUCT_BASE_PRICES = {
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}


def calculate_net_worth(obs: Dict[str, Any], player: Optional[int] = None) -> float:
    """
    Computes total economic Net Worth (NW) for the specified player from the observation.
    
    NW = Cash + ShedValue + FieldValue + SeedValue
    
    Args:
        obs: Environment observation dictionary.
        player: Player index (0 or 1). Defaults to obs["player"].
        
    Returns:
        Total Net Worth scalar in coins ($).
    """
    if player is None:
        player = obs.get("player", 0)

    farm = obs["farms"][player]
    day = obs.get("day", 0)
    market_prices = obs.get("market", {}).get("prices", {})

    # 1. Liquid Cash
    cash = float(farm.get("money", 0.0))

    # 2. Shed Storage Inventory Value
    shed_value = 0.0
    private_state = obs.get("private", {})
    shed = private_state.get("shed", {})
    for item, count in shed.items():
        if count <= 0:
            continue
        if item in ANIMAL_COSTS:
            # Animal sitting in shed (at purchase cost)
            shed_value += count * ANIMAL_COSTS[item]
        elif item in market_prices:
            # Dynamic market price
            shed_value += count * float(market_prices[item])
        elif item in CROP_BASE_PRICES:
            shed_value += count * CROP_BASE_PRICES[item]
        elif item in ANIMAL_PRODUCT_BASE_PRICES:
            shed_value += count * ANIMAL_PRODUCT_BASE_PRICES[item]
        else:
            shed_value += count * 10.0

    # 3. Seed Inventory Value
    seed_value = 0.0
    seeds = private_state.get("seeds", {})
    for crop, count in seeds.items():
        if count > 0 and crop in CROP_SEED_COSTS:
            seed_value += count * CROP_SEED_COSTS[crop]

    # 4. Field Assets Value (Growing Crops + Live Livestock)
    field_value = 0.0
    tiles = farm.get("tiles", [])
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
                
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop", "WHEAT")
                planted_day = tile.get("planted_day", day)
                age = max(0, day - planted_day)
                max_age = CROP_MAX_YIELD_AGE.get(crop, 4)
                current_yield = tile.get("yield_units", 0)
                
                # Unit price on market
                unit_price = float(market_prices.get(crop, CROP_BASE_PRICES.get(crop, 25.0)))
                
                # Valuation: immediate harvestable yield + discounted growth progress
                if current_yield > 0:
                    field_value += current_yield * unit_price
                else:
                    # Seed cost + progress fraction towards base yield
                    growth_frac = min(1.0, age / max(1, max_age))
                    field_value += CROP_SEED_COSTS.get(crop, 10.0) + (unit_price - CROP_SEED_COSTS.get(crop, 10.0)) * 0.7 * growth_frac

            elif kind in ["COOP", "PASTURE"]:
                animal = tile.get("animal")
                if animal in ANIMAL_COSTS:
                    # Live animal valuation (retains purchase value as productive asset)
                    field_value += ANIMAL_COSTS[animal]
                    
                # Unharvested eggs/milk/wool held on tile
                held_yield = tile.get("yield_units", 0)
                if held_yield > 0 and animal:
                    product = "EGG" if animal == "GOOSE" else ("MILK" if animal == "COW" else "WOOL")
                    p_price = float(market_prices.get(product, ANIMAL_PRODUCT_BASE_PRICES.get(product, 50.0)))
                    field_value += held_yield * p_price

    # 5. Field Carried Inventory
    inventories = private_state.get("inventories", [])
    carried_value = 0.0
    for inv in inventories:
        if isinstance(inv, dict):
            for itm, cnt in inv.items():
                if cnt > 0:
                    unit_p = float(market_prices.get(itm, CROP_BASE_PRICES.get(itm, 25.0)))
                    carried_value += cnt * unit_p

    return cash + shed_value + seed_value + field_value + carried_value


class NetWorthRewardTracker:
    """
    Tracks turn-by-turn Net Worth deltas and calculates the living-penalized step reward.
    """

    def __init__(self, living_penalty: float = 0.05, terminal_bonus_scale: float = 500.0):
        self.living_penalty = living_penalty
        self.terminal_bonus_scale = terminal_bonus_scale
        self.last_net_worth: Optional[float] = None

    def reset(self, initial_obs: Dict[str, Any]):
        """Resets tracker at the start of an episode."""
        self.last_net_worth = calculate_net_worth(initial_obs)

    def compute_step_reward(self, current_obs: Dict[str, Any], is_done: bool = False) -> float:
        """
        Computes R_t = (NW_t - NW_{t-1}) - lambda.
        If episode is done, adds terminal margin bonus.
        """
        current_nw = calculate_net_worth(current_obs)
        if self.last_net_worth is None:
            self.last_net_worth = current_nw

        delta_nw = current_nw - self.last_net_worth
        reward = delta_nw - self.living_penalty
        self.last_net_worth = current_nw

        if is_done:
            player = current_obs.get("player", 0)
            opp = 1 - player
            my_money = current_obs["farms"][player]["money"]
            opp_money = current_obs["farms"][opp]["money"]
            margin = my_money - opp_money
            # Terminal outcome sign bonus
            if margin > 0:
                reward += self.terminal_bonus_scale
            elif margin < 0:
                reward -= self.terminal_bonus_scale

        return float(reward)
