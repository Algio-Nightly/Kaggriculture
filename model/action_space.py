"""
Factored Multi-Discrete Action Space and Masking Utilities for Kaggriculture.

This module defines:
1. Factored Multi-Discrete Action Head Enums and mappings.
2. State-dependent action mask computation (Adaptive Logit Masking).
3. Strategy and action decoders.
"""
from enum import IntEnum
from typing import Dict, Any, List, Tuple
import numpy as np


# =============================================================================
# 1. Factored Action Head Enumerations
# =============================================================================

class FarmingStrategy(IntEnum):
    """Macro farming protocol for the in-game day."""
    SUSTAIN = 0           # Maintain crops/animals (water, feed, care, collect)
    EXPAND_CROPS = 1      # Purchase seeds & plant target crop on empty tiles
    EXPAND_LIVESTOCK = 2  # Build coop/pasture & buy animals
    HARVEST_ALL = 3       # Focus on harvesting all mature crops and dumping to shed
    HOARD_RESOURCES = 4   # Store inputs/yields without selling
    IDLE = 5              # Conserve bank balance and pass non-essential turns


class SellingStrategy(IntEnum):
    """Market selling posture for the in-game day."""
    HOLD = 0              # Store all produce in shed
    DRIP_FEED_TOWN = 1    # Sell only amounts demanded by active town shops
    MARKET_DUMP = 2       # Liquidate all shed inventory onto market
    SELL_EXCESS = 3       # Sell stock exceeding a safety threshold
    EMERGENCY_CASH = 4    # Liquidate highest-value stock if bank balance < threshold


class CropParam(IntEnum):
    """Target crop argument for planting and expansion."""
    WHEAT = 0
    CARROT = 1
    TOMATO = 2
    STRAWBERRY = 3
    MELON = 4


class LaborParam(IntEnum):
    """Number of farm hands to hire for the day."""
    HIRE_0 = 0
    HIRE_1 = 1
    HIRE_2 = 2
    HIRE_3 = 3
    HIRE_4 = 4
    HIRE_5 = 5


class QuadrantParam(IntEnum):
    """Target quadrant for land expansion."""
    NW = 0
    NE = 1
    SW = 2
    SE = 3


class TacticalAction(IntEnum):
    """Micro turn-level tactical actions."""
    PASS = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    WATER = 5
    HARVEST = 6
    FERTILIZE = 7
    FEED = 8
    CARE = 9
    COLLECT_FERTILIZER = 10
    DIG = 11
    PLANT_WHEAT = 12
    PLANT_CARROT = 13
    BUILD_COOP = 14


# Head Dimension Constants
HEAD_DIMS = {
    "farming": len(FarmingStrategy),      # 6
    "selling": len(SellingStrategy),      # 5
    "crop": len(CropParam),                # 5
    "labor": len(LaborParam),              # 6
    "quadrant": len(QuadrantParam),        # 4
    "tactical": len(TacticalAction),      # 15
}

CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
CROP_SEED_PRICES = {"WHEAT": 10, "CARROT": 20, "TOMATO": 30, "STRAWBERRY": 40, "MELON": 50}
CROP_MATURITY_DAYS = {"WHEAT": 2, "CARROT": 3, "TOMATO": 4, "STRAWBERRY": 5, "MELON": 6}
QUADRANT_COSTS = {"NW": 0, "NE": 1000, "SW": 2000, "SE": 4000}
QUADRANT_NAMES = ["NW", "NE", "SW", "SE"]

# Fibonacci hire cost schedule: fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, fib(5)=8
FIB_HIRE_COSTS = [1, 1, 2, 3, 5, 8]


# =============================================================================
# 2. Adaptive Forward Logit Mask Computation
# =============================================================================

def compute_action_masks(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Computes boolean validity masks for all factored action heads from the observation.
    
    True = Valid Action, False = Invalid (Penalized with -1e9 in logits).
    
    Args:
        obs: Environment observation dictionary.
        
    Returns:
        Dict mapping head names to boolean numpy arrays of shape (head_dim,).
    """
    player = obs["player"]
    me = obs["farms"][player]
    money = me.get("money", 0)
    day = obs.get("day", 0)
    shed = obs["private"].get("shed", {})
    unlocked_quads = me.get("unlocked_quadrants", ["NW"])
    
    total_shed_items = sum(shed.values())

    # 1. Farming Strategy Mask
    farming_mask = np.ones(HEAD_DIMS["farming"], dtype=bool)
    # EXPAND_CROPS requires at least minimum seed money ($10)
    if money < min(CROP_SEED_PRICES.values()):
        farming_mask[FarmingStrategy.EXPAND_CROPS] = False
    # EXPAND_LIVESTOCK requires money for coop/pasture ($100 minimum)
    if money < 100:
        farming_mask[FarmingStrategy.EXPAND_LIVESTOCK] = False
    # HARVEST_ALL: only if shed has capacity
    if total_shed_items >= 100:
        farming_mask[FarmingStrategy.HARVEST_ALL] = False

    # 2. Selling Strategy Mask
    selling_mask = np.ones(HEAD_DIMS["selling"], dtype=bool)
    if total_shed_items == 0:
        # Cannot sell if shed is completely empty
        selling_mask[SellingStrategy.DRIP_FEED_TOWN] = False
        selling_mask[SellingStrategy.MARKET_DUMP] = False
        selling_mask[SellingStrategy.SELL_EXCESS] = False
        selling_mask[SellingStrategy.EMERGENCY_CASH] = False

    # 3. Crop Parameter Mask
    crop_mask = np.ones(HEAD_DIMS["crop"], dtype=bool)
    days_left = 30 - day
    for crop_idx, crop_name in enumerate(CROP_NAMES):
        seed_cost = CROP_SEED_PRICES[crop_name]
        maturity = CROP_MATURITY_DAYS[crop_name]
        # Mask if cannot afford seed or not enough days left in season to yield
        if money < seed_cost or days_left < maturity:
            crop_mask[crop_idx] = False
    if not np.any(crop_mask):
        crop_mask[CropParam.WHEAT] = True  # Always allow cheapest fallback

    # 4. Labor Parameter Mask
    labor_mask = np.ones(HEAD_DIMS["labor"], dtype=bool)
    cumulative_hire_cost = 0
    for hire_n in range(len(LaborParam)):
        if hire_n > 0:
            cumulative_hire_cost += FIB_HIRE_COSTS[hire_n - 1]
        if money < cumulative_hire_cost:
            labor_mask[hire_n] = False
    labor_mask[LaborParam.HIRE_0] = True  # 0 hires is always free and valid

    # 5. Quadrant Parameter Mask
    quadrant_mask = np.zeros(HEAD_DIMS["quadrant"], dtype=bool)
    for q_idx, q_name in enumerate(QUADRANT_NAMES):
        if q_name not in unlocked_quads and money >= QUADRANT_COSTS[q_name]:
            quadrant_mask[q_idx] = True
    if not np.any(quadrant_mask):
        # If all unlocked or cannot afford, allow current unlocked as no-op
        quadrant_mask[QuadrantParam.NW] = True

    # 6. Tactical Action Mask
    tactical_mask = np.ones(HEAD_DIMS["tactical"], dtype=bool)
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]
    
    # Context-sensitive tactical validity
    is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
    is_animal_tile = isinstance(tile, dict) and tile.get("kind") in ["COOP", "PASTURE"]
    is_empty = tile is None
    
    if not is_plant:
        tactical_mask[TacticalAction.WATER] = False
        tactical_mask[TacticalAction.HARVEST] = False
        tactical_mask[TacticalAction.FERTILIZE] = False
    if not is_animal_tile:
        tactical_mask[TacticalAction.FEED] = False
        tactical_mask[TacticalAction.CARE] = False
        tactical_mask[TacticalAction.COLLECT_FERTILIZER] = False
    if not is_empty:
        tactical_mask[TacticalAction.PLANT_WHEAT] = False
        tactical_mask[TacticalAction.PLANT_CARROT] = False
        tactical_mask[TacticalAction.BUILD_COOP] = False
    if money < 10 or obs["private"].get("seeds", {}).get("WHEAT", 0) == 0:
        tactical_mask[TacticalAction.PLANT_WHEAT] = False
    if money < 20 or obs["private"].get("seeds", {}).get("CARROT", 0) == 0:
        tactical_mask[TacticalAction.PLANT_CARROT] = False

    return {
        "farming": farming_mask,
        "selling": selling_mask,
        "crop": crop_mask,
        "labor": labor_mask,
        "quadrant": quadrant_mask,
        "tactical": tactical_mask,
    }


# =============================================================================
# 3. Action Translation & Decoders
# =============================================================================

TACTICAL_OPS = [
    ["PASS"],
    ["NORTH"],
    ["SOUTH"],
    ["EAST"],
    ["WEST"],
    ["WATER"],
    ["HARVEST"],
    ["FERTILIZE"],
    ["FEED"],
    ["CARE"],
    ["COLLECT_FERTILIZER"],
    ["DIG"],
    ["PLANT", "WHEAT"],
    ["PLANT", "CARROT"],
    ["BUILD_COOP"],
]


def tactical_index_to_dict(action_idx: int) -> Dict[str, Any]:
    """Converts a tactical action index to a game action dictionary."""
    if 0 <= action_idx < len(TACTICAL_OPS):
        farmer_op = TACTICAL_OPS[action_idx]
    else:
        farmer_op = ["PASS"]
    return {
        "farmer": farmer_op,
        "hands": [],
        "market": []
    }


def decode_factored_decisions(decisions: Dict[str, int]) -> Dict[str, Any]:
    """
    Decodes integer head selections into human-readable strategy names and parameters.
    """
    return {
        "farming_strategy": FarmingStrategy(decisions.get("farming", 0)).name,
        "selling_strategy": SellingStrategy(decisions.get("selling", 0)).name,
        "crop_param": CropParam(decisions.get("crop", 0)).name,
        "labor_param": LaborParam(decisions.get("labor", 0)).name,
        "quadrant_param": QuadrantParam(decisions.get("quadrant", 0)).name,
        "tactical_action": TacticalAction(decisions.get("tactical", 0)).name,
    }
