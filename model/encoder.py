"""
Observation Encoder module for Kaggriculture.
Converts raw dictionary observations into Dual-Tower spatial grid tensors and global scalar tensors.
Includes derived product town demand rates.
"""
from typing import Dict, Any, Tuple
import jax.numpy as jnp
import numpy as np

# Product indices mapping (9 items)
PRODUCTS = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK", "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"]
PRODUCT_TO_IDX = {p: i for i, p in enumerate(PRODUCTS)}

# Seed indices mapping (5 crops)
CROPS = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"]
CROP_TO_IDX = {c: i for i, c in enumerate(CROPS)}

# Animal indices mapping (3 animals)
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ANIMAL_TO_IDX = {a: i for i, a in enumerate(ANIMALS)}

# Town Shop Demand Rates (units/day per active shop instance)
SHOP_DEMANDS = {
    "BAKERY": {"EGG": 6.0, "WHEAT": 6.0},
    "PIZZA_SHOP": {"MILK": 6.0, "TOMATO": 6.0, "WHEAT": 6.0},
    "BRUNCH_SPOT": {"EGG": 6.0, "WHEAT": 6.0, "STRAWBERRY": 6.0},
    "YARN_STORE": {"WOOL": 12.0},
    "ICE_CREAM_SHOP": {"STRAWBERRY": 6.0, "MILK": 6.0, "WHEAT": 6.0},
    "PET_CAFE": {"CARROT": 12.0},
    "SMOOTHIE_SHOP": {"STRAWBERRY": 6.0, "MILK": 6.0},
    "FARMERS_MARKET": {"WHEAT": 6.0, "CARROT": 6.0, "TOMATO": 6.0, "STRAWBERRY": 6.0},
}

GRID_SIZE = 10
FARM_GRID_CHANNELS = 21
GLOBAL_FEATURES_DIM = 47


def encode_farm_grid(farm: Dict[str, Any], day: int) -> np.ndarray:
    """
    Constructs a 10x10x21 spatial grid for a single player's farm.

    Channels:
    0: Empty/Unlocked (1.0)
    1: Locked Quadrant (1.0)
    2: Weed (1.0)
    3-7: Plant Crop One-Hot (5 channels: CARROT, MELON, STRAWBERRY, TOMATO, WHEAT)
    8: Plant Watered Today (1.0)
    9: Plant Consecutive Unwatered (normalized count: val / 2.0)
    10: Plant Yield Units (normalized count: val / 6.0)
    11: Plant Fertilized Active (1.0)
    12-14: Structure Animal One-Hot (3 channels: GOOSE, COW, SHEEP)
    15: Animal Fed Today (1.0)
    16: Animal Consecutive Unfed (normalized count: val / 2.0)
    17: Animal Cared Today (1.0)
    18: Animal Yield Units (normalized count: val / 6.0)
    19: Farmer Position (1.0)
    20: Hired Hands Count (float count)
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE, FARM_GRID_CHANNELS), dtype=np.float32)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = farm["tiles"][y][x]
            if tile is None:
                grid[y, x, 0] = 1.0
            elif tile == "LOCKED":
                grid[y, x, 1] = 1.0
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    grid[y, x, 2] = 1.0
                elif kind == "PLANT":
                    crop_idx = CROP_TO_IDX.get(tile.get("crop", ""), -1)
                    if crop_idx >= 0:
                        grid[y, x, 3 + crop_idx] = 1.0
                    grid[y, x, 8] = 1.0 if tile.get("watered_today") else 0.0
                    grid[y, x, 9] = float(tile.get("consecutive_unwatered", 0)) / 2.0
                    grid[y, x, 10] = float(tile.get("yield_units", 0)) / 6.0
                    grid[y, x, 11] = 1.0 if (tile.get("fertilized_until_day", -1) >= day) else 0.0
                elif kind in ("COOP", "PASTURE"):
                    animal = tile.get("animal")
                    if animal in ANIMAL_TO_IDX:
                        grid[y, x, 12 + ANIMAL_TO_IDX[animal]] = 1.0
                    grid[y, x, 15] = 1.0 if tile.get("fed_today") else 0.0
                    grid[y, x, 16] = float(tile.get("consecutive_unfed", 0)) / 2.0
                    grid[y, x, 17] = 1.0 if tile.get("cared_today") else 0.0
                    grid[y, x, 18] = float(tile.get("yield_units", 0)) / 6.0

    # Main Farmer Position (Channel 19)
    fx, fy = farm["farmer"]
    grid[fy, fx, 19] = 1.0

    # Hired Hands Positions (Channel 20)
    for hx, hy in farm.get("hands", []):
        grid[hy, hx, 20] += 1.0

    return grid


def encode_observation(obs: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Encodes raw observation dictionary into Dual-Tower spatial grid tensors and global feature tensor.

    Args:
        obs: Dictionary observation from environment.

    Returns:
        own_grid_tensor: Array of shape (10, 10, 21) representing own farm state.
        opp_grid_tensor: Array of shape (10, 10, 21) representing opponent farm state.
        global_tensor: Array of shape (47,) representing non-spatial scalar state.
    """
    player = obs["player"]
    me = obs["farms"][player]
    opponent = obs["farms"][1 - player]
    private = obs["private"]
    market = obs["market"]
    town = obs.get("town", {})
    day = obs["day"]

    # 1. Construct Dual 10x10 Spatial Feature Grids (21 channels each)
    own_grid = encode_farm_grid(me, day)
    opp_grid = encode_farm_grid(opponent, day)

    # 2. Construct Global Feature Vector (47 features total)
    # Timers, money, hire metrics (6 features)
    globals_list = [
        float(obs["day"]) / 30.0,
        float(obs["hour"]) / 24.0,
        float(obs["step"]) / 720.0,
        float(me["money"]) / 1000.0,
        float(opponent["money"]) / 1000.0,
        float(me["hires_today"]) / 10.0,
    ]

    # Market inventories and prices (9 + 9 = 18 features)
    for p in PRODUCTS:
        globals_list.append(float(market.get("inventory", {}).get(p, 0)) / 10000.0)
    for p in PRODUCTS:
        globals_list.append(float(market.get("prices", {}).get(p, 0)) / 100.0)

    # Private shed inventory (9 features) and seed counts (5 features)
    for p in PRODUCTS:
        globals_list.append(float(private.get("shed", {}).get(p, 0)) / 100.0)
    for c in CROPS:
        globals_list.append(float(private.get("seeds", {}).get(c, 0)) / 50.0)

    # Derived Product Demand Rates Vector (9 features: daily units consumed by Town Center + Shops)
    daily_demand = {p: 1.0 for p in PRODUCTS if p != "FERTILIZER"}
    daily_demand["FERTILIZER"] = 0.0

    active_shops = town.get("unlocked_shops", [])
    for shop_name in active_shops:
        if shop_name in SHOP_DEMANDS:
            for item, rate in SHOP_DEMANDS[shop_name].items():
                daily_demand[item] += rate

    for p in PRODUCTS:
        globals_list.append(daily_demand[p] / 100.0)

    own_grid_tensor = jnp.array(own_grid, dtype=jnp.float32)
    opp_grid_tensor = jnp.array(opp_grid, dtype=jnp.float32)
    global_tensor = jnp.array(globals_list, dtype=jnp.float32)

    return own_grid_tensor, opp_grid_tensor, global_tensor
