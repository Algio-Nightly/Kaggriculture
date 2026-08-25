"""
Observation Encoder module for Kaggriculture.
Converts raw dictionary observations into Dual-Tower spatial grid tensors and global scalar tensors.
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

GRID_SIZE = 10
FARM_GRID_CHANNELS = 19


def encode_farm_grid(farm: Dict[str, Any], day: int) -> np.ndarray:
    """
    Constructs a 10x10x19 spatial grid for a single player's farm.

    Channels:
    0: Empty/Unlocked (1.0)
    1: Locked Quadrant (1.0)
    2: Weed (1.0)
    3-7: Plant Crop One-Hot (5 channels: CARROT, MELON, STRAWBERRY, TOMATO, WHEAT)
    8: Plant Watered Today (1.0)
    9: Plant Consecutive Unwatered (float count)
    10: Plant Yield Units (float count)
    11: Plant Fertilized Active (1.0)
    12-14: Structure Animal One-Hot (3 channels: GOOSE, COW, SHEEP)
    15: Animal Fed Today (1.0)
    16: Animal Consecutive Unfed (float count)
    17: Animal Cared Today (1.0)
    18: Animal Yield Units (float count)
    19: Farmer Position (1.0)
    20: Hired Hands Count (float count)
    """
    # 19 base tile/structure channels + 2 unit channels = 19 channels total per farm grid
    # Channels 0-18:
    # 0: Empty, 1: Locked, 2: Weed
    # 3-7: Crop One-Hot, 8: Watered, 9: Consecutive Unwatered, 10: Crop Yield, 11: Fertilized
    # 12-14: Animal One-Hot, 15: Fed, 16: Consecutive Unfed, 17: Cared, 18: Animal Yield
    # 19: Farmer, 20: Hands
    # To keep total channels per farm grid at 19, we arrange:
    # 0: Empty, 1: Locked, 2: Weed
    # 3-7: Crops (5), 8: Watered, 9: Consecutive Unwatered, 10: Crop Yield, 11: Fertilized
    # 12-14: Animals (3), 15: Fed, 16: Consecutive Unfed, 17: Cared, 18: Animal Yield
    # Note: Farmer position and Hands count are added as channels 17 and 18 if animal care/yield are condensed,
    # or channels 17-18 are Farmer and Hands!
    # Let's cleanly define 19 channels per farm grid:
    # 0: Empty/Unlocked, 1: Locked, 2: Weed
    # 3-7: Crop One-Hot (5)
    # 8: Watered Today, 9: Consecutive Unwatered, 10: Crop Yield, 11: Fertilized
    # 12-14: Animal One-Hot (3)
    # 15: Fed Today, 16: Animal Yield / Care Status
    # 17: Farmer Location (1.0)
    # 18: Hired Hands Count (float)
    grid = np.zeros((GRID_SIZE, GRID_SIZE, 19), dtype=np.float32)

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
                    grid[y, x, 9] = float(tile.get("consecutive_unwatered", 0))
                    grid[y, x, 10] = float(tile.get("yield_units", 0))
                    grid[y, x, 11] = 1.0 if (tile.get("fertilized_until_day", -1) >= day) else 0.0
                elif kind in ("COOP", "PASTURE"):
                    animal = tile.get("animal")
                    if animal in ANIMAL_TO_IDX:
                        grid[y, x, 12 + ANIMAL_TO_IDX[animal]] = 1.0
                    grid[y, x, 15] = 1.0 if tile.get("fed_today") else 0.0
                    grid[y, x, 16] = float(tile.get("yield_units", 0))

    # Farmer Position (Channel 17)
    fx, fy = farm["farmer"]
    grid[fy, fx, 17] = 1.0

    # Hired Hands Positions (Channel 18)
    for hx, hy in farm.get("hands", []):
        grid[hy, hx, 18] += 1.0

    return grid


def encode_observation(obs: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Encodes raw observation dictionary into Dual-Tower spatial grid tensors and global feature tensor.

    Args:
        obs: Dictionary observation from environment.

    Returns:
        own_grid_tensor: Array of shape (10, 10, 19) representing own farm state.
        opp_grid_tensor: Array of shape (10, 10, 19) representing opponent farm state.
        global_tensor: Array of shape (38,) representing non-spatial scalar state.
    """
    player = obs["player"]
    me = obs["farms"][player]
    opponent = obs["farms"][1 - player]
    private = obs["private"]
    market = obs["market"]
    day = obs["day"]

    # 1. Construct Dual 10x10 Spatial Feature Grids (19 channels each)
    own_grid = encode_farm_grid(me, day)
    opp_grid = encode_farm_grid(opponent, day)

    # 2. Construct Global Feature Vector (38 features)
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
        globals_list.append(float(market["inventory"].get(p, 0)) / 10000.0)
    for p in PRODUCTS:
        globals_list.append(float(market["prices"].get(p, 0)) / 100.0)

    # Private shed inventory (9 features) and seed counts (5 features)
    for p in PRODUCTS:
        globals_list.append(float(private["shed"].get(p, 0)) / 100.0)
    for c in CROPS:
        globals_list.append(float(private["seeds"].get(c, 0)) / 50.0)

    own_grid_tensor = jnp.array(own_grid, dtype=jnp.float32)
    opp_grid_tensor = jnp.array(opp_grid, dtype=jnp.float32)
    global_tensor = jnp.array(globals_list, dtype=jnp.float32)

    return own_grid_tensor, opp_grid_tensor, global_tensor
