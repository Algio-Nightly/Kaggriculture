"""
Parameterized Single-Turn Crop Farming Primitives (Abstraction Level 1).
"""
from typing import List, Tuple, Any, Optional
from scripts.abs_level_1.pathfinding import get_cardinal_direction_towards


def water_tile() -> List[str]:
    """Issues WATER action on the current tile."""
    return ["WATER"]


def harvest_tile() -> List[str]:
    """Issues HARVEST action on the current tile."""
    return ["HARVEST"]


def plant_tile(crop: str) -> List[str]:
    """Issues PLANT action on the current tile for a specified crop."""
    return ["PLANT", str(crop)]


def fertilize_tile() -> List[str]:
    """Issues FERTILIZE action on the current tile."""
    return ["FERTILIZE"]


def dig_tile() -> List[str]:
    """Issues DIG action on the current tile (clears weeds/dead plants)."""
    return ["DIG"]


def step_or_act(unit_pos: Tuple[int, int], target_pos: Tuple[int, int], action_if_there: List[str]) -> List[str]:
    """
    Executes a 1-step parameterized action primitive:
    If unit is already at target_pos -> executes action_if_there.
    Else -> takes 1 cardinal step towards target_pos.
    """
    if unit_pos[0] == target_pos[0] and unit_pos[1] == target_pos[1]:
        return action_if_there
    
    direction = get_cardinal_direction_towards(unit_pos, target_pos)
    return [direction]
