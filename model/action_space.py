"""
Action space mapping utilities for Kaggriculture JAX model.
Maps model discrete action indices to game action dictionary format.
"""
from typing import Dict, Any, List

# Discrete Farmer Action Space Mapping
FARMER_ACTIONS = [
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


def action_index_to_dict(action_idx: int) -> Dict[str, Any]:
    """
    Converts a discrete action index into the required action dictionary structure.
    
    Args:
        action_idx: Integer index of selected action.
        
    Returns:
        Action dictionary: {"farmer": [...], "hands": [], "market": []}
    """
    if 0 <= action_idx < len(FARMER_ACTIONS):
        farmer_op = FARMER_ACTIONS[action_idx]
    else:
        farmer_op = ["PASS"]

    return {
        "farmer": farmer_op,
        "hands": [],
        "market": []
    }
