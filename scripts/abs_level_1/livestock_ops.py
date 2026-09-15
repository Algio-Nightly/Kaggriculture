"""
Parameterized Single-Turn Livestock Primitives (Abstraction Level 1).
"""
from typing import List, Tuple, Any, Optional


def build_structure(structure_type: str = "COOP") -> List[str]:
    """Issues BUILD_COOP or BUILD_PASTURE on the current empty tile."""
    if structure_type.upper() == "PASTURE":
        return ["BUILD_PASTURE"]
    return ["BUILD_COOP"]


def place_animal(animal: str) -> List[str]:
    """Places an animal from carried inventory onto the current structure."""
    return ["PLACE", str(animal).upper()]


def feed_animal() -> List[str]:
    """Feeds the animal on the current tile with wheat from carried inventory."""
    return ["FEED"]


def care_animal() -> List[str]:
    """Performs daily CARE on the animal on the current tile."""
    return ["CARE"]


def collect_fertilizer() -> List[str]:
    """Collects generated fertilizer from the animal on the current tile."""
    return ["COLLECT_FERTILIZER"]
