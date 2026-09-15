"""
Parameterized Pathfinding and Spatial Geometry Primitives (Abstraction Level 1).
"""
from typing import Tuple, List, Optional, Dict, Any

SHED_ADJACENT_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]

QUADRANT_RANGES = {
    "NW": (0, 5, 0, 5),    # x: 0..4, y: 0..4
    "NE": (5, 10, 0, 5),   # x: 5..9, y: 0..4
    "SW": (0, 5, 5, 10),   # x: 0..4, y: 5..9
    "SE": (5, 10, 5, 10),  # x: 5..9, y: 5..9
}


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
    """Calculates Manhattan distance between two grid coordinates."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def is_shed_adjacent(pos: Tuple[int, int]) -> bool:
    """Checks if a given coordinate is directly adjacent to the center shed."""
    return tuple(pos) in SHED_ADJACENT_TILES


def get_cardinal_direction_towards(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """
    Returns the single-step cardinal movement string to navigate from from_pos towards to_pos.
    Prioritizes horizontal or vertical move based on distance delta.
    """
    fx, fy = from_pos
    tx, ty = to_pos

    dx = tx - fx
    dy = ty - fy

    if dx == 0 and dy == 0:
        return "PASS"

    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    else:
        return "SOUTH" if dy > 0 else "NORTH"


def get_nearest_shed_tile(from_pos: Tuple[int, int]) -> Tuple[int, int]:
    """Finds the nearest shed-adjacent tile from the given position."""
    return min(SHED_ADJACENT_TILES, key=lambda p: manhattan_distance(from_pos, p))


def get_nearest_empty_tile(
    from_pos: Tuple[int, int],
    tiles: List[List[Any]],
    unlocked_quads: Optional[List[str]] = None,
    target_quadrant: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """
    Finds the nearest empty unlocked tile to from_pos, optionally filtered by quadrant.
    """
    candidates = []
    for y in range(10):
        for x in range(10):
            tile = tiles[y][x]
            if tile is None:
                # Check quadrant filter if specified
                if target_quadrant and target_quadrant in QUADRANT_RANGES:
                    x0, x1, y0, y1 = QUADRANT_RANGES[target_quadrant]
                    if not (x0 <= x < x1 and y0 <= y < y1):
                        continue
                candidates.append((x, y))

    if not candidates:
        return None
    return min(candidates, key=lambda p: manhattan_distance(from_pos, p))


def get_nearest_mature_crop_tile(from_pos: Tuple[int, int], tiles: List[List[Any]]) -> Optional[Tuple[int, int]]:
    """Finds the nearest tile with a harvestable mature crop."""
    candidates = []
    for y in range(10):
        for x in range(10):
            tile = tiles[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if tile.get("yield_units", 0) > 0:
                    candidates.append((x, y))

    if not candidates:
        return None
    return min(candidates, key=lambda p: manhattan_distance(from_pos, p))


def get_nearest_unwatered_crop_tile(from_pos: Tuple[int, int], tiles: List[List[Any]]) -> Optional[Tuple[int, int]]:
    """Finds the nearest tile with a living plant that has not been watered today."""
    candidates = []
    for y in range(10):
        for x in range(10):
            tile = tiles[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if not tile.get("watered_today", False):
                    candidates.append((x, y))

    if not candidates:
        return None
    return min(candidates, key=lambda p: manhattan_distance(from_pos, p))


def get_nearest_weed_tile(from_pos: Tuple[int, int], tiles: List[List[Any]]) -> Optional[Tuple[int, int]]:
    """Finds the nearest tile infested with a weed."""
    candidates = []
    for y in range(10):
        for x in range(10):
            tile = tiles[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                candidates.append((x, y))

    if not candidates:
        return None
    return min(candidates, key=lambda p: manhattan_distance(from_pos, p))
