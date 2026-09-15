"""
Parameterized Inventory and Shed Management Primitives (Abstraction Level 1).
"""
from typing import List, Tuple, Any, Optional


def drop_to_shed() -> List[str]:
    """Issues DROP action when shed-adjacent to deposit full field inventory into storage."""
    return ["DROP"]


def pickup_from_shed(item: str, count: int = 1) -> List[Any]:
    """Issues PICKUP action when shed-adjacent to retrieve items/animals from storage."""
    return ["PICKUP", str(item).upper(), int(count)]
