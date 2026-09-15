"""
Parameterized Market and Economic Primitives (Abstraction Level 1).
"""
from typing import List, Any


def queue_buy_seed(crop: str, count: int = 1) -> List[Any]:
    """Creates a BUY_SEED market order."""
    return ["BUY_SEED", str(crop).upper(), int(count)]


def queue_buy_product(product: str, count: int = 1) -> List[Any]:
    """Creates a BUY_PRODUCT market order (e.g. WHEAT or FERTILIZER)."""
    return ["BUY_PRODUCT", str(product).upper(), int(count)]


def queue_buy_animal(animal: str, count: int = 1) -> List[Any]:
    """Creates a BUY_ANIMAL market order (GOOSE, COW, SHEEP)."""
    return ["BUY_ANIMAL", str(animal).upper(), int(count)]


def queue_sell(item: str, count: int = 1) -> List[Any]:
    """Creates a SELL market order for produce or resources."""
    return ["SELL", str(item).upper(), int(count)]


def queue_hire() -> List[str]:
    """Creates a HIRE market order for 1 additional farm hand today."""
    return ["HIRE"]


def queue_buy_land() -> List[str]:
    """Creates a BUY_LAND market order to unlock the next available quadrant."""
    return ["BUY_LAND"]
