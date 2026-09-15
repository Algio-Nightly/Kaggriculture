# Action Scripts & Level 1 Parameterized Primitives Specification

## 1. Overview & Abstraction Level 1 Concept

In Kaggriculture, every single in-game turn requires an agent to return a 3-component composite dictionary:
```python
{
    "farmer": [farmer_op, ...args],      # 1 action for the main farmer
    "hands":  [[hand_op, ...args], ...],   # 1 action per active hired hand
    "market": [[order, ...args], ...]      # up to 10 market orders
}
```

**Abstraction Level 1** provides a comprehensive library of **1-turn parameterized action primitives**. Instead of requiring multi-turn pathfinding or complex loops, these scripts allow an RL model or rule-based policy to emit complete, coordinated, and rule-compliant actions in **1 single step**.

---

## 2. Cross-Validation with Official Game Rules (`README.md` & `AGENTS.md`)

All scripts in `scripts/abs_level_1/` have been cross-referenced and validated against official mechanics:

| Action Primitive | Valid Arguments | Preconditions & Mechanics | Official Rule Reference |
| :--- | :--- | :--- | :--- |
| **`PLANT`** | `crop` (`WHEAT`, `CARROT`, `TOMATO`, `STRAWBERRY`, `MELON`) | Current tile must be empty (`None`) and unlocked. Consumes 1 seed from player's seed pool. (If multiple units plant same turn without enough seeds, both fail). | `README.md:L54-57` |
| **`WATER`** | None | Tile must have a plant. Done once per day; subsequent daily waterings no-op. (2 consecutive missed days $\to$ turns into weed). | `README.md:L58, L111` |
| **`HARVEST`** | None | Tile must have mature plant (`yield_units > 0`) or animal with product. Yield is added to the acting unit's inventory. | `README.md:L59, L69` |
| **`FERTILIZE`** | None | Tile must have plant. Consumes 1 fertilizer from shed. Doubles per-day watering bonus for 3 days. | `README.md:L60-61` |
| **`FEED`** | None | Tile must have animal. Consumes 1 wheat (from inventory or shed). Done once/day. (2 consecutive unfed days $\to$ animal escapes). | `README.md:L68, L111` |
| **`CARE`** | None | Tile must have animal. Banks +1 bonus yield paid on the next scheduled production tick if fed. | `README.md:L71-80` |
| **`COLLECT_FERTILIZER`** | None | Tile must have animal with `fertilizer_available == True`. Yields 1 fertilizer into unit inventory. | `README.md:L70` |
| **`BUILD_COOP` / `BUILD_PASTURE`** | None | Tile must be empty (`None`) and unlocked. Coop costs $0 to build (holds goose); Pasture costs $0 (holds cow/sheep). | `README.md:L84-85` |
| **`PLACE`** | `item`, `n` | Standing on empty coop/pasture places animal from inventory onto tile. Standing on shed-adjacent tile drops items into shed. | `README.md:L65-67` |
| **`PICKUP`** | `item`, `n` | Must be orthogonally adjacent to shed: `(4,4), (5,4), (4,5), (5,5)`. Moves `n` items from shed to unit inventory. | `README.md:L49` |
| **`DROP`** | None | Must be orthogonally adjacent to shed. Dumps unit's entire inventory into shed. Excess past 100 cap discarded. | `README.md:L50` |
| **`DIG`** | None | Clears weed, removes living/dead plant, or removes *empty* coop/pasture. Occupied coop/pasture cannot be dug. | `README.md:L86` |
| **`BUY_SEED`** | `crop`, `n` | Fixed price: Wheat $10, Carrot $20, Tomato $50, Strawberry $100, Melon $80. Adds directly to `seeds`. | `README.md:L11-17, L96` |
| **`BUY_ANIMAL`** | `animal`, `n` | Fixed price: Goose $300, Cow $400, Sheep $500. Added to `shed`. | `README.md:L18-20, L98` |
| **`BUY_PRODUCT`** | `product`, `n` | **Restricted**: ONLY `WHEAT` and `FERTILIZER` can be bought back from the dynamic market. | `README.md:L100, L200` |
| **`SELL`** | `item`, `n` | Unrestricted: Any produce, animal product, or fertilizer in shed sold at current dynamic market price. | `README.md:L103, L200` |
| **`HIRE`** | None | Hires 1 farm hand for today. Cost is $1 \times \text{fib}(n) = [1, 1, 2, 3, 5, 8, 13 \dots]$. Resets daily. | `README.md:L105, L155` |
| **`BUY_LAND`** | None | Unlocks next quadrant: `NE` ($1k), `SW` ($2k), `SE` ($4k). | `README.md:L106-107` |

---

## 3. Parameter Schema Required for 1-Step Abstraction

In a single RL decision step at Abstraction Level 1, the parameter heads configure:

### 3.1. Market Parameters (Processed up to 10 per turn)
- **`hire_decision`**: $\{0, 1\}$ binary flag.
- **`buy_land_decision`**: $\{0, 1\}$ binary flag.
- **`buy_seed_order`**: `(crop: 0..4, count: 1..5)`.
- **`buy_animal_order`**: `(animal: 0..2, count: 1)`.
- **`buy_product_order`**: `(product: WHEAT/FERTILIZER, count: 1..10)`.
- **`sell_order`**: `(item: 0..8, count: 1..100)`.

### 3.2. Farmer Parameters (1 Action / turn)
- Categorical choice over 16 tactical actions:
  `["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER", "DIG", "PLANT_WHEAT", "PLANT_CARROT", "BUILD_COOP", "DROP"]`.

### 3.3. Hired Hands Parameters (1 Action / active hand)
- Priority work rule parameter for hands:
  $$\text{HandPriority} \in \{\text{HARVEST} \to \text{WATER} \to \text{PLANT} \to \text{DIG} \to \text{STEP}\}$$
