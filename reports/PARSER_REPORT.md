# Observation Parser & Encoding Module Report

This document details the **Observation Parser & Encoder module** implemented in [`model/encoder.py`](file:///c:/Applications%20and%20Development/Kaggriculture/model/encoder.py). 

---

## 1. Module Overview & Responsibilities

The Parser serves as the bridge between raw game simulation data and JAX neural network tensors.

```
                    Raw Kaggle JSON Dict (obs)
                                │
                    encode_observation(obs)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   own_grid_tensor       opp_grid_tensor          global_tensor
 (10 × 10 × 21 float32) (10 × 10 × 21 float32)  (47-dim float32 vector)
        │                       │                       │
        ▼                       ▼                       ▼
  Own 2D-CNN Tower      Opponent 2D-CNN Tower       Global MLP Branch
```

### Key Responsibilities:
1. **Dictionary Traversal**: Safely inspects raw observation dictionaries (`farms`, `market`, `private`, `town`, `step`, `day`, `hour`).
2. **Dual-Tower Spatial Grid Generation**: Constructs two independent $10 \times 10 \times 21$ float32 feature grids (`own_grid_tensor` and `opp_grid_tensor`).
3. **Global Feature Normalization & Derivation**: Extracts and scales 47 non-spatial scalar features into `global_tensor` (including **Derived Per-Product Town Demand Rates**).
4. **JAX Device Array Conversion**: Casts NumPy arrays into `jnp.float32` device tensors ready for JAX `@jit` execution.

---

## 2. Extraction & Parsing Architecture

### Function Entrypoints

#### 1. `encode_farm_grid(farm: Dict[str, Any], day: int) -> np.ndarray`
Extracts a $10 \times 10 \times 21$ spatial matrix for a single player's farm grid.

* **Grid Dimensions**: `(10, 10, 21)`
* **Channel Allocation**:
  - **Soil & Quadrants (Channels 0–1)**: Checks if tile is `None` (empty soil) or `"LOCKED"` (unbought 5×5 quadrant).
  - **Weeds (Channel 2)**: Checks if `tile.kind == "WEED"`.
  - **Crops (Channels 3–11)**: Maps crop types (`WHEAT`, `CARROT`, `TOMATO`, `STRAWBERRY`, `MELON`), watering status (`watered_today`), unwatered risk count (`consecutive_unwatered`), harvestable yield (`yield_units`), and active fertilizer (`fertilized_until_day`).
  - **Livestock (Channels 12–18)**: Maps animal types (`GOOSE`, `COW`, `SHEEP`), feeding status (`fed_today`), unfed risk count (`consecutive_unfed`), cared status (`cared_today`), and uncollected produce (`yield_units`).
  - **Units (Channels 19–20)**: Maps main farmer tile position (`Channel 19`) and active hired hands count per tile (`Channel 20`).

#### 2. `encode_observation(obs: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]`
Extracts the complete game state for the active player perspective:
- Identifies active player `p = obs["player"]` and opponent `1 - p`.
- Calls `encode_farm_grid(obs["farms"][p])` $\rightarrow$ `own_grid_tensor`.
- Calls `encode_farm_grid(obs["farms"][1-p])` $\rightarrow$ `opp_grid_tensor`.
- Derives product town demand rates and normalizes global scalars $\rightarrow$ `global_tensor`.

---

## 3. Real-World Replay Parsing Benchmark

Tested against Step 50 of actual Kaggle competition replay `91889284.json`:

```python
=== FULL INPUT ENCODING DEMO ===
Own Grid Tensor Shape: (10, 10, 21) dtype: float32
Opp Grid Tensor Shape: (10, 10, 21) dtype: float32
Global Vector Shape:   (47,) dtype: float32

--- Active Spatial Channels in Own Grid (10x10) ---
Channel  1 (Locked Quadrant):  75 active cells (3 unbought 5x5 quadrants)
Channel  3 (Wheat Crop):        3 active cells
Channel  4 (Carrot Crop):       7 active cells
Channel  5 (Tomato Crop):       2 active cells
Channel  6 (Strawberry Crop):   1 active cells
Channel  7 (Melon Crop):        6 active cells
Channel 10 (Crop Yield Units): 16 active cells with harvestable produce
Channel 12 (Goose Coop):        1 active cell
Channel 14 (Sheep Pasture):     1 active cell
Channel 19 (Main Farmer):       1 active cell (Farmer position)
Channel 20 (Hired Hands):       4 active cells (Hired worker positions)

--- Global Vector Normalized Values ---
Timers (Day/30, Hour/24, Step/720):      [0.0667, 0.0833, 0.0694]
Money (Own/1000, Opp/1000):               [1.0220, 0.1730]
Hires Today (Hires/10):                   [0.6000] (6 hands hired)
Market Supply (Carrot, Egg, Fert):        [0.9997, 0.9997, 1.0005]
Market Prices (Carrot, Egg, Fert):        [0.3700, 0.5000, 0.9900]
Derived Town Demand Rates (9 products):   [0.0100, 0.0700, 0.0000, ...]  # Wheat + Egg demand from Bakery!
```

---

## 4. Current Status & Verification

- **Parser Status**: **100% Complete & Operational**
- **Test Suite**: Verified via `kaggle-environments` simulations and offline replay JSON parsing.
- **Next Step Integration**: Ready to feed into Supervised Pre-Training (Behavioral Cloning) and PPO RL Rollouts.
