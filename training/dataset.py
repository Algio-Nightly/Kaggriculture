"""
Replay Dataset Preprocessor and DataLoader for Phase 1 Behavioral Cloning.

Supports:
1. Parsing Kaggle episode replay JSON files and snapshot dumps.
2. Extracting winning player state-action transitions, action masks, and final return targets.
3. Binary NPZ caching for instant memory-mapped loading.
4. JAX minibatch generator.
"""
import os
import json
import glob
from typing import Dict, Any, List, Tuple, Optional, Generator
import numpy as np
import jax.numpy as jnp

from model.encoder import encode_observation
from model.action_space import (
    compute_action_masks,
    HEAD_DIMS,
    TacticalAction,
    FarmingStrategy,
    SellingStrategy,
    CropParam,
    LaborParam,
    QuadrantParam,
    CROP_NAMES,
    QUADRANT_NAMES,
)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_REPLAY_DIR = os.path.join(PROJECT_ROOT, "replays")
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "dataset_cache.npz")


def extract_action_labels(action_dict: Dict[str, Any], obs: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Extracts multi-head integer target labels and head activity weights from a player's action dict.
    
    Returns:
        targets: Dict mapping head name to target integer index.
        weights: Dict mapping head name to 1.0 (active) or 0.0 (inactive).
    """
    targets = {}
    weights = {}

    farmer_op = action_dict.get("farmer", ["PASS"])
    market_orders = action_dict.get("market", [])
    
    op_name = farmer_op[0] if len(farmer_op) > 0 else "PASS"

    # Tactical action mapping
    tactical_idx = TacticalAction.PASS
    if op_name in TacticalAction.__members__:
        tactical_idx = TacticalAction[op_name]
    elif op_name == "PLANT":
        crop = farmer_op[1] if len(farmer_op) > 1 else "WHEAT"
        if crop == "WHEAT":
            tactical_idx = TacticalAction.PLANT_WHEAT
        elif crop == "CARROT":
            tactical_idx = TacticalAction.PLANT_CARROT
        else:
            tactical_idx = TacticalAction.PLANT_WHEAT
    elif op_name == "BUILD_COOP":
        tactical_idx = TacticalAction.BUILD_COOP

    targets["tactical"] = int(tactical_idx)
    weights["tactical"] = 1.0

    # Macro action inference (for day start hour == 0)
    hour = obs.get("hour", 0)
    if hour == 0:
        # Determine farming strategy from market orders and tile ops
        has_plant = any(o[0] == "BUY_SEED" for o in market_orders) or op_name == "PLANT"
        has_animal = any(o[0] == "BUY_ANIMAL" for o in market_orders) or op_name in ["BUILD_COOP", "BUILD_PASTURE"]
        has_sell = any(o[0] == "SELL" for o in market_orders)

        if has_plant:
            targets["farming"] = int(FarmingStrategy.EXPAND_CROPS)
        elif has_animal:
            targets["farming"] = int(FarmingStrategy.EXPAND_LIVESTOCK)
        elif op_name == "HARVEST":
            targets["farming"] = int(FarmingStrategy.HARVEST_ALL)
        else:
            targets["farming"] = int(FarmingStrategy.SUSTAIN)
        weights["farming"] = 1.0

        # Selling Strategy
        if has_sell:
            targets["selling"] = int(SellingStrategy.DRIP_FEED_TOWN)
        else:
            targets["selling"] = int(SellingStrategy.HOLD)
        weights["selling"] = 1.0

        # Crop Param
        seed_orders = [o for o in market_orders if o[0] == "BUY_SEED"]
        if seed_orders and seed_orders[0][1] in CROP_NAMES:
            targets["crop"] = CROP_NAMES.index(seed_orders[0][1])
            weights["crop"] = 1.0
        else:
            targets["crop"] = int(CropParam.WHEAT)
            weights["crop"] = 1.0 if has_plant else 0.0

        # Labor Param
        hire_count = sum(1 for o in market_orders if o[0] == "HIRE")
        targets["labor"] = min(hire_count, len(LaborParam) - 1)
        weights["labor"] = 1.0

        # Quadrant Param
        targets["quadrant"] = int(QuadrantParam.NW)
        weights["quadrant"] = 1.0
    else:
        # Intra-day: inactive macro heads
        targets["farming"] = 0
        weights["farming"] = 0.0
        targets["selling"] = 0
        weights["selling"] = 0.0
        targets["crop"] = 0
        weights["crop"] = 0.0
        targets["labor"] = 0
        weights["labor"] = 0.0
        targets["quadrant"] = 0
        weights["quadrant"] = 0.0

    return targets, weights


def process_replay_json(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses a single Kaggle replay JSON file or snapshot dump into a sequence of transition dictionaries.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        records = []

        # Format A: Kaggle Environment Replay ("steps" list)
        if isinstance(data, dict) and "steps" in data:
            steps = data["steps"]
            if not steps or len(steps) < 2:
                return None

            # Determine winning player index from final step rewards
            final_step = steps[-1]
            p0_rew = final_step[0].get("reward", 0.0) or 0.0
            p1_rew = final_step[1].get("reward", 0.0) or 0.0
            winner_idx = 0 if p0_rew >= p1_rew else 1
            final_coins = max(p0_rew, p1_rew)

            for step_tuple in steps:
                player_state = step_tuple[winner_idx]
                obs = player_state.get("observation")
                action = player_state.get("action")
                if not obs or action is None:
                    continue

                obs["player"] = winner_idx
                targets, weights = extract_action_labels(action if isinstance(action, dict) else {}, obs)
                
                records.append({
                    "obs": obs,
                    "targets": targets,
                    "weights": weights,
                    "final_coins": float(final_coins),
                })

        # Format B: Direct Observation JSON dictionary (e.g. obs_populated.json)
        elif isinstance(data, dict) and "farms" in data:
            p_idx = data.get("player", 0)
            my_money = data["farms"][p_idx].get("money", 3000)
            sample_action = {"farmer": ["WATER"], "hands": [], "market": []}
            targets, weights = extract_action_labels(sample_action, data)
            records.append({
                "obs": data,
                "targets": targets,
                "weights": weights,
                "final_coins": float(my_money + 1000.0),
            })

        # Format C: Multi-observation nested dictionary
        elif isinstance(data, dict):
            for key, obs in data.items():
                if isinstance(obs, dict) and "farms" in obs:
                    p_idx = obs.get("player", 0)
                    my_money = obs["farms"][p_idx].get("money", 3000)
                    sample_action = {"farmer": ["WATER"], "hands": [], "market": []}
                    targets, weights = extract_action_labels(sample_action, obs)
                    records.append({
                        "obs": obs,
                        "targets": targets,
                        "weights": weights,
                        "final_coins": float(my_money + 1000.0),
                    })

        return records if records else None

    except Exception as e:
        return None


def build_dataset_arrays(replay_files: List[str]) -> Dict[str, np.ndarray]:
    """
    Processes all replay files and converts them into batched NumPy arrays.
    """
    own_grids = []
    opp_grids = []
    global_vecs = []
    final_coins_list = []

    all_targets = {head: [] for head in HEAD_DIMS.keys()}
    all_weights = {head: [] for head in HEAD_DIMS.keys()}
    all_masks = {head: [] for head in HEAD_DIMS.keys()}

    for fpath in replay_files:
        records = process_replay_json(fpath)
        if not records:
            continue

        for r in records:
            obs = r["obs"]
            own_g, opp_g, glob_v = encode_observation(obs)
            masks = compute_action_masks(obs)

            own_grids.append(np.asarray(own_g, dtype=np.float32))
            opp_grids.append(np.asarray(opp_g, dtype=np.float32))
            global_vecs.append(np.asarray(glob_v, dtype=np.float32))
            final_coins_list.append(r["final_coins"])

            for head in HEAD_DIMS.keys():
                all_targets[head].append(r["targets"].get(head, 0))
                all_weights[head].append(r["weights"].get(head, 1.0))
                all_masks[head].append(masks.get(head, np.ones(HEAD_DIMS[head], dtype=bool)))

    dataset = {
        "own_grid": np.stack(own_grids, axis=0),
        "opp_grid": np.stack(opp_grids, axis=0),
        "global_vec": np.stack(global_vecs, axis=0),
        "final_coins": np.array(final_coins_list, dtype=np.float32),
    }

    for head in HEAD_DIMS.keys():
        dataset[f"target_{head}"] = np.array(all_targets[head], dtype=np.int32)
        dataset[f"weight_{head}"] = np.array(all_weights[head], dtype=np.float32)
        dataset[f"mask_{head}"] = np.array(all_masks[head], dtype=bool)

    return dataset


def load_or_build_dataset(
    replay_dir: str = DEFAULT_REPLAY_DIR,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_rebuild: bool = False,
    rebuild: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads dataset from NPZ binary cache if available, or compiles it from replay JSONs.
    All relative paths are automatically resolved relative to PROJECT_ROOT.
    """
    if rebuild is not None:
        force_rebuild = rebuild

    # Resolve relative paths against PROJECT_ROOT
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(PROJECT_ROOT, cache_path)
    if not os.path.isabs(replay_dir):
        replay_dir = os.path.join(PROJECT_ROOT, replay_dir)

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached dataset from {cache_path}...")
        return dict(np.load(cache_path))

    print(f"Compiling dataset from {replay_dir}...")
    json_files = glob.glob(os.path.join(replay_dir, "*.json"))

    # Also check project root JSON dumps if replay dir is empty
    if not json_files:
        json_files = glob.glob(os.path.join(PROJECT_ROOT, "*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON replay files found in '{replay_dir}' or project root '{PROJECT_ROOT}'!\n"
            "Please place replay JSON files in the 'replays/' directory or download with Kaggle CLI."
        )

    dataset = build_dataset_arrays(json_files)
    np.savez_compressed(cache_path, **dataset)
    print(f"Saved {dataset['own_grid'].shape[0]} samples to {cache_path}!")
    return dataset


class ReplayBatchGenerator:
    """Minibatch iterator over the preprocessed dataset for JAX training."""

    def __init__(self, dataset: Dict[str, np.ndarray], batch_size: int = 64, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = dataset["own_grid"].shape[0]
        self.indices = np.arange(self.num_samples)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_idx = self.indices[start_idx : start_idx + self.batch_size]
            
            batch_targets = {h: jnp.asarray(self.dataset[f"target_{h}"][batch_idx]) for h in HEAD_DIMS.keys()}
            batch_weights = {h: jnp.asarray(self.dataset[f"weight_{h}"][batch_idx]) for h in HEAD_DIMS.keys()}
            batch_masks = {h: jnp.asarray(self.dataset[f"mask_{h}"][batch_idx]) for h in HEAD_DIMS.keys()}

            yield {
                "own_grid": jnp.asarray(self.dataset["own_grid"][batch_idx]),
                "opp_grid": jnp.asarray(self.dataset["opp_grid"][batch_idx]),
                "global_vec": jnp.asarray(self.dataset["global_vec"][batch_idx]),
                "final_coins": jnp.asarray(self.dataset["final_coins"][batch_idx]),
                "targets": batch_targets,
                "weights": batch_weights,
                "masks": batch_masks,
            }