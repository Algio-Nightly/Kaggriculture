import json
from kaggle_environments import make


def to_serializable(obj):
    """Recursively convert dict/Struct/AttrDict/lists for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    return obj


captured_states = {}


def agent(obs, config=None):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    step = obs["step"]
    day = obs["day"]
    hour = obs["hour"]

    # Market queue for this turn
    market_orders = []
    farmer_action = ["PASS"]
    hands_actions = []

    # Get current tile under farmer
    tile = me["tiles"][fy][fx] if (0 <= fy < 10 and 0 <= fx < 10) else None

    # --- Market Strategy ---
    if step == 0:
        market_orders.append(["BUY_SEED", "WHEAT", 5])
        market_orders.append(["BUY_SEED", "CARROT", 3])
        market_orders.append(["BUY_SEED", "TOMATO", 2])
        market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
        market_orders.append(["HIRE"])
    elif step == 24:  # Day 1 start
        market_orders.append(["HIRE"])
        market_orders.append(["BUY_PRODUCT", "WHEAT", 5])  # Feed for animals
    elif step == 48:  # Day 2 start
        market_orders.append(["BUY_LAND"])  # Unlock NE quadrant
        market_orders.append(["HIRE"])
    elif step == 72:  # Day 3 start
        # Sell produce in shed
        wheat_count = private["shed"].get("WHEAT", 0)
        carrot_count = private["shed"].get("CARROT", 0)
        if wheat_count > 0:
            market_orders.append(["SELL", "WHEAT", wheat_count])
        if carrot_count > 0:
            market_orders.append(["SELL", "CARROT", carrot_count])

    # --- Farmer Actions Sequence ---
    # Goal: Move around NW quadrant (0..4, 0..4), plant, water, build coop, place goose, feed, harvest

    # Turn 1: Farmer at (4,4) picks up GOOSE from shed
    if step == 1 and private["shed"].get("GOOSE", 0) > 0:
        farmer_action = ["PICKUP", "GOOSE", 1]
    # Move farmer towards (1,1) to build coop and plant seeds
    elif fx > 1:
        farmer_action = ["WEST"]
    elif fy > 1:
        farmer_action = ["NORTH"]
    elif fx < 1:
        farmer_action = ["EAST"]
    elif fy < 1:
        farmer_action = ["SOUTH"]
    else:
        # Farmer is at (1,1)
        if tile is None:
            # Build COOP at (1,1)
            farmer_action = ["BUILD_COOP"]
        elif isinstance(tile, dict) and tile.get("kind") == "COOP":
            if tile.get("animal") is None and "GOOSE" in private["inventories"][0]:
                farmer_action = ["PLACE", "GOOSE"]
            elif tile.get("animal") == "GOOSE":
                if not tile.get("fed_today"):
                    if private["seeds"].get("WHEAT", 0) > 0 or private["shed"].get("WHEAT", 0) > 0:
                        farmer_action = ["FEED"]
                    else:
                        farmer_action = ["CARE"]
                elif not tile.get("cared_today"):
                    farmer_action = ["CARE"]
                elif tile.get("fertilizer_available", 0) > 0:
                    farmer_action = ["COLLECT_FERTILIZER"]
                elif tile.get("yield_units", 0) > 0:
                    farmer_action = ["HARVEST"]
                else:
                    farmer_action = ["WEST"]  # Move away to plant elsewhere
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                farmer_action = ["HARVEST"]
            elif not tile.get("watered_today"):
                farmer_action = ["WATER"]
            else:
                farmer_action = ["WEST"]
        else:
            farmer_action = ["WEST"]

    # If farmer is at (0,1), plant or manage crop
    if (fx, fy) == (0, 1):
        if tile is None:
            if private["seeds"].get("WHEAT", 0) > 0:
                farmer_action = ["PLANT", "WHEAT"]
            elif private["seeds"].get("CARROT", 0) > 0:
                farmer_action = ["PLANT", "CARROT"]
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                farmer_action = ["HARVEST"]
            elif not tile.get("watered_today"):
                farmer_action = ["WATER"]
            else:
                farmer_action = ["EAST"]

    # If farmer is at (0,0), plant TOMATO
    if (fx, fy) == (0, 0):
        if tile is None and private["seeds"].get("TOMATO", 0) > 0:
            farmer_action = ["PLANT", "TOMATO"]
        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                farmer_action = ["HARVEST"]
            elif not tile.get("watered_today"):
                farmer_action = ["WATER"]
            else:
                farmer_action = ["SOUTH"]

    # --- Hired Hands Actions ---
    for hand_idx, hand_pos in enumerate(me["hands"]):
        hx, hy = hand_pos
        hand_tile = me["tiles"][hy][hx] if (0 <= hy < 10 and 0 <= hx < 10) else None
        
        # Hand logic: Move around, plant remaining seeds, water unwatered plants, harvest
        if hand_tile is None:
            if private["seeds"].get("CARROT", 0) > 0:
                hands_actions.append(["PLANT", "CARROT"])
            elif private["seeds"].get("WHEAT", 0) > 0:
                hands_actions.append(["PLANT", "WHEAT"])
            else:
                hands_actions.append(["EAST" if hx < 4 else "SOUTH"])
        elif isinstance(hand_tile, dict) and hand_tile.get("kind") == "PLANT":
            if hand_tile.get("yield_units", 0) > 0:
                hands_actions.append(["HARVEST"])
            elif not hand_tile.get("watered_today"):
                hands_actions.append(["WATER"])
            else:
                hands_actions.append(["SOUTH" if hy < 4 else "WEST"])
        elif isinstance(hand_tile, dict) and hand_tile.get("kind") == "WEED":
            hands_actions.append(["DIG"])
        else:
            hands_actions.append(["PASS"])

    # Drop inventory into shed if farmer is adjacent to shed and holding harvested items
    if (fx, fy) in [(4, 4), (5, 4), (4, 5), (5, 5)]:
        # Check if farmer is carrying crops/produce
        inv = private["inventories"][0] if len(private["inventories"]) > 0 else {}
        if any(count > 0 for item, count in inv.items() if item != "GOOSE"):
            farmer_action = ["DROP"]

    # --- Record Snapshot Observations at key steps ---
    # Step 0 (Start), Step 23 (End of Day 0), Step 47 (End of Day 1), Step 71 (End of Day 2), Step 95 (End of Day 3 / Turn 96)
    if step in [0, 23, 47, 71, 95]:
        captured_states[f"day_{day}_turn_{hour}_step_{step}"] = to_serializable(obs)

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}


if __name__ == "__main__":
    total_turns = 24 * 4  # 96 turns (4 full days)
    print(f"Running kaggriculture environment for {total_turns} turns (4 days)...")
    env = make("kaggriculture", configuration={"episodeSteps": total_turns}, debug=True)
    env.run([agent, "starter"])

    # Add final end-of-game summary structure to JSON output
    final_obs = env.steps[-1][0]["observation"]
    captured_states["final_state_step_95"] = to_serializable(final_obs)

    # Save to obs_populated.json and obs_syn.json
    with open("obs_populated.json", "w") as f:
        json.dump(captured_states, f, indent=2, default=str)
    
    with open("obs_syn.json", "w") as f:
        json.dump(captured_states, f, indent=2, default=str)

    print("Successfully completed 96 turns!")
    print("Saved populated game states to obs_populated.json and obs_syn.json.")
