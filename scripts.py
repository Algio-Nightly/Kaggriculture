import json

def analyze_obs():
    with open('obs_populated.json', 'r') as f:
        data = json.load(f)
    
    print("=== Top-level Snapshots in obs_populated.json ===")
    for key, obs in data.items():
        print(f"\nSnapshot Key: {key}")
        print(f"  step: {obs.get('step')}, day: {obs.get('day')}, hour: {obs.get('hour')}, player: {obs.get('player')}")
        
        # Farms summary
        farms = obs.get("farms", [])
        for p_idx, farm in enumerate(farms):
            money = farm.get("money")
            farmer = farm.get("farmer")
            hands = farm.get("hands")
            quads = farm.get("unlocked_quadrants")
            hires = farm.get("hires_today")
            
            # Count tile types
            tiles = farm.get("tiles", [])
            tile_counts = {"None": 0, "LOCKED": 0, "PLANT": 0, "WEED": 0, "COOP": 0, "PASTURE": 0}
            sample_dicts = []
            
            for row in tiles:
                for cell in row:
                    if cell is None:
                        tile_counts["None"] += 1
                    elif cell == "LOCKED":
                        tile_counts["LOCKED"] += 1
                    elif isinstance(cell, dict):
                        k = cell.get("kind")
                        tile_counts[k] = tile_counts.get(k, 0) + 1
                        sample_dicts.append(cell)
            
            print(f"  Player {p_idx} Farm:")
            print(f"    Money: ${money}")
            print(f"    Farmer pos: {farmer}, Hands: {hands}, Hires today: {hires}")
            print(f"    Unlocked Quadrants: {quads}")
            print(f"    Tile summary: {tile_counts}")
            if sample_dicts:
                print("    Populated tile objects:")
                for sd in sample_dicts:
                    print(f"      {json.dumps(sd)}")

        # Market
        mkt = obs.get("market", {})
        print(f"  Market Prices: {mkt.get('prices')}")
        print(f"  Market Inventory (non-default items): { {k: v for k, v in mkt.get('inventory', {}).items() if v != 10000} }")
        
        # Town
        town = obs.get("town", {})
        print(f"  Town Unlocked Shops: {town.get('unlocked_shops')}")
        
        # Private
        priv = obs.get("private", {})
        if priv:
            print(f"  Private state - Shed: {priv.get('shed')}, Seeds: {priv.get('seeds')}, Inventories: {priv.get('inventories')}")

if __name__ == "__main__":
    analyze_obs()
