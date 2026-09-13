"""
Official Kaggle Competition Submission Entrypoint for Kaggriculture.

This file is the root submission file expected by Kaggle Environments and the Kaggle CLI:
`kaggle competitions submit kaggriculture -f main.py -m "Dual-Tower Siamese NNX Agent"`
"""
import sys
import os

# Ensure local directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.agent import agent, init_agent

# Pre-initialize weights on module import for zero-delay turn 0
try:
    init_agent()
except Exception:
    pass


if __name__ == "__main__":
    from kaggle_environments import make

    print("Running 48-turn (2 in-game days) local validation match against starter agent...")
    env = make("kaggriculture", configuration={"episodeSteps": 48}, debug=True)
    env.run([agent, "starter"])

    final_step = env.steps[-1]
    p0_reward = final_step[0].get("reward", 0)
    p1_reward = final_step[1].get("reward", 0)
    print(f"Match Finished! Steps: {len(env.steps)}")
    print(f"Player 0 (Our NNX Agent): Reward = {p0_reward}")
    print(f"Player 1 (Starter Baseline): Reward = {p1_reward}")
