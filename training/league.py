"""
Phase 3: Self-Play League and Opponent Pool Management for Kaggriculture.

Implements:
1. SnapshotPool: Manages historical agent checkpoints (e.g. pi_{t-50}, pi_{t-100}).
2. LeagueMatchmaker: Samples opponents according to competitive distribution:
   - 70% Current self-play policy
   - 20% Past historical checkpoint snapshot
   - 10% Deterministic baseline / Starter agent
3. SelfPlayArena: Runs multi-turn head-to-head matches between two ActorCriticNet policies
   or against baseline agents with randomized seeds.
"""
import os
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from kaggle_environments import make

from training.rewards import NetWorthRewardTracker, calculate_net_worth


class SnapshotPool:
    """
    Manages a ring buffer of historical model checkpoint snapshots.
    """

    def __init__(self, max_snapshots: int = 20):
        self.max_snapshots = max_snapshots
        self.snapshots: List[str] = []

    def add_snapshot(self, checkpoint_path: str):
        """Adds a saved model checkpoint path to the pool."""
        if checkpoint_path not in self.snapshots:
            self.snapshots.append(checkpoint_path)
            if len(self.snapshots) > self.max_snapshots:
                # Evict oldest snapshot
                oldest = self.snapshots.pop(0)
                if os.path.exists(oldest):
                    try:
                        os.remove(oldest)
                    except OSError:
                        pass

    def sample_snapshot(self) -> Optional[str]:
        """Uniformly samples a historical snapshot from the pool."""
        if not self.snapshots:
            return None
        return random.choice(self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)


class SelfPlayArena:
    """
    Executes head-to-head games between two agent policies with Delta NW rewards.
    """

    def __init__(
        self,
        living_penalty: float = 0.05,
        episode_steps: int = 720,
    ):
        self.living_penalty = living_penalty
        self.episode_steps = episode_steps
        self.p0_tracker = NetWorthRewardTracker(living_penalty=living_penalty)
        self.p1_tracker = NetWorthRewardTracker(living_penalty=living_penalty)

    def run_match(
        self,
        p0_agent: Callable[[Dict[str, Any]], Dict[str, Any]],
        p1_agent: Callable[[Dict[str, Any]], Dict[str, Any]],
        seed: Optional[int] = None,
        p0_step_callback: Optional[Callable[[Dict[str, Any], float, bool], None]] = None,
        p1_step_callback: Optional[Callable[[Dict[str, Any], float, bool], None]] = None,
    ) -> Tuple[float, float, int]:
        """
        Runs a complete match between Player 0 and Player 1.
        
        Returns:
            p0_reward: Final liquid coins of Player 0.
            p1_reward: Final liquid coins of Player 1.
            steps_played: Number of turns completed.
        """
        config = {"episodeSteps": self.episode_steps}
        if seed is not None:
            config["seed"] = seed

        env = make("kaggriculture", configuration=config, debug=False)
        env.reset()

        # Resolve string agent names (e.g. 'starter', 'random', 'pass')
        p0_fn = env.agents.get(p0_agent) if isinstance(p0_agent, str) else p0_agent
        p1_fn = env.agents.get(p1_agent) if isinstance(p1_agent, str) else p1_agent

        init_p0_obs = env.steps[0][0]["observation"]
        init_p1_obs = env.steps[0][1]["observation"]
        self.p0_tracker.reset(init_p0_obs)
        self.p1_tracker.reset(init_p1_obs)

        for step in range(self.episode_steps):
            p0_obs = env.steps[-1][0]["observation"]
            p1_obs = env.steps[-1][1]["observation"]

            # Query policies
            p0_act = p0_fn(p0_obs)
            p1_act = p1_fn(p1_obs)

            env.step([p0_act, p1_act])

            is_done = (step == self.episode_steps - 1) or env.done
            next_p0_obs = env.steps[-1][0]["observation"]
            next_p1_obs = env.steps[-1][1]["observation"]

            r0 = self.p0_tracker.compute_step_reward(next_p0_obs, is_done=is_done)
            r1 = self.p1_tracker.compute_step_reward(next_p1_obs, is_done=is_done)

            if p0_step_callback:
                p0_step_callback(next_p0_obs, r0, is_done)
            if p1_step_callback:
                p1_step_callback(next_p1_obs, r1, is_done)

            if is_done:
                break

        final_step = env.steps[-1]
        p0_coins = float(final_step[0].get("reward", 0.0) or 0.0)
        p1_coins = float(final_step[1].get("reward", 0.0) or 0.0)

        return p0_coins, p1_coins, len(env.steps)
