"""
Phase 2: Ghost-Play Environment and Historical Replay Opponent Engine.

Enables online Reinforcement Learning against historical Grandmaster matches:
1. Reconstructs exact historical episode seeds.
2. Scripts GhostReplayOpponent to faithfully execute the exact historical move sequence.
3. Pairs learning agent against Ghost with real-time Delta Net Worth (Delta NW) step rewards.
"""
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from kaggle_environments import make

from training.rewards import NetWorthRewardTracker, calculate_net_worth


class GhostReplayOpponent:
    """
    A deterministic agent that replays the exact sequence of historical actions
    from a downloaded replay file.
    """

    def __init__(self, replay_path: str, ghost_player_idx: int = 1):
        self.replay_path = replay_path
        self.ghost_player_idx = ghost_player_idx
        self.action_history: List[Dict[str, Any]] = []
        self.seed: Optional[int] = None
        self._load_replay()

    def _load_replay(self):
        try:
            with open(self.replay_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Extract configuration seed if available
                config = data.get("configuration", {})
                self.seed = config.get("seed", None)

                steps = data.get("steps", [])
                for step_tuple in steps:
                    if len(step_tuple) > self.ghost_player_idx:
                        act = step_tuple[self.ghost_player_idx].get("action", {})
                        if isinstance(act, dict):
                            self.action_history.append(act)
                        else:
                            self.action_history.append({"farmer": ["PASS"], "hands": [], "market": []})
        except Exception:
            self.action_history = []

    def get_action(self, step: int) -> Dict[str, Any]:
        """Returns the recorded action for step t."""
        if 0 <= step < len(self.action_history):
            return self.action_history[step]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        step = obs.get("step", 0)
        return self.get_action(step)


class GhostPlayEnvironment:
    """
    Simulates a live match pairing a learning RL agent (Player 0)
    against a historical Ghost replay player (Player 1) with Delta NW rewards.
    """

    def __init__(
        self,
        ghost_opponent: GhostReplayOpponent,
        living_penalty: float = 0.05,
        episode_steps: int = 720,
    ):
        self.ghost = ghost_opponent
        self.episode_steps = episode_steps
        self.reward_tracker = NetWorthRewardTracker(living_penalty=living_penalty)
        self.env = None
        self.current_step = 0

    def run_match(
        self,
        agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        on_step_callback: Optional[Callable[[Dict[str, Any], float, bool], None]] = None,
    ) -> Tuple[float, float, int]:
        """
        Runs a complete match between the agent and the ghost opponent.
        
        Returns:
            p0_coins: Final liquid coins of our agent.
            p1_coins: Final liquid coins of the ghost opponent.
            total_steps: Number of steps played.
        """
        config = {"episodeSteps": self.episode_steps}
        if self.ghost.seed is not None:
            config["seed"] = self.ghost.seed

        self.env = make("kaggriculture", configuration=config, debug=False)
        self.env.reset()

        initial_obs = self.env.steps[0][0]["observation"]
        self.reward_tracker.reset(initial_obs)

        # Run episode turn by turn
        for step in range(self.episode_steps):
            p0_obs = self.env.steps[-1][0]["observation"]
            p1_obs = self.env.steps[-1][1]["observation"]

            # Query agent and ghost actions
            p0_action = agent_fn(p0_obs)
            p1_action = self.ghost(p1_obs)

            self.env.step([p0_action, p1_action])

            is_done = (step == self.episode_steps - 1) or self.env.done
            next_obs = self.env.steps[-1][0]["observation"]
            step_reward = self.reward_tracker.compute_step_reward(next_obs, is_done=is_done)

            if on_step_callback:
                on_step_callback(next_obs, step_reward, is_done)

            if is_done:
                break

        final_step = self.env.steps[-1]
        p0_coins = final_step[0].get("reward", 0.0) or 0.0
        p1_coins = final_step[1].get("reward", 0.0) or 0.0

        return float(p0_coins), float(p1_coins), len(self.env.steps)
