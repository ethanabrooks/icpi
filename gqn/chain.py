from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep

REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


@dataclass
class Env(base_env.Env[int, int]):

    goal: int
    n: int
    random_seed: int
    status: bool

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )
        self.observation_space = gym.spaces.Discrete(self.n)

    @staticmethod
    def action_stop() -> str:
        return ":"

    def actions(self):
        return [
            "Left",
            "Try goal",
            "Right",
        ]

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        return state_or_reward.rstrip(self.state_stop()) in REWARDS.values()

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.9

    def partially_observable(self) -> bool:
        return False

    @classmethod
    def quantify(cls, prompt: str) -> float:
        success = prompt.endswith(REWARDS[1.0] + cls.state_stop())
        length = prompt.count(cls.action_stop()) - 1
        value = cls.gamma() ** length
        if success:
            return value
        elif prompt.endswith(REWARDS[0.0] + cls.state_stop()):
            return 0
        return 0

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> int:
        self._state = self._start_state = self.random.choice(self.n)
        return self._start_state

    def start_states(self) -> Optional[Iterable[int]]:
        return range(self.n)

    def _state_str(self, state: int) -> str:
        if not self.status:
            return str(state)
        status = f"at {self.goal}" if state == self.goal else f"not at {self.goal}"
        return f"{state} [{status}]"

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        optimal = self.gamma() ** abs(self._start_state - self.goal)
        info = dict(optimal=optimal)
        self._state += action - 1
        self._state = np.clip(self._state, 0, self.n - 1)
        done = action == 1
        success = done and self._state == self.goal
        state = int(self._state)
        return state, float(success), done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += " " + REWARDS[ts.reward] + self.state_stop()
        return description
