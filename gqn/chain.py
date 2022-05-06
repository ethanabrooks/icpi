from dataclasses import dataclass
from typing import Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep

REWARDS = {
    1.0: "Success",
    -1.0: "Failure",
}


@dataclass
class Env(base_env.Env[int, int]):
    gamma: float
    goal: int
    n: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )
        self.observation_space = gym.spaces.Discrete(self.n)

    @staticmethod
    def action_stop() -> str:
        return "."

    def actions(self):
        return [
            "Left",
            "Try goal",
            "Right",
        ]

    def done(self, state_or_reward: str) -> bool:
        return state_or_reward.rstrip(self.state_stop()) in REWARDS.values()

    @classmethod
    def quantify(cls, prompt: str, gamma: Optional[float]) -> float:
        success = prompt.endswith(REWARDS[1.0] + cls.state_stop())
        length = prompt.count(cls.state_stop()) // 2 - 1
        value = gamma ** length
        if success:
            return value
        elif prompt.endswith(REWARDS[-1.0] + cls.state_stop()):
            return -value
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

    @classmethod
    def _state_str(cls, state: int) -> str:
        return str(state)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        optimal = self.gamma ** abs(self._start_state - self.goal)
        info = dict(optimal=optimal)
        self._state += action - 1
        self._state = np.clip(self._state, 0, self.n - 1)
        done = action == 1
        success = done and self._state == self.goal
        state = int(self._state)
        if done:
            reward = 1 if success else -1
        else:
            reward = 0
        return state, reward, done, info

    def successor_feature(self, state: int) -> np.ndarray:
        one_hot = np.zeros(self.n)
        one_hot[state] = 1
        return one_hot

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += " " + REWARDS[ts.reward] + self.state_stop()
        return description
