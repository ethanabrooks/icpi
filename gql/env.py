from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import gym
import gym.spaces
import numpy as np

ACTIONS = [
    "Go left.",
    "Try reward",
    "Go right.",
]


@dataclass
class Env(gym.Env[int, int]):
    goal: int
    n: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(3, seed=self.random_seed)

    @staticmethod
    def success_str():
        return "and succeed!"

    @staticmethod
    def action_str(action: int) -> str:
        return ACTIONS[action]

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        state = self.random.choice(self.n)
        reward = 0
        done = False
        info = {}
        while True:
            action = yield state, reward, done, info
            state += action - 1
            state = np.clip(state, 0, self.n - 1)
            done = action == 1
            success = done and state == self.goal
            reward = float(success)
            state = int(state)

    @classmethod
    def quantify(cls, value: str) -> float:
        if value.endswith(cls.success_str()):
            return 0.9 ** value.count(".")
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
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    @classmethod
    def reward_str(cls, reward: float) -> str:
        if reward:
            return cls.success_str()
        else:
            return "and fail."

    @staticmethod
    def state_str(state: int) -> str:
        return f"You are at state {state}."

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)