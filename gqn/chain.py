from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np


@dataclass
class Chain(base_env.Env[int, int]):
    random_seed: int
    n: int = 8
    goal: int = 4

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(3, seed=self.random_seed)
        self.observation_space = gym.spaces.Discrete(self.n)

    @staticmethod
    def actions():
        return [
            "Left.",
            "Try goal.",
            "Right.",
        ]

    @classmethod
    def time_out_str(cls) -> str:
        return cls.rewards()[0.0]

    @classmethod
    def done(cls, state_or_reward: str) -> bool:
        return state_or_reward in cls.rewards().values()

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
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        success = value.endswith(cls.rewards()[1.0])
        value = gamma ** value.count(".")
        return value if success else (gamma - 1) * value

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

    @staticmethod
    def rewards() -> "dict[float, str]":
        return {
            1.0: "Success.",
            0.0: "Failure.",
        }

    @staticmethod
    def state_str(state: int) -> str:
        return f"{state}."

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)

    def successor_feature(self, state: int) -> np.ndarray:
        one_hot = np.zeros(self.n)
        one_hot[state] = 1
        return one_hot
