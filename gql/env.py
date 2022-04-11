import itertools
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import gym
import gym.spaces
import numpy as np

ACTIONS = [
    "Left.",
    "Try goal.",
    "Right.",
]

REWARDS = {
    1.0: "Success.",
    0.0: "Failure.",
}

lengths = [len(a) + len(r) for a, r in itertools.product(ACTIONS, REWARDS.values())]
MAX_TOKENS = max(lengths)


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
        return REWARDS[1.0]

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
    def quantify(cls, value: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = 1
        success = value.endswith(cls.success_str())
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

    @classmethod
    def reward_str(cls, reward: float) -> str:
        return REWARDS[reward]

    @staticmethod
    def state_str(state: int) -> str:
        return f"{state}."

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)
