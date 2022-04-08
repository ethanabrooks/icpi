from dataclasses import dataclass
from typing import Generator, Iterator, Optional, Tuple, Union
import gym
import gym.spaces
import numpy as np


@dataclass
class Env(gym.Env[int, int]):
    n: int
    goal: int

    def __post_init__(self):
        self.random = np.random.default_rng()
        self.action_space = gym.spaces.Discrete(3)

    @staticmethod
    def action_str(action: int) -> str:
        if action == 0:
            return "left"
        if action == 1:
            return "Try to get reward."
        if action == 2:
            return "right"
        raise RuntimeError()

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        state = self.random.choice(self.n)
        info = {}
        action = yield state, 0, False, info
        while True:
            action -= 1
            done = action == 0
            success = done and state == self.goal
            reward = float(success)
            state = int(state)
            action = yield state, reward, done, info
            state += action
            state = np.clip(state, 0, self.n - 1)

    @staticmethod
    def quantify(value: str) -> float:
        if value.endswith("Receive a reward."):
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

    @staticmethod
    def reward_str(reward: float) -> str:
        if reward:
            return "Got reward."
        else:
            return "No reward."

    @staticmethod
    def state_str(state: int) -> str:
        return f"You are at state {state}."

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)
