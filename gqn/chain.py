from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np

REWARDS = {
    1.0: "Success.",
    0.0: "Failure.",
}


@dataclass
class Chain(base_env.Env[int, int]):
    gamma: float
    goal: int
    n: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(3, seed=self.random_seed)
        self.observation_space = gym.spaces.Discrete(self.n)

    def actions(self):
        return [
            "Left.",
            "Try goal.",
            "Right.",
        ]

    @staticmethod
    def default_reward_str() -> str:
        return REWARDS[0.0]

    def done(self, state_or_reward: str) -> bool:
        return state_or_reward in REWARDS.values()

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        start_state = state = self.random.choice(self.n)
        reward = 0
        done = False
        info = {}
        while True:
            if done:
                optimal = self.gamma ** abs(start_state - self.goal)
                info.update(regret=optimal - (reward * optimal))
            action = yield state, reward, done, info
            state += action - 1
            state = np.clip(state, 0, self.n - 1)
            done = action == 1
            success = done and state == self.goal
            reward = float(success)
            state = int(state)

    @classmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        success = value.endswith(REWARDS[1.0])
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
    def state_str(state: int) -> str:
        return f"{state}."

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)

    def ts_to_string(self, ts: base_env.TimeStep) -> str:
        if ts.done:
            reward_str = " " + REWARDS[ts.reward]
        else:
            reward_str = ""
        return f"{self.state_str(ts.state)} {self.action_str(ts.action)}{reward_str}"
