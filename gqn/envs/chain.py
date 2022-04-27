from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import envs.base_env
import gym
import gym.spaces
import numpy as np
from envs.base_env import TimeStep

REWARDS = {
    1.0: "Success",
    -1.0: "Failure",
}


@dataclass
class Env(envs.base_env.Env[int, int]):
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

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        start_state = state = self.random.choice(self.n)
        reward = 0
        done = False
        optimal = self.gamma ** abs(start_state - self.goal)
        while True:
            info = dict(optimal=optimal)
            action = yield state, reward, done, info
            state += action - 1
            state = np.clip(state, 0, self.n - 1)
            done = action == 1
            success = done and state == self.goal
            if done:
                reward = 1 if success else -1
            state = int(state)

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
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    @classmethod
    def _state_str(cls, state: int) -> str:
        return str(state)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        return self.iterator.send(action)

    def successor_feature(self, state: int) -> np.ndarray:
        one_hot = np.zeros(self.n)
        one_hot[state] = 1
        return one_hot

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += " " + REWARDS[ts.reward] + self.state_stop()
        return description
