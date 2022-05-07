# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple diagnostic bandit environment.
Observation is a single pixel of 0 - this is an independent arm bandit problem!
Rewards are [0, 0.1, .. 1] assigned randomly to 11 arms and deterministic
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from base_env import Env as BaseEnv
from base_env import TimeStep
from gym.spaces import Discrete


@dataclass
class Env(BaseEnv[np.ndarray, int]):
    num_steps: int
    random_seed: int

    def __post_init__(self):
        self.distributions = None
        self.rng = np.random.default_rng(seed=self.random_seed)
        self.action_space = Discrete(2, seed=self.random_seed)

    def actions(self) -> "list[str]":
        assert isinstance(self.action_space, Discrete)
        return [str(i) for i in range(self.action_space.n)]

    def done(self, *completions: str) -> bool:
        trajectory_length = " ".join(completions).count(self.action_stop())
        return trajectory_length >= self.num_steps

    def partially_observable(self) -> bool:
        return True

    def quantify(self, prompt: str, gamma: Optional[float]) -> float:
        rewards = re.findall(r"\d: ([.\d]+);", prompt)
        rewards = rewards[: self.num_steps]
        return sum(map(float, rewards))

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        self.distributions = np.array([[0, 0.5], [0.5, 1]])
        self.rng.shuffle(self.distributions)
        self.t = 0
        return self.distributions

    @staticmethod
    def state_stop() -> str:
        return ";"

    @classmethod
    def _state_str(cls, obs: np.ndarray) -> str:
        return ""

    def state_str(self, state: np.ndarray) -> str:
        return self._state_str(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        dist = self.distributions[action]
        reward = self.rng.choice(dist)
        optimal = self.distributions.max() * self.num_steps
        self.t += 1
        done = self.t == self.num_steps
        return self.distributions, reward, done, dict(optimal=optimal)

    def successor_feature(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()

    def ts_to_string(self, ts: TimeStep) -> str:
        return f"{self.actions()[ts.action]}: {str(round(ts.reward, ndigits=2))}{self.state_stop()}"


if __name__ == "__main__":
    env = Env(num_steps=5, random_seed=0)
    while True:
        s = env.reset()
        t = False
        while not t:
            a = env.action_space.sample()
            s, r, t, i = env.step(a)
            print(s)
            print(env.ts_to_string(TimeStep(s, a, r, t, s)))
            breakpoint()
