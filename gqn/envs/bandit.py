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
from typing import Optional, Tuple, cast

import dm_env
import envs.base_env
import gym
import numpy as np
from bsuite.environments import base
from bsuite.experiments.bandit import sweep
from dm_env import specs
from envs.base_env import TimeStep
from gym.spaces import Discrete


class Env(base.Environment):
    """SimpleBandit environment."""

    def __init__(self, mapping_seed: Optional[int] = None, num_actions: int = 11):
        """Builds a simple bandit environment.
        Args:
          mapping_seed: Optional integer. Seed for action mapping.
          num_actions: number of actions available, defaults to 11.
        """
        super(Env, self).__init__()
        self.random_seed = mapping_seed
        self._rng = np.random.RandomState(mapping_seed)
        self._num_actions = num_actions
        action_mask = self._rng.choice(
            range(self._num_actions), size=self._num_actions, replace=False
        )
        self.means = np.linspace(0, 1, self._num_actions)[action_mask]
        breakpoint()

        self._total_regret = 0.0
        self._optimal_return = 1.0
        self.bsuite_num_episodes = sweep.NUM_EPISODES

    @staticmethod
    def _get_observation():
        return np.ones(shape=(1, 1), dtype=np.float32)

    def _reset(self) -> dm_env.TimeStep:
        observation = self._get_observation()
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        reward = self._rng.normal(self.means[action], 1)
        self._total_regret += self._optimal_return - reward
        observation = self._get_observation()
        return dm_env.termination(reward=reward, observation=observation)

    def observation_spec(self):
        return specs.Array(shape=(1, 1), dtype=np.float32, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(self._num_actions, name="action")

    def bsuite_info(self):
        return dict(total_regret=self._total_regret)


class Wrapper(gym.Wrapper, envs.base_env.Env[np.ndarray, int]):
    def __init__(self, env: Env):
        super().__init__(cast(gym.Env, env))
        self.action_space = Discrete(3, seed=env.random_seed)

    def actions(self) -> "list[str]":
        assert isinstance(self.env, Env)
        return [str(i) for i in range(self.env.action_spec().num_values)]

    def done(self, state_or_reward: str) -> bool:
        return True

    @classmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        breakpoint()
        raise NotImplementedError

    def reset(self):
        assert isinstance(self.env, Env)
        return self.env.reset().observation

    @classmethod
    def _state_str(cls, obs: np.ndarray) -> str:
        return ""

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert isinstance(self.env, Env)
        time_step: dm_env.TimeStep = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last(),
            self.bsuite_info(),
        )

    def successor_feature(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()

    def ts_to_string(self, ts: TimeStep) -> str:
        return f"{self.actions()[ts.action]}: {str(round(ts.reward, ndigits=2))}{self.state_stop()}"


if __name__ == "__main__":
    env = Wrapper(Env(mapping_seed=0, num_actions=3))
    while True:
        env.reset()
        a = env.action_space.sample()
        s, r, t, i = env.step(a)
        print(env.ts_to_string(TimeStep(s, a, r, t, s)))
        breakpoint()
