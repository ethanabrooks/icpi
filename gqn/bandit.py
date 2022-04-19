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
from typing import Optional, Tuple

import base_env
import dm_env
import gym
import numpy as np
from bsuite.environments import base
from bsuite.experiments.bandit import sweep
from dm_env import specs
from gym.spaces import Discrete


class Bandit(base.Environment):
    """SimpleBandit environment."""

    def __init__(self, mapping_seed: Optional[int] = None, num_actions: int = 11):
        """Builds a simple bandit environment.
        Args:
          mapping_seed: Optional integer. Seed for action mapping.
          num_actions: number of actions available, defaults to 11.
        """
        super(Bandit, self).__init__()
        self.random_seed = mapping_seed
        self._rng = np.random.RandomState(mapping_seed)
        self._num_actions = num_actions
        action_mask = self._rng.choice(
            range(self._num_actions), size=self._num_actions, replace=False
        )
        self.rewards = np.linspace(0, 1, self._num_actions)[action_mask]

        self._total_regret = 0.0
        self._optimal_return = 1.0
        self.bsuite_num_episodes = sweep.NUM_EPISODES

    def _get_observation(self):
        return np.ones(shape=(1, 1), dtype=np.float32)

    def _reset(self) -> dm_env.TimeStep:
        observation = self._get_observation()
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        reward = self.rewards[action]
        self._total_regret += self._optimal_return - reward
        observation = self._get_observation()
        return dm_env.termination(reward=reward, observation=observation)

    def observation_spec(self):
        return specs.Array(shape=(1, 1), dtype=np.float32, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(self._num_actions, name="action")

    def bsuite_info(self):
        return dict(total_regret=self._total_regret)


class Wrapper(gym.Wrapper, base_env.Env[np.ndarray, int]):
    def __init__(self, env: Bandit):
        super().__init__(env)
        self.action_space = Discrete(3, seed=env.random_seed)
        self.observation_space = Discrete(1)

    def actions(self) -> "list[str]":
        assert isinstance(self.env, Bandit)
        return [f"Action: {i}." for i in range(self.env.action_spec().num_values)]

    def done(self, state_or_reward: str) -> bool:
        return True

    def longest_reward(self) -> str:
        assert isinstance(self.env, Bandit)
        return max(map(str, self.env.rewards), key=len)

    @classmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        success = value.endswith(cls.rewards()[1.0])
        value = gamma ** value.count(".")
        return value if success else (gamma - 1) * value

    @staticmethod
    def _reward_str(reward: float) -> "str":
        return f"Reward: {reward:.2f}."

    def state_str(self, obs: np.ndarray) -> str:
        return ""

    def reset(self):
        assert isinstance(self.env, Bandit)
        return self.env.reset().observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert isinstance(self.env, Bandit)
        time_step: dm_env.TimeStep = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last(),
            self.bsuite_info(),
        )

    def successor_feature(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()

    def time_out_str(self) -> str:
        assert isinstance(self.env, Bandit)
        return self._reward_str(min(self.env.rewards))
