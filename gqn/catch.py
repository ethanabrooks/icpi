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
"""Catch reinforcement learning environment."""

from typing import Optional, Tuple, cast

import base_env
import dm_env
import gym
import numpy as np
from base_env import TimeStep
from bsuite.environments import base
from bsuite.experiments.catch import sweep
from dm_env import specs
from gym.spaces import Discrete, MultiDiscrete

_ACTIONS = (-1, 0, 1)  # Left, no-op, right.

BALL_CODE = 1.0
PADDLE_CODE = 2.0


class Catch(base.Environment):
    """A Catch environment built on the dm_env.Environment class.

    The agent must move a paddle to intercept falling balls. Falling balls only
    move downwards on the column they are in.

    The observation is an array shape (rows, columns), with binary values:
    zero if a space is empty; 1 if it contains the paddle or a ball.

    The actions are discrete, and by default there are three available:
    stay, move left, and move right.

    The episode terminates when the ball reaches the bottom of the screen.
    """

    # noinspection PyMissingConstructor
    def __init__(
        self, gamma: float, rows: int = 10, columns: int = 5, seed: Optional[int] = None
    ):
        """Initializes a new Catch environment.

        Args:
          rows: number of rows.
          columns: number of columns.
          seed: random seed for the RNG.
        """
        self.random_seed = seed
        self._rows = rows
        self._columns = columns
        self._rng = np.random.RandomState(seed)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = None
        self.bsuite_num_episodes = sweep.NUM_EPISODES
        self._optimal = gamma ** rows

    def render(self, mode="human"):
        pass

    def _reset(self) -> dm_env.TimeStep:
        """Returns the first `TimeStep` of a new episode."""
        self._ball_x = self._rng.randint(self._columns)
        self._ball_y = 0
        self._paddle_x = self._columns // 2
        self._paddle_y = self._rows - 1

        return dm_env.restart(self._observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

        # Drop the ball.
        self._ball_y += 1

        # Check for termination.
        if self._ball_y == self._paddle_y:
            reward = 1.0 if self._paddle_x == self._ball_x else 0
            return dm_env.termination(reward=reward, observation=self._observation())

        return dm_env.transition(reward=0.0, observation=self._observation())

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            dtype=np.int,
            minimum=0,
            maximum=max(self._rows, self._columns),
            name="observation",
            shape=(3,),
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=np.int, num_values=len(_ACTIONS), name="action"
        )

    def _observation(self) -> np.ndarray:
        return np.array([self._paddle_x, self._ball_x, self._paddle_y - self._ball_y])

    def bsuite_info(self):
        return dict(optimal=self._optimal)


REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


class Wrapper(gym.Wrapper, base_env.Env[np.ndarray, int]):
    def __init__(self, env: Catch):
        super().__init__(cast(gym.Env, env))
        self.action_space = Discrete(3, seed=env.random_seed)
        self.observation_space = MultiDiscrete(
            np.ones_like(env.observation_spec().shape), seed=env.random_seed
        )

    def actions(self) -> "list[str]":
        return [
            "Left.",
            "Stay.",
            "Right.",
        ]

    @classmethod
    def done(cls, state_or_reward: str) -> bool:
        return any(r in state_or_reward for r in REWARDS.values())

    @classmethod
    def quantify(cls, prompt: str, gamma: Optional[float]) -> float:
        success = prompt.endswith(f"({REWARDS[1.0]}).")
        prompt = gamma ** prompt.count(".")
        return prompt if success else (gamma - 1) * prompt

    def reset(self):
        assert isinstance(self.env, Catch)
        return self.env.reset().observation

    def state_str(self, obs: np.ndarray) -> str:
        assert isinstance(obs, np.ndarray)
        paddle_x, ball_x, ball_y = obs
        assert isinstance(self.env, Catch)
        height = ball_y
        part1 = f"{paddle_x},{ball_x}"
        if height == 0:
            return part1
        return f"{part1} ({height} to go)."

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert isinstance(self.env, Catch)
        time_step: dm_env.TimeStep = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last(),
            self.bsuite_info(),
        )

    def successor_feature(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()

    @classmethod
    def time_out_str(cls) -> str:
        return f"Out of time ({REWARDS[0.0]})."

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += f" {self.state_str(ts.next_state)} ({REWARDS[ts.reward]})."
        return description
