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

from typing import Optional, Tuple

import base_env
import dm_env
import gym
import numpy as np
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
    def __init__(self, rows: int = 10, columns: int = 5, seed: Optional[int] = None):
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
        self._board = np.zeros((rows, columns), dtype=np.float32)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = None
        self._total_regret = 0.0
        self._regret = 0.0
        self.bsuite_num_episodes = sweep.NUM_EPISODES

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
            self._total_regret += 1.0 - reward
            self._regret = 1.0 - reward
            return dm_env.termination(reward=reward, observation=self._observation())

        return dm_env.transition(reward=0.0, observation=self._observation())

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=self._board.shape,
            dtype=self._board.dtype,
            name="board",
            minimum=0,
            maximum=1,
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=np.int, num_values=len(_ACTIONS), name="action"
        )

    def _observation(self) -> np.ndarray:
        self._board.fill(0.0)
        self._board[self._ball_y, self._ball_x] = BALL_CODE
        self._board[self._paddle_y, self._paddle_x] += PADDLE_CODE

        return self._board.copy()

    def bsuite_info(self):
        return dict(total_regret=self._total_regret, regret=self._regret)


REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


class Wrapper(gym.Wrapper, base_env.Env[np.ndarray, int]):
    def __init__(self, env: Catch):
        super().__init__(env)
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
    def time_out_str(cls) -> str:
        return f"Out of time ({REWARDS[0.0]})."

    @classmethod
    def done(cls, state_or_reward: str) -> bool:
        return any(r in state_or_reward for r in REWARDS.values())

    @staticmethod
    def get_coordinates(obs):
        bottom = obs[-1]
        if np.all(obs[:-1] == 0.0):
            height = 0
            if np.any(bottom == (BALL_CODE + PADDLE_CODE)):
                paddle_pos = ball_pos = np.argmax(bottom)
            else:
                paddle_pos = (bottom == PADDLE_CODE).argmax()
                ball_pos = (bottom == BALL_CODE).argmax()
        else:
            paddle_pos = (bottom == PADDLE_CODE).argmax()
            ball_idx = (obs[:-1] == 1.0).argmax()
            height, ball_pos = np.unravel_index(ball_idx, obs[:-1].shape)
            height = len(obs) - height - 1
        return height, ball_pos, paddle_pos

    def longest_reward(self) -> str:
        return "1,2 (4 to go) (Success)."  # TODO

    @classmethod
    def quantify(cls, prompt: str, gamma: Optional[float]) -> float:
        success = prompt.endswith(f"({REWARDS[1.0]}).")
        prompt = gamma ** prompt.count(".")
        return prompt if success else (gamma - 1) * prompt

    def reward_str(self, reward: float, done: bool, next_state: np.ndarray) -> "str":
        next_state_str = self.state_str(next_state)[:-1]  # remove '.'
        return next_state_str + f" ({REWARDS[reward]})." if done else "."

    def state_str(self, obs: np.ndarray) -> str:
        assert isinstance(obs, np.ndarray)
        height, ball_pos, paddle_pos = self.get_coordinates(obs)
        return f"{paddle_pos},{ball_pos}" + (
            f" ({height} to go)." if height > 0 else "."
        )

    def reset(self):
        assert isinstance(self.env, Catch)
        return self.env.reset().observation

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
        _, ball_pos, _ = self.get_coordinates(obs)
        _, d = obs.shape
        one_hot = np.zeros(d)
        one_hot[ball_pos] = 1
        return one_hot
