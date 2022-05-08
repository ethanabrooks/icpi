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

from typing import NamedTuple, Optional, Tuple, cast

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


class Obs(NamedTuple):
    paddle_x: int
    ball_x: int
    ball_y: int


class Env(base.Environment):
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
        self._ball_y = self._rows - 1
        self._paddle_x = self._columns // 2
        self._paddle_y = 0

        return dm_env.restart(self._observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

        # Drop the ball.
        self._ball_y -= 1

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

    def _observation(self) -> Obs:
        return Obs(paddle_x=self._paddle_x, ball_x=self._ball_x, ball_y=self._ball_y)

    def bsuite_info(self):
        return dict(optimal=self._optimal)


REWARDS = {
    1.0: "P.x==B.x, B.y==0, success",
    0.0: "P.x!=B.x, B.y==0, failure",
}


class Wrapper(gym.Wrapper, base_env.Env[Obs, int]):
    def __init__(self, env: Env):
        super().__init__(cast(gym.Env, env))
        self.action_space = Discrete(3, seed=env.random_seed)
        self.observation_space = MultiDiscrete(
            np.ones_like(env.observation_spec().shape), seed=env.random_seed
        )

    def actions(self) -> "list[str]":
        return [
            "Left",
            "Stay",
            "Right",
        ]

    @classmethod
    def done(cls, *completions: str) -> bool:
        *_, state_or_reward = completions
        return any(r in state_or_reward for r in REWARDS.values())

    def failure_threshold(self) -> float:
        return 0

    def partially_observable(self) -> bool:
        return False

    @classmethod
    def quantify(cls, prompt: str, gamma: Optional[float]) -> float:
        success = prompt.endswith(f"[{REWARDS[1.0]}];")
        value = gamma ** (prompt.count(":") - 1)
        return value if success else 0

    def reset(self):
        assert isinstance(self.env, Env)
        return self.env.reset().observation

    @staticmethod
    def state_stop() -> str:
        return ";"

    @classmethod
    def _state_str(cls, obs: Obs) -> str:
        assert isinstance(obs, Obs)
        paddle_x, ball_x, ball_y = Obs(*obs)
        return f"P=({paddle_x},0) B=({ball_x},{ball_y}) [{cls._status(obs)}]"

    @classmethod
    def _status(cls, obs: Obs) -> str:
        paddle_x, ball_x, ball_y = Obs(*obs)
        x_status = "P.x==B.x" if paddle_x == ball_x else "P.x!=B.x"
        y_status = "B.y==0" if ball_y == 0 else "B.y>0"
        if ball_y == 0:
            reward = (
                "success"
                if x_status == "P.x==B.x" and y_status == "B.y==0"
                else "failure"
            )
        else:
            reward = "in progress"
        return f"{x_status}, {y_status}, {reward}"

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        assert isinstance(self.env, Env)
        time_step: dm_env.TimeStep = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last(),
            self.bsuite_info(),
        )

    def successor_feature(self, obs: Obs) -> np.ndarray:
        return np.array(obs)

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += f" {self.state_str(ts.next_state)}"
        return description
