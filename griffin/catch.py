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
import re
from typing import List, NamedTuple, Optional, Tuple, cast

import base_env
import dm_env
import gym
import numpy as np
from base_env import TimeStep
from bsuite.environments import base
from bsuite.experiments.catch import sweep
from dm_env import specs
from gym.spaces import Discrete, MultiDiscrete
from rl.lm import Data

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
    def __init__(self, rows: int = 10, columns: int = 5, seed: Optional[int] = None):
        """Initializes a new Catch environment.

        Args:
          rows: number of rows.
          columns: number of columns.
          seed: random seed for the RNG.
        """
        self.random_seed = seed
        self.rows = rows
        self.columns = columns
        self._rng = np.random.RandomState(seed)
        self._ball_x = None
        self._ball_y = None
        self._paddle_x = None
        self._paddle_y = None
        self.bsuite_num_episodes = sweep.NUM_EPISODES

    def render(self, mode="human"):
        pass

    def _reset(self) -> dm_env.TimeStep:
        """Returns the first `TimeStep` of a new episode."""
        self._ball_x = self._rng.randint(self.columns)
        self._ball_y = self.rows - 1
        self._paddle_x = self.columns // 2
        self._paddle_y = 0

        return dm_env.restart(self._observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        """Updates the environment according to the action."""

        # Check for termination.
        if self._ball_y == self._paddle_y:
            reward = 1.0 if self._paddle_x == self._ball_x else 0
            return dm_env.termination(reward=reward, observation=self._observation())

        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self.columns - 1)

        # Drop the ball.
        self._ball_y -= 1

        return dm_env.transition(reward=0.0, observation=self._observation())

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            dtype=np.int,
            minimum=0,
            maximum=max(self.rows, self.columns),
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
        return dict(optimal=1)


class Wrapper(gym.Wrapper, base_env.Env[Obs, int]):
    metadata = {"render.modes": []}

    def __init__(self, data: Data, env: Env, hint: bool):
        super().__init__(cast(gym.Env, env))
        self.data = data
        self.hint = hint
        self.action_space = Discrete(3, seed=env.random_seed)
        spec = env.observation_spec()
        self.observation_space = MultiDiscrete(
            np.full(spec.shape, spec.maximum - spec.minimum), seed=env.random_seed
        )

    def action_stop(self) -> str:
        return "\nball.descend()\n"

    def action_str(self, action: int) -> str:
        return f"reward = paddle.{self.actions()[action]}(){self.action_stop()}"

    def actions(self) -> "list[str]":
        return [
            "left",
            "stay",
            "right",
        ]

    @staticmethod
    def done_stop() -> str:
        return "\n"

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 1

    @classmethod
    def hint_str(cls, obs: Obs) -> str:
        if obs.paddle_x < obs.ball_x:
            operator = "<"
        elif obs.paddle_x > obs.ball_x:
            operator = ">"
        else:
            assert obs.paddle_x == obs.ball_x
            operator = "=="
        return " and ".join(
            [
                f"paddle.x == {obs.paddle_x}",
                f"ball.x == {obs.ball_x}",
                "paddle.x " + operator + " ball.x",
                f"ball.y == {int(obs.ball_y)}",
            ]
        )

    @staticmethod
    def initial_str() -> str:
        return "\npaddle, ball = reset()\n"

    def max_q_steps(self) -> int:
        return self.env.rows

    def reset(self):
        assert isinstance(self.env, Env)
        return self.env.reset().observation

    @staticmethod
    def reward_stop() -> str:
        return "\n"

    def reward_str(self, reward: float) -> str:
        return f"assert reward == {int(reward)}"

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    def start_states(self) -> Optional[List[Obs]]:
        return [
            Obs(self.env.columns // 2, ball_x, self.env.rows - 1)
            for ball_x in range(self.env.columns)
        ]

    def state_str(self, state: Obs) -> str:
        state_str = f"assert paddle == C({state.paddle_x}, 0) and ball == C({state.ball_x}, {state.ball_y})"
        if self.hint:
            state_str += f" and {self.hint_str(state)}"
        return state_str + self.state_stop()

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        assert isinstance(self.env, Env)
        time_step: dm_env.TimeStep = self.env.step(action)
        return (
            time_step.observation,
            time_step.reward,
            time_step.last(),
            self.bsuite_info(),
        )

    def termination_str(self, ts: TimeStep) -> str:
        s = super().termination_str(ts)
        if (
            self.hint
            and ts.reward == 0
            and ts.done
            and (
                ("paddle.x < ball.x" not in s or "paddle.x > ball.x" not in s)
                or "ball.y == 0" not in s
            )
        ):
            breakpoint()
        if (
            self.hint
            and ts.reward == 1
            and ("paddle.x == ball.x" not in s and "ball.y == 0" not in s)
        ):
            breakpoint()
        return s

    def ts_to_string(self, ts: TimeStep) -> str:
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
            self.reward_str(ts.reward),
            self.reward_stop(),
            self.done_str(ts.done),
            self.done_stop(),
        ]
        if ts.done:
            parts += [self.state_str(ts.next_state)]
        s = "".join(parts)
        if self.hint and ts.reward == 1 and "paddle.x == ball.x" not in s:
            breakpoint()
        if (
            self.hint
            and ts.reward == 0
            and ts.done
            and ("paddle.x < ball.x" not in s and "paddle.x > ball.x" not in s)
        ):
            breakpoint()
        return s

    def valid_done(self, done_str: str) -> bool:
        return (
            done_str.startswith("assert")
            and "done" in done_str
            and done_str.endswith(self.done_stop())
        )

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.findall(r"assert reward == [0-9]+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(state_str.startswith("assert paddle == C(")) and state_str.endswith(
            self.state_stop()
        )


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    max_step = 8
    env = Wrapper(env=Env(columns=4, rows=5, seed=0), data=Data.code, hint=True)
    while True:
        s = env.reset()
        print(env.initial_str() + env.state_str(s))
        t = False
        trajectory = []
        while not t:
            a = env.action_space.sample()
            # a = int(input("Action: ")) - 1
            s_, r, t, i = env.step(a)
            ts = TimeStep(s, a, r, t, s_)
            print(
                env.initial_str()
                + env.state_str(ts.state)
                + env.action_str(ts.action)
                + env.reward_str(ts.reward)
                + env.reward_stop()
                + env.done_str(ts.done)
                + env.done_stop()
            )
            breakpoint()
            trajectory.append(ts)
            completions = [env.ts_to_string(ts) for ts in trajectory]
            with_termination = [*completions, env.termination_str(ts)]
            done_estimate = env.done(*with_termination)
            if not done_estimate == t:
                breakpoint()
                env.done(*with_termination)
            prompt = "".join(with_termination)
            value_from_trajectory = get_value(*trajectory, gamma=env.gamma())
            print(env.ts_to_string(ts) + env.state_str(ts.next_state))
            # breakpoint()
            s = s_
        # breakpoint()
