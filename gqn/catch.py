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
        self._optimal = 1

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

        # Move the paddle.
        dx = _ACTIONS[action]
        self._paddle_x = np.clip(self._paddle_x + dx, 0, self.columns - 1)

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
        return dict(optimal=self._optimal)


class Wrapper(gym.Wrapper, base_env.Env[Obs, int]):
    def __init__(self, env: Env, hint: bool):
        super().__init__(cast(gym.Env, env))
        self.hint = hint
        self.action_space = Discrete(3, seed=env.random_seed)
        self.observation_space = MultiDiscrete(
            np.ones_like(env.observation_spec().shape), seed=env.random_seed
        )
        self.rewards = {
            1.0: "P.x==B.x, B.y==0, success" if hint else "success",
            0.0: "P.x!=B.x, B.y==0, failure" if hint else "failure",
        }

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

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        # return bool(re.findall(r"ball.y == 0", state_or_reward))
        return bool(re.findall(r"ball == C\(\d+, 0\)", state_or_reward))

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.9

    @classmethod
    def hint_str(cls, obs: Obs) -> str:
        return " and ".join(
            [
                "paddle.x "
                + ("==" if obs.paddle_x == obs.ball_x else "!=")
                + " ball.x",
            ]
        )

    @staticmethod
    def initial_str() -> str:
        return "\npaddle, ball = reset()\n"

    def max_trajectory(self) -> int:
        return self.env.rows

    def partially_observable(self) -> bool:
        return False

    def reset(self):
        assert isinstance(self.env, Env)
        return self.env.reset().observation

    @staticmethod
    def reward_stop() -> str:
        return "\n"

    def start_states(self) -> Optional[List[Obs]]:
        return [
            Obs(self.env.columns // 2, ball_x, self.env.rows - 1)
            for ball_x in range(self.env.columns)
        ]

    def _state_str(self, obs: Obs) -> str:
        return f"assert paddle == C({obs.paddle_x}, 0) and ball == C({obs.ball_x}, {obs.ball_y})"

    def state_str(self, state: Obs) -> str:
        state_str = self._state_str(state)
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

    def ts_to_string(self, ts: TimeStep) -> str:
        reward_str = " and ".join(
            (
                [
                    "ball.x " + ("==" if ts.reward else "!=") + " paddle.x",
                    "ball.y " + ("==" if ts.done else ">") + " 0",
                ]
                if self.hint
                else []
            )
            + [f"assert reward == {ts.reward}"]
        )
        s = "".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
                reward_str,
                self.reward_stop(),
            ]
        )
        if (
            self.hint
            and ts.reward == 1
            and ("ball.x == paddle.x" not in s or "ball.y == 0" not in s)
        ):
            breakpoint()
        if (
            self.hint
            and ts.reward == 0
            and ts.done
            and ("ball.x != paddle.x" not in s or "ball.y == 0" not in s)
        ):
            breakpoint()
        return s

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
    env = Wrapper(Env(columns=4, rows=5, seed=0), hint=False)
    while True:
        s = env.reset()
        print(env.initial_str() + env.state_str(s))
        t = False
        trajectory = []
        while not t:
            # a = env.action_space.sample()
            a = int(input("Action: ")) - 1
            s_, r, t, i = env.step(a)
            ts = TimeStep(s, a, r, t, s_)
            trajectory.append(ts)
            completions = [env.ts_to_string(ts) for ts in trajectory]
            done_estimate = env.done(*completions, env.state_str(s_))
            if not done_estimate == t:
                breakpoint()
                env.done(*completions, env.state_str(s_))
            prompt = "".join(completions)
            value_from_prompt = env.quantify(prompt)
            value_from_trajectory = get_value(*trajectory, gamma=env.gamma())
            if not value_from_prompt == value_from_trajectory:
                print(value_from_prompt, value_from_trajectory)
                breakpoint()
                env.quantify(prompt)
                get_value(*trajectory, gamma=env.gamma())
            print(env.ts_to_string(ts) + env.state_str(ts.next_state))
            # breakpoint()
            s = s_
        # breakpoint()
