import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep
from rl.lm import Data

REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


@dataclass
class Env(base_env.Env[int, int]):
    goal: int
    n: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )
        self.observation_space = gym.spaces.Discrete(self.n)

    def action_stop(self) -> str:
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

    def action_str(self, action: int) -> str:
        action_str = self.actions()[action]
        if self.data == Data.code:
            if action == 1:
                return f"reward = {action_str}(state){self.action_stop()}"
            else:
                return f"state = {action_str}(){self.action_stop()}"
        elif self.data == Data.natural_language:
            if action == 1:
                return f"Check if you are at the goal{self.action_stop()}"
            else:
                return f"Move {action_str}{self.action_stop()}"
        raise RuntimeError("Invalid data")

    def actions(self):
        return [
            "left",
            "try_goal",
            "right",
        ]

    def done_stop(self) -> str:
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

    def done_str(self, done: bool) -> str:
        if self.data == Data.code:
            return "assert done" if done else "assert not done"
        elif self.data == Data.natural_language:
            return "You are " + ("" if done else "not ") + "done"
        raise RuntimeError("Invalid data")

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: int) -> str:
        if self.data == Data.code:
            return "state " + ("==" if state == self.goal else "!=") + f" {self.goal}"
        elif self.data == Data.natural_language:
            if state == self.goal:
                return ""
            else:
                return f", not state {self.goal}"
        raise RuntimeError("Invalid data")

    def initial_str(self) -> str:
        if self.data == Data.code:
            return "\nstate, reward = reset()\n"
        elif self.data == Data.natural_language:
            return "\n"
        raise RuntimeError("Invalid data")

    @classmethod
    def log_gamma(cls) -> float:
        return cls.gamma()

    def max_q_steps(self) -> int:
        return 2 + max(self.n - self.goal, self.goal)

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> int:
        self._state = self._start_state = self.random.choice(self.n)
        return self._start_state

    def reward_str(self, reward: float) -> str:
        if self.data == Data.code:
            return f"assert reward == {int(reward)}"
        elif self.data == Data.natural_language:
            return f"Receive {int(reward)} reward"

    def start_states(self) -> Optional[Iterable[int]]:
        return range(self.n)

    def state_str(self, state: int) -> str:
        if self.data == Data.code:
            state_str = f"assert state == {state}"
            if self.hint:
                state_str += f" and {self.hint_str(state)}"
            return state_str + self.state_stop()
        elif self.data == Data.natural_language:
            state_str = f"You are at state {state}"
            if self.hint:
                state_str += f"{self.hint_str(state)}"
            return state_str + self.state_stop()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        optimal = self.gamma() ** abs(self._start_state - self.goal)
        info = dict(optimal=optimal)
        self._state += action - 1
        self._state = np.clip(self._state, 0, self.n - 1)
        done = action == 1
        success = done and self._state == self.goal
        state = int(self._state)
        return state, float(success), done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        if self.data == Data.code:
            reward_str = f"assert reward == {ts.reward}"
        elif self.data == Data.natural_language:
            reward_str = f"Receive {int(ts.reward)} reward"
        else:
            raise RuntimeError("Invalid data")
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
            reward_str,
            self.reward_stop(),
        ]
        s = "".join(parts)
        if self.data == Data.code:
            if self.hint and ts.reward == 1 and f"state == {self.goal}" not in s:
                breakpoint()
            if (
                self.hint
                and ts.action == 1
                and ts.reward == 0
                and f"state != {self.goal}" not in s
            ):
                breakpoint()
        return s

    def valid_done(self, done_str: str) -> bool:
        if self.data == Data.code:
            return (
                done_str.startswith("assert")
                and "done" in done_str
                and done_str.endswith(self.done_stop())
            )
        elif self.data == Data.natural_language:
            return (
                done_str.startswith("You are")
                and "done" in done_str
                and done_str.endswith(self.done_stop())
            )
        raise RuntimeError("Invalid data")

    def valid_reward(self, reward_str: str) -> bool:
        if self.data == Data.code:
            return bool(
                re.findall(r"assert reward == \d+", reward_str)
            ) and reward_str.endswith(self.reward_stop())
        elif self.data == Data.natural_language:
            return bool(re.findall(r"Receive \d+", reward_str)) and reward_str.endswith(
                self.reward_stop()
            )
        raise RuntimeError("Invalid data")

    def valid_state(self, state_str: str) -> bool:
        if self.data == Data.code:
            return bool(
                state_str.startswith("assert state == ")
            ) and state_str.endswith(self.state_stop())
        elif self.data == Data.natural_language:
            return bool(
                state_str.startswith("You are at state ")
            ) and state_str.endswith(self.state_stop())


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    env = Env(goal=4, n=8, hint=True, random_seed=0, data=Data.code)
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
            # done_estimate = env.done(*completions, env.state_str(s_))
            # if not done_estimate == t:
            #     breakpoint()
            #     env.done(*completions, env.state_str(s_))
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
