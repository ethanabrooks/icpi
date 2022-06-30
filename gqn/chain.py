import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep

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

    @staticmethod
    def action_stop() -> str:
        return "\n"

    def action_str(self, action: int) -> str:
        action_str = self.actions()[action]
        if action == 1:
            return f"reward = {action_str}(state){self.action_stop()}"
        else:
            return f"state = {action_str}(){self.action_stop()}"

    def actions(self):
        return [
            "left",
            "try_goal",
            "right",
        ]

    def done(self, done_str: str) -> bool:
        return "assert done" in done_str

    def done_stop(self) -> str:
        return "\n"

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: int) -> str:
        return "state " + ("==" if state == self.goal else "!=") + f" {self.goal}"

    @classmethod
    def initial_str(cls) -> str:
        return "\nstate, reward = reset()\n"

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
        return f"assert reward == {int(reward)}"

    def start_states(self) -> Optional[Iterable[int]]:
        return range(self.n)

    def state_str(self, state: int) -> str:
        state_str = f"assert state == {state}"
        if self.hint:
            state_str += f" and {self.hint_str(state)}"
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
        reward_str = f"assert reward == {ts.reward}"
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
            reward_str,
            self.reward_stop(),
        ]
        s = "".join(parts)
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
        return (
            done_str.startswith("assert")
            and "done" in done_str
            and done_str.endswith(self.done_stop())
        )

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.findall(r"assert reward == \d+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(state_str.startswith("assert state == ")) and state_str.endswith(
            self.state_stop()
        )


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    env = Env(goal=4, n=8, hint=True, random_seed=0)
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
