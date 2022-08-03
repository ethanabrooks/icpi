import math
import re
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Optional

import base_env
import numpy as np
from base_env import TimeStep
from gym.spaces import Discrete
from rl.lm import Data


class State(NamedTuple):
    pos: float
    vel: float


@dataclass
class Env(base_env.Env):
    max_distance: float
    _max_trajectory: int
    pos_threshold: float
    random_seed: int
    action_space: Discrete = field(init=False)
    info: dict = field(init=False)
    rng: np.random.Generator = field(init=False)
    state: State = field(init=False)
    t: int = field(init=False)

    def __post_init__(self):
        self.action_space = Discrete(2, seed=self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def action_stop() -> str:
        return "\n"

    def action_str(self, action: int) -> str:
        action_str = self.actions()[action]
        return f"pos, vel = {action_str}(pos, vel){self.action_stop()}"

    def actions(self) -> "list[str]":
        return ["decel", "accel"]

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    def done_stop(self) -> str:
        return "\n"

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: State) -> str:
        if state.pos < -self.pos_threshold:
            pos_hint = f"pos < {-self.pos_threshold}"
        elif state.pos > self.pos_threshold:
            pos_hint = f"{self.pos_threshold} < pos"
        elif -self.pos_threshold <= state.pos <= self.pos_threshold:
            pos_hint = f"{-self.pos_threshold} <= pos <= {self.pos_threshold}"
        else:
            raise RuntimeError()
        vel_hint = f"vel {'==' if state.vel == 0 else '!='} 0"
        hint = " and ".join([pos_hint, vel_hint])
        return hint

    @staticmethod
    def initial_str() -> str:
        return "\npos, vel = reset()\n"

    @classmethod
    def log_gamma(cls) -> float:
        return cls.gamma()

    def max_q_steps(self) -> int:
        return self._max_trajectory

    def oob(self, pos):
        return abs(pos) > self.max_distance

    def quantify(self, prompt: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.gamma()
        matches = re.findall(r"reward == (\d)", prompt)
        matches = matches[: self.max_q_steps()]
        return sum([gamma**t * float(x) for t, x in enumerate(matches)])

    def reset(self):
        self.pos = pos = self.rng.choice(
            [
                self.rng.uniform(-self.max_distance, -self.pos_threshold),
                self.rng.uniform(self.pos_threshold, self.max_distance),
            ]
        )
        vel = 0
        self.state = State(pos, vel)
        self.t = 0
        self.dist_to_threshold = dist_to_threshold = abs(pos) - self.pos_threshold
        sqrt = math.sqrt(dist_to_threshold)
        n = math.ceil(sqrt)
        self.min_steps = min_steps = 2 * n
        self.info = dict(min_steps=min_steps, optimal=self.gamma() ** min_steps)
        return self.state

    def render(self, mode="human"):
        pass

    def reward_str(self, reward: float) -> str:
        return f"assert reward == {int(reward)}"

    def start_states(self) -> Optional[Iterable[State]]:
        return None

    def state_str(self, state: State) -> str:
        state_str = f"assert pos == {state.pos:.2f} and vel == {state.vel:.2f}"
        hint_str = self.hint_str(state)
        if self.hint and hint_str:
            state_str += f" and {hint_str}"
        return state_str + self.state_stop()

    def step(self, action: int):
        success = self.success(self.state.pos, self.state.vel)
        act_str = self.actions()[action]
        if act_str == "accel":
            accel = 1
        elif act_str == "decel":
            accel = -1
        else:
            breakpoint()
            raise RuntimeError()

        pos = self.state.pos + self.state.vel
        vel = self.state.vel + accel
        self.state = State(pos, vel)
        reward = float(success)
        self.t += 1
        if success and self.t < self.min_steps:
            breakpoint()
        return self.state, reward, success, self.info

    def success(self, pos, vel):
        return abs(pos) <= self.pos_threshold and vel == 0

    def termination_str(self, ts: TimeStep) -> str:
        return ""

    def ts_to_string(self, ts: TimeStep) -> str:
        reward_str = f"assert reward == {ts.reward}"
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
            reward_str,
            self.reward_stop(),
        ]
        if ts.done:
            parts += [self.state_str(ts.next_state)]
        s = "".join(parts)
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
        return bool(state_str.startswith("assert pos == ")) and state_str.endswith(
            self.state_stop()
        )


if __name__ == "__main__":
    from rl.common import get_value

    env = Env(
        hint=True,
        max_distance=6,
        _max_trajectory=6,
        pos_threshold=2,
        random_seed=0,
        data=Data.code,
    )

    while True:
        s = env.reset()
        print(env.initial_str() + env.state_str(s))
        t = False
        trajectory = []
        while not t:
            a = env.action_space.sample()
            # a = int(input("Action: ")) - 1
            s_, r, t, i = env.step(a)
            ts = base_env.TimeStep(s, a, r, t, s_)
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
            prompt = "".join(completions)
            value_from_prompt = env.quantify(prompt)
            value_from_trajectory = get_value(*trajectory, gamma=env.gamma())
            if not value_from_prompt == value_from_trajectory:
                print(value_from_prompt, value_from_trajectory)
                breakpoint()
                env.quantify(prompt)
                get_value(*trajectory, gamma=env.gamma())
            # if not done_estimate == t:
            #     state_str = env.state_str(s_)
            #     breakpoint()
            #     env.done(*completions, state_str)
            if t:
                print("Min Steps", i["min_steps"])
                print("T", env.t)
                if not i["min_steps"] - 1 < env.t <= i["min_steps"]:
                    breakpoint()
            print(env.ts_to_string(ts) + env.state_str(ts.next_state))
            # breakpoint()
            s = s_
        # breakpoint()
