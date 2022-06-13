import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import base_env
import numpy as np
from base_env import TimeStep
from gym.spaces import Discrete


@dataclass
class State:
    pos: float
    vel: float


@dataclass
class Env(base_env.Env):
    max_distance: float
    max_step: int
    _max_trajectory: int
    pos_threshold: float
    random_seed: int
    vel_threshold: float
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

    def done(self, *completions: str) -> bool:
        *_, last_state = completions
        try:
            [pos] = re.findall(r"pos == (-?\d+\.\d+)", last_state)
            [vel] = re.findall(r"vel == (-?\d+\.\d+)", last_state)
        except ValueError:
            return True
        pos, vel = map(float, [pos, vel])
        t = "".join(completions).count("pos, vel = ")
        return self.oob(pos) or self.success(pos, vel) or self.oot(t)

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
        if state.vel < -self.vel_threshold:
            vel_hint = f"vel < {-self.vel_threshold}"
        elif state.vel > self.vel_threshold:
            vel_hint = f"{self.vel_threshold} < vel"
        elif -self.vel_threshold <= state.vel <= self.vel_threshold:
            vel_hint = f"{-self.vel_threshold} <= vel <= {self.vel_threshold}"
        else:
            raise RuntimeError()
        hint = " and ".join([pos_hint, vel_hint])
        return hint

    @staticmethod
    def initial_str() -> str:
        return "\npos, vel = reset()\n"

    @classmethod
    def log_gamma(cls) -> float:
        return cls.gamma()

    def max_trajectory(self) -> int:
        return self._max_trajectory

    def start_states(self) -> Optional[Iterable[State]]:
        return None

    def state_str(self, state: State) -> str:
        state_str = f"assert pos == {state.pos:.2f} and vel == {state.vel:.2f}"
        hint_str = self.hint_str(state)
        if self.hint and hint_str:
            state_str += f" and {hint_str}"
        reward = f" and reward == {int(self.success(state.pos, state.vel))}"
        return state_str + reward + self.state_stop()

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.findall(r"assert reward == [0-9]+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(state_str.startswith("assert pos == ")) and state_str.endswith(
            self.state_stop()
        )

    def step(self, action: int):
        act_str = self.actions()[action]
        if act_str == "accel":
            accel = 1
        elif act_str == "decel":
            accel = -1
        else:
            breakpoint()
            raise RuntimeError()

        pos = self.state.pos + self.state.vel + accel / 2
        vel = self.state.vel + accel
        self.state = State(pos, vel)
        success = self.success(pos, vel)
        reward = float(success)
        self.t += 1
        done = self.oob(pos) or success or self.oot(self.t)
        if success and self.t < self.min_steps:
            breakpoint()
        return self.state, reward, done, self.info

    def oob(self, pos):
        return abs(pos) > self.max_distance

    def oot(self, t: int):
        return t >= self.max_step

    def quantify(self, prompt: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.gamma()
        matches = re.findall(r"reward == (\d)", prompt)
        return sum([gamma ** (t - 1) * float(x) for t, x in enumerate(matches)])

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
        self.half_dist = half_dist = dist_to_threshold / 2
        self.min_steps = min_steps = math.ceil(2 * math.sqrt(half_dist) - 1)
        self.info = dict(min_steps=min_steps, optimal=self.gamma() ** min_steps)
        return self.state

    def render(self, mode="human"):
        pass

    def success(self, pos, vel):
        return abs(pos) <= self.pos_threshold and abs(vel) <= self.vel_threshold

    def termination_str(self, ts: TimeStep) -> str:
        return ""

    def ts_to_string(self, ts: TimeStep) -> str:
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
        ]
        if ts.done:
            parts += [self.state_str(ts.next_state)]
        s = "".join(parts)
        return s


if __name__ == "__main__":
    from rl.common import get_value

    env = Env(
        hint=True,
        max_distance=10,
        max_step=6,
        _max_trajectory=6,
        pos_threshold=1,
        random_seed=0,
        vel_threshold=1,
    )

    while True:
        s = env.reset()
        print(env.initial_str() + env.state_str(s))
        t = False
        trajectory = []
        while not t:
            # a = env.action_space.sample()
            a = int(input("Action: ")) - 1
            s_, r, t, i = env.step(a)
            ts = base_env.TimeStep(s, a, r, t, s_)
            trajectory.append(ts)
            completions = [env.ts_to_string(ts) for ts in trajectory]
            done_estimate = env.done(*completions, env.state_str(s_))
            prompt = "".join(completions)
            value_from_prompt = env.quantify(prompt)
            value_from_trajectory = get_value(*trajectory, gamma=env.gamma())
            if not value_from_prompt == value_from_trajectory:
                print(value_from_prompt, value_from_trajectory)
                breakpoint()
                env.quantify(prompt)
                get_value(*trajectory, gamma=env.gamma())
            if not done_estimate == t:
                state_str = env.state_str(s_)
                breakpoint()
                env.done(*completions, state_str)
            if t:
                print("Min Steps", i["min_steps"])
                print("T", env.t)
                if not i["min_steps"] - 1 < env.t <= i["min_steps"]:
                    breakpoint()
            print(env.ts_to_string(ts) + env.state_str(ts.next_state))
            # breakpoint()
            s = s_
        # breakpoint()
