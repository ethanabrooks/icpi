import re
from dataclasses import astuple, dataclass, field
from typing import Generic, Iterable, Iterator, NamedTuple, Optional, Tuple, TypeVar

import base_env
import numpy as np
from base_env import TimeStep
from gym.spaces import Discrete

WALL = "█"
GOAL = "*"
MAP = """\
█████████████
█ ·   ·   · █
█   █       █
█ * █ ·   · █
█   █████   █
█ ·   ·   · █
█████████████\
"""


class C(NamedTuple):
    i: int
    j: int

    def __add__(self, other: "C"):
        return C(self.i + other.i, self.j + other.j)

    def clip(self, max_i: int, max_j: int) -> "C":
        return C(
            max(0, min(max_i - 1, self.i)),
            max(0, min(max_j - 1, self.j)),
        )


T = TypeVar("T")


@dataclass
class Actions(Generic[T]):
    left: T
    down: T
    up: T
    right: T


REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


@dataclass
class Env(base_env.Env[C, int]):
    max_step: int
    random_seed: int
    goal: C = field(init=False)
    t: int = field(init=False)

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = Discrete(len(self.actions()), seed=self.random_seed)

        def chunk(*seq: T) -> Iterator[Tuple[T, T, T]]:
            """
            :param seq: █ * █ ·   · █
            """
            while True:
                try:
                    first, second, third, *tail = seq
                except ValueError:
                    return
                seq = [third, *tail]
                yield first, second, third  # e.g. (█*█),(█· ),( ·█)

        def parse_triple(
            top_str: str, middle_str: str, bottom_str: str
        ) -> Iterator[np.ndarray]:
            chunks = [
                list(chunk(*string[::2]))
                for string in (top_str, middle_str, bottom_str)
            ]  # e.g. [
            # [(█ █),(   ),(  █), ...],
            # [(█*█),(█· ),( ·█), ...],
            # [(█ █),(███),(  █), ...],
            # ]
            yield from zip(*chunks)  # e.g. [
            # [(█ █),(█*█),(█ █)],
            # [(   ),(█· ),(███)],
            # [(  █),( ·█),(  █)],
            # ...

        rows = MAP.split("\n")

        def parse_map() -> Iterator[np.array]:
            chunks = list(chunk(*rows))
            for c in chunks:
                yield np.array(list(parse_triple(*c)))

        self.map = np.array(list(parse_map()))
        rows = rows[1::2]
        [goal_row] = [r for r in rows if GOAL in r]
        self.goal = C(i=rows.index(goal_row), j=goal_row.index(GOAL))
        self.height = len(rows[::2])
        self.width = None
        for row in rows:
            width = len(row[2::4])
            if self.width is None:
                self.width = width
            assert self.width == width, f"{self.width} != {width}"

    @staticmethod
    def action_stop() -> str:
        return "\n"

    def action_str(self, action: int) -> str:
        action_str = self.actions()[action]
        return f"state, reward = {action_str}(){self.action_stop()}"

    def actions(self):
        return list(astuple(Actions(left="left", down="down", up="up", right="right")))

    def done(self, *completions: str) -> bool:
        *_, last_state, _ = completions
        if not len(completions) == self.t + 1:
            breakpoint()
        return (f"state == {self.goal}" in last_state) or (
            len(completions) > self.max_step
        )

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: C) -> str:
        return "state " + ("==" if state == self.goal else "!=") + f" {self.goal}"

    @classmethod
    def initial_str(cls) -> str:
        return "\nstate, reward = reset()\n"

    @classmethod
    def log_gamma(cls) -> float:
        return cls.gamma()

    def max_trajectory(self) -> int:
        return self.height + self.width

    def oot(self, t: int):
        return t >= self.max_step

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> C:
        low = np.zeros(2)
        high = np.array([self.height, self.width]) - 1
        self._state = self._start_state = C(*self.random.integers(low=low, high=high))
        self.t = 0
        return self._start_state

    def start_states(self) -> Optional[Iterable[C]]:
        for i in range(self.height):
            for j in range(self.width):
                coord = C(i, j)
                if coord == self.goal:
                    yield coord

    def state_str(self, state: C) -> str:
        state_str = f"assert state == {state}"
        if self.hint:
            state_str += f" and {self.hint_str(state)}"
        return state_str + self.state_stop()

    def step(self, action: int) -> Tuple[C, float, bool, dict]:
        optimal = None  # TODO
        info = dict(optimal=optimal)
        success = self.success(self._state)
        deltas = Actions[C](
            left=C(0, -1),
            down=C(1, 0),
            up=C(-1, 0),
            right=C(0, 1),
        )
        delta = astuple(deltas)[action]
        chunk = self.map[self._state]
        obstructed = chunk[delta + C(1, 1)] == WALL
        if not obstructed:
            self._state += delta
            self._state = self._state.clip(self.height, self.width)
        self.t += 1
        done = success or self.oot(self.t)
        return self._state, float(success), done, info

    def success(self, state: C) -> bool:
        return state == self.goal

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

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.findall(r"assert reward == [0-9]+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(state_str.startswith("assert state == ")) and state_str.endswith(
            self.state_stop()
        )


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    env = Env(hint=True, max_step=8, random_seed=0)
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
                state_str = env.state_str(s_)
                breakpoint()
                env.done(*completions, state_str)
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
