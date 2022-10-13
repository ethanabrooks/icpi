import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep
from rl.lm import Data

DEAD = "dead"


class C(NamedTuple):
    x: int
    y: int

    def __str__(self):
        return f"C{str(tuple(self))}"


class Alien(NamedTuple):
    xy: Optional[C]

    def is_dead(self) -> bool:
        return self.xy is None

    @classmethod
    def dead(cls) -> "Alien":
        return cls(None)

    def descend(self) -> "Alien":
        return self if self.is_dead() else Alien(C(self.xy.x, self.xy.y - 1))

    def escaped(self, ship: int) -> bool:
        if self.is_dead():
            return False
        return abs(ship - self.xy.x) > self.xy.y

    def landed(self) -> bool:
        return not self.is_dead() and self.xy.y == 0

    def over(self, x: int) -> bool:
        return not self.is_dead() and self.xy.x == x

    def __str__(self) -> str:
        return str(None) if self.is_dead() else str(self.xy)

    def __repr__(self):
        return str(self)

    @classmethod
    def spawn(cls, x: int, y: int) -> "Alien":
        return cls(C(x, y))

    def take_fire(self, ship: int) -> "Alien":
        return Alien(None if self.over(ship) else self.xy)

    @property
    def x(self):
        return self.xy.x

    @property
    def y(self):
        return self.xy.y


class Obs(NamedTuple):
    agent: int
    aliens: Tuple[Alien]

    def num_shot_down(self):
        return sum(1 for a in self.aliens if a.is_dead())


@dataclass
class Env(base_env.Env[Obs, int]):
    height: int
    n_aliens: int
    random_seed: int
    width: int
    aliens: Tuple[Alien] = field(init=False)
    random: np.random.Generator = field(init=False)
    action_space: gym.spaces.Discrete = field(init=False)

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )

    @staticmethod
    def action_stop() -> str:
        return "\n"

    def action_str(self, action: int) -> str:
        if action == 1:
            return f"reward = {self.ship()}.{self.actions()[action]}(aliens){self.action_stop()}"
        else:
            return f"{self.ship()}.{self.actions()[action]}(){self.action_stop()}"

    def actions(self):
        return [
            "left",
            "shoot",
            "right",
        ]

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    def done_stop(self) -> str:
        return "\n"

    def failure_threshold(self) -> float:
        return 1

    @staticmethod
    def gamma() -> float:
        return 0.8

    @classmethod
    def initial_str(cls) -> str:
        return f"\n{cls.ship()}, aliens = reset()\n"

    def max_q_steps(self) -> int:
        return self.width * self.n_aliens

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Obs:
        self.agent, *alien_xs = self.random.choice(
            self.width, size=self.n_aliens + 1, replace=False
        )
        self.aliens = tuple([Alien.spawn(x, self.height) for x in alien_xs])
        return Obs(agent=self.agent, aliens=self.aliens)

    def reward_str(self, reward: float) -> str:
        return f"assert reward == {int(reward)}\nfor a in aliens:\n    a.descend"

    def reward_stop(self) -> str:
        return "()\n"

    @staticmethod
    def ship() -> str:
        return "ship"

    def state_stop(self) -> str:
        return "\n"

    def state_str(self, state: Obs) -> str:
        assertions = [
            f"{self.ship()} == {C(state.agent, 0)}",
            f"aliens == {list(state.aliens)}",
        ]

        if self.hint:
            lhs = ["ship.x"]
            rhs = [str(state.agent)]
            for i, a in enumerate(state.aliens):
                if not a.is_dead():
                    lhs.append(f"aliens[{i}].x")
                    rhs.append(str(a.x))
            assertions.append(f"({', '.join(lhs)}) == ({', '.join(rhs)})")

            for i, a in enumerate(state.aliens):
                if not a.is_dead():
                    if state.agent < a.x:
                        operator = "<"
                    elif state.agent > a.x:
                        operator = ">"
                    else:
                        assert a.over(state.agent)
                        operator = "=="
                    assertions.append("ship.x " + operator + f" aliens[{i}].x")

        state_str = "assert " + " and ".join(assertions)
        return state_str + self.state_stop()

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent, *aliens in itertools.permutations(range(self.n_aliens)):
            aliens = [Alien.spawn(x, self.height) for x in aliens]
            yield Obs(agent, aliens)

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        new_aliens = []
        reward = 0
        done = False
        for alien in self.aliens:
            if action == 1:
                reward += alien.over(self.agent)
                alien = alien.take_fire(self.agent)
            if alien.landed():
                done = True
            new_aliens.append(alien.descend())

        self.aliens = tuple(new_aliens)
        if all(a.is_dead() for a in self.aliens):
            done = True
        info = dict(optimal=self.n_aliens)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        # print(f"landed={landed}, return={self.r}, done={done}")
        obs = Obs(self.agent, self.aliens)
        return obs, reward, done, info

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
            parts += [self.state_str(ts.next_state), self.state_stop()]
        s = "".join(parts)
        # if (
        #     self.hint
        #     and ts.reward == 1
        #     and f"{self.alien_str()}.x == {self.ship()}.x" not in s
        # ):
        #     breakpoint()
        # if (
        #     self.hint
        #     and ts.reward == 0
        #     and ts.action == 1
        #     and f"{self.alien_str()}.x != {self.ship()}.x" not in s
        # ):
        #     breakpoint()
        # if ts.reward == 1:
        #     if "spawn()" not in s:
        #         breakpoint()
        # else:
        #     if "descend()" not in s:
        #         breakpoint()
        return s

    def valid_done(self, done_str: str) -> bool:
        return (
            done_str.startswith("assert")
            and "done" in done_str
            and done_str.endswith(self.done_stop())
        )

    def valid_reward(self, reward_str: str) -> bool:
        return bool(re.findall(r"reward == \d+", reward_str)) and reward_str.endswith(
            self.reward_stop()
        )

    def valid_state(self, state_str: str) -> bool:
        return bool(
            state_str.startswith(f"assert {self.ship()} == C(")
        ) and state_str.endswith(self.state_stop())


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    max_step = 8
    env = Env(
        width=4,
        height=5,
        n_aliens=2,
        random_seed=0,
        hint=True,
        data=Data.code,
    )
    while True:
        s = env.reset()
        print(env.initial_str() + env.state_str(s))
        t = False
        trajectory = []
        completions = []
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
            print(env.ts_to_string(ts) + "\n" + env.state_str(ts.next_state))
            if t:
                breakpoint()
            s = s_
        # breakpoint()
