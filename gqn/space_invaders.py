import itertools
import re
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep
from gym.wrappers import TimeLimit

DEAD = "dead"


class C(NamedTuple):
    x: int
    y: int


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
        return str(None) if self.is_dead() else f"C{tuple(self.xy)}"

    @classmethod
    def spawn(cls, x: int, y: int) -> "Alien":
        return cls(C(x, y))

    def take_fire(self, ship: int) -> "Alien":
        return Alien(None if self.over(ship) else self.xy)


class Obs(NamedTuple):
    agent: int
    aliens: Tuple[Alien, ...]


@dataclass
class Env(base_env.Env[Obs, int]):
    height: int
    max_aliens: int
    max_step: int
    random_seed: int
    width: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )

    def action_str(self, action: int) -> str:
        if action == 1:
            return f"reward = {self.ship()}.{self.actions()[action]}({self.alien()}){self.action_stop()}"
        else:
            return f"{self.ship()} = {self.ship()}.{self.actions()[action]}(){self.action_stop()}"

    def actions(self):
        return [
            "left",
            "shoot",
            "right",
        ]

    @staticmethod
    def alien() -> str:
        return "aliens"

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        return bool(re.findall(r"aliens == \[.*C\(\d, 0\).*] and", state_or_reward))

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 1.0

    def _hint_str(self, state: Obs) -> str:
        hint = " and ".join(
            [
                f"{self.ship()}.x == {self.alien()}[{i}].x"
                if a.over(state.agent)
                else f"{self.ship()}.x != {self.alien()}[{i}].x"
                for i, a in enumerate(state.aliens)
                if not a.is_dead()
            ]
        )
        return hint

    @classmethod
    def initial_str(cls) -> str:
        return f"\n{cls.ship()}, {cls.alien()} = reset()\n"

    def max_trajectory(self) -> int:
        return self.max_step

    def partially_observable(self) -> bool:
        return False

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Obs:
        num_aliens = 1 + self.random.choice(self.max_aliens)
        self.agent, *alien_xs = self.random.choice(self.width, size=1 + self.max_aliens)
        self.aliens = [
            (Alien.spawn(x, self.height) if i < num_aliens else Alien.dead())
            for i, x in enumerate(alien_xs)
        ]
        self.optimal = num_aliens
        self.t = 0
        return Obs(self.agent, tuple(self.aliens))

    @staticmethod
    def reward_stop() -> str:
        return "aliens.descend()\n"

    @staticmethod
    def ship() -> str:
        return "ship"

    def _state_str(self, state: Obs) -> str:
        aliens = ", ".join([f"{a}" for a in state.aliens])
        return f"assert {self.ship()} == C{(state.agent, 0)} and {self.alien()} == [{aliens}]"

    def state_str(self, state: Obs) -> str:
        state_str = self._state_str(state)
        if self.hint:
            state_str += f" and {self._hint_str(state)}"
        return state_str + self.state_stop()

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent in range(self.width):
            for xs in itertools.product(range(self.width), repeat=self.max_aliens):
                aliens = [Alien.spawn(x, self.height) for x in xs]
                yield Obs(agent, tuple(aliens))

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if action == 1:
            reward = sum(a.over(self.agent) for a in self.aliens)
            self.aliens = [a.take_fire(self.agent) for a in self.aliens]
        else:
            reward = 0

        dead = [a.is_dead() for a in self.aliens]
        if reward == 0 and any(dead):
            i = dead.index(True)
            alien = self.aliens[i]
            self.aliens[i] = alien.spawn(self.random.choice(self.width), self.height)
            if abs(self.aliens[i].xy.x - self.agent) < (self.max_step - self.t):
                self.optimal += 1

        self.t += 1
        self.aliens = [a.descend() for a in self.aliens]
        info = dict(optimal=self.optimal)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        done = any(a.landed() for a in self.aliens)
        state = Obs(self.agent, tuple(self.aliens))
        # print("agent", self.agent, "aliens", self.aliens, "action", action)
        return state, reward, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        reward_str = " and ".join(
            [f"assert reward == {ts.reward}"]
            + (
                [
                    f"alien[{i}] is None"
                    for i, a in enumerate(ts.next_state.aliens)
                    if a.dead()
                ]
                if ts.reward > 0
                else []
            )
        )
        s = "".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
                reward_str + "\n",
                self.reward_stop(),
            ]
        )
        if ts.reward == 1 and "x == aliens" not in s:
            breakpoint()
        if ts.action == 1 and ts.reward == 0 and "x == aliens" in s:
            breakpoint()
        if ts.done:
            s += self.state_str(ts.next_state)
        return s

    def valid_reward(self, reward_str: str) -> bool:
        return bool(
            re.match(r"assert reward == [0-9]+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(
            state_str.startswith(f"assert {self.ship()} == C(")
        ) and state_str.endswith(self.state_stop())


if __name__ == "__main__":
    max_step = 8
    env = TimeLimit(
        Env(
            width=3,
            height=4,
            max_aliens=2,
            max_step=max_step,
            random_seed=0,
            hint=False,
        ),
        max_episode_steps=max_step,
    )
    while True:
        s = env.reset()
        print(env.state_str(s))
        t = False
        while not t:
            a = int(input("Action: ")) - 1
            s_, r, t, i = env.step(a)
            print(env.ts_to_string(TimeStep(s, a, r, t, s_)), env.state_str(s_))
            s = s_
        breakpoint()
