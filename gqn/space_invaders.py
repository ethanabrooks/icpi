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


class Alien(NamedTuple):
    x: int
    y: int

    def landed(self) -> bool:
        return self.y == 0

    def over(self, x: int) -> bool:
        return self.x == x


class Obs(NamedTuple):
    agent: int
    aliens: Tuple[Alien, ...]


@dataclass
class Env(base_env.Env[Obs, int]):
    height: int
    max_aliens: int
    max_step: int
    random_seed: int
    status: bool
    width: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )

    def action_str(self, action: int) -> str:
        return f"{self.ship()}, {self.alien()}, reward = {self.actions()[action]}({self.ship()}, {self.alien()}){self.action_stop()}"

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
        return bool(re.findall(r"\[C\(\d, 0\)", state_or_reward))

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 1.0

    @staticmethod
    def hint_stop() -> Optional[str]:
        return "\n"

    def _hint_str(self, state: Obs) -> str:
        hint = " and ".join(
            [
                f"{self.ship()}.x == {self.alien()}[{i}].x"
                if a.over(state.agent)
                else f"{self.ship()}.x != {self.alien()}[{i}].x"
                for i, a in enumerate(state.aliens)
            ]
            + [f"len({self.alien()}) == {len(state.aliens)}"]
        )
        if hint:
            return f"assert {hint}"
        return ""

    @classmethod
    def initial_str(cls) -> str:
        return f"{cls.ship()}, {cls.alien()} = reset()\n"

    def partially_observable(self) -> bool:
        return False

    def quantify(self, prompt: str) -> float:
        matches = re.findall(r"assert reward == (\d)", prompt)
        return sum([float(x) for x in matches])

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
        self.reward = 0
        self.agent, *alien_xs = self.random.choice(self.width, size=1 + num_aliens)
        self.aliens = [Alien(x, self.height) for i, x in enumerate(alien_xs)]
        return Obs(self.agent, tuple(self.aliens))

    @staticmethod
    def ship() -> str:
        return "ship"

    def _state_str(self, state: Obs) -> str:
        aliens = ", ".join([f"C{tuple(a)}" for a in state.aliens])
        return f"assert {self.ship()} == C{(state.agent, 0)} and {self.alien()} == [{aliens}]"

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent in range(self.width):
            for xs in itertools.product(range(self.width), repeat=self.max_aliens):
                aliens = [Alien(x, self.height) for x in xs]
                yield Obs(agent, tuple(aliens))

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if (
            self.reward == 0
            and len(self.aliens) < self.max_aliens
            and self.random.choice(2)
        ):
            self.aliens.append(Alien(self.random.choice(self.width), self.height))

        if action == 1:
            num_aliens = len(self.aliens)
            self.aliens = [Alien(a.x, a.y) for a in self.aliens if a.x != self.agent]
            self.reward = num_aliens - len(self.aliens)
        else:
            self.reward = 0
        self.aliens = [Alien(a.x, a.y - 1) for a in self.aliens]
        info = dict(optimal=self.max_step)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        done = any(a.landed() for a in self.aliens)
        state = Obs(self.agent, tuple(self.aliens))
        return state, self.reward, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        return "".join(
            [
                self.state_str(ts.state),
                self._hint_str(ts.state),
                self.hint_stop(),
                self.action_str(ts.action),
                f"assert reward == {ts.reward}",
                self.reward_stop(),
            ]
        )


if __name__ == "__main__":
    max_step = 8
    env = TimeLimit(
        Env(
            width=3,
            height=4,
            max_aliens=2,
            max_step=max_step,
            random_seed=0,
            status=False,
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
