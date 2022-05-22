import re
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep

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
    alien: Alien
    num_shot_down: int


@dataclass
class Env(base_env.Env[Obs, int]):
    height: int
    optimal_undiscounted: int
    random_seed: int
    width: int

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
            return f"reward = {self.ship()}.{self.actions()[action]}({self.alien_str()}){self.action_stop()}"
        else:
            return f"{self.ship()}.{self.actions()[action]}(){self.action_stop()}"

    def actions(self):
        return [
            "left",
            "shoot",
            "right",
        ]

    @staticmethod
    def alien_str() -> str:
        return "alien"

    def done(self, done_str: str) -> bool:
        return "assert done" in done_str

    def done_str(self, done: bool) -> str:
        return f"assert{' ' if done else ' not '}done"

    def done_stop(self) -> str:
        return "\n"

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def hint_str(self, state: Obs) -> str:
        if state.alien.is_dead():
            return ""
        hint = " and ".join(
            [
                f"{self.alien_str()}.x "
                + ("==" if state.alien.over(state.agent) else "!=")
                + f" {self.ship()}.x",
                f"{self.alien_str()}.y "
                + ("==" if state.alien.landed() else ">")
                + " 0",
            ]
        )
        # if state.alien.xy is not None:
        #     if state.agent == state.alien.xy.x:
        #         breakpoint()
        #     if 0 == state.alien.xy.y:
        #         breakpoint()
        return hint

    @classmethod
    def initial_str(cls) -> str:
        return f"{cls.ship()}, {cls.alien_str()} = reset()\n"

    def max_trajectory(self) -> int:
        return self.width * self.optimal_undiscounted

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Obs:
        self.agent, alien_x = self.random.choice(self.width, size=2)
        self.alien = Alien.spawn(alien_x, self.height)
        self.t = 0
        self.r = 0
        return Obs(agent=self.agent, alien=self.alien, num_shot_down=self.r)

    def reward_str(self, reward: float) -> str:
        string = f"assert reward == {int(reward)}\nalien."
        if reward == 1:
            string += "spawn"
        else:
            string += "descend"
        return string

    @staticmethod
    def reward_stop() -> str:
        return "()\n"

    @staticmethod
    def ship() -> str:
        return "ship"

    @staticmethod
    def state_stop() -> str:
        return "\n"

    def state_str(self, state: Obs) -> str:
        assertions = [
            f"{self.ship()} == C{(state.agent, 0)}",
            f"{self.alien_str()} == {str(state.alien)}",
            f"num_shot_down == {state.num_shot_down}",
        ]
        hint_str = self.hint_str(state)
        if self.hint and hint_str:
            assertions += [hint_str]
        state_str = "assert " + " and ".join(assertions)
        return state_str + self.state_stop()

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent in range(self.width):
            for alien_x in range(self.width):
                alien = Alien.spawn(alien_x, self.height)
                yield Obs(agent, alien, num_shot_down=0)

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if action == 1:
            reward = float(self.alien.over(self.agent))
            self.r += reward
            self.alien = self.alien.take_fire(self.agent)
        else:
            reward = 0

        dead = self.alien.is_dead()
        if dead:
            self.alien = self.alien.spawn(self.random.choice(self.width), self.height)

        self.t += 1
        max_return = self.r >= self.optimal_undiscounted
        done = self.alien.landed() or max_return
        self.alien = self.alien.descend()
        info = dict(optimal=self.optimal_undiscounted)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        # print(f"landed={landed}, return={self.r}, done={done}")
        state = Obs(self.agent, self.alien, self.r)
        return state, reward, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        parts = [
            self.state_str(ts.state),
            self.action_str(ts.action),
            self.reward_str(ts.reward),
            self.reward_stop(),
            self.done_str(ts.done),
        ]
        if ts.done:
            parts += [self.state_str(ts.next_state), self.state_stop()]
        s = "".join(parts)
        if (
            self.hint
            and ts.reward == 1
            and f"{self.alien_str()}.x == {self.ship()}.x" not in s
        ):
            breakpoint()
        if (
            self.hint
            and ts.reward == 0
            and ts.action == 1
            and f"{self.alien_str()}.x != {self.ship()}.x" not in s
        ):
            breakpoint()
        if ts.reward == 1:
            if "spawn()" not in s:
                breakpoint()
        else:
            if "descend()" not in s:
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
            re.findall(r"reward == [0-9]+", reward_str)
        ) and reward_str.endswith(self.reward_stop())

    def valid_state(self, state_str: str) -> bool:
        return bool(
            state_str.startswith(f"assert {self.ship()} == C(")
        ) and state_str.endswith(self.state_stop())


if __name__ == "__main__":

    def get_value(*trajectory: TimeStep, gamma: float) -> float:
        return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])

    max_step = 8
    env = Env(
        width=3,
        height=4,
        optimal_undiscounted=3,
        random_seed=0,
        hint=True,
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
            s = s_
        # breakpoint()
