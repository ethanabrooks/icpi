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

    def descend(self):
        return Alien(self.x, self.y - 1)

    def landed(self) -> bool:
        return self.y == 0

    def over(self, x: int) -> bool:
        return self.x == x

    @staticmethod
    def spawn(x: int, y: int) -> "Alien":
        return Alien(x, y)

    def take_fire(self, agent: int) -> Optional["Alien"]:
        if self.over(agent):
            return None
        else:
            return self


class Obs(NamedTuple):
    agent: int
    alien: Alien


@dataclass
class Env(base_env.Env[Obs, int]):
    height: int
    max_return: int
    random_seed: int
    width: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )

    def actions(self):
        return [
            "Left",
            "Shoot",
            "Right",
        ]

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        if (
            bool(re.findall(r"A=\(\d+, 0\)", state_or_reward))
            or self.quantify(" ".join(completions), gamma=1) >= self.max_return
        ):
            return True
        return False

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.8

    def quantify(self, prompt: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.gamma()
        _, *steps = prompt.split(":")
        discounted_rewards = [
            gamma**t * float("got 1" in p) for t, p in enumerate(steps)
        ]
        return sum(discounted_rewards)

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
        self.alien = Alien(alien_x, self.height)
        self.r = 0
        self.t = 0
        return Obs(self.agent, self.alien)

    def reward_stop(self) -> Optional[str]:
        return "}"

    def state_stop(self) -> str:
        return ";"

    def _state_str(self, state: Obs) -> str:
        state_str = self._state_without_status_str(state)
        status = self._status_str(state)
        if not self.hint:
            return state_str
        return f"{state_str} [{status}]"

    @staticmethod
    def _status_str(state: Obs) -> str:

        return ", ".join(
            [
                "S.x" + ("==" if state.agent == state.alien.x else "!=") + "A.x",
                "A.y" + ("==" if 0 == state.alien.y else "!=") + "0",
            ]
        )

    @staticmethod
    def _state_without_status_str(state: Obs) -> str:
        return f"S={(state.agent, 0)}, A={tuple(state.alien)}"

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent in range(self.width):
            for x in range(self.width):
                yield Obs(agent, Alien(x, self.height))

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if action == 1:
            reward = float(self.alien.over(self.agent))
            self.r += reward
            self.alien = self.alien.take_fire(self.agent)
        else:
            reward = 0

        if self.alien is None:
            self.alien = Alien.spawn(self.random.choice(self.width), self.height)

        self.t += 1
        self.alien = self.alien.descend()
        info = dict(optimal=self.max_return)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        max_return = self.r >= self.max_return
        done = self.alien.landed() or max_return
        # print(f"landed={landed}, return={self.r}, done={done}")
        state = Obs(self.agent, self.alien)
        return state, reward, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.action == 1:
            if ts.state.alien.over(ts.state.agent):
                description += " {got 1}"
            else:
                description += " {missed}"
        else:
            description += " {didn't shoot}"
        if ts.done:
            description += f" {self._state_str(ts.next_state)}"
        return description


if __name__ == "__main__":
    max_return = 8
    env = TimeLimit(
        Env(
            width=3,
            height=4,
            max_return=3,
            random_seed=0,
            hint=True,
        ),
        max_episode_steps=max_return,
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
