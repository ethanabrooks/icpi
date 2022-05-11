import itertools
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep
from gym.wrappers import TimeLimit

DEAD = "dead"


class Coord(NamedTuple):
    x: int
    y: int

    def aligned_with(self, x: int) -> bool:
        return self.x == x


class Alien(NamedTuple):
    i: int
    xy: Optional[Coord]

    def dead(self) -> bool:
        return self.xy is None

    def landed(self) -> bool:
        return not self.dead() and self.xy.y == 0

    def over(self, x: int) -> bool:
        return not self.dead() and self.xy.aligned_with(x)


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

    def actions(self):
        return [
            "left",
            "shoot",
            "right",
        ]

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        if "landed" in state_or_reward or "survived" in state_or_reward:
            return True
        return False

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 1.0

    def partially_observable(self) -> bool:
        return True

    def quantify(self, prompt: str) -> float:
        return_ = 0
        for n in range(self.max_aliens):
            for permutation in itertools.permutations(
                range(1, 1 + self.max_aliens), n + 1
            ):
                search = "shot down " + " and ".join(f"A{i}" for i in permutation) + ","
                return_ += (n + 1) * prompt.count(search)

        return return_

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
        self.aliens = [
            Alien(i + 1, Coord(x, self.height)) for i, x in enumerate(alien_xs)
        ]
        return Obs(self.agent, tuple(self.aliens))

    def state_stop(self) -> str:
        return ";"

    def _state_str(self, state: Obs) -> str:
        state_str = self._state_without_status_str(state)
        status = self._status_str(state)
        if not self.status:
            return state_str
        return f"{state_str} [{status}]"

    @staticmethod
    def _status_str(state: Obs) -> str:
        in_range = [a.i for a in state.aliens if a.over(state.agent)]
        statuses = []
        if in_range:
            statuses.append("C.x=" + "=".join(f"A{i}.x" for i in in_range))
        return ", ".join(statuses)

    @staticmethod
    def _state_without_status_str(state: Obs) -> str:
        return ", ".join(
            [f"C={(state.agent, 0)}"]
            + [f"A{i}={tuple(pos)}" for i, *pos in state.aliens]
        )

    def start_states(self) -> Optional[Iterable[Obs]]:
        for agent in range(self.width):
            for xs in itertools.product(range(self.width), repeat=self.max_aliens):
                aliens = [Alien(i + 1, Coord(x, self.height)) for i, x in enumerate(xs)]
                yield Obs(agent, tuple(aliens))

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        reward_ = self.reward == 0
        max_aliens = len(self.aliens) < self.max_aliens
        choice = self.random.choice(2)
        if reward_ and max_aliens and choice:
            alien_ids = [i for i, *_ in self.aliens]
            alien_id = next(
                i for i in range(1, 1 + self.max_aliens) if i not in alien_ids
            )
            self.aliens.append(
                Alien(alien_id, Coord(self.random.choice(self.width), self.height))
            )
        # else:
        #     if not reward_:
        #         print(f"reward={self.reward}")
        #     elif not max_aliens:
        #         print(f"self.aliens={self.aliens}")
        #     elif not choice:
        #         print(f"choice={choice}")

        if action == 1:
            num_aliens = len(self.aliens)
            self.aliens = [
                Alien(i, Coord(xy.x, xy.y))
                for i, xy in self.aliens
                if xy.x != self.agent
            ]
            self.reward = num_aliens - len(self.aliens)
        else:
            self.reward = 0
        self.aliens = [Alien(i, Coord(xy.x, xy.y - 1)) for i, xy in self.aliens]
        info = dict(optimal=self.max_step)
        self.agent += action - 1
        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        done = any(a.landed() for a in self.aliens)
        state = Obs(self.agent, tuple(self.aliens))
        return state, self.reward, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.action == 1:
            shot = [a.i for a in ts.state.aliens if a.x == ts.state.agent]
            if shot:
                shot = " and ".join(f"A{i}" for i in shot)
                description += f" shot down {shot},"
            else:
                description += " missed,"
        if ts.done:
            description += f" {self._state_without_status_str(ts.next_state)}"
            landed = [a.i for a in ts.next_state.aliens if a.y == 0]
            landed = " and ".join(f"A{i}" for i in landed)
            if landed:
                description += f" [{landed} landed]"
            else:
                description += " [survived]"
            description += self.state_stop()
        return description


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
