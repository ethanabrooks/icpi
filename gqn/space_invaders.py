import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep
from gym.core import ObsType

DEAD = "dead"

Obs = Tuple[Tuple[int, int], ...]


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
            "Left",
            "shoot",
            "Right",
        ]

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        if DEAD in state_or_reward.rstrip(self.state_stop()):
            return True  # dead
        if len(completions) // 2 > self.max_step:
            return True  # survived
        return False

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 1.0

    def partially_observable(self) -> bool:
        return True

    @classmethod
    def quantify(cls, prompt: str) -> float:
        return prompt.count(cls.action_stop())

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
        self.agent, *alien_xs = self.random.choice(self.width, size=1 + num_aliens)
        self.aliens = [(x, self.height) for x in alien_xs]
        return tuple([(self.agent, 0)] + self.aliens)

    def state_stop(self) -> str:
        return ";"

    def _state_str(self, state: List[Tuple[int, int]]) -> str:
        state_str = self._state_without_status_str(state)
        if not self.status:
            return state_str
        status = self._status_str(state)
        return f"{state_str} [{status}]"

    @staticmethod
    def _status_str(state: List[Tuple[int, int]]) -> str:
        _, *aliens = state
        alive = all([y > 0 for x, y in aliens])
        return ", ".join(
            ["A.y>0" if y > 0 else "A.y=0" for _, y in aliens]
            + ["alive" if alive else DEAD]
        )

    @staticmethod
    def _state_without_status_str(state: List[Tuple[int, int]]) -> str:
        you, *aliens = state
        return f"C={you}, " + ", ".join(f"A={x}" for x in aliens)

    def start_states(self) -> Optional[Iterable[ObsType]]:
        for n_aliens in range(self.max_aliens):
            for agent in range(self.width):
                for xs in itertools.product(range(self.width), repeat=n_aliens):
                    yield tuple([(agent, 0)] + [(x, self.height) for x in xs])

    def step(self, action: int) -> Tuple[Obs, float, bool, dict]:
        if len(self.aliens) < self.max_aliens and self.random.choice(2):
            self.aliens.append((self.random.choice(self.width), self.height))

        if action == 1:
            self.aliens = [(x, y) for x, y in self.aliens if x != self.agent]
        self.aliens = [(x, y - 1) for x, y in self.aliens]
        info = dict(optimal=self.max_step)

        self.agent = int(np.clip(self.agent, 0, self.width - 1))
        done = any(y == 0 for x, y in self.aliens)
        state = [(self.agent, 0)] + self.aliens
        return tuple(state), 1.0, done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += f" {self.state_str(ts.next_state)}"
        return description
