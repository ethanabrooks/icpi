import abc
import re
from dataclasses import dataclass
from typing import Generic, Iterable, Optional

import gym
from gym.core import ActType, ObsType


@dataclass
class TimeStep(Generic[ObsType, ActType]):
    state: ObsType
    action: ActType
    reward: float
    done: bool
    next_state: ObsType


@dataclass
class Env(gym.Env[ObsType, ActType], abc.ABC):
    hint: bool

    def action(self, action_str: Optional[str]) -> Optional[ActType]:
        action_space = self.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        try:
            actions = [
                a for a in range(action_space.n) if self.action_str(a) == action_str
            ]
            [action] = actions
            return action
        except ValueError:
            return None

    @staticmethod
    @abc.abstractmethod
    def action_stop() -> str:
        ...

    @abc.abstractmethod
    def action_str(self, action: ActType) -> str:
        ...

    @abc.abstractmethod
    def actions(self) -> "list[str]":
        ...

    @abc.abstractmethod
    def done(self, *completions: str) -> bool:
        ...

    @abc.abstractmethod
    def failure_threshold(self) -> float:
        ...

    @staticmethod
    @abc.abstractmethod
    def gamma() -> float:
        ...

    @staticmethod
    def hint_stop() -> Optional[str]:
        return "\n"

    @staticmethod
    @abc.abstractmethod
    def initial_str() -> str:
        ...

    @abc.abstractmethod
    def max_trajectory(self) -> int:
        ...

    @abc.abstractmethod
    def partially_observable(self) -> bool:
        ...

    def quantify(self, prompt: str) -> float:
        matches = re.findall(r"assert reward == (\d)", prompt)
        return sum([float(x) for x in matches])

    @staticmethod
    def reward_stop() -> Optional[str]:
        return "\n"

    @staticmethod
    def state_stop() -> str:
        return "\n"

    @abc.abstractmethod
    def start_states(self) -> Optional[Iterable[ObsType]]:
        ...

    def state_str(self, state: ObsType) -> str:
        return self._state_str(state) + self.state_stop()

    @classmethod
    @abc.abstractmethod
    def _state_str(cls, state: ObsType) -> str:
        ...

    def ts_to_string(self, ts: TimeStep) -> str:
        ...

    @abc.abstractmethod
    def valid_reward(self, reward_str: str) -> bool:
        ...

    @abc.abstractmethod
    def valid_state(self, state_str: str) -> bool:
        ...
