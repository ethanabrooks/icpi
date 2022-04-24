import abc
from dataclasses import dataclass
from typing import Generic, Optional

import gym
from gym.core import ActType, ObsType


@dataclass
class TimeStep(Generic[ObsType, ActType]):
    state: ObsType
    action: ActType
    reward: float
    done: bool
    next_state: ObsType


class Env(gym.Env[ObsType, ActType], abc.ABC):
    def action(self, action_str: str) -> Optional[ActType]:
        try:
            return self.actions().index(action_str)
        except ValueError:
            return None

    def action_str(self, action: ActType) -> str:
        return self.actions()[action]

    @abc.abstractmethod
    def actions(self) -> "list[str]":
        ...

    @staticmethod
    @abc.abstractmethod
    def default_reward_str() -> str:
        ...

    @abc.abstractmethod
    def done(self, state_or_reward: str) -> bool:
        ...

    @classmethod
    @abc.abstractmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        ...

    @staticmethod
    @abc.abstractmethod
    def state_str(state: ObsType) -> str:
        ...

    def ts_to_string(self, ts: TimeStep) -> str:
        ...
