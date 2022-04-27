import abc
from dataclasses import dataclass
from typing import Generic, Optional

import gym
import numpy as np
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
            [action] = [
                a
                for a, _action_str in enumerate(self.actions())
                if _action_str == (action_str + _action_str[-1])
            ]
            return action
        except ValueError:
            return None

    def action_str(self, action: ActType) -> str:
        try:
            return self.actions()[action]
        except IndexError:
            breakpoint()

    @abc.abstractmethod
    def actions(self) -> "list[str]":
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

    def successor_feature(self, state: int) -> np.ndarray:
        ...

    def ts_to_string(self, ts: TimeStep) -> str:
        ...
