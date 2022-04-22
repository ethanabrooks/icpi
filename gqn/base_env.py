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
            return self.actions().index(action_str)
        except ValueError:
            return None

    @abc.abstractmethod
    def actions(self) -> "list[str]":
        ...

    def action_str(self, action: ActType) -> str:
        return self.actions()[action]

    @abc.abstractmethod
    def done(self, state_or_reward: str) -> bool:
        ...

    @abc.abstractmethod
    def quantify(self, value: str, gamma: Optional[float]) -> float:
        ...

    @abc.abstractmethod
    def state_str(self, state: ObsType) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def successor_feature(state: ObsType) -> np.ndarray:
        ...

    @abc.abstractmethod
    def time_out_str(self) -> str:
        ...

    @abc.abstractmethod
    def ts_to_string(self, ts: TimeStep) -> str:
        ...
