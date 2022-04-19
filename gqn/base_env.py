import abc
from typing import Optional

import gym
import numpy as np
from gym.core import ActType, ObsType


class Env(gym.Env[ObsType, ActType], abc.ABC):
    @classmethod
    def action(cls, action_str: str) -> Optional[ActType]:
        try:
            return cls.actions().index(action_str)
        except ValueError:
            return None

    @staticmethod
    @abc.abstractmethod
    def actions() -> "list[str]":
        ...

    @classmethod
    def action_str(cls, action: ActType) -> str:
        return cls.actions()[action]

    @staticmethod
    @abc.abstractmethod
    def time_out_str() -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def done(state_or_reward: str) -> bool:
        ...

    @classmethod
    @abc.abstractmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        ...

    @staticmethod
    @abc.abstractmethod
    def rewards() -> "dict[float, str]":
        ...

    @classmethod
    def reward_str(cls, reward: float, next_state: Optional[str]) -> str:
        return cls.rewards()[reward] if next_state is None else ""

    @staticmethod
    @abc.abstractmethod
    def state_str(state: ObsType) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def successor_feature(state: ObsType) -> np.ndarray:
        ...
