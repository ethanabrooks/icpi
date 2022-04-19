import abc
from typing import Optional

import gym
from gym.core import ActType, ObsType


class Env(gym.Env[ObsType, ActType], abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def action(action_str: str) -> int:
        ...

    @staticmethod
    @abc.abstractmethod
    def action_str(action: int) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def default_reward_str() -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def done(state_or_reward: str) -> bool:
        ...

    @classmethod
    @abc.abstractmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        ...

    @classmethod
    @abc.abstractmethod
    def reward_str(cls, reward: float, next_state: Optional[str]) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def state_str(state: int) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def success_str() -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def value_str(cls, value: float) -> str:
        ...
