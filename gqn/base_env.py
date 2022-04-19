import abc
from typing import Optional

import gym
import numpy as np
from gym.core import ActType, ObsType


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
    def longest_reward(self):
        ...

    @abc.abstractmethod
    def quantify(self, value: str, gamma: Optional[float]) -> float:
        ...

    @abc.abstractmethod
    def _reward_str(self, reward: float) -> "str":
        ...

    def reward_str(self, reward: float, next_state: Optional[str]) -> str:
        return self._reward_str(reward) if next_state is None else ""

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
