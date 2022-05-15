import abc
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

import gym

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TimeStep(Generic[ObsType, ActType]):
    state: ObsType
    action: ActType
    reward: float
    done: bool
    next_state: ObsType


@dataclass
class Env(gym.Env, Generic[ObsType, ActType], abc.ABC):
    def action(self, action_str: Optional[str]) -> Optional[ActType]:
        try:
            actions = [
                a
                for a, _action_str in enumerate(self.actions())
                if _action_str == action_str
            ]
            [action] = actions
            return action
        except ValueError:
            return None

    @staticmethod
    def action_stop() -> str:
        return ":"

    def action_str(self, action: ActType) -> str:
        try:
            return self.actions()[action] + self.action_stop()
        except IndexError:
            breakpoint()

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

    @abc.abstractmethod
    def quantify(self, value: str) -> float:
        ...

    @staticmethod
    def state_stop() -> str:
        return "."

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
