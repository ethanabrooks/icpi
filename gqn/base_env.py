import abc
import re
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
    def done(self, done_str: str) -> bool:
        ...

    @staticmethod
    @abc.abstractmethod
    def done_stop() -> str:
        ...

    @abc.abstractmethod
    def done_str(self, done: bool) -> str:
        ...

    @abc.abstractmethod
    def failure_threshold(self) -> float:
        ...

    @staticmethod
    @abc.abstractmethod
    def gamma() -> float:
        ...

    def hint_stop(self) -> Optional[str]:
        return "\n" if self.hint else None

    @staticmethod
    @abc.abstractmethod
    def initial_str() -> str:
        ...

    @classmethod
    def log_gamma(cls) -> float:
        return 1.0

    @abc.abstractmethod
    def max_q_steps(self) -> int:
        ...

    def quantify(self, prompt: str, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.gamma()
        matches = re.findall(r"reward == (\d)", prompt)
        return sum([gamma**t * float(x) for t, x in enumerate(matches)])

    @abc.abstractmethod
    def reward_str(self, reward: float) -> str:
        ...

    @staticmethod
    def reward_stop() -> str:
        return "\n"

    @staticmethod
    def state_stop() -> str:
        return "\n"

    @abc.abstractmethod
    def start_states(self) -> Optional[Iterable[ObsType]]:
        ...

    @abc.abstractmethod
    def state_str(self, state: ObsType) -> str:
        ...

    def termination_str(self, ts: TimeStep) -> str:
        return self.state_str(ts.next_state)

    def ts_to_string(self, ts: TimeStep) -> str:
        ...

    @abc.abstractmethod
    def valid_done(self, done_str: str) -> bool:
        ...

    @abc.abstractmethod
    def valid_reward(self, reward_str: str) -> bool:
        ...

    @abc.abstractmethod
    def valid_state(self, state_str: str) -> bool:
        ...
