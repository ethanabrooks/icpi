import abc
import re
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

import gym
from rl.lm import Data

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
    data: Data
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

    def done(self, done_str: str) -> bool:
        return "assert done" in done_str

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
        if not self.hint:
            return None
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

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
        if self.data == Data.code:
            matches = re.findall(r"reward == (\d+)", prompt)
        elif self.data == Data.natural_language:
            matches = re.findall(r"Receive (\d+)", prompt)
        else:
            raise RuntimeError("Invalid data")
        return sum([gamma**t * float(x) for t, x in enumerate(matches)])

    @staticmethod
    def reward(reward_str: str) -> float:
        matches = re.findall(r"reward == (\d+)", reward_str)
        try:
            [reward] = matches
        except ValueError:
            return 0.0
        return float(reward)

    @abc.abstractmethod
    def reward_str(self, reward: float) -> str:
        ...

    def reward_stop(self) -> str:
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

    def state_stop(self) -> str:
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

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

    @staticmethod
    def valid_transition(transition_str: str) -> bool:
        return (
            transition_str.startswith("assert")
            and "reward" in transition_str
            and "done" in transition_str
        )

    def transition_stop(self) -> str:
        return "done" + self.done_stop()
