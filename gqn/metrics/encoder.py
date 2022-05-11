import abc
from typing import Generic, List

from base_env import TimeStep
from gym.core import ActType, ObsType


class Encoder(Generic[ObsType, ActType], abc.ABC):
    def action(self, action_str: str):
        matches = (
            a for a in self.actions() if self.action_str(a).rstrip(".:") == action_str
        )
        return next(matches, None)

    @abc.abstractmethod
    def actions(self):
        ...

    @abc.abstractmethod
    def action_query(self, state: ObsType) -> str:
        ...

    @abc.abstractmethod
    def action_str(self, state: ObsType, action: int) -> str:
        ...

    def name(self) -> str:
        ...

    @abc.abstractmethod
    def nonterminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
        ...

    @abc.abstractmethod
    def reward_query(self, ts: TimeStep[ObsType, ActType]) -> str:
        ...

    @abc.abstractmethod
    def state_str(self, state: ObsType) -> str:
        ...

    @abc.abstractmethod
    def stop(self) -> List[str]:
        ...

    @abc.abstractmethod
    def terminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
        ...

    def time_step_str(self, ts: TimeStep[ObsType, ActType]) -> str:
        ...

    @abc.abstractmethod
    def transition_query(self, ts: TimeStep[ObsType, ActType]) -> str:
        ...

    def get_prompt(
        self,
        trajectories: "list[list[TimeStep]]",
    ) -> str:
        return "\n".join(
            [
                "\n".join([self.time_step_str(ts) for ts in trajectory])
                for trajectory in trajectories
            ]
        )
