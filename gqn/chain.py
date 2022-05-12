from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env
import gym
import gym.spaces
import numpy as np
from base_env import TimeStep

REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


@dataclass
class Env(base_env.Env[int, int]):
    goal: int
    n: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )
        self.observation_space = gym.spaces.Discrete(self.n)

    def action_str(self, action: int) -> str:
        return f"state, reward = {self.actions()[action]}(state){self.action_stop()}"

    def actions(self):
        return [
            "left",
            "try_goal",
            "right",
        ]

    def done(self, *completions: str) -> bool:
        *_, state_or_reward = completions
        breakpoint()
        return "try_goal" in state_or_reward

    def failure_threshold(self) -> float:
        return 0

    @staticmethod
    def gamma() -> float:
        return 0.9

    def _hint_str(self, state: int) -> str:
        return (
            "assert state " + ("==" if state == self.goal else "!=") + f" {self.goal}"
        )

    @classmethod
    def initial_str(cls) -> str:
        return "state = reset()\n"

    def partially_observable(self) -> bool:
        return False

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> int:
        self._state = self._start_state = self.random.choice(self.n)
        return self._start_state

    def start_states(self) -> Optional[Iterable[int]]:
        return range(self.n)

    def _state_str(self, state: int) -> str:
        return f"assert state == {state}"

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        optimal = self.gamma() ** abs(self._start_state - self.goal)
        info = dict(optimal=optimal)
        self._state += action - 1
        self._state = np.clip(self._state, 0, self.n - 1)
        done = action == 1
        success = done and self._state == self.goal
        state = int(self._state)
        return state, float(success), done, info

    def ts_to_string(self, ts: TimeStep) -> str:
        parts = [
            self.state_str(ts.state),
            *([self._hint_str(ts.state), self.hint_stop()] if self.hint else []),
            self.action_str(ts.action),
            f"assert reward == {ts.reward}",
            self.reward_stop(),
        ]
        s = "".join(parts)
        if ts.reward == 1 and f"state == {self.goal}" not in s:
            breakpoint()
        if ts.action == 1 and ts.reward == 0 and f"state != {self.goal}" in s:
            breakpoint()
        return s
