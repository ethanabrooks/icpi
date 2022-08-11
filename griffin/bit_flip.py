from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env

ObsType = Tuple[int, ...]


@dataclass
class Env(base_env.Env[ObsType, int]):
    time_steps: int

    def action_str(self, action: int) -> str:
        action_str = ["stay", "flip"][action]
        return f"reward = {action_str}(state){self.action_stop()}"

    def failure_threshold(self) -> float:
        return self.time_steps - 1

    @staticmethod
    def initial_str() -> str:
        return "assert state == reset()"

    def max_q_steps(self) -> int:
        return 8

    def start_states(self) -> Optional[Iterable[ObsType]]:
        return None

    def state_str(self, state: ObsType) -> str:
        pass

    def valid_done(self, done_str: str) -> bool:
        pass

    def valid_reward(self, reward_str: str) -> bool:
        pass

    def valid_state(self, state_str: str) -> bool:
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
