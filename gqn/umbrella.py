from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import base_env
import numpy as np
from base_env import TimeStep
from gym.spaces import Discrete

REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]


@dataclass
class Env(base_env.Env[int, int]):
    num_colors: int
    num_steps: int
    random_seed: int

    def __post_init__(self):
        self.means = None
        self.rng = np.random.default_rng(seed=self.random_seed)
        self.action_space = Discrete(self.num_colors, seed=self.random_seed)

    def action_stop(self) -> str:
        return "."

    def actions(self) -> "list[str]":
        assert isinstance(self.action_space, Discrete)
        return [str(i) for i in range(self.action_space.n)]

    def done(self, *completions: str) -> bool:
        return len(completions) // 2 == self.num_steps

    def failure_threshold(self) -> float:
        return 0

    def gamma(self) -> float:
        return 1.0

    def partially_observable(self) -> bool:
        return True

    def quantify(self, prompt: str) -> float:
        return prompt.endswith("Success.")

    def render(self, mode="human"):
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        *self.observations, self.last = self.rng.choice(
            self.num_colors, size=self.num_steps + 1
        )
        self.t = 0
        self.first = self.observations[0]
        return self.first

    def start_state(self) -> np.ndarray:
        return self.first

    def start_states(self) -> Optional[Iterable[np.ndarray]]:
        return None

    @classmethod
    def _state_str(cls, obs: int) -> str:
        return COLORS[obs]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        try:
            self.t += 1
            return self.observations[self.t], 0, False, {}
        except IndexError:
            success = action == self.first
            return self.last, float(success), True, dict(optimal=1)

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += " " + REWARDS[ts.reward] + self.state_stop()
        return description


if __name__ == "__main__":
    env = Env(num_colors=2, num_steps=3, random_seed=0)
    while True:
        s = env.reset()
        t = False
        while not t:
            a = env.action_space.sample()
            s_, r, t, i = env.step(a)
            go_home = s_ == 2
            print(env.ts_to_string(TimeStep(s, a, r, t, s_)))
            s = s_
        breakpoint()
