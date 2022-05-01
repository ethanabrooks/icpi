from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, Union

import envs.base_env
import gym
import numpy as np
from envs.base_env import TimeStep
from gym.core import ActType, ObsType

REWARDS = {
    1.0: "Success",
    0.0: "Failure",
}


@dataclass
class Env(envs.base_env.Env[int, int]):
    gamma: float
    max_steps: int
    random_seed: int

    def __post_init__(self):
        self.random = np.random.default_rng(self.random_seed)
        self.action_space = gym.spaces.Discrete(
            len(self.actions()), seed=self.random_seed
        )
        self.observation_space = gym.spaces.Discrete(len(self.states()))

    def done(self, state_or_reward: str) -> bool:
        return state_or_reward in REWARDS.values()

    @classmethod
    def quantify(cls, value: str, gamma: Optional[float]) -> float:
        success = value.endswith(REWARDS[1.0])
        value = gamma ** value.count(".")
        return value if success else (gamma - 1) * value

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return self.iterator.send(action)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def render(self, mode="human"):
        pass

    def actions(self):
        return ["Look busy.", "Relax.", "Leave Umbrella.", "Take umbrella."]

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        n_chain = self.random.integers(2, self.max_steps)
        weather_step = self.random.choice(n_chain)
        need_umbrella = self.random.choice(2)
        optimal = self.gamma ** n_chain
        done = False
        overworked = False
        reward = 0.0
        info = dict(optimal=optimal)
        go_home_state = 2
        for i in range(n_chain - 1):
            state = self.random.choice(2)
            if i == weather_step:
                state = go_home_state + need_umbrella + 1
            action = yield state, reward, done, info
            if state == 0 and action != 0:  # boss is in and you were not looking busy
                done = True
            elif state == 1 and action != 1:  # you worked unnecessarily
                overworked = True

        action = yield go_home_state, reward, done, info
        reward = 0 if overworked else ((action - 2) == need_umbrella)
        yield go_home_state, reward, True, info

    def starting_states(self) -> Iterable[ObsType]:
        raise RuntimeError()

    @classmethod
    def state_str(cls, state: int) -> str:
        return cls.states()[state]

    @staticmethod
    def states():
        return ["Boss is in.", "Boss is out.", "Go home.", "Sunny.", "Raining."]

    def successor_feature(self, state: "int | tuple[int, int]") -> np.ndarray:
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        one_hot = np.zeros(self.observation_space.n)
        one_hot[state] = 1
        return one_hot

    def ts_to_string(self, ts: TimeStep) -> str:
        description = f"{self.state_str(ts.state)} {self.action_str(ts.action)}"
        if ts.done:
            description += " " + REWARDS[ts.reward] + self.state_stop()
        return description


if __name__ == "__main__":
    env = Env(gamma=0.99, max_steps=8, random_seed=0)
    while True:
        s = env.reset()
        t = False
        go_home = False
        while not t:
            if go_home:
                a = 2 + env.random.choice(2)
            else:
                a = env.random.choice(2)
            if a == 1:
                selected_goal = True
            s_, r, t, i = env.step(a)
            go_home = s_ == 2
            print(env.ts_to_string(TimeStep(s, a, r, t, s_)))
            s = s_
        breakpoint()
