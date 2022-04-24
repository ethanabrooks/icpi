import itertools
from dataclasses import dataclass
from typing import Generator, Tuple

import base_env
import numpy as np
from chain import Chain

REWARDS = {
    1.0: "Success.",
    0.0: "Failure.",
}


@dataclass
class Umbrella(Chain):
    def actions(self):
        return [*super().actions(), "Leave Umbrella.", "Take umbrella."]

    def generator(self) -> Generator[Tuple[int, float, bool, dict], int, None]:
        iterator = super().generator()
        state, reward, done, info = next(iterator)
        min_steps = abs(state - self.goal)
        need_umbrella = self.random.choice(2)
        weather_step = self.random.choice(min_steps + 1)
        for i in itertools.count():
            state = (state, need_umbrella) if weather_step == i else state
            if done:
                state = bool(reward)
            action = yield state, reward, False, info
            if done:
                break
            state, reward, done, info = iterator.send(action)
        took_correct_umbrella_action = action > 3 and (action - 3 == need_umbrella)
        reward *= took_correct_umbrella_action
        yield state, reward, True, info

    @classmethod
    def state_str(cls, state: "int | str | tuple[int, int]") -> str:
        if isinstance(state, bool):
            return "Correct goal." if state else "Incorrect goal."
        if isinstance(state, tuple):
            _, need_umbrella = state
            return f"{state[0]} ({['sunny', 'started raining'][need_umbrella]})."
        return super().state_str(state)

    def successor_feature(self, state: "int | tuple[int, int]") -> np.ndarray:
        one_hot = np.zeros(self.n + 2)
        one_hot[state] = 1
        if isinstance(state, tuple):
            _, need_umbrella = state
            one_hot[-(need_umbrella + 1)] = 1
        return one_hot


if __name__ == "__main__":
    env = Umbrella(gamma=0.99, goal=1, n=3, random_seed=0)
    while True:
        s = env.reset()
        t = False
        selected_goal = False
        while not t:
            if selected_goal:
                a = 3 + env.random.choice(2)
            else:
                a = env.random.choice(3)
            if a == 1:
                selected_goal = True
            s_, r, t, i = env.step(a)
            print(env.ts_to_string(base_env.TimeStep(s, a, r, t, s_)))
            s = s_
        breakpoint()
