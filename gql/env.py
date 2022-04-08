import gym
import math


class Env(gym.Env):
    goal: int
    n: int

    def generator(self):
        state = self.random.choice(self.n)
        info = {}
        yield state
        while True:
            success = state == self.goal
            reward = float(success)
            done = success
            action = yield state, reward, done, info
            state += action
            state = min(state, self.n - 1)
            state = max(state, 0)

    def reset(self) -> int:
        self.iterator = self.generator()
        return next(self.iterator)

    def step(self, action: int) -> (int, float, bool, dict):
        return next(self.iterator)
