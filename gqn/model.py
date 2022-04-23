import abc
from dataclasses import dataclass
from typing import Deque, List

import numpy as np
from base_env import Env, TimeStep
from gpt3 import GPT3
from gym.spaces import Discrete
from numpy.linalg import norm
from numpy.random import Generator


def to_string(*_trajectory: TimeStep, env) -> str:
    return " ".join([env.ts_to_string(ts) for ts in _trajectory])


def get_value(*trajectory: TimeStep, gamma: float) -> float:
    return sum([gamma ** t * ts.reward for t, ts in enumerate(trajectory)])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


@dataclass
class Model(abc.ABC):
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    delta: float
    failure_threshold: float
    gamma: float
    gpt3: GPT3
    max_steps: int
    prompt_size: int
    rng: Generator

    def act(self, state: int) -> int:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: int) -> int:
        ...

    def get_good(self):
        return [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]

    def ready(self) -> bool:
        return len(self.buffer) >= self.prompt_size

    def sample(self):
        prompts = [to_string(*t, env=self.env) for t in self.buffer]
        self.rng.shuffle(prompts)
        return prompts[: self.prompt_size]

    def sample_best(self):
        trajectories = sorted(
            self.get_good(), key=lambda t: get_value(*t, gamma=self.gamma), reverse=True
        )
        unique = dict()

        for trajectory in trajectories:
            if len(unique) == self.prompt_size:
                break

            def successor_representation(
                *trajectory: TimeStep, gamma: float
            ) -> np.ndarray:
                representation = 0
                for t, ts in enumerate(trajectory):
                    representation += gamma ** t * self.env.successor_feature(ts.state)
                assert isinstance(representation, np.ndarray)
                return representation

            rep1 = successor_representation(*trajectory, gamma=self.gamma)
            different = True
            for rep2 in unique.values():
                if cosine_similarity(rep1, rep2) > self.delta:
                    different = False
                    break
            if different:
                prompt = to_string(*trajectory, env=self.env)
                unique[prompt] = rep1

        prompts = list(unique)
        self.rng.shuffle(prompts)
        return prompts


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    def _act(self, state) -> int:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_values():
            for a in actions:
                yield self.value(state, action=a)

        values = list(get_values())
        action_values = list(zip(actions, values))
        self.rng.shuffle(action_values)
        action, value = max(
            action_values,
            key=lambda x: (
                self.env.quantify(x[1], gamma=self.gamma),
                self.rng.random(),
            ),
        )

        if self.debug >= 1:
            print("Q")
            print("state", state)
            for a, v in zip(actions, values):
                print("action", a)
                print("value", v)
            print("chosen", action)
        if self.debug >= 3:
            breakpoint()
        return action

    def value(self, state, action: int) -> str:
        t = 0
        action_str = self.env.action_str(action)
        state_str = self.env.state_str(state)
        completions = [state_str, action_str]

        while True:
            if t == self.max_steps:
                state_or_reward = (
                    self.env.time_out_str()
                )  # TODO: can we eliminate this?
                completions.append(state_or_reward)
                break
            else:
                prompts = self.sample()
                new_prompt = "\n".join([*prompts, f"{state_str} {action_str}"])
                if self.debug >= 2:
                    print("Q prompt:")
                    print(new_prompt)

                state_or_reward, *_ = self.gpt3(new_prompt).lstrip().split(".")
                state_or_reward = reformat(state_or_reward)
            if self.debug >= 2:
                print("state/reward", state_or_reward)
            if self.debug >= 4:
                breakpoint()
            completions.append(state_or_reward)
            if self.env.done(state_or_reward):
                break
            state_str = state_or_reward
            prompts = self.sample_best()
            new_prompt = "\n".join([*prompts, state_str])
            if self.debug >= 2:
                print("Q prompt:")
                print(new_prompt)

            action_str, *_ = self.gpt3(new_prompt).lstrip().split(".")
            action_str = reformat(action_str)
            t += 1
            if self.debug >= 2:
                print("action", action_str)
            if self.debug >= 4:
                breakpoint()
            completions.append(action_str)

        return " ".join(completions)


class Pi(Model):
    def _act(self, state) -> int:
        state = self.env.state_str(state)
        action = None
        t = 0
        while action is None:
            if t > self.max_steps:
                return self.env.action_space.sample()
            prompts = self.sample_best()
            prompt = "\n".join([*prompts, state])
            if self.debug >= 1:
                print("pi prompt:")
                print(prompt)
            completion = self.gpt3(prompt).lstrip()
            maybe_action, *_ = completion.split(".")
            if self.debug >= 1:
                print("Action:", maybe_action)
            if self.debug >= 3:
                breakpoint()

            action = self.env.action(maybe_action + ".")
            t += 1

        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
