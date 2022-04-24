import abc
from dataclasses import dataclass
from typing import Deque, Generic, List

import numpy as np
from base_env import Env, TimeStep
from gpt3 import GPT3
from gym.core import ActType, ObsType
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
class Model(abc.ABC, Generic[ObsType, ActType]):
    buffer: Deque[List[TimeStep]]
    env: Env
    delta: float
    failure_threshold: float
    gamma: float
    gpt3: GPT3
    prompt_size: int
    rng: Generator
    debug: bool

    def act(self, state: ObsType) -> ActType:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: ObsType) -> ActType:
        ...

    def get_good(self):
        return [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

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
class Q(Model[ObsType, ActType]):
    max_steps: int

    def _act(self, state: ObsType) -> ActType:
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

        self.print("Q")
        self.print("state", state)
        for a, v in zip(actions, values):
            self.print("action", a)
            self.print("value", v)
        self.print("chosen", action)
        if self.debug:
            breakpoint()
        return action

    def value(self, state: ObsType, action: ActType) -> str:
        # original_state = state
        # original_action = action
        completions = []
        state = self.env.state_str(state)
        action = self.env.action_str(action)
        trajectories = self.sample()
        new_prompt = "\n".join([*trajectories, f"{state} {action}"])
        # print("Q prompt:")
        # print(new_prompt)

        state_or_reward, action, *_ = self.gpt3(new_prompt).lstrip().split(".")
        state_or_reward, action = map(reformat, [state_or_reward, action])
        # print("state/reward", state_or_reward)
        # print("action", action)
        completions.append(state_or_reward)
        t = 1

        while not self.env.done(state_or_reward):
            state = state_or_reward
            trajectories = self.sample_best()

            new_prompt = "\n".join([*trajectories, state])
            # print("Q prompt:")
            # print(new_prompt)

            # print(f"{state} {action}", end=" :: ")
            completion = self.gpt3(new_prompt).lstrip()
            action, state_or_reward, *_ = completion.split(".")
            action, state_or_reward = map(reformat, [action, state_or_reward])
            if t == self.max_steps:
                state_or_reward = (
                    self.env.default_reward_str()
                )  # TODO: can we eliminate this?
            t += 1
            # print("action", action)
            # print("state/reward", state_or_reward)
            completions.extend([action, state_or_reward])

        return " ".join(completions)


class Pi(Model[ObsType, ActType]):
    def _act(self, state: ObsType) -> ActType:
        state = self.env.state_str(state)
        action = None
        while action is None:
            trajectories = self.sample_best()
            prompt = "\n".join([*trajectories, state])
            self.print("pi prompt:")
            self.print(prompt)
            completion = self.gpt3(prompt).lstrip()
            maybe_action, *_ = completion.split(".")
            self.print("Action:", maybe_action)
            if self.debug:
                breakpoint()

            action = self.env.action(maybe_action + ".")

        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
