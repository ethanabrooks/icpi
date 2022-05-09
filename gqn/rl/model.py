import abc
from dataclasses import dataclass
from typing import Deque, Generic, List, Union

import numpy as np
from base_env import Env, TimeStep
from gym.core import ActType, ObsType
from gym.spaces import Discrete
from numpy.linalg import norm
from numpy.random import Generator
from rl.gpt3 import GPT3
from rl.huggingface import HuggingFaceModel


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
    debug: int
    failure_threshold: float
    gamma: float
    lm: Union[GPT3, HuggingFaceModel]
    max_steps: int
    prompt_size: int
    rng: Generator
    success_buffer: Deque[List[TimeStep]]

    def act(self, state: ObsType) -> ActType:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: ObsType) -> ActType:
        ...

    def get_value(self, trajectory: List[TimeStep]) -> float:
        return get_value(*trajectory, gamma=self.gamma)

    def ready(self) -> bool:
        return len(self.success_buffer) >= 1

    def sample(self):
        successful = [
            t for t in self.buffer if self.get_value(t) > self.failure_threshold
        ]
        unsuccessful = [
            t for t in self.buffer if self.get_value(t) <= self.failure_threshold
        ]
        half1 = self.prompt_size // 2
        half2 = self.prompt_size - half1
        successful_choices = [
            successful[i]
            for i in self.rng.choice(
                len(successful), min(half1, len(successful)), replace=False
            )
        ]
        unsuccessful_choices = [
            unsuccessful[i]
            for i in self.rng.choice(
                len(unsuccessful), min(half2, len(unsuccessful)), replace=False
            )
        ]
        trajectories = successful_choices + unsuccessful_choices
        if all(
            [
                len(successful_choices) < len(successful),
                len(unsuccessful_choices) < len(unsuccessful),
            ]
        ):
            assert len(trajectories) == self.prompt_size
        return [to_string(*t, env=self.env) for t in trajectories]

    def sample_best(self):
        trajectories = list(self.success_buffer)
        for trajectory in list(trajectories):
            while trajectory:
                _, *trajectory = trajectory
                if trajectory:
                    trajectories.append(trajectory)
        trajectories = [
            trajectories[i]
            for i in self.rng.choice(len(trajectories), self.prompt_size)
        ]
        assert len(trajectories) == self.prompt_size
        return [to_string(*t, env=self.env) for t in trajectories]


@dataclass
class Q(Model[ObsType, ActType]):
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

    def value(self, state: ObsType, action: ActType) -> str:
        t = 0
        state_str = self.env.state_str(state)
        action_str = self.env.action_str(action)
        completions = [state_str, action_str]

        while True:
            if t == self.max_steps:
                break
            else:
                prompts = self.sample()
                new_prompt = "\n".join([*prompts, " ".join(completions)])
                if self.debug >= 2:
                    print("Q prompt:")
                    print(new_prompt)
                if self.debug >= 4:
                    breakpoint()

                state_or_reward, *_ = self.lm(new_prompt).split(self.env.state_stop())
                state_or_reward = state_or_reward.lstrip() + self.env.state_stop()
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
            if self.debug >= 4:
                breakpoint()

            action_str, *_ = self.lm(new_prompt).split(self.env.state_stop())
            action_str = action_str.lstrip() + self.env.action_stop()
            t += 1

            if self.debug >= 2:
                print("action", action_str)
            if self.debug >= 4:
                breakpoint()
            completions.append(action_str)

        return " ".join(completions)


class Pi(Model[ObsType, ActType]):
    def _act(self, state: ObsType) -> ActType:
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
            maybe_action, *_ = self.lm(prompt).lstrip().split(self.env.action_stop())
            if self.debug >= 1:
                print("Action:", maybe_action)
            if self.debug >= 3:
                breakpoint()

            action = self.env.action(maybe_action)
            t += 1

        return action
