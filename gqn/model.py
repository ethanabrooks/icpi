import abc
from dataclasses import dataclass
from typing import Deque, Generic, List

import numpy as np
from base_env import Env
from gpt3 import GPT3
from gym.core import ActType, ObsType
from gym.spaces import Discrete
from numpy.linalg import norm
from numpy.random import Generator


@dataclass
class TimeStep(Generic[ObsType, ActType]):
    state: ObsType
    action: ActType
    reward: float
    done: bool
    next_state: ObsType


def to_string(*_trajectory: TimeStep, env) -> str:

    if not _trajectory:
        return ""
    head, *tail = _trajectory
    if head.done:
        reward_str = env.reward_str(
            head.reward, done=head.done, next_state=head.next_state
        )
    else:
        reward_str = ""

    tail_trajectory = to_string(*tail, env=env)
    sep = " " if tail_trajectory and reward_str else ""
    value = f"{reward_str}{sep}{tail_trajectory}"
    return f"{env.state_str(head.state)} {env.action_str(head.action)} {value}"


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

    def print(self, *args, **kwargs):
        if self.debug >= 1:
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

        self.print("Q")
        self.print("state", state)
        for a, v in zip(actions, values):
            self.print("action", a)
            self.print("value", v)
        self.print("chosen", action)
        if self.debug >= 2:
            breakpoint()
        return action

    def value(self, state, action: int = None) -> str:
        assert action is not None

        # original_state = state
        # original_action = action
        completions = []
        t = 0
        action = self.env.action_str(action)
        state = self.env.state_str(state)

        while True:
            if t == self.max_steps:
                state_or_reward = (
                    self.env.time_out_str()
                )  # TODO: can we eliminate this?
            else:
                prompts = self.sample()
                new_prompt = "\n".join([*prompts, f"{state} {action}"])
                self.print("Q prompt:")
                self.print(new_prompt)

                state_or_reward, *_ = self.gpt3(new_prompt).lstrip().split(".")
                state_or_reward = reformat(state_or_reward)
            self.print("state/reward", state_or_reward)
            self.print("action", action)
            completions.append(state_or_reward)
            if self.env.done(state_or_reward):
                break
            state = state_or_reward
            prompts = self.sample_best()
            new_prompt = "\n".join([*prompts, state])
            self.print("Q prompt:")
            self.print(new_prompt)

            action, *_ = self.gpt3(new_prompt).lstrip().split(".")
            action = reformat(action)
            t += 1
            self.print("action", action)
            self.print("state/reward", state_or_reward)
            completions.append(action)

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
            self.print("pi prompt:")
            self.print(prompt)
            completion = self.gpt3(prompt).lstrip()
            maybe_action, *_ = completion.split(".")
            self.print("Action:", maybe_action)
            if self.debug >= 2:
                breakpoint()

            action = self.env.action(maybe_action + ".")
            t += 1

        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
