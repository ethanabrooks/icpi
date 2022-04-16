import abc
import shelve
import sys
from dataclasses import dataclass
from typing import Deque, List, Optional, cast

import openai
from env import ACTIONS, MAX_TOKENS, REWARDS, Env
from gym.spaces import Discrete
from numpy.random import Generator


@dataclass
class TimeStep:
    state: int
    action: int
    reward: float
    next_state: Optional[int]


def to_string(*_trajectory: TimeStep, env) -> str:

    if not _trajectory:
        return ""
    head, *tail = _trajectory
    if head.next_state is None:
        reward_str = env.reward_str(head.reward, next_state=None)
    else:
        reward_str = ""

    tail_trajectory = to_string(*tail, env=env)
    sep = " " if tail_trajectory and reward_str else ""
    value = f"{reward_str}{sep}{tail_trajectory}"
    return f"{env.state_str(head.state)} {env.action_str(head.action)} {value}"


def get_value(*trajectory: TimeStep, gamma: float) -> float:
    return sum([gamma ** t * ts.reward for t, ts in enumerate(trajectory)])


def _print(*args, debug: bool, **kwargs):
    if debug:
        print(*args, **kwargs)


@dataclass
class GPT3:
    db: shelve.DbfilenameShelf
    debug: bool = False

    def __call__(self, prompt, pause=True):
        self.print("<", end="")
        if prompt in self.db:
            completion = cast(str, self.db[prompt])
            # print("Completion:")
            # print(value)
            self.print(">", end="")
            return completion

        # print("Prompt:")
        # print(prompt)
        # breakpoint()
        #
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            choice, *_ = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=len(prompt) + MAX_TOKENS + 1,
            ).choices
            completion = choice.text.lstrip()
            if "." in completion:
                self.db[prompt] = completion
                self.print(">", end="")
                # print("Completion:", completion.split("\n")[0])
                # breakpoint()
                return completion

    def print(self, *args, **kwargs):
        _print(*args, debug=self.debug, **kwargs)


@dataclass
class Model(abc.ABC):
    buffer: Deque[List[TimeStep]]
    env: Env
    failure_threshold: float
    gpt3: GPT3
    prompt_size: int
    rng: Generator
    debug: bool

    def act(self, state: int) -> int:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: int) -> int:
        ...

    def print(self, *args, **kwargs):
        _print(*args, debug=self.debug, **kwargs)

    def ready(self) -> bool:
        return len(self.buffer) >= self.prompt_size

    def sample(self):
        prompts = [to_string(*t, env=self.env) for t in self.buffer]
        self.rng.shuffle(prompts)
        return prompts[: self.prompt_size]

    def sample_best(self):
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        self.rng.shuffle(trajectories)
        return [to_string(*t, env=self.env) for t in trajectories][: self.prompt_size]


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    gamma: float
    max_steps: int

    def _act(self, state: int) -> int:
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

    def value(self, state: int, action: Optional[int] = None) -> str:
        assert action is not None

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

        while state_or_reward not in REWARDS.values():
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
                state_or_reward = REWARDS[0.0]
            t += 1
            # print("action", action)
            # print("state/reward", state_or_reward)
            completions.extend([action, state_or_reward])

        return " ".join(completions)


class Pi(Model):
    def _act(self, state: int) -> int:
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

            try:
                action = ACTIONS.index(maybe_action + ".")
            except ValueError:
                pass
        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
