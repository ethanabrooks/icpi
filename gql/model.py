import abc
import shelve
import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import openai
from env import ACTIONS, MAX_TOKENS, REWARDS, Env
from gym.spaces import Discrete
from numpy.random import Generator


@dataclass
class Prompt:
    state: int
    action: int
    value: str

    @staticmethod
    def make(state: int, action: int, value: str):
        return Prompt(state, action, value.lstrip())

    def to_string(self, env: Env) -> str:
        return f"{env.state_str(self.state)} {env.action_str(self.action)} {self.value}"


@dataclass
class GPT3:
    db: shelve.DbfilenameShelf

    def __call__(self, prompt, pause=True):
        if prompt in self.db:
            completion = cast(str, self.db[prompt])
            # print("Completion:")
            # print(value)
            return completion

        # if pause:
        # print("Prompt:")
        # print(prompt)
        # breakpoint()
        #
        # print("<", end="")
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            choice, *_ = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=len(prompt) + MAX_TOKENS + 1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            ).choices
            completion = choice.text.lstrip()
            if "." in completion:
                self.db[prompt] = completion
                # print(">", end="")
                # if pause:
                # print("Completion:", completion.split("\n")[0])
                # breakpoint()
                return completion


@dataclass
class Model(abc.ABC):
    env: Env
    gpt3: GPT3
    prompt_buffer_size: int
    prompt_size: int
    rng: Generator

    def __post_init__(self):
        self.buffer = deque(maxlen=self.prompt_buffer_size)

    def act(self, state: int) -> int:
        if self.ready():
            act, _ = self.action_value(state)
            return act
        return self.env.action_space.sample()

    @abc.abstractmethod
    def action_value(self, state: int) -> Tuple[int, str]:
        ...

    @abc.abstractmethod
    def learn(self, prompt: Prompt):
        ...

    def __len__(self):
        return len(self.buffer)

    def ready(self) -> bool:
        return len(self) >= self.prompt_size

    def sample(self):
        prompts = [p.to_string(self.env) for p in self.buffer]
        self.rng.shuffle(prompts)
        return prompts

    @abc.abstractmethod
    def value(self, state: int, action: Optional[int] = None) -> str:
        ...


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    def action_value(self, state: int) -> Tuple[int, str]:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_values():
            for a in actions:
                yield self.value(state, action=a)

        values = list(get_values())
        action_values = list(zip(actions, values))
        self.rng.shuffle(action_values)
        action, value = max(action_values, key=lambda x: self.env.quantify(x[1]))
        # print("Q")
        # print("state", state)
        # for a, v in zip(actions, values):
        #     print("action", a)
        #     print("value", v)
        # print("chosen", action)
        return action, value

    def learn(self, prompt: Prompt):
        self.buffer.append(prompt)

    def value(self, state: int, action: Optional[int] = None) -> str:
        assert action is not None

        # original_state = state
        # original_action = action
        completions = []
        state = self.env.state_str(state)
        action = self.env.action_str(action)
        while True:
            prompt = self.sample()
            new_prompt = "\n".join([*prompt, f"{state} {action}"])
            # pprint(new_prompt)
            # print(f"{state} {action}", end=" :: ")
            completion = self.gpt3(new_prompt).lstrip()
            state_or_reward, action, *_ = completion.split(".")
            state_or_reward, action = map(reformat, [state_or_reward, action])
            # print(f"{state_or_reward} {action}")
            # breakpoint()

            completions.append(state_or_reward)
            if state_or_reward in REWARDS.values():
                break
            completions.append(action)
            state = state_or_reward

        completion = " ".join(completions)
        # print("state", original_state)
        # print("action", original_action)
        # print(completion)
        # breakpoint()
        return completion


class V(Model):
    def action_value(self, state: int) -> Tuple[int, str]:
        action = None
        value = None
        while action is None:
            # print("\n".join(prompt))
            # breakpoint()
            value = self.value(state).lstrip()
            for maybe_action, action_str in enumerate(ACTIONS):
                # print(value.startswith(action_str), action_str, value)
                if value.startswith(action_str):
                    action = maybe_action
        assert value is not None
        # print("V")
        # print("state", state)
        # print("action", action)
        # print("value", value)
        # breakpoint()
        return action, value

    def learn(self, prompt: Prompt):
        if self.env.quantify(prompt.value) > 0:
            self.buffer.append(prompt)

    def value(self, state: int, action: Optional[int] = None) -> str:
        completions = []
        state = self.env.state_str(state)
        while True:
            prompt = self.sample()
            completion = self.gpt3("\n".join([*prompt, state])).lstrip()
            action, state_or_reward, *_ = completion.split(".")
            action, state_or_reward = map(reformat, [action, state_or_reward])
            completions.extend([action, state_or_reward])
            if state_or_reward in REWARDS.values():
                break
            state = state_or_reward
        completion = " ".join(completions)
        # print(completion)
        # breakpoint()
        return completion
