import abc
import shelve
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np
import openai
from env import ACTIONS, Env
from gym.spaces import Discrete


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
        #     print("Prompt:")
        #     print(prompt)

        # print("Querying...", end=" ")
        choice, *_ = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.1,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        ).choices
        # print("Received response.")
        completion, *_ = choice.text.split("\n")
        self.db[prompt] = completion
        # if pause:
        #     print("Completion:")
        #     print(completion)
        return completion


@dataclass
class Model(abc.ABC):
    env: Env
    gpt3: GPT3
    optimistic: bool
    prompt_buffer_size: int
    prompt_size: int
    seed: int

    def __post_init__(self):
        self.buffer = deque(maxlen=self.prompt_buffer_size)
        self.rng = np.random.default_rng(seed=self.seed)

    def __len__(self):
        return len(self.buffer)

    def learn(self, prompt: Prompt):
        if not self.optimistic or self.env.quantify(prompt.value) > 0:
            self.buffer.append(prompt)

    def sample(self):
        prompts = [p.to_string(self.env) for p in self.buffer]
        self.rng.shuffle(prompts)
        return prompts

    def ready(self) -> bool:
        return len(self) >= self.prompt_size

    @abc.abstractmethod
    def value(self, state: int, prompt: List[str], action: Optional[int] = None) -> str:
        ...

    @abc.abstractmethod
    def action_value(self, state: int) -> Tuple[int, str]:
        ...

    def act(self, state: int) -> int:
        if self.ready():
            act, _ = self.action_value(state)
            return act
        return self.env.action_space.sample()


@dataclass
class Q(Model):
    def value(self, state: int, prompt: List[str], action: Optional[int] = None) -> str:
        assert action is not None
        return self.gpt3(
            "\n".join(
                [*prompt, f"{self.env.state_str(state)} {self.env.action_str(action)}"]
            )
        )

    def action_value(self, state: int) -> Tuple[int, str]:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_values():
            for a in actions:
                prompt = self.sample()
                # print(prompt)
                yield self.value(state, prompt=prompt, action=a)

        values = list(get_values())
        action, value = max(zip(actions, values), key=lambda x: self.env.quantify(x[1]))
        # print("Q")
        # print("state", state)
        # for a, v in zip(actions, values):
        #     print("action", a)
        #     print("value", v)
        # print("chosen", action)
        # breakpoint()
        return action, value


class V(Model):
    def value(self, state: int, prompt: List[str], action: Optional[int] = None) -> str:
        return self.gpt3(
            "\n".join([*prompt, f"{self.env.state_str(state)}"]),
            pause=True,
        )

    def action_value(self, state: int) -> Tuple[int, str]:
        action = None
        while action is None:
            prompt = self.sample()
            print(prompt)
            value = self.value(state, prompt=prompt).lstrip()
            for maybe_action, action_str in enumerate(ACTIONS):
                # print(value.startswith(action_str), action_str, value)
                if value.startswith(action_str):
                    action = maybe_action
        print("V")
        print("state", state)
        print("action", action)
        print("value", value)
        breakpoint()
        return action, value
