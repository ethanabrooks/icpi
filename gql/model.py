import abc
import shelve
import sys
from dataclasses import dataclass
from typing import Deque, Optional, cast

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

    def to_value_quantity(self, env: Env, gamma: Optional[float] = None) -> float:
        return env.quantify(self.to_string(env), gamma=gamma)

    def to_string(self, env: Env) -> str:
        return f"{env.state_str(self.state)} {env.action_str(self.action)} {self.value}"


@dataclass
class GPT3:
    db: shelve.DbfilenameShelf

    def __call__(self, prompt, pause=True):
        print("<", end="")
        if prompt in self.db:
            completion = cast(str, self.db[prompt])
            # print("Completion:")
            # print(value)
            print(">", end="")
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
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            ).choices
            completion = choice.text.lstrip()
            if "." in completion:
                self.db[prompt] = completion
                print(">", end="")
                # print("Completion:", completion.split("\n")[0])
                # breakpoint()
                return completion


@dataclass
class Model(abc.ABC):
    buffer: Deque[Prompt]
    env: Env
    failure_threshold: float
    gpt3: GPT3
    prompt_size: int
    rng: Generator

    def act(self, state: int) -> int:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: int) -> int:
        ...

    def ready(self) -> bool:
        return len(self.buffer) >= self.prompt_size

    def sample(self):
        prompts = [p.to_string(self.env) for p in self.buffer]
        self.rng.shuffle(prompts)
        return prompts[: self.prompt_size]

    def sample_best(self):
        # buffer = sorted(
        #     self.buffer,
        #     key=lambda p: (p.to_value_quantity(self.env), self.rng.random()),
        #     reverse=True,
        # )
        success_prompts = [p for p in self.buffer if p.to_value_quantity(self.env) > self.failure_threshold]
        self.rng.shuffle(success_prompts)
        return [p.to_string(self.env) for p in success_prompts][: self.prompt_size]


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    max_trajectory: int

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
            key=lambda x: (self.env.quantify(x[1], gamma=0.9), self.rng.random()),
        )

        print("Q")
        print("state", state)
        for a, v in zip(actions, values):
            print("action", a)
            print("value", v)
        print("chosen", action)
        breakpoint()
        return action

    def learn(self, prompt: Prompt):
        self.buffer.append(prompt)

    def value(self, state: int, action: Optional[int] = None) -> str:
        assert action is not None

        # original_state = state
        # original_action = action
        completions = []
        state = self.env.state_str(state)
        action = self.env.action_str(action)
        prompt = self.sample()
        new_prompt = "\n".join([*prompt, f"{state} {action}"])
        print("Q prompt:")
        print(new_prompt)

        completion = self.gpt3(new_prompt).lstrip()
        state_or_reward, action, *_ = completion.split(".")
        state_or_reward, action = map(reformat, [state_or_reward, action])
        print("state/reward", state_or_reward)
        print("action", action)
        completions.append(state_or_reward)
        t = 1

        while state_or_reward not in REWARDS.values():
            state = state_or_reward
            prompt = self.sample_best()

            new_prompt = "\n".join([*prompt, state])
            print("Q prompt:")
            print(new_prompt)

            # print(f"{state} {action}", end=" :: ")
            completion = self.gpt3(new_prompt).lstrip()
            action, state_or_reward, *_ = completion.split(".")
            action, state_or_reward = map(reformat, [action, state_or_reward])
            if t == self.max_trajectory:
                state_or_reward = REWARDS[0.0]
            t += 1
            print("action", action)
            print("state/reward", state_or_reward)
            completions.extend([action, state_or_reward])

        completion = " ".join(completions)
        # print("state", original_state)
        # print("action", original_action)
        # print(completion)
        # breakpoint()
        return completion


class Pi(Model):
    def _act(self, state: int) -> int:
        state = self.env.state_str(state)
        action = None
        while action is None:
            prompt = self.sample_best()
            prompt = "\n".join([*prompt, state])
            print("pi prompt:")
            print(prompt)
            completion = self.gpt3(prompt).lstrip()
            maybe_action, *_ = completion.split(".")
            print("Action:", maybe_action)
            breakpoint()

            try:
                action = ACTIONS.index(maybe_action + ".")
            except ValueError:
                pass
        return action
