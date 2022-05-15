import abc
import itertools
from dataclasses import dataclass
from typing import Callable, Deque, Generic, List, Optional, Union

from base_env import ActType, Env, ObsType, TimeStep
from gym.spaces import Discrete
from numpy.random import Generator
from rl.common import Colorize, get_value
from rl.gpt3 import GPT3
from rl.huggingface import HuggingFaceModel


def to_string(*trajectory: TimeStep, env) -> str:
    return " ".join([env.ts_to_string(ts) for ts in trajectory])


@dataclass
class Model(abc.ABC, Generic[ObsType, ActType]):
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    lm: Union[GPT3, HuggingFaceModel]
    max_steps: int
    prompt_size: int
    rng: Generator
    success_buffer: Deque[List[TimeStep]]
    temperature: float

    def act(self, trajectory: List[TimeStep], state: ObsType) -> ActType:
        if self.ready():
            return self._act(trajectory, state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, trajectory: List[TimeStep], state: ObsType) -> ActType:
        ...

    def get_value(self, trajectory: List[TimeStep]) -> float:
        return get_value(*trajectory, gamma=self.env.gamma())

    def ready(self) -> bool:
        return len(self.success_buffer) > 0

    def predict(
        self,
        query: List[str],
        get_prompts: Callable[[], List[str]],
        name: str,
        stop: List[str],
    ) -> Optional[str]:
        prompts = get_prompts()

        new_prompt = "\n".join([*prompts, " ".join(query)])
        if self.debug >= 2:
            print()
            print(" ".join(prompts), end="")
            Colorize.print_bold("".join(query))
        if self.debug >= 4:
            breakpoint()
        completion = self.lm(
            new_prompt, stop=stop, temperature=self.temperature, use_cache=True
        )

        if self.debug >= 2:
            Colorize.print_blue(name)
            Colorize.print_cyan(completion)
        if self.debug >= 4:
            breakpoint()
        return completion

    def sample(self):
        successful = list(self.success_buffer)
        unsuccessful = [
            t for t in self.buffer if self.get_value(t) <= self.env.failure_threshold()
        ]
        half = self.prompt_size // 2
        half = min([half, len(successful), len(unsuccessful)])
        successful_choices = [
            successful[i]
            for i in self.rng.choice(
                len(successful), min(half, len(successful)), replace=False
            )
        ]
        unsuccessful_choices = [
            unsuccessful[i]
            for i in self.rng.choice(
                len(unsuccessful), min(half, len(unsuccessful)), replace=False
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
        self.rng.shuffle(trajectories)
        return [to_string(*t, env=self.env) for t in trajectories]

    def sample_best(self):
        trajectories = list(self.success_buffer)
        trajectories = [
            t[i:j]
            for t in trajectories
            for i, j in itertools.combinations(range(len(t) + 1), 2)
            if self.get_value(t[i:j]) > self.env.failure_threshold()
        ]
        self.rng.shuffle(trajectories)
        prompts = [to_string(*t, env=self.env) for t in trajectories]
        return list(prompts)[: self.prompt_size]

    def generate_action(self, completions: List[str]) -> Optional[str]:
        maybe_action = self.predict(
            completions,
            get_prompts=self.sample_best,
            name="action",
            stop=[self.env.state_stop(), self.env.action_stop()],
        )
        if maybe_action is None:
            return self.env.action_str(self.env.action_space.sample())
        return maybe_action


@dataclass
class Q(Model[ObsType, ActType]):
    def _act(self, trajectory: List[TimeStep], state: ObsType) -> ActType:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_values():
            for action in actions:
                yield self.value(trajectory, state, action)

        values = list(get_values())
        action_values = list(zip(actions, values))
        self.rng.shuffle(action_values)
        action, value = max(
            action_values,
            key=lambda x: (self.env.quantify(x[1]), self.rng.random()),
        )

        if self.debug >= 1:
            print()
            Colorize.print_header("Q prompts")
            Colorize.print_blue("state:", end=" ")
            Colorize.print_cyan(state)
            for a, v in zip(actions, values):
                Colorize.print_blue("action:", end=" ")
                Colorize.print_cyan(a)
                trajectory_strings = [
                    self.env.state_str(state),
                    self.env.action_str(a),
                ]
                trajectory_str = " ".join(trajectory_strings)
                print("value:", trajectory_str, end="")
                if not v.startswith(trajectory_str):
                    print(trajectory_str)
                    breakpoint()
                Colorize.print_cyan(v[len(trajectory_str) :])
            Colorize.print_blue("chosen", end=" ")
            Colorize.print_cyan(action)
        if self.debug >= 3:
            breakpoint()
        return action

    def value(self, trajectory: List[TimeStep], state: ObsType, action: ActType) -> str:
        t = 0
        state_str = self.env.state_str(state)
        action_str = self.env.action_str(action)
        completions = [s for s in [state_str, action_str] if s]

        if self.debug >= 2:
            print()
            Colorize.print_header(
                "Computing value for state", state, "and action", action
            )

        while True:
            if t == self.max_steps:
                break
            else:
                state_or_reward = self.predict(
                    completions,
                    get_prompts=self.sample,
                    name="state/reward",
                    stop=[self.env.action_stop(), self.env.state_stop()],
                )
                state_or_reward = state_or_reward.lstrip() + self.env.state_stop()
            if self.debug >= 2:
                Colorize.print_blue("state/reward", end=" ")
                Colorize.print_cyan(state_or_reward)
            if self.debug >= 4:
                breakpoint()
            completions.append(state_or_reward)
            if self.env.done(*completions):
                break
            action_str = self.predict(
                [state_or_reward],
                get_prompts=self.sample_best,
                name="action",
                stop=[self.env.action_stop(), self.env.state_stop()],
            )
            if self.env.action(action_str) is None and self.debug >= 3:
                print(self.env.actions())
                print(action_str)
                breakpoint()
                break

            action_str = action_str.lstrip() + self.env.action_stop()
            t += 1

            if self.debug >= 2:
                Colorize.print_blue("action", end=" ")
                Colorize.print_cyan(action_str)
            if self.debug >= 4:
                breakpoint()
            completions.append(action_str)

        return " ".join(completions)


class Pi(Model[ObsType, ActType]):
    def _act(self, trajectory: List[TimeStep], state: ObsType) -> ActType:
        state = self.env.state_str(state)
        action = None
        t = 0
        while action is None:
            if t > self.max_steps:
                return self.env.action_space.sample()
            prompts = self.sample_best()
            prompt = "\n".join([*prompts, state])
            if self.debug >= 1:
                Colorize.print_header("pi prompt:")
                print(prompt)
            maybe_action, *_ = (
                self.lm(
                    prompt,
                    stop=[self.env.action_stop(), self.env.state_stop()],
                    temperature=self.temperature,
                )
                .lstrip()
                .split(self.env.action_stop())
            )
            if self.debug >= 1:
                Colorize.print_blue("Action:", end=" ")
                Colorize.print_cyan(maybe_action)
            if self.debug >= 4:
                breakpoint()

            action = self.env.action(maybe_action)
            t += 1

        return action
