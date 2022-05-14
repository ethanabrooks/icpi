import abc
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Deque, Generic, List, Optional, Union

from base_env import ActType, Env, ObsType, TimeStep
from gym.spaces import Discrete
from numpy.random import Generator
from rl.common import Colorize, get_value
from rl.gpt3 import GPT3
from rl.huggingface import HuggingFaceModel


def to_string(*trajectory: TimeStep, env) -> str:
    return "".join(
        [env.initial_str()]
        + [env.ts_to_string(ts) for ts in trajectory]
        + [env.termination_str(trajectory[-1])]
    )


@dataclass
class Model(abc.ABC, Generic[ObsType, ActType]):
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    lm: Union[GPT3, HuggingFaceModel]
    max_resamples: int
    max_steps: int
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

    def predict(
        self,
        query: List[str],
        get_prompts: Callable[[], List[str]],
        max_prompts: int,
        name: str,
        stop: str,
        valid: Callable[[str], bool],
    ) -> Optional[str]:
        previous_prompts = set()
        for _ in range(self.max_resamples):
            prompts = get_prompts()
            while "".join(prompts) in previous_prompts and len(prompts) < max_prompts:
                prompts = get_prompts()
            previous_prompts.add("".join(prompts))

            new_prompt = "".join([*prompts, "".join(query)])
            if self.debug >= 2:
                print()
                print("".join(prompts), end="")
                Colorize.print_bold("".join(query))
            if self.debug >= 4:
                breakpoint()
            completion = self.lm(
                new_prompt, stop=[stop], temperature=self.temperature, use_cache=True
            )

            if self.debug >= 2:
                Colorize.print_blue(name)
                Colorize.print_cyan(completion)
            if self.debug >= 4:
                breakpoint()
            completion += stop
            if valid(completion):
                return completion
            else:
                if self.debug >= 3:
                    Colorize.print_warning(f"Invalid {name}:", end=" ")
                    Colorize.print_cyan(completion)
                    breakpoint()
                    valid(completion)
        return None

    def ready(self) -> bool:
        return bool(self.sample_best())

    def sample(self, action: ActType) -> List[str]:
        trajectories = [
            trajectory
            for trajectory in self.buffer
            if any(ts.action == action for ts in trajectory)
        ]
        self.rng.shuffle(trajectories)
        trajectories_by_success = defaultdict(list)
        for trajectory in trajectories:
            trajectories_by_success[
                self.get_value(trajectory) > self.env.failure_threshold()
            ].append(trajectory)
        trajectories = [
            trajectory
            for trajectories in zip(*trajectories_by_success.values())
            for trajectory in trajectories
        ]
        self.rng.shuffle(trajectories)
        return [to_string(*t, env=self.env) for t in trajectories]

    def sample_best(self):
        trajectories = list(self.success_buffer)
        trajectories = [
            trajectory[start:stop]
            for trajectory in trajectories
            for start, stop in itertools.combinations(range(len(trajectory) + 1), 2)
            if self.get_value(trajectory[start:stop]) > self.env.failure_threshold()
        ]
        self.rng.shuffle(trajectories)
        prompts = [to_string(*t, env=self.env) for t in trajectories]
        return list(prompts)

    def generate_action(self, completions: List[str]) -> Optional[str]:
        maybe_action = self.predict(
            completions,
            get_prompts=self.sample_best,
            max_prompts=math.factorial(len(self.success_buffer)),
            name="action",
            stop=self.env.action_stop(),
            valid=lambda s: self.env.action(s) is not None,
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
                    self.env.initial_str(),
                    self.env.state_str(state),
                    self.env.action_str(a),
                ]
                trajectory_str = "".join(trajectory_strings)
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

    def ready(self) -> bool:
        return (
            all(bool(self.sample(a)) for a in range(self.env.action_space.n))
            and super().ready()
        )

    def value(self, trajectory: List[TimeStep], state: ObsType, action: ActType) -> str:
        if self.debug >= 2:
            Colorize.print_header(
                f"Computing Q value for state {state} and action {action}:"
            )
        t = 0
        initial_str = self.env.initial_str()
        state_str = self.env.state_str(state)
        action_str = self.env.action_str(action)
        completions = [s for s in [initial_str, state_str, action_str] if s]
        query = list(completions)

        def sample() -> List[str]:
            return self.sample(action=action)

        max_prompts = math.factorial(len(sample()))
        while True:
            if t == self.max_steps:
                break
            if self.env.reward_stop():
                reward_str = self.predict(
                    query,
                    max_prompts=max_prompts,
                    name="reward",
                    get_prompts=sample,
                    stop=self.env.reward_stop(),
                    valid=self.env.valid_reward,
                )
                if reward_str is None:
                    break
                completions.append(reward_str)
                query.append(reward_str)
            state_str = self.predict(
                query,
                max_prompts=max_prompts,
                name="state",
                get_prompts=sample,
                stop=self.env.state_stop(),
                valid=self.env.valid_state,
            )
            if state_str is None:
                break
            completions.append(state_str)
            query = [initial_str, state_str]
            if self.env.done(*completions):
                break

            action_str = self.generate_action(query)
            action = self.env.action(action_str)
            completions.append(action_str)
            query.append(action_str)
            t += 1

        return "".join(completions)


class Pi(Model[ObsType, ActType]):
    def _act(self, trajectory: List[TimeStep], state: ObsType) -> ActType:
        if self.debug >= 2:
            Colorize.print_header(f"Computing pi action for state {state}:")
        state = self.env.state_str(state)

        completions = [self.env.initial_str(), state]

        action_str = self.generate_action(completions)
        action = self.env.action(action_str)
        assert action is not None
        return action
