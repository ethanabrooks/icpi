import abc
import itertools
import math
import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Deque, Generic, Hashable, Iterable, List, Optional, Union

from base_env import ActType, Env, ObsType, TimeStep
from gym.spaces import Discrete
from numpy.random import Generator
from rl.common import Colorize, get_value
from rl.gpt3 import GPT3
from rl.huggingface import HuggingFaceModel


def to_string(*trajectory: TimeStep, env) -> str:
    return "".join([env.initial_str()] + [env.ts_to_string(ts) for ts in trajectory])


def product(x):
    return reduce(operator.mul, x, 1)


def unique_permutations(population: Iterable[Hashable]) -> int:
    counts = list(Counter(population).values())
    return math.factorial(sum(counts)) // product(math.factorial(c) for c in counts)


@dataclass
class Model(abc.ABC, Generic[ObsType, ActType]):
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    lm: Union[GPT3, HuggingFaceModel]
    max_prompts: int
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
        name: str,
        stop: str,
        valid: Callable[[str], bool],
    ) -> Optional[str]:
        previous_prompts = set()
        for _ in range(self.max_resamples):
            prompts = get_prompts()
            for _ in range(self.max_prompts):
                prompts = get_prompts()
                if "".join(prompts) not in previous_prompts:
                    break
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
            completion += stop

            if self.debug >= 2:
                Colorize.print_blue(name)
                Colorize.print_cyan(completion)
            if self.debug >= 4:
                breakpoint()
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

    def sample_done(self, action: int) -> List[str]:
        time_steps = [ts for t in self.buffer for ts in t if ts.action == action]
        done = [ts for ts in time_steps if ts.done]
        not_done = [ts for ts in time_steps if not ts.done]
        balanced = [ts for (d, nd) in zip(done, not_done) for ts in [d, nd]]
        self.rng.shuffle(balanced)
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.done_str(ts.done)
            + self.env.done_stop()
            + "\n"
            for ts in balanced
        ]

    def sample_next_state(self, action: int) -> List[str]:
        def get_time_steps(*trajectories: List[TimeStep]) -> List[TimeStep]:
            return [
                ts
                for t in trajectories
                for ts in t
                if ts.action == action and not ts.done
            ]

        successful = get_time_steps(*self.success_buffer)
        unsuccessful = get_time_steps(
            *[
                t
                for t in self.buffer
                if self.get_value(t) <= self.env.failure_threshold()
            ]
        )
        if not successful:
            balanced = unsuccessful
        elif not unsuccessful:
            balanced = successful
        else:
            balanced = [ts for (s, u) in zip(successful, unsuccessful) for ts in [s, u]]
        self.rng.shuffle(balanced)
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.state_str(ts.next_state)
            + "\n"
            for ts in balanced
        ]

    def sample_reward(self, action: int, done: bool) -> List[str]:
        rewards = defaultdict(list)
        for t in self.buffer:
            for ts in t:
                if ts.action == action and ts.done == done:
                    rewards[ts.reward].append(ts)
        balanced = [ts for time_steps in zip(*rewards.values()) for ts in time_steps]
        self.rng.shuffle(balanced)
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.reward_str(ts.reward)
            + self.env.reward_stop()
            + "\n"
            for ts in balanced
        ]

    def sample_best(self) -> List[str]:
        trajectories = list(self.success_buffer)
        trajectories = [
            trajectory[start:stop]
            for trajectory in trajectories
            for start, stop in itertools.combinations(range(len(trajectory) + 1), 2)
            if self.get_value(trajectory[start:stop]) > self.env.failure_threshold()
        ]
        self.rng.shuffle(trajectories)

        return [to_string(*t, env=self.env) for t in trajectories]

    def generate_action(self, state: str) -> Optional[str]:
        maybe_action = self.predict(
            ["", self.env.initial_str(), state],
            get_prompts=self.sample_best,
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
        actions = list(range(self.env.action_space.n))
        return (
            all(
                [bool(self.sample_done(a)) for a in actions]
                + [bool(self.sample_next_state(a)) for a in actions]
                + [
                    bool(self.sample_reward(a, d))
                    for a in actions
                    for d in [True, False]
                ]
            )
            and super().ready()
        )

    def value(self, trajectory: List[TimeStep], state: ObsType, action: ActType) -> str:
        if self.debug >= 2:
            Colorize.print_header(
                f"Computing Q value for state {state} and action {action}:"
            )
        t = 0
        state_str = self.env.state_str(state)
        action_str = self.env.action_str(action)
        completions = [s for s in [state_str, action_str] if s]
        while True:
            if t == self.max_steps:
                break
            query = [state_str + action_str]
            done_str = self.predict(
                query,
                name="state",
                get_prompts=lambda: self.sample_done(action),
                stop=self.env.done_stop(),
                valid=self.env.valid_done,
            )
            completions.append(done_str)
            done = self.env.done(done_str)
            reward_str = self.predict(
                query,
                name="reward",
                get_prompts=lambda: self.sample_reward(action=action, done=done),
                stop=self.env.reward_stop(),
                valid=self.env.valid_reward,
            )
            if reward_str is None:
                break
            completions.append(reward_str)
            if done:
                break
            state_str = self.predict(
                query,
                name="state",
                get_prompts=lambda: self.sample_next_state(action=action),
                stop=self.env.state_stop(),
                valid=self.env.valid_state,
            )
            if state_str is None:
                break
            completions.append(state_str)
            action_str = self.generate_action(state_str)
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

        action_str = self.generate_action(state)
        action = self.env.action(action_str)
        assert action is not None
        return action
