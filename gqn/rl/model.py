import abc
import itertools
import math
import operator
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Deque, Generic, Hashable, Iterable, List, Optional

from base_env import ActType, Env, ObsType, TimeStep
from gym.spaces import Discrete
from numpy.random import Generator
from rl.common import Colorize, get_value
from rl.lm import LM


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
    lm: LM
    max_prompts: int
    max_resamples: int
    rng: Generator
    success_buffer: Deque[List[TimeStep]]
    temperature: float
    t_threshold: Optional[int]
    use_cache: bool

    def act(self, state: ObsType, T: int) -> ActType:
        if self.ready():
            return self._act(state, T)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: ObsType, T: int) -> ActType:
        ...

    def balance(self, *lists: List) -> List[List]:
        if not lists:
            return [[]]
        max_len = max(len(l) for l in lists)
        return [self.extend(l, max_len) for l in lists]

    def breakpoint(self, T: int, threshold: int):
        if self.debug >= threshold and (
            self.t_threshold is None or T >= self.t_threshold
        ):
            breakpoint()

    def extend(self, lst: List, length: int) -> List:
        lst = list(lst)
        self.rng.shuffle(lst)
        lst = list(itertools.islice(itertools.cycle(lst), length))
        self.rng.shuffle(lst)
        assert len(lst) == length
        return lst

    def get_value(self, trajectory: List[TimeStep]) -> float:
        return get_value(*trajectory, gamma=self.env.gamma())

    def predict(
        self,
        query: List[str],
        get_prompts: Callable[[], List[str]],
        name: str,
        stop: str,
        T: int,
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
            self.breakpoint(T, 4)
            completion = self.lm(
                new_prompt,
                stop=[stop],
                temperature=self.temperature,
                use_cache=self.use_cache,
            )
            completion = completion.replace("ball.x!=", "ball.x !=")
            completion += stop

            if self.debug >= 2:
                Colorize.print_blue(name)
                Colorize.print_cyan(completion)
            self.breakpoint(T, 4)
            if valid(completion):
                return completion
            else:
                if self.debug >= 3:
                    Colorize.print_warning(f"Invalid {name}:", end=" ")
                    Colorize.print_cyan(completion)
                self.breakpoint(T, 3)
                valid(completion)
        return None

    def ready(self) -> bool:
        return bool(self.sample_best())

    def sample_done(self, action: int) -> List[str]:
        time_steps = [ts for t in self.buffer for ts in t if ts.action == action]
        self.rng.shuffle(time_steps)
        done = [ts for ts in time_steps if ts.done]
        not_done = [ts for ts in time_steps if not ts.done]
        if len(done) == 0:
            balanced = not_done
        elif len(not_done) == 0:
            balanced = done
        else:
            done, not_done = self.balance(done, not_done)
            # if len(done) > 3:
            #     breakpoint()
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

        buffer = [t for t in self.buffer]
        self.rng.shuffle(buffer)
        successful = get_time_steps(
            *[t for t in buffer if self.get_value(t) > self.env.failure_threshold()]
        )
        unsuccessful = get_time_steps(
            *[t for t in buffer if self.get_value(t) <= self.env.failure_threshold()]
        )
        if not successful:
            balanced = unsuccessful
        elif not unsuccessful:
            balanced = successful
        else:
            successful, unsuccessful = self.balance(successful, unsuccessful)
            # if len(successful) > 3:
            #     breakpoint()
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
        buffer = [t for t in self.buffer]
        self.rng.shuffle(buffer)
        for t in buffer:
            for ts in t:
                if ts.action == action and ts.done == done:
                    rewards[ts.reward].append(ts)
        balanced = self.balance(*rewards.values())
        # if rewards and max(len(r) for r in rewards.values()) > 3:
        #     breakpoint()
        balanced = [ts for time_steps in zip(*balanced) for ts in time_steps]
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

    def generate_action(self, state: str, T: int) -> Optional[str]:
        maybe_action = self.predict(
            ["", self.env.initial_str(), state],
            get_prompts=self.sample_best,
            name="action",
            stop=self.env.action_stop(),
            T=T,
            valid=lambda s: self.env.action(s) is not None,
        )
        if maybe_action is None:
            return self.env.action_str(self.env.action_space.sample())
        return maybe_action


@dataclass
class Q(Model[ObsType, ActType]):
    max_steps: int
    oracle_transitions: bool

    def _act(self, state: ObsType, T: int) -> ActType:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_rollouts():
            for action in actions:
                yield self.rollout(state, action, T)

        rollouts = list(get_rollouts())
        action_rollouts = list(zip(actions, rollouts))
        self.rng.shuffle(action_rollouts)
        action, value = max(
            action_rollouts,
            key=lambda x: (self.env.quantify(x[1]), self.rng.random()),
        )

        if self.debug >= 1:
            print()
            Colorize.print_header("Q prompts")
            Colorize.print_blue("state:", end=" ")
            Colorize.print_cyan(state)
            for a, v in zip(actions, rollouts):
                Colorize.print_blue("action:", end=" ")
                Colorize.print_cyan(a)
                trajectory_strings = [
                    self.env.state_str(state),
                    self.env.action_str(a),
                ]
                trajectory_str = "".join(trajectory_strings)
                print("rollout:", trajectory_str, end="")
                if not v.startswith(trajectory_str):
                    print(trajectory_str)
                    breakpoint()
                Colorize.print_cyan(v[len(trajectory_str) :])
            Colorize.print_blue("chosen", end=" ")
            Colorize.print_cyan(action)

        threshold = 3
        self.breakpoint(T, threshold)
        return action

    def ready(self) -> bool:
        actions = list(range(self.env.action_space.n))
        done_ready = [bool(self.sample_done(a)) for a in actions]
        next_state_ready = any([bool(self.sample_next_state(a)) for a in actions])
        reward_ready_done = [bool(self.sample_reward(a, True)) for a in actions]
        reward_ready_not_done = [bool(self.sample_reward(a, False)) for a in actions]
        reward_ready = [
            done or not_done
            for done, not_done in zip(reward_ready_done, reward_ready_not_done)
        ]
        return all(done_ready + reward_ready + [next_state_ready]) and super().ready()

    def rollout(self, state: ObsType, action: ActType, T: int) -> str:
        if self.debug >= 2:
            Colorize.print_header(
                f"Computing Q rollout for state {state} and action {action}:"
            )
        t = 0
        state_str = self.env.state_str(state)
        action_str = self.env.action_str(action)
        completions = [s for s in [state_str, action_str] if s]
        env = deepcopy(self.env)
        while True:
            query = [state_str + action_str]
            if self.oracle_transitions:
                next_state, reward, done, _ = env.step(action)
                completions.extend(
                    [
                        env.done_str(done) + env.done_stop(),
                        env.reward_str(reward) + env.reward_stop(),
                    ]
                )
                if done:
                    break
                completions.append(env.state_str(next_state) + env.state_stop())
            else:
                if t == self.max_steps:
                    break
                done_str = self.predict(
                    query,
                    name="done",
                    get_prompts=lambda: self.sample_done(action),
                    stop=self.env.done_stop(),
                    T=T,
                    valid=self.env.valid_done,
                )
                if done_str is None:
                    break
                completions.append(done_str)
                done = self.env.done(done_str)
                reward_str = self.predict(
                    query,
                    name="reward",
                    get_prompts=lambda: self.sample_reward(action=action, done=done),
                    stop=self.env.reward_stop(),
                    T=T,
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
                    T=T,
                    valid=self.env.valid_state,
                )
                if state_str is None:
                    break
                completions.append(state_str)
            action_str = self.generate_action(state_str, T)
            action = self.env.action(action_str)
            completions.append(action_str)
            query.append(action_str)
            t += 1

        return "".join(completions)


class Pi(Model[ObsType, ActType]):
    def _act(self, state: ObsType, T: int) -> ActType:
        if self.debug >= 2:
            Colorize.print_header(f"Computing pi action for state {state}:")
        state = self.env.state_str(state)

        action_str = self.generate_action(state, T)
        action = self.env.action(action_str)
        assert action is not None
        return action
