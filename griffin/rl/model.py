import abc
import itertools
import math
import operator
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Deque,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
)

from base_env import ActType, Env, ObsType, TimeStep
from gym.spaces import Discrete
from numpy.random import Generator
from rich.syntax import Syntax
from rl.common import Colorize, console, get_value
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
    break_on_invalid: bool
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    lm: LM
    max_prompts: int
    max_resamples: int
    rng: Generator
    sil: bool
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
        query,
        get_prompts: Callable[[], List[str]],
        ground_truth: Optional[str],
        name: str,
        stop: str,
        T: int,
        valid: Callable[[str], bool],
    ) -> Optional[Any]:
        if self.lm is None:
            for t in reversed(self.buffer):
                for ts in reversed(t):
                    if name == "action":
                        if ts.state == query:
                            return ts.action
                    else:
                        state, action = query
                        if ts.state == state and ts.action == action:
                            return dict(
                                done=ts.done, reward=ts.reward, state=ts.next_state
                            )[name]
            return None

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
                console.print(
                    Syntax("".join(prompts).rstrip("\n"), "python", theme="ansi_dark"),
                    end="",
                )
                console.rule(characters="·")
                console.print(Syntax("".join(query), "python", theme="ansi_light"))
            self.breakpoint(T, 4)
            completion = self.lm(
                new_prompt,
                stop=[stop],
                temperature=self.temperature,
                use_cache=self.use_cache,
            )
            if "state!=" in completion:
                breakpoint()
            completion = completion.replace("ball.x!=", "ball.x !=")
            completion += stop

            if self.debug >= 2:
                Colorize.print_prediction_type(name)
                Colorize.print_completion(completion)
                if ground_truth is not None:
                    Colorize.print_ground_truth(ground_truth)
            self.breakpoint(T, 4)
            if valid(completion):
                if "state!=" in completion:
                    breakpoint()
                return completion
            else:
                if self.debug >= 3:
                    Colorize.print_warning(f"Invalid {name}:", end=" ")
                    Colorize.print_completion(completion)
                if self.break_on_invalid:
                    self.breakpoint(T, 3)
                valid(completion)
        return None

    def ready(self) -> bool:
        return bool(self.sample_best())

    def sample_best(self) -> List[str]:
        trajectories = list(self.success_buffer)
        trajectories = [
            trajectory[start:stop]
            for trajectory in trajectories
            for start, stop in itertools.combinations(range(len(trajectory) + 1), 2)
            if self.successful(trajectory[start:stop])
        ]
        self.rng.shuffle(trajectories)
        return [to_string(*t, env=self.env) for t in trajectories]

    def successful(self, trajectory: List[TimeStep]) -> bool:
        return not self.sil or self.get_value(trajectory) > self.env.failure_threshold()

    def generate_action(self, state: "ObsType | str", T: int) -> Optional[str]:
        query = state if self.lm is None else ["", self.env.initial_str(), state]
        maybe_action = self.predict(
            ground_truth=None,
            query=query,
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
    balance_prompts: bool
    constrain_prompts: bool
    max_steps: int
    predict_transitions: bool

    def _act(self, state: ObsType, T: int) -> ActType:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_rollouts():
            for action in actions:
                yield self.rollout(state, action, T)

        rollouts, returns = zip(*get_rollouts())
        action_returns = list(zip(actions, returns))
        self.rng.shuffle(action_returns)
        action, value = max(
            action_returns,
            key=lambda x: (x[1], self.rng.random()),
        )

        if self.debug >= 1:
            print()
            Colorize.print_header("Q prompts")
            Colorize.print_prediction_type("state:", end=" ")
            Colorize.print_green(state)
            for a, v in zip(actions, rollouts):
                Colorize.print_prediction_type("action:", end=" ")
                Colorize.print_green(a)
                trajectory_strings = [
                    self.env.state_str(state),
                    self.env.action_str(a),
                ]
                trajectory_str = "".join(trajectory_strings)
                console.print("rollout:", trajectory_str, end="")
                if not v.startswith(trajectory_str):
                    console.print(trajectory_str)
                    breakpoint()
                Colorize.print_completion(v[len(trajectory_str) :])
            Colorize.print_prediction_type("chosen", end=" ")
            Colorize.print_green(action)

        threshold = 3
        self.breakpoint(T, threshold)
        return action

    def balance(self, *lists: List) -> List[List]:
        assert self.balance_prompts
        if not lists:
            return [[]]
        max_len = max(len(l) for l in lists)
        return [self.extend(l, max_len) for l in lists]

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

    def rollout(self, state: ObsType, action: ActType, T: int) -> Tuple[str, float]:
        if self.debug >= 2:
            Colorize.print_header(
                f"Computing Q rollout for state {state} and action {action}"
            )
        t = 0
        state_u = state
        action_u = action
        completions = []
        if self.lm is not None:
            state_u = self.env.state_str(state)
            action_u = self.env.action_str(action)
            completions = [x for x in [state_u, action_u] if x not in ("", None)]
        discounted_return = 0

        def update_return(r: float):
            return discounted_return + self.env.gamma() ** t * r

        true_done = False
        env = deepcopy(self.env)
        while True:
            query = [state_u, action_u] if self.lm is None else [state_u + action_u]
            if not true_done:
                true_state, true_reward, true_done, _ = env.step(action)
            if self.predict_transitions:
                if t == self.max_steps:
                    break
                true_done_str = self.env.done_str(true_done)
                done_u = self.predict(
                    ground_truth=None if true_done else true_done_str,
                    query=query,
                    name="done",
                    get_prompts=lambda: self.sample_done(action),
                    stop=self.env.done_stop(),
                    T=T,
                    valid=self.env.valid_done,
                )
                if done_u is None:
                    break
                completions.append(done_u)
                done = done_u if self.lm is None else self.env.done(done_u)
                # noinspection PyUnboundLocalVariable
                true_reward_str = self.env.reward_str(true_reward)
                reward_u = self.predict(
                    ground_truth=None if true_done else true_reward_str,
                    query=query,
                    name="reward",
                    get_prompts=lambda: self.sample_reward(action=action, done=done),
                    stop=self.env.reward_stop(),
                    T=T,
                    valid=self.env.valid_reward,
                )
                if reward_u is None:
                    break
                completions.append(reward_u)
                discounted_return = update_return(self.env.reward(reward_u))
                if done:
                    break
                # noinspection PyUnboundLocalVariable
                true_state_str = self.env.state_str(true_state)
                state_u = self.predict(
                    ground_truth=None if true_done else true_state_str,
                    query=query,
                    name="state",
                    get_prompts=lambda: self.sample_next_state(action=action),
                    stop=self.env.state_stop(),
                    T=T,
                    valid=self.env.valid_state,
                )
                if state_u is None:
                    break
                completions.append(state_u)
            elif not self.predict_transitions:
                if true_done:
                    break
                discounted_return = update_return(true_reward)
            else:
                raise RuntimeError("Unhandled case")
            action_u = self.generate_action(state_u, T)
            action = self.env.action(action_u)
            completions.append(action_u)
            t += 1

        return "" if self.lm is None else "".join(completions), discounted_return

    def sample_done(self, action: int) -> List[str]:
        time_steps = [
            ts
            for t in self.buffer
            for ts in t
            if (not self.constrain_prompts) or (ts.action == action)
        ]
        self.rng.shuffle(time_steps)
        if self.balance_prompts:
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
            time_steps = balanced
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.done_str(ts.done)
            + self.env.done_stop()
            + "\n"
            for ts in time_steps
        ]

    def sample_next_state(self, action: int) -> List[str]:
        def get_time_steps(*trajectories: List[TimeStep]) -> List[TimeStep]:
            return [
                ts
                for t in trajectories
                for ts in t
                if (not self.constrain_prompts) or (ts.action == action and not ts.done)
            ]

        if self.balance_prompts:
            buffer = [t for t in self.buffer]
            self.rng.shuffle(buffer)
            successful = get_time_steps(*[t for t in buffer if self.successful(t)])
            unsuccessful = get_time_steps(
                *[t for t in buffer if not self.successful(t)]
            )
            if not successful:
                balanced = unsuccessful
            elif not unsuccessful:
                balanced = successful
            else:
                successful, unsuccessful = self.balance(successful, unsuccessful)
                # if len(successful) > 3:
                #     breakpoint()
                balanced = [
                    ts for (s, u) in zip(successful, unsuccessful) for ts in [s, u]
                ]
            buffer = balanced
        else:
            buffer = list(get_time_steps(*self.buffer))
        self.rng.shuffle(buffer)
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.state_str(ts.next_state)
            + "\n"
            for ts in buffer
        ]

    def sample_reward(self, action: int, done: bool) -> List[str]:
        if self.balance_prompts:
            rewards = defaultdict(list)
            buffer = [t for t in self.buffer]
            self.rng.shuffle(buffer)
            for t in buffer:
                for ts in t:
                    if (not self.constrain_prompts) or (
                        ts.action == action and ts.done == done
                    ):
                        rewards[ts.reward].append(ts)
            balanced = self.balance(*rewards.values())
            # if rewards and max(len(r) for r in rewards.values()) > 3:
            #     breakpoint()
            balanced = [ts for time_steps in zip(*balanced) for ts in time_steps]
            buffer = balanced
        else:
            buffer = [
                ts
                for t in self.buffer
                for ts in t
                if (not self.constrain_prompts)
                or (ts.action == action and ts.done == done)
            ]
        self.rng.shuffle(buffer)
        return [
            self.env.state_str(ts.state)
            + self.env.action_str(ts.action)
            + self.env.reward_str(ts.reward)
            + self.env.reward_stop()
            + "\n"
            for ts in buffer
        ]


class Pi(Model[ObsType, ActType]):
    def _act(self, state: ObsType, T: int) -> ActType:
        if self.debug >= 2:
            Colorize.print_header(f"Computing pi action for state {state}")
        action_str = self.generate_action(state, T)
        action = self.env.action(action_str)
        assert action is not None
        return action
