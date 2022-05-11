import abc
import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Tuple

import numpy as np
from base_env import Env, TimeStep
from gym.core import ActType, ObsType
from metrics.encoder import Encoder
from rl.gpt3 import GPT3


def _get_prob(target: str, logprobs: List[Dict[str, float]]) -> Tuple[float, str]:
    if not target or not logprobs:
        return 1, target

    def get_prob_rec(logprobs):
        while logprobs:
            head, *logprobs = logprobs
            for token, lp in head.items():
                prob = np.exp(lp)
                if target.startswith(token):
                    prob_rest, leftover = _get_prob(target[len(token) :], logprobs)
                    yield prob * prob_rest, leftover
            # logprobs = [
            #     {k: v for k, v in lp.items() if k not in head} for lp in logprobs
            # ]

    rec = list(get_prob_rec(logprobs))
    if not rec:
        return 0, target
    if all([leftover for prob, leftover in rec]):
        smallest_leftover = min([leftover for prob, leftover in rec], key=len)
        probs = [
            (prob, leftover) for prob, leftover in rec if leftover == smallest_leftover
        ]
    else:
        probs = [(prob, leftover) for prob, leftover in rec if not leftover]
    return max(probs)


def get_prob(target: str, logprobs: List[Dict[str, float]]) -> float:
    prob, leftover = _get_prob(target, logprobs)
    return prob


Trajectory = List[TimeStep[ObsType, ActType]]


@dataclass
class TimeStepWithActions(Generic[ObsType, ActType]):
    time_step: TimeStep[ObsType, ActType]
    actions: List[ActType]


@dataclass
class Metric(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def get_prompt(
        self,
        encoder: Encoder,
        failure_trajectories: List[List[Trajectory]],
        prompt_size: int,
        rng: np.random.Generator,
        success_trajectories: List[List[Trajectory]],
    ) -> str:
        def get_trajectories():
            rng.shuffle(failure_trajectories)
            rng.shuffle(success_trajectories)
            yield from self.prompt_trajectory_generator(
                failure_trajectories, success_trajectories
            )

        trajectories = [
            trajectories[rng.choice(len(trajectories))]
            for trajectories in itertools.islice(
                itertools.cycle(get_trajectories()), prompt_size
            )
        ]
        rng.shuffle(trajectories)
        return encoder.get_prompt(trajectories)

    @abc.abstractmethod
    def get_query(self, encoder: Encoder, trajectory: List[TimeStep]):
        ...

    def name(self) -> str:
        name = self.__class__.__name__
        return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()

    @abc.abstractmethod
    def prompt_trajectory_generator(
        self,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[List[Trajectory]]:
        ...

    @abc.abstractmethod
    def take_measurement(
        self,
        debug: int,
        encoder: Encoder,
        gpt3: GPT3,
        max_logprobs: int,
        prompt_size: int,
        rng: np.random.Generator,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[float]:
        ...


TrajectoryWithActions = List[TimeStepWithActions]


def get_trajectory(trajectory_with_actions: TrajectoryWithActions) -> Trajectory:
    return [step.time_step for step in trajectory_with_actions]


N = 30


@dataclass
class ProbabilityMetric(Metric, abc.ABC):
    queries: Dict[Any, List[TrajectoryWithActions]]

    def __hash__(self):
        return hash(self.name())

    def __len__(self) -> int:
        return N

    @abc.abstractmethod
    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        ...

    def get_query_trajectories(
        self, rng: np.random.Generator
    ) -> List[TrajectoryWithActions]:
        def get_trajectories() -> Iterator[TrajectoryWithActions]:
            while True:
                trajectories = [
                    bucket[rng.choice(len(bucket))] for bucket in self.queries.values()
                ]
                rng.shuffle(trajectories)
                yield from self._get_query_trajectories(trajectories)

        return list(itertools.islice(get_trajectories(), N))

    def take_measurement(
        self,
        debug: int,
        encoder: Encoder,
        gpt3: GPT3,
        max_logprobs: int,
        prompt_size: int,
        rng: np.random.Generator,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[float]:

        for trajectory in self.get_query_trajectories(rng):
            output = self.get_output(encoder, trajectory[-1])
            if not output:
                continue
            prompt = self.get_prompt(
                encoder=encoder,
                failure_trajectories=failure_trajectories,
                prompt_size=prompt_size,
                rng=rng,
                success_trajectories=success_trajectories,
            )
            query = self.get_query(encoder, get_trajectory(trajectory))
            full_prompt = f"{prompt}\n{query}"
            if debug >= 1:
                print(full_prompt)
                print()
                print("Output:")
                print(output)
            if debug >= 1:
                breakpoint()
            if debug >= 3:
                yield 0
                continue
            if debug >= 0:
                print("<", end="")
            stop = [o[-1] for o in output]
            completion = gpt3.get_full_completion(full_prompt, best_of=False, stop=stop)
            logprobs = completion["top_logprobs"]
            if debug >= 0:
                print(">", end="")
            prob = self.get_prob(debug, encoder, logprobs[:max_logprobs], output)
            if debug >= 1:
                print(prob)
                breakpoint()
            if prob is not None:
                yield prob

    @staticmethod
    def get_prob(
        debug: int,
        encoder: Encoder,
        logprobs: List[Dict[str, float]],
        output: List[str],
    ):
        output_probs = [get_prob(" " + o, logprobs) for o in output]
        if debug >= 2:
            print(output_probs)
            breakpoint()
        return sum(output_probs)

    @abc.abstractmethod
    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> List[str]:
        ...


@dataclass
class ActMetric(Metric, abc.ABC):
    @classmethod
    def get_query(cls, encoder: Encoder, trajectory: Trajectory) -> str:
        *_, last = trajectory
        return encoder.action_query(last.state)

    def prompt_trajectory_generator(
        self,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[List[Trajectory]]:
        if self._remove_query_prefix():
            for trajectories in success_trajectories:
                yield [
                    trajectory[i:]
                    for trajectory in trajectories
                    for i in range(len(trajectory))
                ]
        else:
            yield from success_trajectories

    @abc.abstractmethod
    def _remove_query_prefix(self) -> bool:
        ...

    @staticmethod
    def _success_threshold() -> float:
        return 0.0


@dataclass
class ModelMetric(ProbabilityMetric, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        ...

    @classmethod
    def get_query(cls, encoder: Encoder, trajectory: Trajectory) -> str:
        *rest, last = trajectory
        query = encoder.get_prompt([rest])
        if query:
            query += " "
        return query + cls._get_query(encoder, last)

    def prompt_trajectory_generator(
        self,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[List[Trajectory]]:
        for f, s in zip(failure_trajectories, success_trajectories):
            yield f
            yield s

    @staticmethod
    def _success_threshold() -> float:
        return 0.0


@dataclass
class Action(ProbabilityMetric, ActMetric):
    num_actions: int
    remove_query_prefix: bool = True

    def actions(self) -> List[int]:
        return list(range(self.num_actions))

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            actions = query[-1].actions
            if 0 < len(actions) < self.num_actions:
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return [encoder.action_str(a) for a in last_step.actions]

    def get_prob(
        self,
        debug: int,
        encoder: Encoder,
        logprobs: List[Dict[str, float]],
        output: List[str],
    ):

        prob = super().get_prob(
            debug=debug, encoder=encoder, logprobs=logprobs, output=output
        )
        all_probs = [
            get_prob(" " + encoder.action_str(a), logprobs) for a in self.actions()
        ]
        if sum(all_probs) > 1:
            breakpoint()
        return prob / sum(all_probs)

    def _remove_query_prefix(self) -> bool:
        return self.remove_query_prefix


@dataclass
class Episode(ActMetric):
    envs: List[Env]
    remove_query_prefix: bool = True

    def __len__(self) -> int:
        return N

    def _remove_query_prefix(self) -> bool:
        return self.remove_query_prefix

    def get_envs(self, rng: np.random.Generator) -> Iterator[Env]:
        while True:
            envs = list(self.envs)
            rng.shuffle(envs)
            yield from envs

    def take_measurement(
        self,
        debug: int,
        encoder: Encoder,
        gpt3: GPT3,
        max_logprobs: int,
        prompt_size: int,
        rng: np.random.Generator,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[float]:
        for env in itertools.islice(self.get_envs(rng), N):
            state = env.reset()
            return_ = 0
            done = False
            trajectory = []
            while not done:
                action = None
                while action is None:
                    prompt = self.get_prompt(
                        encoder=encoder,
                        failure_trajectories=failure_trajectories,
                        prompt_size=prompt_size,
                        rng=rng,
                        success_trajectories=success_trajectories,
                    )
                    prompt_with_query = (
                        prompt
                        + "\n"
                        + (
                            f"{encoder.get_prompt([trajectory])} "
                            if self._remove_query_prefix()
                            else ""
                        )
                        + encoder.action_query(state)
                    )
                    if debug >= 1:
                        print(prompt_with_query)
                    if debug >= 2:
                        breakpoint()
                        action = env.action_space.sample()
                    else:
                        if debug >= 0:
                            print("<", end="")
                        completion = gpt3.get_full_completion(
                            prompt_with_query, best_of=False, stop=encoder.stop()
                        )
                        action_str = completion["completion"]
                        if debug >= 0:
                            print(">", end="")
                        action = encoder.action(action_str)
                        if debug >= 1:
                            print("action:")
                            print(action_str)
                            breakpoint()
                next_state, reward, done, _ = env.step(action)
                trajectory.append(TimeStep(state, action, reward, done, next_state))
                state = next_state
                return_ += reward
            yield return_


@dataclass
class TerminalReward(ModelMetric, abc.ABC):
    include_criterion: Callable[[float], bool]

    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.reward_query(last_step)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if last_step.done and self.include_criterion(last_step.reward):
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return [encoder.terminal_reward_str(last_step.time_step)]


@dataclass
class FailureReward(TerminalReward):
    include_criterion: Callable[[float], bool] = lambda r: r <= 0


@dataclass
class SuccessReward(TerminalReward):
    include_criterion: Callable[[float], bool] = lambda r: r > 0


@dataclass
class NonterminalReward(ModelMetric):
    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.reward_query(last_step)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if not last_step.done:
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        reward_str = encoder.nonterminal_reward_str(last_step.time_step)
        if reward_str:
            return [reward_str]
        return []


@dataclass
class Transition(ModelMetric):
    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.transition_query(last_step)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if not last_step.done:
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return [encoder.state_str(last_step.time_step.next_state)]
