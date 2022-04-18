import abc
import math
from dataclasses import dataclass
from typing import Deque, List, Optional

from env import ACTIONS, REWARDS, Env
from gpt3 import GPT3
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


@dataclass
class Model(abc.ABC):
    buffer: Deque[List[TimeStep]]
    env: Env
    failure_threshold: float
    gamma: float
    gpt3: GPT3
    prompt_size: int
    rng: Generator
    debug: bool

    def act(
        self,
        state: int,
        best_trajectories: "list[str]",
    ) -> int:
        if self.ready():
            return self._act(state, best_trajectories=best_trajectories)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(
        self,
        state: int,
        best_trajectories: "list[str]",
    ) -> int:
        ...

    def get_good(self):
        return [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def ready(self) -> bool:
        return len(self.buffer) >= self.prompt_size

    def sample(self):
        prompts = [to_string(*t, env=self.env) for t in self.buffer]
        self.rng.shuffle(prompts)
        return prompts[: self.prompt_size]

    def sample_best(self):
        trajectories = sorted(
            self.get_good(), key=lambda t: get_value(*t, gamma=self.gamma), reverse=True
        )
        if len(trajectories) > self.prompt_size:
            advantages = [
                (to_string(*t, env=self.env), -1e5)
                for t in trajectories[: self.prompt_size]
            ]

            for trajectory in trajectories:
                worst_prompt, worst_adv = min(advantages, key=lambda x: x[1])
                prompt_list = [p for p, _ in advantages]
                self.rng.shuffle(prompt_list)
                first_state, *_ = trajectory
                prompt = "\n".join(
                    [*prompt_list, self.env.state_str(first_state.state)]
                )
                rollout = self.gpt3(prompt).lstrip().split(".")
                first_episode = []
                for word in rollout:
                    word = word.lstrip() + "."
                    first_episode.append(word)
                    if word in REWARDS.values():
                        break
                value = self.env.quantify(" ".join(first_episode), gamma=self.gamma)
                monte_carlo = get_value(*trajectory, gamma=self.gamma)
                adv = squash(value, self.gamma) - squash(monte_carlo, self.gamma)
                if (adv, self.rng.random()) > (worst_adv, self.rng.random()):
                    advantages.remove((worst_prompt, worst_adv))
                    advantages.append((to_string(*trajectory, env=self.env), adv))

            prompt_list = [p for p, _ in advantages][: self.prompt_size]
        else:
            prompt_list = [to_string(*t, env=self.env) for t in trajectories]
        self.rng.shuffle(prompt_list)
        return prompt_list


def squash(discounted_return: float, gamma: float) -> float:
    if discounted_return > 0:
        log_return = 4 + math.log(discounted_return, gamma)
        return max(log_return, 0)
    elif discounted_return < 0:
        log_return = -4 - math.log(-discounted_return, gamma)
        return min(log_return, 0)
    else:
        return 0


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    max_steps: int

    def _act(
        self,
        state: int,
        best_trajectories: "list[str]",
    ) -> int:
        assert isinstance(self.env.action_space, Discrete)
        actions = range(self.env.action_space.n)

        def get_values():
            for a in actions:
                yield self.value(
                    state,
                    action=a,
                    best_trajectories=best_trajectories,
                )

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

    def value(
        self,
        state: int,
        action: int,
        best_trajectories: "list[str]",
    ) -> str:
        # original_state = state
        # original_action = action
        completions = []
        state = self.env.state_str(state)
        action = self.env.action_str(action)
        random_trajectories = self.sample()
        new_prompt = "\n".join([*random_trajectories, f"{state} {action}"])
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
            self.rng.shuffle(best_trajectories)
            new_prompt = "\n".join([*best_trajectories, state])
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
    def _act(
        self,
        state: int,
        best_trajectories: "list[str]",
    ) -> int:
        state = self.env.state_str(state)
        action = None
        while action is None:
            prompt = "\n".join([*best_trajectories, state])
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
                best_trajectories = self.sample_best()
        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
