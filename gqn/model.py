import abc
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

import numpy as np
from base_env import Env, TimeStep
from gpt3 import GPT3
from gym.spaces import Discrete
from numpy.linalg import norm
from numpy.random import Generator


def to_string(*_trajectory: TimeStep, env) -> str:
    return " ".join([env.ts_to_string(ts) for ts in _trajectory])


def get_value(*trajectory: TimeStep, gamma: float) -> float:
    return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


@dataclass
class Model(abc.ABC):
    buffer: Deque[List[TimeStep]]
    env: Env
    debug: int
    delta: float
    failure_threshold: float
    gamma: float
    gpt3: GPT3
    max_steps: int
    prompt_size: int
    rng: Generator

    def act(self, state: int) -> int:
        if self.ready():
            return self._act(state)
        return self.env.action_space.sample()

    @abc.abstractmethod
    def _act(self, state: int) -> int:
        ...

    def get_good(self):
        return [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]

    def ready(self) -> bool:
        return len(self.buffer) >= self.prompt_size

    def sample(self):
        prompts = list(self.buffer)
        self.rng.shuffle(prompts)
        return prompts[: self.prompt_size]

    def sample_best(self):
        trajectories = sorted(
            self.get_good(), key=lambda t: get_value(*t, gamma=self.gamma), reverse=True
        )
        unique = dict()

        for trajectory in trajectories:
            if len(unique) == self.prompt_size:
                break

            def successor_representation(
                *trajectory: TimeStep, gamma: float
            ) -> np.ndarray:
                representation = 0
                for t, ts in enumerate(trajectory):
                    representation += gamma**t * self.env.successor_feature(ts.state)
                assert isinstance(representation, np.ndarray)
                return representation

            rep1 = successor_representation(*trajectory, gamma=self.gamma)
            different = True
            for (rep2, _) in unique.values():
                if cosine_similarity(rep1, rep2) > self.delta:
                    different = False
                    break
            if different:
                prompt = to_string(*trajectory, env=self.env)
                unique[prompt] = (rep1, trajectory)

        prompts = [(k, v1, v2) for k, (v1, v2) in unique.items()]
        self.rng.shuffle(prompts)
        return prompts


def reformat(completion: str) -> str:
    return f"{completion.lstrip()}."


@dataclass
class Q(Model):
    def _act(self, state) -> int:
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
            key=lambda x: (
                self.env.quantify(x[1], gamma=self.gamma),
                self.rng.random(),
            ),
        )

        if self.debug >= 1:
            print("Q")
            print("state", state)
            for a, v in zip(actions, values):
                print("action", a)
                print("value", v)
            print("chosen", action)
        if self.debug >= 3:
            breakpoint()
        return action

    def value(self, state, action: int) -> str:
        t = 0
        action_str = self.env.action_str(action)
        state_str = self.env.state_str(state)
        completions = [state_str, action_str]
        env = deepcopy(self.env)

        path = Path("logs/transitions.pkl")
        if path.exists():
            with path.open("rb") as f:
                transitions_to_serialize = pickle.load(f)
        else:
            transitions_to_serialize = []
        path = Path("logs/actions.pkl")
        if path.exists():
            with path.open("rb") as f:
                actions_to_serialize = pickle.load(f)
        else:
            actions_to_serialize = []

        recorded_transition = False
        recorded_action = False

        while True:
            if t == self.max_steps:
                state_or_reward = (
                    self.env.time_out_str()
                )  # TODO: can we eliminate this?
                completions.append(state_or_reward)
                break
            else:
                trajectories = self.sample()
                prompts = [to_string(*t, env=self.env) for t in trajectories]
                new_prompt = "\n".join([*prompts, f"{state_str} {action_str}"])
                if self.debug >= 2:
                    print("Q prompt:")
                    print(new_prompt)

                state_or_reward, *_ = self.gpt3(new_prompt).lstrip().split(".")
                state_or_reward = reformat(state_or_reward)
                state = env.env._observation()
                next_state, reward, done, _ = env.step(env.action(action_str))
                last_step = TimeStep(
                    state=state,
                    action=env.action(action_str),
                    reward=reward,
                    done=done,
                    next_state=next_state,
                )

            if self.debug >= 2:
                print("state/reward", state_or_reward)
            if self.debug >= 4:
                breakpoint()
            completions.append(state_or_reward)
            if not recorded_transition:
                # print(state_or_reward, "||", env.state_str(next_state))
                # breakpoint()
                transitions_to_serialize.append(trajectories + [[last_step]])
                recorded_transition = True
            if self.env.done(state_or_reward):
                break
            state_str = state_or_reward
            prompts, _, trajectories = zip(*self.sample_best())
            new_prompt = "\n".join([*prompts, state_str])
            if self.debug >= 2:
                print("Q prompt:")
                print(new_prompt)

            good_actions = []

            action_str, *_ = self.gpt3(new_prompt).lstrip().split(".")
            action_str = reformat(action_str)
            if not recorded_action:
                _state = env.env._observation()
                # print("State:", _state)
                for _action_str in env.actions():
                    _env = deepcopy(env)
                    _action = _env.action(_action_str)
                    next_state, reward, done, _ = _env.step(_action)
                    if _action == env.action(action_str):
                        last_step = TimeStep(
                            state=_state,
                            action=_action,
                            reward=reward,
                            done=done,
                            next_state=next_state,
                        )

                    paddle_x, ball_x, ball_y = next_state
                    hopeless = abs(ball_x - paddle_x) > ball_y
                    # print("Next state:", next_state, "hopeless", hopeless)
                    if not hopeless:
                        good_actions.append(_action)
                if 0 < len(good_actions) < 3:
                    actions_to_serialize.append(
                        ([*trajectories, [last_step]], good_actions)
                    )
                    recorded_action = True
            t += 1
            if self.debug >= 2:
                print("action", action_str)
            if self.debug >= 4:
                breakpoint()
            completions.append(action_str)

        with Path("logs/transitions.pkl").open("wb") as f:
            pickle.dump(transitions_to_serialize, f)
        with Path("logs/actions.pkl").open("wb") as f:
            pickle.dump(actions_to_serialize, f)
        return " ".join(completions)


class Pi(Model):
    def _act(self, state) -> int:
        state = self.env.state_str(state)
        action = None
        t = 0
        while action is None:
            if t > self.max_steps:
                return self.env.action_space.sample()
            prompts, _, _ = zip(*self.sample_best())
            prompt = "\n".join([*prompts, state])
            if self.debug >= 1:
                print("pi prompt:")
                print(prompt)
            completion = self.gpt3(prompt).lstrip()
            maybe_action, *_ = completion.split(".")
            if self.debug >= 1:
                print("Action:", maybe_action)
            if self.debug >= 3:
                breakpoint()

            action = self.env.action(maybe_action + ".")
            t += 1

        return action

    def ready(self) -> bool:
        trajectories = [
            t for t in self.buffer if get_value(*t, gamma=1) > self.failure_threshold
        ]
        return len(trajectories) > 0
