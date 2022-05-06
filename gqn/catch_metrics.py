import abc
import itertools
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import catch
from base_env import TimeStep
from catch import Obs
from dollar_lambda import command, option
from gym.spaces import Discrete
from metrics.encoder import Encoder as BaseEncoder
from metrics.metric import (
    Action,
    Episode,
    FailureReward,
    NonterminalReward,
    SuccessReward,
    TimeStepWithActions,
    Transition,
    get_trajectory,
)
from metrics.test_runner import TestRunner

ACTIONS = ["Left", "Stay", "Right"]


class Encoder(BaseEncoder, ABC):
    def actions(self):
        return range(3)

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def action_query(self, state: Obs) -> str:
        return self.state_str(state)

    def name(self) -> str:
        return self.state_action_str(TimeStep(Obs(1, 1, 4), 0, 0, False, Obs(0, 1, 3)))

    def nonterminal_reward_str(self, ts: TimeStep[Obs, int]) -> str:
        return self.status_str(ts.state)

    def reward_query(self, ts: TimeStep[Obs, int]) -> str:
        if ts.done:
            return self.state_action_str(ts) + " " + self.state_str(ts.next_state)
        else:
            return self.state_without_status_str(ts.state)

    def state_str(self, state: Obs) -> str:
        string = self.state_without_status_str(state)
        status = self.status_str(state)
        if status:
            string += " " + status
        return string

    def state_action_str(self, ts: TimeStep[Obs, int]) -> str:
        return " ".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
            ]
        )

    @abc.abstractmethod
    def state_without_status_str(self, state: Obs) -> str:
        ...

    def status_str(self, state: Obs) -> str:
        return ""

    def stop(self) -> List[str]:
        return [":", ";"]

    def transition_query(self, ts: TimeStep[Obs, int]) -> str:
        return self.state_str(ts.state) + " " + self.action_str(ts.action)

    def time_step_str(self, ts: TimeStep[Obs, int]) -> str:
        return " ".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
            ]
            + (
                [
                    self.state_str(ts.next_state),
                    self.terminal_reward_str(ts),
                ]
                if ts.done
                else []
            )
        )


class WithStatus(Encoder, ABC):
    @staticmethod
    @abc.abstractmethod
    def ball() -> str:
        ...

    def nonterminal_reward_str(self, ts: TimeStep[Obs, int]) -> str:
        return self.status_str(ts.state)

    @staticmethod
    @abc.abstractmethod
    def paddle() -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def failure_str() -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def nonterminal_str() -> str:
        ...

    def status_str(self, state: Obs) -> str:
        paddle_x, ball_x, ball_y = state
        same_x = paddle_x == ball_x
        if ball_y > 0:
            nonterminal = self.nonterminal_str()
            if nonterminal:
                nonterminal = ", " + nonterminal
            return f"[{self.paddle()}.x{'==' if same_x else '!='}{self.ball()}.x, {self.paddle()}.y>0{nonterminal}];"
        else:
            return ""

    def state_without_status_str(self, state: Obs) -> str:
        paddle_x, ball_x, ball_y = state
        return f"{self.paddle()}=({paddle_x},0) {self.ball()}=({ball_x},{ball_y})"

    @staticmethod
    @abc.abstractmethod
    def success_str() -> str:
        ...

    def terminal_reward_str(self, ts: TimeStep[Obs, int]) -> str:
        paddle_x, ball_x, ball_y = ts.next_state
        same_x = paddle_x == ball_x
        return f"[P.x{'==' if same_x else '!='}B.x, P.y==0, {self.success_str() if same_x else self.failure_str()}];"


class WithReward(WithStatus, ABC):
    @staticmethod
    def failure_str() -> str:
        return "R=0"

    @staticmethod
    def nonterminal_str() -> str:
        return "R=0"

    @staticmethod
    def success_str() -> str:
        return "R=1"


class WithoutReward(WithStatus, ABC):
    @staticmethod
    def failure_str() -> str:
        return "failure"

    @staticmethod
    def nonterminal_str() -> str:
        return ""

    @staticmethod
    def success_str() -> str:
        return "success"


class Terse(WithStatus, ABC):
    @staticmethod
    def ball() -> str:
        return "B"

    @staticmethod
    def paddle() -> str:
        return "P"


class Verbose(WithStatus, ABC):
    @staticmethod
    def ball() -> str:
        return "Ball"

    @staticmethod
    def paddle() -> str:
        return "Paddle"


class TerseWithReward(WithReward, Terse):
    pass


class VerboseWithReward(WithReward, Verbose):
    pass


class TerseWithoutReward(WithoutReward, Terse):
    pass


class VerboseWithoutReward(WithoutReward, Verbose):
    pass


def hopeless(s: Obs) -> bool:
    return s.ball_y < abs(s.paddle_x - s.ball_x)


def collect_trajectory(
    env: catch.Wrapper,
) -> Tuple[List[TimeStepWithActions], catch.Wrapper]:
    trajectory = []
    _env = deepcopy(env)
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        good_actions = []
        if not hopeless(state):
            for a in range(len(ACTIONS)):
                _env = deepcopy(env)
                s, _, _, _ = _env.step(a)
                if not hopeless(s):
                    good_actions.append(a)

        next_state, reward, done, _ = env.step(action)

        step = TimeStep(state, action, reward, done, next_state)
        trajectory.append(TimeStepWithActions(step, good_actions))
        state = next_state
    return trajectory, _env


COLUMNS = 4
ROWS = 5
ALL_START_STATES = [(COLUMNS // 2, bx, ROWS - 1) for bx in range(COLUMNS)]


def impossible(obs: Obs):
    return abs(obs.paddle_x - COLUMNS // 2) > (ROWS - 1 - obs.ball_y)


@command(
    parsers=dict(
        prompt_sizes=option(
            "prompt_sizes",
            default=(8,),
            type=lambda s: tuple(map(int, s.split(","))),
        )
    )
)
def main(
    prompt_sizes: Tuple[int, ...],
    debug: int = -1,
    encoder: Optional[str] = None,
    logprobs: int = 5,
    metric: Optional[str] = None,
    num_trajectories: int = 5,
    seed: int = 0,
):
    env = catch.Wrapper(catch.Env(gamma=1.0, rows=ROWS, columns=COLUMNS, seed=seed))
    success_trajectories_by_feature = defaultdict(list)
    failure_trajectories_by_feature = defaultdict(list)
    all_states = [
        Obs(px, bx, by)
        for px, bx, by in itertools.product(
            range(COLUMNS), range(COLUMNS), range(1, ROWS)
        )
        if not impossible(Obs(px, bx, by))
    ]
    trajectories_by_last_state_action = {
        (Obs(*state), action): []
        for state in all_states
        for action in range(len(ACTIONS))
    }
    envs_by_first_state = {Obs(*state): None for state in ALL_START_STATES}

    while any(
        [
            len(v) < num_trajectories
            for d in [
                success_trajectories_by_feature,
                failure_trajectories_by_feature,
                trajectories_by_last_state_action,
            ]
            for v in d.values()
        ]
        + [not e for e in envs_by_first_state.values()]
    ):
        trajectory, _env = collect_trajectory(env)
        trajectories_by_feature = (
            success_trajectories_by_feature
            if trajectory[-1].time_step.reward > 0
            else failure_trajectories_by_feature
        )
        trajectories_by_feature[trajectory[0].time_step.state.ball_x].append(
            get_trajectory(trajectory)
        )
        for i in range(len(trajectory)):
            last_time_step: TimeStepWithActions = trajectory[i]
            sub_trajectory = trajectory[: i + 1]
            step = last_time_step.time_step
            trajectories_by_last_state_action[step.state, step.action].append(
                sub_trajectory
            )
            envs_by_first_state[trajectory[0].time_step.state] = _env

    queries = trajectories_by_last_state_action
    envs = list(envs_by_first_state.values())
    failure_trajectories = list(failure_trajectories_by_feature.values())
    success_trajectories = list(success_trajectories_by_feature.values())

    action_space = env.action_space
    assert isinstance(action_space, Discrete)
    TestRunner().run(
        debug=debug,
        encoder_str=encoder,
        encoders=[
            TerseWithReward(),
            TerseWithoutReward(),
            VerboseWithReward(),
            VerboseWithoutReward(),
        ],
        failure_trajectories=failure_trajectories,
        filename="logs/catch-metrics.html",
        logprobs=logprobs,
        metric_str=metric,
        metrics=[
            Action(queries, num_actions=action_space.n),
            Episode(envs=envs),
            FailureReward(queries),
            NonterminalReward(queries),
            SuccessReward(queries),
            Transition(queries),
        ],
        prompt_sizes=list(prompt_sizes),
        seed=seed,
        success_trajectories=success_trajectories,
        title="Catch",
    )


if __name__ == "__main__":
    main()
