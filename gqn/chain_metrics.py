from abc import ABC
from collections import defaultdict
from copy import deepcopy
from typing import DefaultDict, List, Optional, Tuple, cast

import chain
from base_env import Env, TimeStep
from dollar_lambda import command, option
from gym.spaces import Discrete
from gym.wrappers import TimeLimit
from metrics.encoder import Encoder as BaseEncoder
from metrics.metric import (
    Action,
    Episode,
    FailureReward,
    NonterminalReward,
    SuccessReward,
    TimeStepWithActions,
    Trajectory,
    Transition,
    get_trajectory,
)
from metrics.test_runner import TestRunner


class Encoder(BaseEncoder, ABC):
    def actions(self):
        return range(3)

    def action_str(self, action: int) -> str:
        actions = ["Left", "Try goal", "Right"]
        return f"{actions[action]}:"

    def action_query(self, state: int) -> str:
        return self.state_str(state)

    def name(self) -> str:
        return self.time_step_str(TimeStep(3, 1, 0.0, True, 4))

    def nonterminal_reward_str(self, ts: TimeStep[int, int]) -> str:
        return self.status_str(ts.state)

    def reward_query(self, ts: TimeStep[int, int]) -> str:
        if ts.done:
            return (
                self.state_action_str(ts)
                + " "
                + self.state_without_status_str(ts.next_state)
            )
        else:
            return self.state_without_status_str(ts.state)

    def state_str(self, state: int) -> str:
        string = self.state_without_status_str(state)
        status = self.status_str(state)
        if status:
            string += " " + status
        return string

    def state_action_str(self, ts: TimeStep[int, int]) -> str:
        return " ".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
            ]
        )

    @staticmethod
    def state_without_status_str(state: int) -> str:
        return str(state)

    def status_str(self, state: int) -> str:
        return ""

    def stop(self) -> List[str]:
        return [":", "."]

    def transition_query(self, ts: TimeStep[int, int]) -> str:
        return self.state_str(ts.state) + " " + self.action_str(ts.action)

    def time_step_str(self, ts: TimeStep[int, int]) -> str:
        return " ".join(
            [
                self.state_str(ts.state),
                self.action_str(ts.action),
            ]
            + (
                [
                    self.state_without_status_str(ts.next_state),
                    self.terminal_reward_str(ts),
                ]
                if ts.done
                else []
            )
        )


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
    goal: int = 4,
    logprobs: int = 3,
    max_steps: int = 16,
    metric: Optional[str] = None,
    num_trajectories: int = 5,
    n: int = 8,
    seed: int = 0,
):
    class WithTerminalReward(Encoder):
        def status_str(self, state: int) -> str:
            return f"[at {goal}]." if state == goal else f"[not at {goal}]."

        def terminal_reward_str(self, ts: TimeStep[int, int]) -> str:
            assert ts.done
            return f"[{chain.REWARDS[ts.reward]}]."

    class WithNonterminalReward(Encoder):
        @staticmethod
        def _status_str(state: int) -> str:
            return f"at {goal}" if state == goal else f"not at {goal}"

        def status_str(self, state: int) -> str:
            return f"[{self._status_str(state)}, reward=0]."

        def terminal_reward_str(self, ts: TimeStep[int, int]) -> str:
            assert ts.done
            return f"[{self._status_str(ts.state)}, reward={int(ts.reward)}]."

    def collect_trajectory(env: Env) -> Tuple[List[TimeStepWithActions], Env]:
        trajectory = []
        _env = deepcopy(env)
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            if goal < state:
                good_actions = [0]
            elif goal == state:
                good_actions = [1]
            elif state < goal:
                good_actions = [2]
            else:
                raise RuntimeError()

            next_state, reward, done, _ = env.step(action)

            step = TimeStep(state, action, reward, done, next_state)
            trajectory.append(TimeStepWithActions(step, good_actions))
            state = next_state
        return trajectory, _env

    env = cast(
        chain.Env,
        TimeLimit(
            chain.Env(gamma=1.0, goal=goal, n=n, random_seed=seed),
            max_episode_steps=max_steps,
        ),
    )
    action_space = env.action_space
    assert isinstance(action_space, Discrete)

    success_trajectories_by_feature: DefaultDict[int, List[Trajectory]] = defaultdict(
        list
    )
    failure_trajectories_by_feature: DefaultDict[int, List[Trajectory]] = defaultdict(
        list
    )
    all_states = list(range(n))
    trajectories_by_last_state_action = {
        (state, action): [] for state in all_states for action in range(action_space.n)
    }
    envs_by_first_state = {state: None for state in all_states}

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
        trajectories_by_feature[trajectory[0].time_step.state].append(
            get_trajectory(trajectory)
        )
        envs_by_first_state[trajectory[0].time_step.state] = _env
        for i in range(len(trajectory)):
            last_state: TimeStepWithActions = trajectory[i]
            sub_trajectory = trajectory[: i + 1]
            trajectories_by_last_state_action[
                last_state.time_step.state, last_state.time_step.action
            ].append(sub_trajectory)

    queries = trajectories_by_last_state_action
    envs = list(envs_by_first_state.values())
    failure_trajectories = list(failure_trajectories_by_feature.values())
    success_trajectories = list(success_trajectories_by_feature.values())

    TestRunner().run(
        debug=debug,
        encoder_str=encoder,
        encoders=[
            WithTerminalReward(),
            WithNonterminalReward(),
        ],
        failure_trajectories=failure_trajectories,
        filename="logs/chain-metrics.html",
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
        title="Chain",
    )


if __name__ == "__main__":
    main()
