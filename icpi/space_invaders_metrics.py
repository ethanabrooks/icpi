from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import space_invaders
from base_env import TimeStep
from dollar_lambda import command, option
from gym.spaces import Discrete
from metrics.encoder import Encoder as BaseEncoder
from metrics.metric import (
    ModelMetric,
    TimeStepWithActions,
    Trajectory,
    TrajectoryWithActions,
)
from metrics.metric import Transition as BaseTransition
from metrics.metric import get_trajectory
from metrics.test_runner import TestRunner
from rl.common import get_value
from space_invaders import Alien, Obs

ACTIONS = ["left", "shoot", "right"]


class Encoder(BaseEncoder):
    def actions(self):
        return range(3)

    def action_str(self, state: Obs, action: int) -> str:
        if action == 1:
            return f"reward = {self.ship()}.{ACTIONS[action]}({self.alien()})\n{self.alien()}.descend()\n"
        else:
            return f"{self.ship()} = {self.ship()}.{ACTIONS[action]}()\n{self.alien()}.descend()\n"

    def action_query(self, state: Obs) -> str:
        hint = self.hint(state)
        query = self.hint_query(state)
        if hint:
            query += " " + hint
        return query + "\n"

    @staticmethod
    def alien() -> str:
        return "aliens"

    def hint(self, state: Obs) -> str:
        hint = " and ".join(
            [
                (
                    f"{self.ship()}.x == {self.alien()}[{i}].x"
                    if a.over(state.agent)
                    else f"{self.ship()}.x != {self.alien()}[{i}].x"
                )
                for i, a in enumerate(state.aliens)
                if not a.is_dead()
            ]
        )
        if hint:
            hint = f"and {hint}"
        return hint

    def hint_query(self, state: Obs) -> str:
        return self.state_str(state)

    def name(self) -> tuple[str, ...]:
        return tuple(
            self.time_step_str(
                TimeStep(
                    Obs(1, (Alien.spawn(1, 2),)),
                    1,
                    1,
                    False,
                    Obs(1, (Alien.dead(),)),
                )
            ).split("\n")
        )

    def nonterminal_reward_str(self, ts: TimeStep[Obs, int]) -> str:
        reward_str = f"assert reward == {ts.reward}"
        if ts.reward > 0:
            reward_str += " and ".join(
                [""]
                + [
                    f"alien[{i}] is None"
                    for i, a in enumerate(ts.next_state.aliens)
                    if a.is_dead()
                ]
            )
        return reward_str

    def reset_str(self) -> str:
        return f"\n{self.ship()}, {self.alien()} = reset()"

    def reward_query(self, ts: TimeStep[Obs, int]) -> str:
        return self.action_query(ts.state) + self.action_str(ts.state, ts.action)

    @staticmethod
    def ship() -> str:
        return "ship"

    def state_str(self, state: Obs) -> str:
        aliens = ", ".join([str(a) for a in state.aliens])
        return f"assert {self.ship()} == C{(state.agent, 0)} and {self.alien()} == [{aliens}]"

    def stop(self) -> List[str]:
        return [":", ";", "."]

    def terminal_reward_str(self, ts: TimeStep[Obs, int]) -> str:
        raise RuntimeError("Not implemented")

    def transition_query(self, ts: TimeStep[Obs, int]) -> str:
        reward_str = self.nonterminal_reward_str(ts)
        query = self.reward_query(ts)
        if reward_str:
            return query + reward_str
        else:
            return query

    def time_step_str(self, ts: TimeStep[Obs, int]) -> str:
        if ts.done:
            s = self.reward_query(ts) + self.nonterminal_reward_str(ts)
        else:
            s = self.transition_query(ts)
        return s

    def get_prompt(
        self,
        trajectories: "list[list[TimeStep]]",
    ) -> str:
        return "\n".join(
            [
                "\n".join(
                    [self.reset_str()]
                    + [self.time_step_str(ts) for ts in trajectory]
                    + [
                        self.time_step_str(last)
                        + "\n"
                        + self.state_str(last.next_state)
                        + " "
                        + self.hint(last.next_state)
                    ]
                )
                for *trajectory, last in trajectories
            ]
        )


class Terse(Encoder):
    @staticmethod
    def alien() -> str:
        return "a"

    @staticmethod
    def ship() -> str:
        return "s"


class WithNamedTuple(Encoder):
    @staticmethod
    def alien() -> str:
        return "a"

    @staticmethod
    def ship() -> str:
        return "s"


@dataclass
class AllSuccess(ModelMetric, ABC):
    def prompt_trajectory_generator(
        self,
        failure_trajectories: List[List[Trajectory]],
        success_trajectories: List[List[Trajectory]],
    ) -> Iterator[List[Trajectory]]:
        yield from success_trajectories


@dataclass
class Hint(AllSuccess, ModelMetric):
    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.hint_query(last_step.state)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if not all(a.is_dead() for a in last_step.state.aliens):
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return [" " + encoder.hint(last_step.time_step.state)]


@dataclass
class HitReward(AllSuccess, ModelMetric):
    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.reward_query(last_step)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if last_step.reward > 0:
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return ["\n" + encoder.nonterminal_reward_str(last_step.time_step)]


@dataclass
class MissReward(AllSuccess, ModelMetric):
    @classmethod
    def _get_query(cls, encoder: Encoder, last_step: TimeStep) -> str:
        return encoder.reward_query(last_step)

    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            last_step = query[-1].time_step
            if last_step.action == 1 and last_step.reward == 0:
                yield query

    def get_output(self, encoder: Encoder, last_step: TimeStepWithActions) -> list[str]:
        return ["\n" + encoder.nonterminal_reward_str(last_step.time_step)]


@dataclass
class Transition(AllSuccess, BaseTransition):
    def _get_query_trajectories(
        self, queries: List[TrajectoryWithActions]
    ) -> Iterator[Trajectory]:
        for query in queries:
            ts = query[-1].time_step

            spawned = False
            for a1, a2 in zip(ts.state.aliens, ts.next_state.aliens):
                if a1.is_dead() and not a2.is_dead():
                    spawned = True

            if not spawned and self.condition(ts):
                yield query

    @staticmethod
    @abstractmethod
    def condition(ts: TimeStep) -> bool:
        ...


@dataclass
class HitTransition(Transition):
    @staticmethod
    def condition(ts: TimeStep) -> bool:
        return ts.reward > 0


@dataclass
class MissTransition(Transition):
    @staticmethod
    def condition(ts: TimeStep) -> bool:
        return ts.action == 1 and ts.reward == 0


@dataclass
class MoveTransition(Transition):
    @staticmethod
    def condition(ts: TimeStep) -> bool:
        return ts.action != 1


def hopeless(s: Obs) -> bool:
    return any([a.escaped(s.agent) for a in s.aliens])


def collect_trajectory(
    env: space_invaders.Env,
) -> Tuple[List[TimeStepWithActions], space_invaders.Env]:
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


@command(
    parsers=dict(
        prompt_sizes=option(
            "prompt_sizes",
            default=(30,),
            type=lambda s: tuple(map(int, s.split(","))),
        )
    )
)
def main(
    prompt_sizes: Tuple[int, ...],
    debug: int = -1,
    encoder: Optional[str] = None,
    height: int = 4,
    logprobs: int = 5,
    metric: Optional[str] = None,
    max_aliens: int = 2,
    max_logprobs: int = 30,
    model_name: str = "code-davinci-002",
    num_trajectories: int = 30,
    require_cache: bool = False,
    seed: int = 0,
    width: int = 3,
):
    env = space_invaders.Env(
        height=height,
        width=width,
        max_aliens=max_aliens,
        max_step=8,
        random_seed=seed,
        hint=True,
    )
    success_trajectories = []
    failure_trajectories = []
    trajectories_by_last_state_action = {}
    envs_by_first_state = {}

    while any(
        [
            len(l) < num_trajectories
            for l in [
                trajectories_by_last_state_action,
                success_trajectories,
                failure_trajectories,
            ]
        ]
        # + [len(envs_by_first_state) < len(list(env.start_states()))]
    ):
        # print(
        #     [
        #         len(l)
        #         for l in [
        #             trajectories_by_last_state_action,
        #             success_trajectories,
        #             failure_trajectories,
        #             # envs_by_first_state,
        #         ]
        #     ]
        # )
        # print(
        #     [
        #         start_state
        #         for start_state in start_states
        #         if start_state not in envs_by_first_state
        #     ]
        # )

        trajectory, _env = collect_trajectory(env)
        trajectories = (
            success_trajectories
            if get_value(*get_trajectory(trajectory), gamma=1) > 0
            else failure_trajectories
        )
        trajectories.append(get_trajectory(trajectory))
        for i in range(len(trajectory)):
            last_time_step: TimeStepWithActions = trajectory[i]
            sub_trajectory = trajectory[: i + 1]
            step = last_time_step.time_step
            trajectories_by_last_state_action[step.state, step.action] = sub_trajectory
            envs_by_first_state[trajectory[0].time_step.state] = _env

    queries = {k: [v] for k, v in trajectories_by_last_state_action.items()}
    # envs = list(envs_by_first_state.values())

    action_space = env.action_space
    assert isinstance(action_space, Discrete)
    TestRunner().run(
        debug=debug,
        encoder_str=encoder,
        encoders=[Encoder()],
        failure_trajectories=[failure_trajectories],
        filename="logs/space-invader-metrics.html",
        logprobs=logprobs,
        max_logprobs=max_logprobs,
        metric_str=metric,
        metrics=[
            # Action(queries, num_actions=action_space.n),
            # Episode(envs=envs),
            # FailureReward(queries),
            HitReward(queries),
            MissReward(queries),
            Hint(queries),
            HitTransition(queries),
            MissTransition(queries),
            MoveTransition(queries),
        ],
        model_name=model_name,
        prompt_sizes=list(prompt_sizes),
        require_cache=require_cache,
        seed=seed,
        success_trajectories=[success_trajectories],
        title="Space Invaders",
    )


if __name__ == "__main__":
    main()
