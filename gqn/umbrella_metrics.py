import itertools
from abc import ABC
from string import ascii_lowercase
from typing import List, Optional

import numpy as np
import umbrella
from base_env import TimeStep
from dollar_lambda import command, option
from gym.core import ActType, ObsType
from gym.spaces import Discrete
from metrics.encoder import Encoder as BaseEncoder
from metrics.metric import (
    Action,
    Episode,
    FailureReward,
    NonterminalReward,
    SuccessReward,
)
from metrics.test_runner import TestRunner, TimeStepWithActions


@command(
    parsers=dict(
        prompt_sizes=option(
            "prompt_sizes",
            default=(8, 12),
            type=lambda s: tuple(map(int, s.split(","))),
        )
    )
)
def main(
    prompt_sizes: tuple,
    debug: int = -1,
    encoder: Optional[str] = None,
    seed: int = 0,
    logprobs: int = 5,
    metric: Optional[str] = None,
    num_prompts: int = 1,
    num_steps: int = 2,
    num_colors: int = 2,
):
    class Encoder(BaseEncoder, ABC):
        def actions(self):
            return range(num_colors)

        def action_query(self, state: ObsType) -> str:
            return self.state_str(state)

        def name(self) -> str:
            return " ".join(
                [
                    self.state_str(0),
                    self.action_str(0),
                    self.terminal_reward_str(TimeStep(1, 0, 1.0, True, 1)),
                ]
            )

        def reward_query(self, ts: TimeStep[ObsType, ActType]) -> str:
            return self.state_str(ts.state) + " " + self.action_str(ts.action)

        def state_str(self, state: int) -> str:
            return f"{umbrella.COLORS[state]}."

        def stop(self):
            return [".", ":"]

        def transition_query(self, ts: TimeStep[ObsType, ActType]) -> str:
            return " ".join([self.state_str(ts.state), self.action_str(ts.action)])

        def time_step_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            string = self.state_str(ts.state) + " " + self.action_str(ts.action)
            if ts.done:
                string += " " + self.terminal_reward_str(ts)
            else:
                nonterminal_reward = self.nonterminal_reward_str(ts)
                if nonterminal_reward:
                    string += " " + nonterminal_reward
            return string

    class NumericRewards(Encoder):
        def action_str(self, action: int) -> str:
            return f"{action}:"

        def nonterminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return f"reward={int(ts.reward)}."

        def terminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return f"reward={int(ts.reward)}."

    class NoNonterminalRewards(Encoder):
        def action_str(self, action: int) -> str:
            return f"{action}."

        def nonterminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return ""

        def terminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return f"{umbrella.REWARDS[ts.reward]}."

    class TerseStates(Encoder):
        def action_str(self, action: int) -> str:
            return f"{action}."

        def nonterminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return ""

        def state_str(self, state: int) -> str:
            return f"{ascii_lowercase[state]}."

        def terminal_reward_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return f"{umbrella.REWARDS[ts.reward]}."

    def collect_trajectory(env: umbrella.Env) -> List[TimeStepWithActions]:
        trajectory = []
        initial_state = state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            if done:
                good_actions = [initial_state]
            else:
                good_actions = [a for a in range(num_colors)]
            step = TimeStep(state, action, reward, done, next_state)
            trajectory.append(TimeStepWithActions(step, good_actions))
            state = next_state
        return trajectory

    env = umbrella.Env(num_colors=num_colors, num_steps=num_steps, random_seed=seed)
    action_space = env.action_space
    assert isinstance(action_space, Discrete)

    all_sequences = list(
        itertools.product(
            *[
                itertools.product(range(num_colors), range(num_colors))
                for _ in range(num_steps)
            ]
        )
    )

    all_start_states = list(range(num_colors))
    successful_by_start_state = {policy: [] for policy in all_start_states}
    unsuccessful_by_start_state = {policy: [] for policy in all_start_states}
    trajectories_by_sequence = {
        sequence[: i + 1]: None
        for sequence in all_sequences
        for i in range(len(sequence))
    }
    envs_by_start_state = {s: None for s in all_start_states}

    while any(
        [
            not v
            for v in [
                *trajectories_by_sequence.values(),
                *successful_by_start_state.values(),
                *unsuccessful_by_start_state.values(),
            ]
        ]
    ):
        trajectory, _env = collect_trajectory(env)
        start_state = trajectory[0].time_step.state
        trajectories = (
            successful_by_start_state
            if trajectory[-1].time_step.reward == 1
            else unsuccessful_by_start_state
        )
        trajectories[start_state].append(trajectory)
        envs_by_start_state[start_state] = _env
        for i in range(len(trajectory)):
            _trajectory = trajectory[: i + 1]
            sequence = [(ts.time_step.state, ts.time_step.action) for ts in _trajectory]
            trajectories_by_sequence[tuple(sequence)] = _trajectory

    trajectories_by_feature = [
        [*s, *u]
        for s, u in zip(
            successful_by_start_state.values(), unsuccessful_by_start_state.values()
        )
    ]
    queries = list(trajectories_by_sequence.values())
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(queries)
    envs = list(envs_by_start_state.values())

    TestRunner(
        prompt_sizes=list(prompt_sizes),
        remove_query_prefix=[False],
        seed=seed,
        trajectories_by_feature=trajectories_by_feature,
    ).run(
        debug=debug,
        encoder_str=encoder,
        encoders=[
            NumericRewards(),
            NoNonterminalRewards(),
            TerseStates(),
        ],
        filename="logs/umbrella-metrics.html",
        logprobs=logprobs,
        metric_str=metric,
        metrics=[
            Action(queries, num_actions=action_space.n),
            Episode(envs=envs),
            FailureReward(queries),
            NonterminalReward(queries),
            SuccessReward(queries),
        ],
        num_prompts=num_prompts,
        title=f"Umbrella, {num_colors} colors",
    )


if __name__ == "__main__":
    main()
