import itertools
from abc import ABC
from copy import deepcopy
from typing import List, Optional

import bandit
import numpy as np
from base_env import TimeStep, ActType, ObsType
from catch import Env
from dollar_lambda import command
from metrics.encoder import Encoder as BaseEncoder
from metrics.metric import Actions as BaseActions
from metrics.test_runner import TestRunner as BaseTestRunner
from metrics.test_runner import TimeStepWithActions


@command()
def main(
    debug: int = -1,
    seed: int = 0,
    logprobs: int = 3,
    metric: Optional[str] = None,
    num_steps: int = 5,
    num_actions: int = 2,
    num_arms: int = 2,
    num_episodes: int = 20,
    prompt_size: int = 8,
    remove_query_prefix: bool = False,
):
    class Actions(BaseActions):
        @classmethod
        def all_actions(cls) -> List[int]:
            return list(range(num_actions))

    class Encoder(BaseEncoder, ABC):
        def action(self, action_str: str):
            matches = [
                a for a in list(range(num_actions)) if self.action_str(a) == action_str
            ]
            return next(matches, None)

        def action_query(self, state: ObsType) -> str:
            return ""

        def _name(
            self, obs: np.ndarray, reward: float, done: bool, next_obs: ObsType
        ) -> str:
            return self.state_action_str(TimeStep(obs, 0, reward, done, next_obs))

        def name(self) -> str:
            array = np.linspace(0, 1, num_actions)
            return self._name(array, 1.0, True, array)

        def reward_query(self, ts: TimeStep[ObsType, ActType]) -> str:
            return self.state_str(ts.state)

        def state_action_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return " ".join(
                [
                    self.action_str(ts.action),
                    self.nonterminal_reward_str(ts.reward),
                ]
            )

        def terminal_reward_str(self, reward: float, next_state: ObsType) -> str:
            raise RuntimeError()

        def transition_query(self, ts: TimeStep[ObsType, ActType]) -> str:
            raise RuntimeError()

        def time_step_str(self, ts: TimeStep[ObsType, ActType]) -> str:
            return " ".join(
                [
                    self.action_str(ts.action),
                    self.nonterminal_reward_str(ts.reward),
                ]
            )

    class NoMath(Encoder):
        def action_str(self, action: int) -> str:
            actions = ["Left", "Try goal", "Right"]
            return f"{actions[action]}:"

        def nonterminal_reward_str(self, reward: float) -> str:
            return f"{reward}."

        def state_str(self, state: int) -> str:
            return ""

    def collect_trajectory(env: bandit.Env) -> List[TimeStepWithActions]:
        trajectory = []
        state = env.reset()
        done = False
        tried = {}
        while not done:
            action = env.action_space.sample()
            tried |= {action}
            if len(tried) == num_actions:
                good_actions = state.argmax()
            else:
                good_actions = [a for a in range(num_actions) if a not in tried]

            next_state, reward, done, _ = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            trajectory.append(TimeStepWithActions(step, good_actions))
            state = next_state
        return trajectory

    env = bandit.Env(num_steps=num_steps, random_seed=seed)

    class TestRunner(BaseTestRunner):
        def get_envs(self) -> List[Env]:
            return [deepcopy(env) for _ in range(num_episodes)]

    all_policies = list(
        itertools.product(*[range(num_actions) for _ in range(num_steps)])
    )

    trajectories_by_policy = {policy: [] for policy in all_policies}

    while any([not v for v in trajectories_by_policy.values()]):
        trajectory = collect_trajectory(env)
        policy = [ts.time_step.action for ts in trajectory]
        trajectories_by_policy[tuple(policy)].append(trajectory)

    TestRunner(
        all_start_states=[],
        all_states=all_policies,
        env=env,
        prompt_size=prompt_size,
        remove_query_prefix=remove_query_prefix,
        seed=seed,
        trajectories_by_feature=list(trajectories_by_policy.values()),
        trajectories_by_last_state=trajectories_by_policy,
    ).run(
        actions=Actions(),
        debug=debug,
        encoders=[NoMath()],
        filename="logs/chain-metrics.html",
        logprobs=logprobs,
        metric=metric,
        title="Bandit",
    )


if __name__ == "__main__":
    main()
