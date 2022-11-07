import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from numpy.random import Generator, default_rng
from rl.common import evaluate, make_env, make_log
from rl.lm import Data
from run_logger import HasuraLogger


@dataclass
class TabularQAgent:
    discount_factor: float
    exploration_bonus: bool
    initial_q_value: float
    learning_rate: float
    n_actions: int
    seed: int
    n: defaultdict = field(init=False)
    q: defaultdict = field(init=False)
    _rng: Generator = field(init=False)

    def __post_init__(self):
        self.n = defaultdict(
            lambda: self.initial_q_value * np.zeros(self.n_actions, dtype=float)
        )
        self.q = defaultdict(
            lambda: self.initial_q_value * np.ones(self.n_actions, dtype=float)
        )
        self._rng = default_rng(self.seed)

    def act(self, state: Hashable) -> int:
        q_vals = self.q[state]
        if self.exploration_bonus:
            q_vals += 1 / (1 + self.n[state])
        best_actions = np.flatnonzero(q_vals == q_vals.max())
        return self._rng.choice(best_actions)

    def act_random(self) -> int:
        return self._rng.integers(0, self.n_actions)

    def update(
        self,
        cur_state: Hashable,
        action: int,
        reward: float,
        done: bool,
        next_state: Hashable,
    ):
        prev_q = self.q[cur_state][action]
        prediction_error = reward - prev_q
        if not done:
            prediction_error += self.discount_factor * self.q[next_state].max()

        self.q[cur_state][action] = prev_q + self.learning_rate * prediction_error
        self.n[cur_state][action] += 1
        # from rich.pretty import pprint
        #
        # for s, v in self.q.items():
        #     if s == 4 and ((v[1] < v[0] != 1) or (v[1] < v[2] != 1)):
        #         pprint(f"state: {s}; action: {action}; value {v}")
        #         breakpoint()
        #     if s != 4 and ((v[0] < v[1] != 1) or (v[2] < v[1] != 1)):
        #         pprint(f"state: {s}; action: {action}; value {v}")
        #         breakpoint()
        #     if s < 4 and ((v[2] < v[1] != 1) or (v[2] < v[0] != 1)):
        #         pprint(f"state: {s}; action: {action}; value {v}")
        #         breakpoint()
        #     if s > 4 and ((v[0] < v[1] != 1) or (v[0] < v[2] != 1)):
        #         pprint(f"state: {s}; action: {action}; value {v}")
        #         breakpoint()


def tabular_main(
    env_id: str,
    eval_interval: int,
    min_successes: int,
    logger: HasuraLogger,
    seed: int,
    total_steps: int,
    **_,
):
    env = make_env(data=Data.code, env_id=env_id, seed=seed, hint=False)
    agent = TabularQAgent(
        n_actions=env.action_space.n,
        learning_rate=1,
        discount_factor=env.gamma(),
        initial_q_value=1,
        seed=seed,
    )

    num_successes = 0
    start_time = time.time()
    rng = np.random.default_rng(seed)
    T = 0
    for episode in itertools.count():
        use_agent_prob = 1 / (1 + math.exp(2 * (min_successes - num_successes)))

        log_info = dict(
            num_success=num_successes,
            use_model_prob=use_agent_prob,
            gamma=env.log_gamma(),
            seed=seed,
            start_time=start_time,
            step=T,
        )

        if eval_interval is not None and episode % eval_interval == 0:
            evaluate(
                act_fn=lambda _, s: agent.act(s),  # type: ignore
                env=env,
                eval_interval=eval_interval,
                logger=logger,
                **log_info,  # type: ignore
            )
        rewards = []
        t = 0
        info = {}
        state = env.reset()
        done = False
        while not done:
            use_agent = rng.random() < use_agent_prob
            action = agent.act(state) if use_agent else agent.act_random()
            next_state, reward, done, info = env.step(action)
            # timed_out = info.get("TimeLimit.truncated", False)
            agent.update(
                cur_state=state,
                done=False,
                action=action,
                reward=reward,
                next_state=next_state,
            )
            state = next_state
            rewards.append(reward)
            t += 1
            T += 1
            if T >= total_steps:
                print("Done!")
                return
        if (
            sum(env.gamma() ** t * r for t, r in enumerate(rewards))
            >= env.failure_threshold()
        ):
            num_successes += 1

        make_log(
            logger=logger,
            info=info,
            rewards=rewards,
            evaluation=False,
            **log_info,  # type: ignore
        )
