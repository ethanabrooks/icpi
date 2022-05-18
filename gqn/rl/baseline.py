import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Optional

import numpy as np
from base_env import Env
from numpy.random import Generator, default_rng
from rl.common import evaluate, make_env, make_log
from run_logger import HasuraLogger
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
)


@dataclass
class TabularQAgent:
    n_actions: int
    learning_rate: float
    discount_factor: float
    seed: int
    initial_q_value: float
    q: defaultdict = field(init=False)
    _rng: Generator = field(init=False)

    def __post_init__(self):
        self.q = defaultdict(
            lambda: self.initial_q_value * np.ones(self.n_actions, dtype=float)
        )
        self._rng = default_rng(self.seed)

    def update(
        self, cur_state: Hashable, action: int, reward: float, next_state: Hashable
    ):
        prev_q = self.q[cur_state][action]
        prediction_error = (
            reward + self.discount_factor * self.q[next_state].max() - prev_q
        )
        self.q[cur_state][action] = prev_q + self.learning_rate * prediction_error

    def act_random(self) -> int:
        return self._rng.integers(0, self.n_actions)

    def act(self, state: Hashable) -> int:
        q_vals = self.q[state]
        best_actions = np.flatnonzero(q_vals == q_vals.max())
        return self._rng.choice(best_actions)


def tabular_main(
    env_id: str,
    eval_interval: int,
    min_successes: int,
    logger: HasuraLogger,
    seed: int,
    total_steps: int,
    **kwargs,
):
    env = make_env(env_id, seed, hint=False)
    agent = TabularQAgent(
        env.action_space.n,
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

        if episode % eval_interval == 0:
            evaluate(
                logger=logger,
                env=env,
                eval_interval=eval_interval,
                act_fn=lambda t, s: agent.act(s),
                **log_info,
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
            agent.update(
                cur_state=state, action=action, reward=reward, next_state=next_state
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
            **log_info,
        )


class LoggingCallback(BaseCallback):
    def __init__(self, logger: HasuraLogger, start_time: float, verbose=0):
        super().__init__(verbose)
        self.run_logger = logger
        self.start_time = start_time

    def _on_step(self):
        if np.sum(self.locals["dones"]) > 0:
            assert len(self.locals["dones"]) == 1
            make_log(
                logger=self.run_logger,
                info=self.locals["infos"][0],
                rewards=self.locals["rewards"].tolist(),
                evaluation=False,
                use_model_prob=0.0,
                num_success=0,
                gamma=self.training_env.envs[0].gamma(),
                start_time=self.start_time,
                step=self.model.num_timesteps,
            )


class EvalCallback(BaseCallback):
    def __init__(
        self,
        logger: HasuraLogger,
        eval_env: Env,
        eval_interval: int,
        start_time: float,
        verbose=0,
    ):
        super().__init__(verbose)
        self.run_logger = logger
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.start_time = start_time

    def _on_step(self):
        evaluate(
            logger=self.run_logger,
            env=self.eval_env,
            eval_interval=self.eval_interval,
            act_fn=self.act,
            use_model_prob=0.0,
            success_buffer_size=0,
            gamma=self.eval_env.gamma(),
            start_time=self.start_time,
            step=self.model.num_timesteps,
        )

    def act(self, _, state):
        return self.model.predict(np.array(state))[0].item()


def deep_baseline(
    logger: HasuraLogger,
    model_name: str,
    env_id: str,
    total_steps: int,
    eval_interval: Optional[int],
    seed: int,
    **_,
):

    train_env = make_env(env_id, seed, False)
    eval_env = make_env(env_id, seed, False)
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[3, 3]),
        gamma=train_env.gamma(),
        seed=seed,
        verbose=0,
    )

    start_time = time.time()
    eval_callback = EvalCallback(logger, eval_env, eval_interval, start_time)
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList(
            [
                LoggingCallback(logger, start_time=start_time),
                EveryNTimesteps(n_steps=eval_interval, callback=eval_callback),
            ]
        ),
    )
