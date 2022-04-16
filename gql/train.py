import math
import os
import shelve
from collections import deque
from pprint import pprint
from typing import Deque, List, Optional

import numpy as np
import openai
from env import Env
from model import GPT3, Pi, Q, TimeStep, get_value
from run_logger import HasuraLogger


def train(
    failure_threshold: float,
    gamma: float,
    goal: int,
    logger: Optional[HasuraLogger],
    max_steps: int,
    max_trajectory: int,
    min_successes: int,
    n: int,
    q_prompt_size: int,
    pi_prompt_size: int,
    seed: int,
    states: int,
    episodes: int,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = Env(states, goal, seed)

    buffer: Deque[List[TimeStep]] = deque()
    with shelve.open("completions/completions.pkl") as db:
        gpt3 = GPT3(db)
        pi = Pi(
            buffer=buffer,
            env=env,
            failure_threshold=failure_threshold,
            gpt3=gpt3,
            prompt_size=pi_prompt_size,
            rng=rng,
        )
        q = Q(
            buffer=buffer,
            env=env,
            failure_threshold=failure_threshold,
            gamma=gamma,
            gpt3=gpt3,
            max_steps=max_trajectory,
            prompt_size=q_prompt_size,
            rng=rng,
        )

        T = 0
        for i in range(episodes):
            done = False
            state = env.reset()
            optimal = gamma ** (abs(env.goal - state) + 1)
            trajectory: List[TimeStep] = []
            use_pi = i % 2 == 0
            timed_out = False
            t = 0
            while not done:
                value_quantities = [get_value(*t, gamma=1) for t in list(buffer)]
                value_quantities = sorted(value_quantities, reverse=True)[:n]
                value_sum = sum(value_quantities)
                use_model_prob = 1 / (1 + math.exp(2 * (min_successes - value_sum)))
                print("use_model_prob", round(use_model_prob, 3))
                model = pi if use_pi else q
                use_model = (rng.random() < use_model_prob) and model.ready()
                if use_model:
                    action = model.act(state)
                else:
                    action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                step = TimeStep(state, action, reward, None if done else next_state)
                t += 1
                if t >= max_steps:
                    done = timed_out = True
                if done:
                    T += t
                    print("episode", i)
                    print("state", state)
                    print("action", action)
                    print("reward", reward)
                    if use_pi:
                        returns = reward * gamma ** t
                        regrets = optimal - returns
                        log = dict(
                            episode=i,
                            step=T,
                            regret=regrets,
                            **{"return": returns},
                        )
                        pprint(log)
                        if logger is not None:
                            logger.log(**log)
                trajectory.append(step)
                state = next_state

            trajectory = trajectory[-max_trajectory:]
            if not timed_out:
                while trajectory:
                    buffer.append(trajectory)
                    head, *trajectory = trajectory

        print("done!")
