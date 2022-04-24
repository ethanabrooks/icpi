import math
import os
import time
from collections import deque
from pprint import pprint
from typing import Deque, List

import bandit
import catch
import chain
import numpy as np
import openai
from bandit import Bandit
from catch import Catch
from model import GPT3, Pi, Q, TimeStep
from run_logger import HasuraLogger


def train(
    debug: int,
    delta: float,
    env_id: str,
    failure_threshold: float,
    gamma: float,
    logprobs: int,
    logger: HasuraLogger,
    max_steps: int,
    max_trajectory: int,
    min_successes: int,
    q_prompt_size: int,
    pi_prompt_size: int,
    seed: int,
    temperature: float,
    top_p: float,
    total_steps: int,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    if env_id == "bandit":
        env = bandit.Wrapper(Bandit(mapping_seed=seed, num_actions=3))
    elif env_id == "catch":
        env = catch.Wrapper(Catch(columns=4, gamma=gamma, rows=5, seed=seed))
    elif env_id == "chain":
        env = chain.Chain(gamma=gamma, goal=4, n=8, random_seed=seed)
    else:
        raise RuntimeError()

    buffer: Deque[List[TimeStep]] = deque()
    gpt3 = GPT3(
        debug=debug,
        logprobs=logprobs,
        logger=logger,
        temperature=temperature,
        top_p=top_p,
    )
    pi = Pi(
        buffer=buffer,
        delta=delta,
        env=env,
        failure_threshold=failure_threshold,
        gamma=gamma,
        gpt3=gpt3,
        max_steps=max_trajectory,
        prompt_size=pi_prompt_size,
        rng=rng,
        debug=debug,
    )
    q = Q(
        buffer=buffer,
        delta=delta,
        env=env,
        failure_threshold=failure_threshold,
        gamma=gamma,
        gpt3=gpt3,
        max_steps=max_trajectory,
        prompt_size=q_prompt_size,
        rng=rng,
        debug=debug,
    )

    T = 0
    episodes = 0
    start_time = time.time()
    while T < total_steps:
        done = False
        state = env.reset()
        trajectory: List[TimeStep] = []
        use_pi = episodes % 2 == 0
        timed_out = False
        t = 0
        r = 0
        while not done:
            use_model_prob = 1 / (
                1 + math.exp(2 * (min_successes - len(pi.get_good())))
            )
            model = pi if use_pi else q
            use_model = (rng.random() < use_model_prob) and model.ready()
            if use_model:
                action = model.act(state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            r += gamma ** t * reward
            t += 1
            T += 1
            if t >= max_steps:
                done = timed_out = True
            if done:
                episodes += 1
                if use_pi:
                    regret = info["optimal"] - r
                    log = dict(
                        episode=episodes,
                        hours=(time.time() - start_time) / 3600,
                        regret=regret,
                        step=T,
                        use_model_prob=use_model_prob,
                        **{"return": r, "run ID": logger.run_id}
                    )
                    pprint(log)
                    if logger.run_id is not None:
                        logger.log(**log)
            trajectory.append(step)
            state = next_state

        trajectory = trajectory[-max_trajectory:]
        if not timed_out:
            while trajectory:
                buffer.append(trajectory)
                head, *trajectory = trajectory

    print("done!")
