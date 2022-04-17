import math
import os
import time
from collections import deque
from pprint import pprint
from typing import Deque, List

import numpy as np
import openai
from env import Env
from model import GPT3, Pi, Q, TimeStep
from run_logger import HasuraLogger


def train(
    debug: bool,
    failure_threshold: float,
    gamma: float,
    goal: int,
    logger: HasuraLogger,
    max_steps: int,
    max_trajectory: int,
    q_prompt_size: int,
    pi_prompt_size: int,
    prob_scale: float,
    seed: int,
    states: int,
    temperature: float,
    top_p: float,
    total_steps: int,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = Env(states, goal, seed)

    buffer: Deque[List[TimeStep]] = deque()
    gpt3 = GPT3(
        debug=debug,
        logger=logger,
        temperature=temperature,
        top_p=top_p,
    )
    pi = Pi(
        buffer=buffer,
        env=env,
        failure_threshold=failure_threshold,
        gamma=gamma,
        gpt3=gpt3,
        prompt_size=pi_prompt_size,
        rng=rng,
        debug=debug,
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
        debug=debug,
    )

    T = 0
    episodes = 0
    start_time = time.time()
    while T < total_steps:
        done = False
        state = env.reset()
        optimal = gamma ** (abs(env.goal - state) + 1)
        trajectory: List[TimeStep] = []
        use_pi = episodes % 2 == 0
        timed_out = False
        t = 0
        r = 0
        while not done:
            best_trajectories = pi.sample_best()
            model = pi if use_pi else q
            if len(best_trajectories) > 1:
                use_model_prob = 1 / (
                    1 + math.exp(-prob_scale * math.log(len(best_trajectories) - 1))
                )
                print("use_model_prob", use_model_prob)
                use_model = (rng.random() < use_model_prob) and model.ready()
            else:
                use_model = False

            if use_model:
                action = model.act(state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            step = TimeStep(state, action, reward, None if done else next_state)
            r += reward
            t += 1
            T += 1
            if t >= max_steps:
                done = timed_out = True
            if done:
                episodes += 1
                returns = r * gamma ** t
                regrets = optimal - returns
                log = dict(
                    episode=episodes,
                    step=T,
                    time=time.time() - start_time,
                    **{
                        "regret": regrets,
                        "return": returns,
                        "run ID": logger.run_id,
                    }
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
