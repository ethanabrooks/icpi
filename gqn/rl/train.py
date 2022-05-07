import math
import os
import time
from collections import deque
from pprint import pprint
from typing import Deque, List

import bandit
import cartpole
import catch
import chain
import numpy as np
import openai
import umbrella
from base_env import Env
from gym.wrappers import TimeLimit
from rl.model import GPT3, Pi, Q, TimeStep, get_value, to_string
from run_logger import HasuraLogger


def make_env(env_id: str, gamma: float, seed: int, status: bool) -> Env:
    if env_id == "bandit":
        assert gamma == 1.0
        env = bandit.Env(num_steps=5, random_seed=seed)
    elif env_id == "cartpole":
        env = cartpole.Wrapper(
            cartpole.Env(gamma=gamma, max_episode_steps=5, seed=seed)
        )
    elif env_id == "catch":
        assert gamma == 1.0
        env = catch.Wrapper(catch.Env(columns=4, gamma=gamma, rows=5, seed=seed))
    elif env_id == "chain":
        env = TimeLimit(
            chain.Env(gamma=gamma, goal=4, n=8, random_seed=seed, status=status),
            max_episode_steps=8,
        )
    elif env_id == "umbrella":
        assert gamma == 1.0
        env = umbrella.Env(num_colors=2, num_steps=2, random_seed=seed)
    else:
        raise RuntimeError()
    return env


def train(
    debug: int,
    delta: float,
    env_id: str,
    gamma: float,
    logprobs: int,
    logger: HasuraLogger,
    max_trajectory: int,
    min_successes: int,
    prompt_size: int,
    seed: int,
    status: bool,
    temperature: float,
    top_p: float,
    total_steps: int,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = make_env(env_id=env_id, gamma=gamma, seed=seed, status=status)

    buffer: Deque[List[TimeStep]] = deque()
    success_buffer: Deque[List[TimeStep]] = deque()

    gpt3 = GPT3(
        debug=debug,
        logprobs=logprobs,
        logger=logger,
        stop=[env.action_stop(), env.state_stop()],
        temperature=temperature,
        top_p=top_p,
    )
    pi = Pi(
        buffer=buffer,
        debug=debug,
        delta=delta,
        env=env,
        gamma=gamma,
        gpt3=gpt3,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
    )
    q = Q(
        buffer=buffer,
        debug=debug,
        delta=delta,
        env=env,
        gamma=gamma,
        gpt3=gpt3,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
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
                1 + math.exp(2 * (min_successes - len(success_buffer)))
            )
            model = pi if use_pi else q
            use_model = (rng.random() < use_model_prob) and model.ready()
            if use_model:
                action = model.act(trajectory, state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            r += gamma**t * reward
            t += 1
            T += 1
            timed_out = "TimeLimit.truncated" in info
            if done:
                print(".", end="")
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

        # quantify unittest
        prompt = to_string(*trajectory, env=env)
        value_from_prompt = env.quantify(prompt, gamma=gamma)
        value_from_trajectory = get_value(*trajectory, gamma=gamma)
        if not value_from_prompt == value_from_trajectory:
            print(value_from_prompt, value_from_trajectory)
            breakpoint()
            env.quantify(prompt, gamma=gamma)
            get_value(*trajectory, gamma=gamma)

        trajectory = trajectory[-max_trajectory:]
        if not timed_out:
            buffer.append(trajectory)
            while trajectory:
                if get_value(*trajectory, gamma=1) > env.failure_threshold():
                    success_buffer.append(trajectory)
                head, *trajectory = trajectory

    print("done!")
