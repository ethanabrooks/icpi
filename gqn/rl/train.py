import math
import os
import time
from collections import deque
from pprint import pprint
from typing import Deque, List, Optional

import bandit
import cartpole
import catch
import chain
import numpy as np
import openai
import umbrella
from base_env import Env
from gym.wrappers import TimeLimit
from rl.huggingface import HF_MODELS
from rl.model import GPT3, HuggingFaceModel, Pi, Q, TimeStep, get_value, to_string
from run_logger import HasuraLogger


def make_env(env_id: str, seed: int, status: bool) -> Env:
    if env_id == "bandit":
        env = bandit.Env(num_steps=5, random_seed=seed)
    elif env_id == "cartpole":
        env = cartpole.Wrapper(cartpole.Env(max_episode_steps=5, seed=seed))
    elif env_id == "catch":
        env = catch.Wrapper(catch.Env(columns=4, rows=5, seed=seed), status=status)
    elif env_id == "chain":
        env = TimeLimit(
            chain.Env(goal=4, n=8, random_seed=seed, status=status),
            max_episode_steps=8,
        )
    elif env_id == "umbrella":
        env = umbrella.Env(num_colors=2, num_steps=2, random_seed=seed)
    else:
        raise RuntimeError()
    return env


def print_rank0(local_rank: Optional[int], *args, pretty=False, **kwargs):
    if local_rank is None or local_rank == 0:
        if pretty:
            pprint(*args, **kwargs)
        else:
            print(*args, **kwargs)


def train(
    debug: int,
    model_name: str,
    env_id: str,
    eval_interval: Optional[int],
    logprobs: int,
    logger: HasuraLogger,
    max_trajectory: int,
    min_successes: int,
    prompt_size: int,
    require_cache: bool,
    seed: int,
    status: bool,
    success_buffer_size: int,
    temperature: float,
    top_p: float,
    total_steps: int,
):
    local_rank = os.getenv("LOCAL_RANK", None)
    if local_rank is not None:
        local_rank = int(local_rank)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = make_env(env_id=env_id, seed=seed, status=status)

    buffer: Deque[List[TimeStep]] = deque()
    success_buffer: Deque[List[TimeStep]] = deque(maxlen=success_buffer_size)

    if model_name == "gpt3":
        lm = GPT3(
            debug=debug,
            logprobs=logprobs,
            logger=logger,
            require_cache=require_cache,
            stop=[env.action_stop(), env.state_stop()],
            temperature=temperature,
            top_p=top_p,
        )
    elif model_name in HF_MODELS:
        lm = HuggingFaceModel(
            model_name=HF_MODELS[model_name],
            debug=debug,
            logprobs=logprobs,
            logger=logger,
            seed=seed,
            stop=[env.action_stop(), env.state_stop()],
            temperature=temperature,
            top_p=top_p,
        )

    pi = Pi(
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
    )
    q = Q(
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
    )

    T = 0
    episodes = 0
    start_time = time.time()

    def make_log(info: dict, return_: float, return_key: str, regret_key: str):
        regret = info["optimal"] - return_
        log = dict(
            hours=(time.time() - start_time) / 3600,
            step=T,
            use_model_prob=use_model_prob,
            **{
                return_key: return_,
                regret_key: regret,
                "run ID": logger.run_id,
                "success buffer": len(success_buffer),
            }
        )
        print_rank0(local_rank, log, pretty=True)
        if logger.run_id is not None:
            logger.log(**log)

    def evaluate():
        start_states = env.start_states()
        finite_start_states = start_states is not None
        if finite_start_states:
            start_states = list(start_states)
        for _ in range(eval_interval):
            state = env.reset()
            if finite_start_states:
                if not start_states:
                    return
                while state not in start_states:
                    state = env.reset()
                start_states.remove(state)
            trajectory: List[TimeStep] = []
            done = False
            r = 0
            t = 0
            while not done:
                action = pi.act(trajectory, state)
                next_state, reward, done, info = env.step(action)
                step = TimeStep(state, action, reward, done, next_state)
                trajectory.append(step)
                state = next_state
                r += env.gamma() ** t * reward
                t += 1
                if done:
                    make_log(
                        info=info,
                        return_=r,
                        return_key="eval return",
                        regret_key="eval regret",
                    )

    while T < total_steps:
        use_model_prob = 1 / (1 + math.exp(2 * (min_successes - len(success_buffer))))
        if eval_interval is not None and episodes % eval_interval == 0:
            evaluate()

        done = False
        state = env.reset()
        trajectory: List[TimeStep] = []
        timed_out = False
        t = 0
        r = 0
        while not done:
            use_model = (rng.random() < use_model_prob) and q.ready()
            if use_model:
                action = q.act(trajectory, state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            r += env.gamma() ** t * reward
            t += 1
            T += 1
            timed_out = info.get("TimeLimit.truncated", False)
            if done:
                print_rank0(local_rank, ".", end="")
                episodes += 1
                make_log(
                    info=info,
                    return_=r,
                    return_key="return",
                    regret_key="regret",
                )
            trajectory.append(step)
            state = next_state

        # quantify unittest
        prompt = to_string(*trajectory, env=env)
        value_from_prompt = env.quantify(prompt)
        value_from_trajectory = get_value(*trajectory, gamma=env.gamma())
        if not value_from_prompt == value_from_trajectory:
            print_rank0(local_rank, value_from_prompt, value_from_trajectory)
            breakpoint()
            env.quantify(prompt)
            get_value(*trajectory, gamma=env.gamma())

        trajectory = trajectory[-max_trajectory:]
        if not timed_out:
            buffer.append(trajectory)
            if get_value(*trajectory, gamma=env.gamma()) > env.failure_threshold():
                success_buffer.append(trajectory)

    print("done!")
