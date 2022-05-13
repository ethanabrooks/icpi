import math
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import openai
from rl.common import evaluate, get_value, make_env, make_log, print_rank0
from rl.huggingface import HF_MODELS
from rl.model import GPT3, HuggingFaceModel, Pi, Q, TimeStep, to_string
from run_logger import HasuraLogger


def train(
    balance_successful_and_failed: bool,
    debug: int,
    env_id: str,
    eval_interval: Optional[int],
    hint: bool,
    logprobs: int,
    logger: HasuraLogger,
    max_trajectory: int,
    min_successes: int,
    model_name: str,
    prompt_size: int,
    require_cache: bool,
    seed: int,
    success_buffer_size: int,
    temperature: float,
    top_p: float,
    total_steps: int,
    wait_time: int,
):
    local_rank = os.getenv("LOCAL_RANK", None)
    if local_rank is not None:
        local_rank = int(local_rank)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = make_env(env_id=env_id, seed=seed, hint=hint)

    buffer: Deque[List[TimeStep]] = deque()
    success_buffer: Deque[List[TimeStep]] = deque(maxlen=success_buffer_size)

    if model_name == "gpt3":
        lm = GPT3(
            debug=debug,
            logprobs=logprobs,
            logger=logger,
            require_cache=require_cache,
            top_p=top_p,
            wait_time=wait_time,
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
    else:
        raise RuntimeError(f"Unknown model {model_name}")

    pi = Pi(
        balance_successful_and_failed=balance_successful_and_failed,
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
        temperature=temperature,
    )
    q = Q(
        balance_successful_and_failed=balance_successful_and_failed,
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
        temperature=temperature,
    )

    T = 0
    episodes = 0
    start_time = time.time()

    while T < total_steps:
        use_model_prob = 1 / (1 + math.exp(2 * (min_successes - len(success_buffer))))
        log_info = dict(
            success_buffer_size=len(success_buffer),
            use_model_prob=use_model_prob,
            gamma=env.gamma(),
            start_time=start_time,
            step=T,
            local_rank=local_rank,
        )
        if eval_interval is not None and episodes % eval_interval == 0:
            evaluate(logger, env, eval_interval, pi.act, **log_info)

        done = False
        state = env.reset()
        trajectory: List[TimeStep] = []
        timed_out = False
        t = 0
        rewards = []
        while not done:
            use_model = (rng.random() < use_model_prob) and q.ready()
            if use_model:
                action = q.act(trajectory, state)
            else:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            rewards.append(reward)
            t += 1
            T += 1
            timed_out = info.get("TimeLimit.truncated", False)
            if done:
                print_rank0(local_rank, ".", end="")
                episodes += 1
                make_log(
                    logger=logger,
                    info=info,
                    rewards=rewards,
                    evaluation=False,
                    **log_info,
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
