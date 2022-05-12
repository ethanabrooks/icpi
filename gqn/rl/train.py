import math
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import openai
from rl.gpt3 import OPENAI_MODELS
from rl.huggingface import HF_MODELS
from rl.model import GPT3, HuggingFaceModel, Pi, Q, TimeStep, get_value, to_string
from run_logger import HasuraLogger
from util import make_env, print_rank0


def train(
    debug: int,
    env_id: str,
    eval_interval: Optional[int],
    hint: bool,
    logprobs: int,
    logger: HasuraLogger,
    max_resamples: int,
    max_trajectory: int,
    min_successes: int,
    model_name: str,
    prompt_size: int,
    require_cache: bool,
    seed: int,
    success_buffer_size: int,
    success_fraction: float,
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

    if model_name in OPENAI_MODELS:
        lm = GPT3(
            debug=debug,
            logprobs=logprobs,
            logger=logger,
            model_name=model_name,
            require_cache=require_cache,
            stop=[env.action_stop(), env.state_stop()],
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
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_resamples=max_resamples,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
        success_fraction=success_fraction,
        temperature=0,
    )
    q = Q(
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_resamples=max_resamples,
        max_steps=max_trajectory,
        prompt_size=prompt_size,
        rng=rng,
        success_buffer=success_buffer,
        success_fraction=success_fraction,
        temperature=temperature,
    )

    T = 0
    episodes = 0
    start_time = time.time()

    def make_log(info: dict, rewards: List[float], evaluation: bool):
        discounted = sum([env.gamma() ** t * r for t, r in enumerate(rewards)])
        undiscounted = sum(rewards)
        regret = info["optimal"] - discounted

        prefix = "eval " if evaluation else ""

        log = dict(
            hours=(time.time() - start_time) / 3600,
            step=T,
            use_model_prob=use_model_prob,
            **{
                prefix + "return": discounted,
                prefix + "undiscounted return": undiscounted,
                prefix + "regret": regret,
                "run ID": logger.run_id,
                "success buffer": len(success_buffer),
            },
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
            rewards = []
            t = 0
            while not done:
                action = pi.act(trajectory, state)
                next_state, reward, done, info = env.step(action)
                step = TimeStep(state, action, reward, done, next_state)
                trajectory.append(step)
                state = next_state
                rewards.append(reward)
                t += 1
                if done:
                    make_log(
                        info=info,
                        rewards=rewards,
                        evaluation=True,
                    )
                    if pi.ready() and debug >= 3:
                        breakpoint()

    while T < total_steps:
        use_model_prob = 1 / (1 + math.exp(2 * (min_successes - len(success_buffer))))
        if eval_interval is not None and episodes % eval_interval == 0:
            evaluate()

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
            env.ts_to_string(step)
            rewards.append(reward)
            t += 1
            T += 1
            timed_out = info.get("TimeLimit.truncated", False)
            if done:
                print_rank0(local_rank, ".", end="")
                episodes += 1
                make_log(
                    info=info,
                    rewards=rewards,
                    evaluation=False,
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
