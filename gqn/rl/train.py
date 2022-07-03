import math
import os
import time
from collections import deque
from copy import deepcopy
from typing import Deque, List, Optional

import numpy as np
import openai
from rl.api.fast import Fast
from rl.api.open_ai import OPENAI_MODELS, OpenAi
from rl.common import evaluate, get_value, make_env, make_log, print_rank0
from rl.huggingface import HF_MODELS, HuggingFaceModel
from rl.model import Pi, Q, TimeStep, to_string
from run_logger import HasuraLogger


def train(
    argmax: bool,
    debug: int,
    env_id: str,
    eval_interval: Optional[int],
    hint: bool,
    logprobs: int,
    logger: HasuraLogger,
    max_prompts: int,
    max_resamples: int,
    max_tokens: int,
    min_successes: int,
    model_name: str,
    require_cache: bool,
    seed: int,
    sil: bool,
    success_buffer_size: int,
    t_threshold: Optional[int],
    temperature: float,
    top_p: float,
    total_steps: int,
    use_cache: bool,
    wait_time: Optional[float],
):
    local_rank = os.getenv("LOCAL_RANK", None)
    if local_rank is not None:
        local_rank = int(local_rank)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)

    buffer: Deque[List[TimeStep]] = deque()
    success_buffer: Deque[List[TimeStep]] = deque(maxlen=success_buffer_size)

    kwargs = dict(
        debug=debug,
        logprobs=logprobs,
        logger=logger,
        max_tokens_in_completion=max_tokens,
        model_name=model_name,
        require_cache=require_cache,
        top_p=top_p,
    )
    if model_name in OPENAI_MODELS:
        lm = OpenAi(**kwargs, wait_time=wait_time)
    elif model_name == "fast":
        fast_url = os.getenv("FAST_URL")
        assert fast_url is not None, "FAST_URL must be set"
        lm = Fast(**kwargs, seed=seed, url=fast_url)
    elif model_name in HF_MODELS:
        del kwargs["wait_time"]
        kwargs["model_name"] = HF_MODELS[kwargs["model_name"]]
        lm = HuggingFaceModel(seed=seed, **kwargs)
    else:
        raise RuntimeError(f"Unknown model {model_name}")

    env = make_env(data=lm.trained_on(), env_id=env_id, seed=seed, hint=hint)
    eval_env = deepcopy(env)

    pi = Pi(
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_prompts=max_prompts,
        max_resamples=max_resamples,
        rng=rng,
        success_buffer=success_buffer,
        t_threshold=t_threshold,
        temperature=0,
        use_cache=use_cache,
    )
    q = Q(
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_prompts=max_prompts,
        max_resamples=max_resamples,
        max_steps=env.max_q_steps(),
        rng=rng,
        success_buffer=success_buffer,
        t_threshold=t_threshold,
        temperature=temperature,
        use_cache=use_cache,
    )

    T = 0
    episodes = 0
    start_time = time.time()

    while T < total_steps:
        use_model_prob = 1 / (1 + math.exp(2 * (min_successes - len(success_buffer))))
        log_info = dict(
            num_success=len(success_buffer),
            use_model_prob=use_model_prob,
            gamma=env.log_gamma(),
            seed=seed,
            start_time=start_time,
            step=T,
            local_rank=local_rank,
        )
        if eval_interval is not None and episodes % eval_interval == 0:
            evaluate(
                act_fn=pi.act,
                env=eval_env,
                eval_interval=eval_interval,
                logger=logger,
                T=T,
                **log_info,
            )

        done = False
        state = env.reset()
        trajectory: List[TimeStep] = []
        timed_out = False
        t = 0
        rewards = []
        while not done:
            use_model = (rng.random() < use_model_prob) and q.ready()
            if use_model:
                action = q.act(state, T) if argmax else pi.act(state, T)
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

        if timed_out:
            trajectory[-1].done = False
        buffer.append(trajectory)
        if (
            not sil
            or get_value(*trajectory, gamma=env.gamma()) > env.failure_threshold()
        ):
            success_buffer.append(trajectory)

    print("done!")
