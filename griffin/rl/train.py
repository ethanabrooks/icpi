import math
import os
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import openai
import yaml
from rl.api.local import Local
from rl.api.open_ai import OPENAI_MODELS, OpenAi
from rl.common import evaluate, get_value, make_env, make_log, print_rank0
from rl.lm import Data
from rl.model import Pi, Q, TimeStep, to_string
from run_logger import HasuraLogger


def train(
    argmax: bool,
    balance_prompts: bool,
    break_on_invalid: bool,
    constrain_prompts: bool,
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
    model_name: Optional[str],
    predict_transitions: bool,
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
    local_models_path = Path(".local-models.yml")
    if local_models_path.exists():
        with local_models_path.open() as f:
            local_models = yaml.load(f, yaml.FullLoader)
    else:
        local_models = {}

    if model_name in OPENAI_MODELS:
        lm = OpenAi(**kwargs, wait_time=wait_time)
    elif model_name in local_models:
        lm = Local(**kwargs, seed=seed, url=local_models[model_name])
    elif model_name is None:
        lm = None
    else:
        raise RuntimeError(
            f"Unknown model name: {model_name}. For local models, use one of: {', '.join(local_models)}"
        )

    env = make_env(data=Data.code, env_id=env_id, seed=seed, hint=hint)
    eval_env = deepcopy(env)

    pi = Pi(
        break_on_invalid=break_on_invalid,
        buffer=buffer,
        debug=debug,
        env=env,
        lm=lm,
        max_prompts=max_prompts,
        max_resamples=max_resamples,
        rng=rng,
        sil=sil,
        success_buffer=success_buffer,
        t_threshold=t_threshold,
        temperature=0,
        use_cache=use_cache,
    )
    q = Q(
        balance_prompts=balance_prompts,
        break_on_invalid=break_on_invalid,
        buffer=buffer,
        constrain_prompts=constrain_prompts,
        debug=debug,
        env=env,
        lm=lm,
        max_prompts=max_prompts,
        max_resamples=max_resamples,
        max_steps=env.max_q_steps(),
        predict_transitions=predict_transitions,
        rng=rng,
        sil=sil,
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
            use_model_prob=use_model_prob,
            gamma=env.log_gamma(),
            seed=seed,
            start_time=start_time,
            step=T,
            local_rank=local_rank,
        )
        if sil:
            log_info.update(success_buffer_size=len(success_buffer))
        if eval_interval is not None and episodes % eval_interval == 0:
            evaluate(
                act_fn=pi.act,  # type: ignore
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
                    argmax=argmax,
                    balance_prompts=balance_prompts,
                    constrain_prompts=constrain_prompts,
                    env_id=env_id,
                    evaluation=False,
                    hint=hint,
                    info=info,
                    logger=logger,
                    model_name=model_name,
                    rewards=rewards,
                    sil=sil,
                    total_steps=total_steps,
                    **log_info,  # type: ignore
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
