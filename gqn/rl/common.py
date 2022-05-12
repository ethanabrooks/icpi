from typing import Optional, List, Callable, Any
from pprint import pprint
import time

from gym.wrappers import TimeLimit
from run_logger import HasuraLogger

import bandit
import cartpole
import catch
import chain
import umbrella
from base_env import Env
from rl.model import TimeStep


def print_rank0(local_rank: Optional[int], *args, pretty=False, **kwargs):
    if local_rank is None or local_rank == 0:
        if pretty:
            pprint(*args, **kwargs)
        else:
            print(*args, **kwargs)


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


def make_log(
        logger: HasuraLogger,
        info: dict,
        rewards: List[float],
        success_buffer_size: int,
        use_model_prob: float,
        gamma: float,
        start_time: float,
        step: int,
        evaluation: bool,
        local_rank: int = 0,
    ):
    discounted = sum([gamma ** t * r for t, r in enumerate(rewards)])
    undiscounted = sum(rewards)
    regret = info["optimal"] - discounted

    prefix = "eval " if evaluation else ""

    log = dict(
        hours=(time.time() - start_time) / 3600,
        step=step,
        use_model_prob=use_model_prob,
        **{
            prefix + "return": discounted,
            prefix + "undiscounted return": undiscounted,
            prefix + "regret": regret,
            "run ID": logger.run_id,
            "success buffer": success_buffer_size,
        },
    )
    print_rank0(local_rank, log, pretty=True)
    if logger.run_id is not None:
        logger.log(**log)

def evaluate(
    logger: HasuraLogger,
    env: Env,
    eval_interval: int,
    act_fn: Callable[[List[TimeStep], Any], int],
    **kwargs,
):
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
            action = act_fn(trajectory, state)
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            trajectory.append(step)
            state = next_state
            rewards.append(reward)
            t += 1
            if done:
                make_log(
                    logger=logger,
                    info=info,
                    rewards=rewards,
                    evaluation=True,
                    **kwargs
                )

