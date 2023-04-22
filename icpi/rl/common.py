import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, List, Optional

import bandit
import cartpole
import catch
import chain
import maze
import point_mass
import space_invaders
from base_env import Env, TimeStep
from gym.wrappers import TimeLimit
from rich.console import Console
from rich.pretty import pprint
from rich.syntax import Syntax
from rl.lm import Data
from run_logger import RunLogger

console = Console()


class Debug(Enum):
    print_api_call_indicator = auto()  # 1
    debug_rollouts = auto()  # 2
    debug_rollouts_and_print_inferences = auto()  # 3
    debug_inferences = auto()  # 4
    debug_api_calls = auto()  # 5

    def meets_threshold(self, threshold: int) -> bool:
        return threshold >= self.value


@dataclass
class Colorize:
    @staticmethod
    def print_header(*args, **kwargs):
        console.rule(*args, **kwargs)

    @staticmethod
    def print_prediction_type(*args, **kwargs):
        return console.print(*args, **kwargs, style="bold green")

    @staticmethod
    def print_with_comment(code: str, comment: str):
        code = code.rstrip("\n")
        console.print(Syntax(f"{code}  # {comment}", "python"))

    @staticmethod
    def print_completion(completion: str):
        Colorize.print_with_comment(completion, "completion")

    @staticmethod
    def print_ground_truth(completion: str):
        Colorize.print_with_comment(completion, "ground truth")

    @staticmethod
    def print_green(*args, **kwargs):
        console.print(*args, **kwargs, style="green")

    @staticmethod
    def print_warning(*args, **kwargs):
        console.print(*args, **kwargs, style="yellow")


def make_env(data: Data, env_id: str, seed: int, hint: bool) -> Env:
    if env_id == "bandit":
        env = bandit.Env(data=data, num_steps=5, random_seed=seed, hint=hint)
    elif env_id == "cartpole":
        env = cartpole.Wrapper(cartpole.CartPoleEnv(random_seed=seed), hint=hint)
    elif env_id == "catch":
        env = catch.Wrapper(
            data=data, env=catch.Env(columns=5, rows=10, seed=seed), hint=hint
        )
    elif env_id == "chain":
        env = TimeLimit(
            chain.Env(d=1, data=data, goal=4, n=8, random_seed=seed, hint=hint),
            max_episode_steps=8,
        )
    elif env_id == "distractor-chain":
        env = TimeLimit(
            chain.Env(d=2, data=data, goal=4, n=8, random_seed=seed, hint=hint),
            max_episode_steps=8,
        )
    elif env_id == "maze":
        env = TimeLimit(
            maze.Env(data=data, hint=hint, random_seed=seed), max_episode_steps=8
        )
    elif env_id == "mini-catch":
        env = catch.Wrapper(
            data=data, env=catch.Env(columns=4, rows=5, seed=seed), hint=hint
        )
    elif env_id == "point-mass":
        max_steps = 8
        env = TimeLimit(
            point_mass.Env(
                data=data,
                hint=hint,
                max_distance=6,
                _max_trajectory=max_steps,
                pos_threshold=2,
                random_seed=seed,
            ),
            max_episode_steps=max_steps,
        )
    elif env_id == "space-invaders":
        env = space_invaders.Env(
            data=data,
            width=4,
            height=5,
            n_aliens=2,
            random_seed=seed,
            hint=hint,
        )
    elif env_id == "umbrella":
        raise NotImplementedError()
    else:
        raise RuntimeError()
    return env


def print_rank0(local_rank: Optional[int], *args, pretty=False, **kwargs):
    if local_rank is None or local_rank == 0:
        if pretty:
            pprint(*args, **kwargs)
        else:
            print(*args, **kwargs)


def make_log(
    evaluation: bool,
    gamma: float,
    info: dict,
    logger: RunLogger,
    rewards: List[float],
    seed: int,
    start_time: float,
    step: int,
    local_rank: int = 0,
    total_steps: Optional[int] = None,
    **kwargs,
):
    discounted = sum([gamma**t * r for t, r in enumerate(rewards)])
    optimal = info.get("optimal")
    if optimal is not None:
        regret = info["optimal"] - discounted
        if regret < 0:
            breakpoint()
    else:
        regret = None

    prefix = "eval " if evaluation else ""

    log = dict(
        seed=seed,
        hours=(time.time() - start_time) / 3600,
        **{
            prefix + "return": discounted,
            "run ID": logger.run_id,
        },
        **kwargs,
    )
    if regret is not None:
        log.update({prefix + "regret": regret})
    if logger.run_id is not None:
        # noinspection PyTypeChecker
        logger.log(**log, step=step)
    log.update(step=step if total_steps is None else f"{step} / {total_steps}")
    # noinspection PyTypeChecker
    log = dict(sorted(list(log.items())))
    print_rank0(local_rank, log, pretty=True)


def evaluate(
    act_fn: Callable[[List[TimeStep], Any, int], int],
    env: Env,
    eval_interval: int,
    logger: RunLogger,
    T: int,
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
            action = act_fn(trajectory, state, T)
            next_state, reward, done, info = env.step(action)
            step = TimeStep(state, action, reward, done, next_state)
            trajectory.append(step)
            state = next_state
            rewards.append(reward)
            t += 1
            if done:
                make_log(
                    logger=logger, info=info, rewards=rewards, evaluation=True, **kwargs
                )


def get_value(*trajectory: TimeStep, gamma: float) -> float:
    return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])
