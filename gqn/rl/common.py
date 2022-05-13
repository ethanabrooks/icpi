import sys
import time
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Any, Callable, List, Optional

import bandit
import cartpole
import catch
import chain
import space_invaders
from base_env import Env, TimeStep
from gym.wrappers import TimeLimit
from run_logger import HasuraLogger
from transformers import PreTrainedTokenizer


class Color(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class Colorize:
    color: Color

    def print(self, *objects, sep=" ", end="\n", file=sys.stdout):
        string = sep.join(map(str, objects))
        print(self.color.value + string + Color.ENDC.value, end=end, file=file)

    @staticmethod
    def print_header(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.HEADER).print(*objects, sep=sep, end=end, file=file)

    @staticmethod
    def print_bold(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.BOLD).print(*objects, sep=sep, end=end, file=file)

    @staticmethod
    def print_blue(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.OKBLUE).print(*objects, sep=sep, end=end, file=file)

    @staticmethod
    def print_cyan(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.OKCYAN).print(*objects, sep=sep, end=end, file=file)

    @staticmethod
    def print_green(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.OKGREEN).print(*objects, sep=sep, end=end, file=file)

    @staticmethod
    def print_warning(*objects, sep=" ", end="\n", file=sys.stdout):
        return Colorize(Color.WARNING).print(*objects, sep=sep, end=end, file=file)


def make_env(env_id: str, seed: int, hint: bool) -> Env:
    if env_id == "bandit":
        env = bandit.Env(num_steps=5, random_seed=seed, hint=hint)
    elif env_id == "cartpole":
        env = cartpole.Wrapper(cartpole.Env(max_episode_steps=5, seed=seed))
    elif env_id == "catch":
        env = catch.Wrapper(catch.Env(columns=4, rows=5, seed=seed), hint=hint)
    elif env_id == "chain":
        env = TimeLimit(
            chain.Env(goal=4, n=8, random_seed=seed, hint=hint),
            max_episode_steps=8,
        )
    elif env_id == "space-invaders":
        max_step = 8
        env = TimeLimit(
            space_invaders.Env(
                width=3,
                height=4,
                max_aliens=2,
                max_step=max_step,
                random_seed=seed,
                hint=hint,
            ),
            max_episode_steps=max_step,
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
    discounted = sum([gamma**t * r for t, r in enumerate(rewards)])
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
                    logger=logger, info=info, rewards=rewards, evaluation=True, **kwargs
                )


def get_value(*trajectory: TimeStep, gamma: float) -> float:
    return sum([gamma**t * ts.reward for t, ts in enumerate(trajectory)])


def clip_prompt(max_tokens: int, prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    tokens = tokenizer(prompt)["input_ids"]
    tokens = tokens[-max_tokens:]
    prompt = tokenizer.decode(tokens)
    return prompt
