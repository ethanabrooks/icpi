import sys
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Optional

import bandit
import cartpole
import catch
import chain
import space_invaders
import umbrella
from base_env import Env
from gym.wrappers import TimeLimit


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


def make_env(env_id: str, seed: int, hint: bool) -> Env:
    if env_id == "bandit":
        env = bandit.Env(num_steps=5, random_seed=seed)
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
