import os
from math import ceil
from typing import List

import numpy as np
import pandas as pd
from agent.gpt3 import GPT3
from compute_probabilities import (
    Encoder,
    Trajectory,
    collect_trajectory,
    get_transition_probs,
    save_plot,
)
from dollar_lambda import command
from envs import bandit
from run_logger import HasuraLogger


class BanditEncoder(Encoder):
    def name(self) -> str:
        return ""

    def state_str(self, state: np.ndarray) -> str:
        return ""

    def action_str(self, action: int) -> str:
        return f"{action}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        return f"{reward}."


@command()
def main(
    n: int = 40,
    seed: int = 0,
    logprobs: int = 3,
    prompt_size=8,
    random_trajectories: int = 500,
):
    env = bandit.Wrapper(bandit.Env(mapping_seed=seed, num_actions=3))

    def get_trajectories():
        for _ in range(random_trajectories):
            yield collect_trajectory(env)

    trajectories = list(get_trajectories())
    random = np.random.default_rng(seed=seed)
    successful = [t for t in trajectories if t[-1].time_step.reward > 0.5]
    unsuccessful = [t for t in trajectories if t[-1].time_step.reward < 0.5]

    logger = HasuraLogger(graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"))
    gpt3 = GPT3(
        debug=-1,
        logprobs=logprobs,
        logger=logger,
        stop=[".", ":"],
        temperature=0.1,
        top_p=1,
    )
    encoder = Encoder()
    transition_probs = {}

    def get_transition_trajectories() -> List[Trajectory]:
        half = ceil((prompt_size + 1) / 2)
        prompt_trajectories = [
            [ts.time_step for ts in successful[i]]
            for i in random.choice(len(successful), half, replace=False)
        ] + [
            [ts.time_step for ts in unsuccessful[i]]
            for i in random.choice(len(unsuccessful), half, replace=False)
        ]
        random.shuffle(prompt_trajectories)
        prompt_trajectories = prompt_trajectories[: prompt_size + 1]
        return prompt_trajectories

    transitions = [get_transition_trajectories() for _ in range(n)]
    transition_probs[encoder.name()] = list(
        get_transition_probs(encoder=encoder, gpt3=gpt3, transitions=transitions)
    )

    data = [
        dict(encoding=k, probability=np.mean(v), inference="transition", std=np.std(v))
        for k, v in transition_probs.items()
    ]
    df = pd.DataFrame.from_records(data)
    save_plot(df, "logs/bandit.html")


if __name__ == "__main__":
    main()
