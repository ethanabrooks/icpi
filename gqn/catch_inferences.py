import os
from abc import ABC
from math import ceil
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
from agent.gpt3 import GPT3
from compute_probabilities import Encoder as BaseEncoder
from compute_probabilities import (
    Trajectory,
    collect_trajectory,
    get_transition_probs,
    save_plot,
)
from dollar_lambda import command
from envs import catch
from envs.base_env import TimeStep
from envs.catch import Obs
from run_logger import HasuraLogger

ACTIONS = ["Left", "Stay", "Right"]


class Encoder(BaseEncoder, ABC):
    def transition_pair(self, trajectories: List["Trajectory"]) -> Tuple[str, str]:
        body = self.prompt_body(trajectories, "transitions")
        *query, ground_truth = self.ts_str(trajectories[-1][-1]).split("[")
        prompt = body + ("" if body.endswith("\n") else " ") + "[".join(query)
        ground_truth = "[" + ground_truth
        return prompt.rstrip(), ground_truth


class Math(Encoder):
    def name(self) -> str:
        return "P=(3,0) B=(3,0) [P.x==B.x, B.y==0];"

    def state_str(self, state: Obs) -> str:
        paddle_x, ball_x, ball_y = state
        return f"P=({paddle_x},0) B=({ball_x},{ball_y}) [{self.status(*state)}];"

    @staticmethod
    def status(paddle_x: float, ball_x: float, ball_y: float) -> str:
        x_status = "P.x==B.x" if paddle_x == ball_x else "P.x!=B.x"
        y_status = "B.y==0" if ball_y == 0 else "B.y>0"
        return f"{x_status}, {y_status}"

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: Obs) -> str:
        return self.state_str(next_state)


class MathWithReward(Math):
    def name(self) -> str:
        return "P=(3,0) B=(3,0) [P.x==B.x, B.y==0, success];"

    def status(self, paddle_x: float, ball_x: float, ball_y: float) -> str:
        status = super().status(paddle_x, ball_x, ball_y)
        if ball_y == 0:
            status += ", Success" if paddle_x == ball_x else ", Failure"
        return status


class Names(Encoder):
    def name(self) -> str:
        return "paddle=(3,0) ball=(3,0) [caught the ball]."

    def state_str(self, state: Obs) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: Obs) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


def get_prob(target, logprobs):
    if not target:
        return 1
    if not logprobs:
        return 0

    def get_prob_rec(logprobs):
        while logprobs:
            head, *logprobs = logprobs
            for token, lp in head.items():
                prob = np.exp(lp)
                if target.startswith(token):
                    yield prob * get_prob(target[len(token) :], logprobs)

    rec = list(get_prob_rec(logprobs))
    return max(rec, default=0)


def hopeless(s: Obs) -> bool:
    return s.ball_y < abs(s.paddle_x - s.ball_x)


class TimeStepGoodActions(NamedTuple):
    time_step: TimeStep[Obs, int]
    good_actions: List[int]


# def collect_trajectory(env: catch.Wrapper) -> List[TimeStepGoodActions]:
#     trajectory = []
#     state = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         good_actions = []
#         if not hopeless(state):
#             for a in range(len(ACTIONS)):
#                 _env = deepcopy(env)
#                 s, _, _, _ = _env.step(a)
#                 if not hopeless(s):
#                     good_actions.append(a)
#
#         next_state, reward, done, _ = env.step(action)
#         step = TimeStep(state, action, reward, done, next_state)
#         trajectory.append(TimeStepGoodActions(step, good_actions))
#         state = next_state
#     return trajectory


@command()
def main(
    debug: int = -1,
    n: int = 40,
    seed: int = 0,
    logprobs: int = 3,
    prompt_size=8,
    random_trajectories: int = 500,
):
    env = catch.Wrapper(catch.Env(gamma=1.0, rows=5, columns=4, seed=seed))

    def get_trajectories():
        for _ in range(random_trajectories):
            yield collect_trajectory(env)

    trajectories = list(get_trajectories())
    random = np.random.default_rng(seed=seed)
    successful = [t for t in trajectories if t[-1].reward == 1]
    unsuccessful = [t for t in trajectories if t[-1].reward < 1]

    logger = HasuraLogger(graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"))
    gpt3 = GPT3(
        debug=-1,
        logprobs=logprobs,
        logger=logger,
        stop=[";", ":"],
        temperature=0.1,
        top_p=1,
    )
    probs = {}

    for encoder in [
        Math(),
        MathWithReward(),
        Names(),
    ]:

        def get_transition_trajectories() -> List[Trajectory]:
            half = ceil(prompt_size / 2)
            prompt_trajectories = [
                [ts for ts in successful[i]]
                for i in random.choice(len(successful), half, replace=False)
            ] + [
                [ts for ts in unsuccessful[i]]
                for i in random.choice(len(unsuccessful), half, replace=False)
            ]
            random.shuffle(prompt_trajectories)
            prompt_trajectories = prompt_trajectories[:prompt_size]
            return prompt_trajectories

        transitions = [
            get_transition_trajectories() + [random.choice(trajectories)]
            for _ in range(n)
        ]
        # name = f"{encoder.ts_str(transitions[-1][-1][-1])}"
        probs[encoder.name()] = list(
            get_transition_probs(
                encoder=encoder, gpt3=gpt3, transitions=transitions, debug=debug
            )
        )

    data = [
        dict(encoding=k, probability=np.mean(v), inference="transition", std=np.std(v))
        for k, v in probs.items()
    ]
    df = pd.DataFrame.from_records(data)
    save_plot(df, "logs/catch-encodings.html")


if __name__ == "__main__":
    main()
