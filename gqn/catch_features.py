import os
from copy import deepcopy
from math import ceil
from typing import List, Literal, NamedTuple, Optional

import numpy as np
import pandas as pd
from agent.gpt3 import GPT3
from compare_features import (
    Encoder,
    TrajectoriesGoodActions,
    Trajectory,
    get_good_action_probs,
    get_transition_probs,
    save_plot,
)
from dollar_lambda import command
from envs import catch
from envs.base_env import TimeStep
from envs.catch import Obs
from run_logger import HasuraLogger

ACTIONS = ["Left", "Stay", "Right"]


class PaddleXBallXParensBallY(Encoder):
    def name(self) -> str:
        return "{paddle_x},{ball_x} ({ball_y} to go). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"{paddle_x},{ball_x} ({ball_y} to go)."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, _ = next_state
        return f"{paddle_x},{ball_x} ({'success' if reward == 1 else 'failure'})."


class ParensPaddleParensBall(Encoder):
    def name(self) -> str:
        return "({paddle_x},0) ({ball_x},{ball_y}). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"({paddle_x},0) ({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"({paddle_x},0) ({ball_x},{ball_y}) [{'success' if reward == 1 else 'failure'}]."


class ParensPaddleParensBallWithNames(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}). Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndStart(ParensPaddleParensBallWithNames):
    def name(self) -> str:
        return "Start: Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}). Right:"

    def prefix(self) -> str:
        return "Start:"


class ParensPaddleParensBallWithNamesAndPreface(ParensPaddleParensBallWithNames):
    def name(self) -> str:
        return "A paddle that sometimes catches a falling ball:"

    @staticmethod
    def first_line(prediction: Literal["actions", "transitions"]) -> Optional[str]:
        if prediction == "actions":
            return "A paddle that catches a falling ball:"
        elif prediction == "transitions":
            return "A paddle that sometimes catches a falling ball:"
        else:
            raise RuntimeError()


class ParensPaddleParensBallWithNamesAndVerboseActions(ParensPaddleParensBallWithNames):
    def name(self) -> str:
        return "({paddle_x},0) ({ball_x},{ball_y}). Move the paddle right:"

    def action_str(self, action: int) -> str:
        return f"{'Do not move paddle' if action == 1 else ('Move paddle ' + ACTIONS[action].lower())}:"


class ParensPaddleParensBallWithNamesAndFalling(ParensPaddleParensBallWithNames):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]."


class ParensPaddleParensBallWithNamesAndCanCatch(ParensPaddleParensBallWithNames):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [can catch the ball]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        can_catch = ball_y >= abs(paddle_x - ball_x)
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'can catch the ball' if can_catch else 'cannot catch the ball'}]."


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


def collect_trajectory(env: catch.Wrapper) -> List[TimeStepGoodActions]:
    trajectory = []
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        good_actions = []
        if not hopeless(state):
            for a in range(len(ACTIONS)):
                _env = deepcopy(env)
                s, _, _, _ = _env.step(a)
                if not hopeless(s):
                    good_actions.append(a)

        next_state, reward, done, _ = env.step(action)
        step = TimeStep(state, action, reward, done, next_state)
        trajectory.append(TimeStepGoodActions(step, good_actions))
        state = next_state
    return trajectory


@command()
def main(
    n: int = 40,
    seed: int = 0,
    logprobs: int = 3,
    random_trajectories: int = 500,
):
    env = catch.Wrapper(catch.Env(gamma=1.0, rows=5, columns=4, seed=seed))

    trajectories = [collect_trajectory(env) for _ in range(random_trajectories)]
    rng = np.random.default_rng(seed=seed)
    successful = [t for t in trajectories if t[-1].time_step.reward == 1]
    unsuccessful = [t for t in trajectories if t[-1].time_step.reward < 1]
    action_time_steps = [
        ts for t in trajectories for ts in t if 0 < len(ts.good_actions) < len(ACTIONS)
    ]

    logger = HasuraLogger(graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"))
    gpt3 = GPT3(
        debug=-1,
        logprobs=logprobs,
        logger=logger,
        stop=[".", ":"],
        temperature=0.1,
        top_p=1,
    )
    encoder = ParensPaddleParensBallWithNamesAndStart()
    transition_probs = {}
    action_probs = {}

    for prompt_size in range(5, 15):

        def get_action_trajectories() -> TrajectoriesGoodActions:
            prompt_trajectories = [
                [ts.time_step for ts in successful[i]]
                for i in rng.choice(len(successful), prompt_size, replace=False)
            ]
            ts = action_time_steps[rng.choice(len(action_time_steps))]
            prompt_trajectories = prompt_trajectories + [[ts.time_step]]
            return TrajectoriesGoodActions(prompt_trajectories, ts.good_actions)

        def get_transition_trajectories() -> List[Trajectory]:
            half = ceil((prompt_size + 1) / 2)
            prompt_trajectories = [
                [ts.time_step for ts in successful[i]]
                for i in rng.choice(len(successful), half, replace=False)
            ] + [
                [ts.time_step for ts in unsuccessful[i]]
                for i in rng.choice(len(unsuccessful), half, replace=False)
            ]
            rng.shuffle(prompt_trajectories)
            prompt_trajectories = prompt_trajectories[: prompt_size + 1]
            prompt_trajectories[-1] = prompt_trajectories[-1][:1]
            return prompt_trajectories

        actions = [get_action_trajectories() for _ in range(n)]
        action_probs[prompt_size] = list(
            get_good_action_probs(actions=actions, encoder=encoder, gpt3=gpt3)
        )
        transitions = [get_transition_trajectories() for _ in range(n)]
        transition_probs[prompt_size] = list(
            get_transition_probs(encoder=encoder, gpt3=gpt3, transitions=transitions)
        )

    data = [
        dict(encoding=k, probability=np.mean(v), inference="transition", std=np.std(v))
        for k, v in transition_probs.items()
    ] + [
        dict(encoding=k, probability=np.mean(v), inference="action", std=np.std(v))
        for k, v in action_probs.items()
    ]
    df = pd.DataFrame.from_records(data)
    save_plot(df, "logs/catch-prompt-sizes.html")


if __name__ == "__main__":
    main()
