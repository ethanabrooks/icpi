import abc
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
from base_env import TimeStep
from dollar_lambda import command
from gpt3 import GPT3
from run_logger import HasuraLogger
from tqdm import tqdm

ACTIONS = ["Left", "Stay", "Right"]


class Encoder(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def state_str(self, state: np.ndarray) -> str:
        ...

    @abc.abstractmethod
    def action_str(self, action: int) -> str:
        ...

    @abc.abstractmethod
    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        ...

    def state_action_str(self, ts: TimeStep) -> str:
        return self.state_str(ts.state) + " " + self.action_str(ts.action)

    def ts_str(self, ts: TimeStep) -> str:
        string = self.state_action_str(ts)
        if ts.done:
            string += " " + self.done_str(ts.reward, ts.next_state)
        return string


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


class ParensPaddleParensBallWithNamesAndVerboseActions(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) Move paddle right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y})."

    def action_str(self, action: int) -> str:
        return f"{'Do not move paddle' if action == 1 else ('Move paddle ' + ACTIONS[action].lower())}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndFalling(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [falling]."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}."

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = next_state
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'caught the ball' if reward == 1 else 'missed the ball'}]."


class ParensPaddleParensBallWithNamesAndCanCatch(Encoder):
    def name(self) -> str:
        return "Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [can catch the ball]. Right:"

    def state_str(self, state: np.ndarray) -> str:
        paddle_x, ball_x, ball_y = state
        can_catch = ball_y >= abs(paddle_x - ball_x)
        return f"Paddle=({paddle_x},0) Ball=({ball_x},{ball_y}) [{'can catch the ball' if can_catch else 'cannot catch the ball'}]."

    def action_str(self, action: int) -> str:
        return f"{ACTIONS[action]}:"

    def done_str(self, reward: float, next_state: np.ndarray) -> str:
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


@command()
def main(
    actions_path: str,
    transitions_path: str,
    n: int = 40,
    seed: int = 0,
    logprobs: int = 3,
):
    with open(transitions_path, "rb") as f:
        transitions = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    def get_balanced_transitions():
        for trajectories in transitions:
            successful = [t for t in trajectories if t[-1].reward == 1]
            unsuccessful = [t for t in trajectories if t[-1].reward == 0]
            if abs(len(successful) - len(unsuccessful)) <= 1:
                yield trajectories

    transitions = list(get_balanced_transitions())
    rng = np.random.default_rng(seed)
    rng.shuffle(transitions)
    rng.shuffle(actions)

    transitions = transitions[:n]
    actions = actions[:n]

    logger = HasuraLogger(graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"))
    gpt3 = GPT3(debug=-1, logprobs=logprobs, logger=logger, temperature=0.1, top_p=1)
    transition_probs = {}
    transition_stds = {}
    action_probs = {}
    action_stds = {}
    for encoder in [
        PaddleXBallXParensBallY(),
        ParensPaddleParensBall(),
        ParensPaddleParensBallWithNames(),
        ParensPaddleParensBallWithNamesAndVerboseActions(),
        ParensPaddleParensBallWithNamesAndFalling(),
        ParensPaddleParensBallWithNamesAndCanCatch(),
    ]:
        probs = []
        for trajectories in tqdm(transitions, desc=encoder.name()):
            prompt = "\n".join(
                [
                    " ".join([encoder.ts_str(ts) for ts in trajectory])
                    for trajectory in trajectories[:-1]
                ]
                + [encoder.state_action_str(trajectories[-1][0])]
            )
            last_step = trajectories[-1][0]
            if last_step.done:
                ground_truth = encoder.done_str(last_step.reward, last_step.next_state)
            else:
                ground_truth = encoder.state_str(last_step.next_state)
            logprobs = gpt3.get_full_completion(prompt)["logprobs"]
            prob = get_prob(" " + ground_truth, logprobs)
            probs.append(prob)
        transition_probs[encoder.name()] = np.mean(probs)
        transition_stds[encoder.name()] = np.std(probs)

        probs = []
        for trajectories, good_actions in tqdm(actions, desc=encoder.name()):
            prompt = "\n".join(
                [
                    " ".join([encoder.ts_str(ts) for ts in trajectory])
                    for trajectory in trajectories[:-1]
                ]
                + [encoder.state_str(trajectories[-1][0].state)]
            )

            logprobs = gpt3.get_full_completion(prompt)["logprobs"]
            probs_per_action = [
                get_prob(" " + encoder.action_str(action), logprobs)
                for action in range(3)
            ]
            prob = sum(
                [p for a, p in enumerate(probs_per_action) if a in good_actions]
            ) / sum(probs_per_action)
            probs.append(prob)
        action_probs[encoder.name()] = np.mean(probs)
        action_stds[encoder.name()] = np.std(probs)
    data = [
        dict(encoding=k, probability=v, inference="transition", std=transition_stds[k])
        for k, v in transition_probs.items()
    ] + [
        dict(encoding=k, probability=v, inference="action", std=action_stds[k])
        for k, v in action_probs.items()
    ]
    df = pd.DataFrame.from_records(data)
    df["lower"] = df["probability"] - df["std"]
    df["upper"] = df["probability"] + df["std"]
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("inference:N", title=""),
            color=alt.Color("inference:N"),
        )
    )
    error_bars = (
        alt.Chart()
        .mark_errorbar()
        .encode(x="lower", x2="upper", y=alt.Y("inference:N", title=""))
    )
    alt.layer(bars, error_bars, data=df).facet(
        row=alt.Row("encoding:N", header=alt.Header(labelAngle=0, labelAlign="left")),
    ).save("logs/encodings.html")


if __name__ == "__main__":
    main()
