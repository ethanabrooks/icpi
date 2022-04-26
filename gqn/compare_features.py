import abc
from dataclasses import dataclass
from typing import Dict, Generic, Iterator, List, Literal, Optional

import altair as alt
import numpy as np
import pandas as pd
from agent.gpt3 import GPT3
from envs.base_env import TimeStep
from gym.core import ObsType
from tqdm import tqdm


class Encoder(abc.ABC):
    @staticmethod
    def first_line(prediction: Literal["actions", "transitions"]) -> Optional[str]:
        return None

    @abc.abstractmethod
    def name(self) -> str:
        ...

    @staticmethod
    def prefix() -> Optional[str]:
        return None

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

    def _ts_str(self, ts: TimeStep) -> str:
        string = self.state_action_str(ts)
        if ts.done:
            string += " " + self.done_str(ts.reward, ts.next_state)
        return string

    def prompt_body(
        self,
        trajectories: "list[list[TimeStep]]",
        prediction: Literal["actions", "transitions"],
    ) -> str:
        return "\n".join(
            ([self.first_line(prediction)] if self.first_line(prediction) else [])
            + [
                " ".join(
                    ([self.prefix()] if self.prefix() else [])
                    + [self._ts_str(ts) for ts in trajectory]
                )
                for trajectory in (trajectories[:-1] + [trajectories[-1][:-1]])
            ]
        )


def get_prob(target: str, logprobs: List[Dict[str, float]]) -> float:
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


Trajectory = List[TimeStep]
Action = int


@dataclass
class TrajectoriesGoodActions(Generic[ObsType]):
    trajectories: List[List[TimeStep[ObsType, int]]]
    good_actions: List[int]


def get_good_action_probs(
    actions: List[TrajectoriesGoodActions], encoder: Encoder, gpt3: GPT3
) -> Iterator[List[float]]:
    for tga in tqdm(actions, desc=encoder.name()):
        trajectories = tga.trajectories
        good_actions = tga.good_actions
        prompt = (
            encoder.prompt_body(trajectories, "actions")
            + " "
            + encoder.state_str(trajectories[-1][-1].state)
        )
        # print(prompt)
        # for a in good_actions:
        #     print(encoder.action_str(a))
        # breakpoint()
        logprobs = gpt3.get_full_completion(prompt)["logprobs"]
        probs_per_action = [
            get_prob(" " + encoder.action_str(action), logprobs) for action in range(3)
        ]
        yield sum(
            [p for a, p in enumerate(probs_per_action) if a in good_actions]
        ) / sum(probs_per_action)


def get_transition_probs(
    encoder: Encoder, gpt3: GPT3, transitions: List[List[Trajectory]]
) -> Iterator[float]:
    for trajectories in tqdm(transitions, desc=encoder.name()):
        prompt = (
            encoder.prompt_body(trajectories, "transitions")
            + " "
            + encoder.state_action_str(trajectories[-1][-1])
        )
        last_step = trajectories[-1][-1]
        if last_step.done:
            ground_truth = encoder.done_str(last_step.reward, last_step.next_state)
        else:
            ground_truth = encoder.state_str(last_step.next_state)
        # print(prompt)
        # print()
        # print(ground_truth)
        # breakpoint()

        logprobs = gpt3.get_full_completion(prompt)["logprobs"]
        yield get_prob(" " + ground_truth, logprobs)


def save_plot(df: pd.DataFrame, filename: str):
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
    ).save(filename)
