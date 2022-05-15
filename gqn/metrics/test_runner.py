import os
from dataclasses import dataclass
from typing import Generic, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from base_env import ActType, ObsType
from metrics.encoder import Encoder
from metrics.metric import Metric, Trajectory
from rl.gpt3 import GPT3
from run_logger import HasuraLogger
from tqdm import tqdm


def save_plot(df: pd.DataFrame, filename: str, title: str, y: str = "inference:N"):
    df["lower"] = df["probability"] - df["std"]
    df["upper"] = df["probability"] + df["std"]
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(y, title=""),
            color=alt.Color(
                y,
                legend=alt.Legend(
                    orient="none",
                    legendY=-40,
                    direction="horizontal",
                    titleAnchor="middle",
                ),
            ),
        )
    )
    error_bars = (
        alt.Chart()
        .mark_errorbar(clip=True)
        .encode(x=alt.X("lower", title=""), x2="upper", y=alt.Y(y, title=""))
    )
    alt.layer(bars, error_bars, data=df).facet(
        row=alt.Row(
            "encoding:N", title="", header=alt.Header(labelAngle=0, labelAlign="left")
        ),
    ).properties(title=title).save(filename)


@dataclass
class TestRunner(Generic[ObsType, ActType]):
    @staticmethod
    def run(
        debug: int,
        encoder_str: Optional[str],
        encoders: List[Encoder],
        failure_trajectories: List[List[Trajectory]],
        filename: str,
        logprobs: int,
        max_logprobs: int,
        metric_str: Optional[str],
        metrics: List[Metric],
        prompt_sizes: List[int],
        require_cache: bool,
        seed: int,
        success_trajectories: List[List[Trajectory]],
        title: str,
    ):

        logger = HasuraLogger(graphql_endpoint=os.getenv("GRAPHQL_ENDPOINT"))
        gpt3 = GPT3(
            debug=-1,
            logprobs=logprobs,
            logger=logger,
            require_cache=require_cache,
            stop=None,
            top_p=1,
            wait_time=4,
        )
        rng = np.random.default_rng(seed)

        results = dict(episode={})
        for encoder in encoders:
            for prompt_size in prompt_sizes:
                if (
                    encoder_str is not None
                    and encoder.__class__.__name__ != encoder_str
                ):
                    continue
                name = (encoder.name(), f"prompt size: {prompt_size}")
                print(name)
                loop: Metric
                for metric in metrics:
                    if metric.name() not in results:
                        results[metric.name()] = {}
                    if metric_str is not None and metric_str != metric.name():
                        continue

                    results[metric.name()][name] = list(
                        tqdm(
                            metric.take_measurement(
                                debug=debug,
                                encoder=encoder,
                                failure_trajectories=failure_trajectories,
                                gpt3=gpt3,
                                max_logprobs=max_logprobs,
                                prompt_size=prompt_size,
                                rng=rng,
                                success_trajectories=success_trajectories,
                            ),
                            total=len(metric),
                            desc=metric.name(),
                        )
                    )

        data = [
            dict(
                encoding=list(k),
                probability=np.mean(v),
                inference=inference.replace("-", " "),
                std=np.std(v),
            )
            for inference, prob in results.items()
            for k, v in prob.items()
            if len(v) > 0
        ]
        df = pd.DataFrame.from_records(data)
        save_plot(df, filename, title=title)
