import math
import os
import shelve
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import altair as alt
import numpy as np
import openai
import pandas as pd
from env import Env
from model import GPT3, Pi, Prompt, Q


@dataclass
class TimeStep:
    state: int
    action: int
    reward: float
    next_state: Optional[int]


def main(
    failure_threshold: float = 0.0,
    goal: int = 4,
    max_steps: int = 16,
    max_trajectory: int = 8,
    min_successes: int = 3,
    n=10,
    q_prompt_size: int = 10,
    pi_prompt_size: int = 8,
    seed: int = 1,
    states: int = 8,
    episodes: int = 400,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = Env(states, goal, seed)

    returns = deque()

    def to_string(_trajectory: List[TimeStep]) -> str:
        if not _trajectory:
            return ""
        head, *tail = _trajectory
        if head.next_state is None:
            reward_str = env.reward_str(head.reward)
        else:
            reward_str = ""

        tail_trajectory = to_string(tail)
        sep = " " if tail_trajectory and reward_str else ""
        return Prompt.make(
            head.state, head.action, f"{reward_str}{sep}{tail_trajectory}"
        ).to_string(env)

    buffer = deque()
    with shelve.open("completions/completions.pkl") as db:
        gpt3 = GPT3(db)
        pi = Pi(
            buffer=buffer,
            env=env,
            failure_threshold=failure_threshold,
            gpt3=gpt3,
            prompt_size=pi_prompt_size,
            rng=rng,
        )
        q = Q(
            buffer=buffer,
            env=env,
            failure_threshold=failure_threshold,
            gpt3=gpt3,
            max_trajectory=max_trajectory,
            prompt_size=q_prompt_size,
            rng=rng,
        )

        for i in range(episodes):
            done = False
            state = env.reset()
            trajectory: List[TimeStep] = []
            use_pi = i % 2 == 0
            timed_out = False
            t = 0
            while not done:
                value_quantities = [p.to_value_quantity(env) for p in list(buffer)]
                value_quantities = sorted(value_quantities, reverse=True)[:n]
                value_sum = sum(value_quantities)
                use_model_prob = 1 / (1 + math.exp(2 * (min_successes - value_sum)))
                print("use_model_prob", round(use_model_prob, 3))
                use_model = rng.random() < use_model_prob
                if use_model:
                    model = pi if use_pi else q
                    action = model.act(state)
                else:
                    action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                step = TimeStep(state, action, reward, None if done else next_state)
                t += 1
                if t >= max_steps:
                    done = timed_out = True
                if done:
                    if use_pi:
                        print(i)
                        print("state", state)
                        print("action", action)
                        print("reward", reward)
                        if reward > 0:
                            breakpoint()
                        returns.append(reward)
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$ Reward:", reward)
                trajectory.append(step)
                state = next_state

            # if model.ready():
            #     _last10 = list(returns) + [0] * (10 - len(returns))
            #     _last10 = sorted(_last10, reverse=True)
            #     # print()
            #     print(i, "".join(["#" if r > 0 else " " for r in _last10]) + "|")

            if not timed_out:
                head, *tail = trajectory[-max_trajectory:]
                value = to_string(tail)
                if not value:
                    value = env.reward_str(head.reward)
                prompt = Prompt.make(head.state, head.action, value)
                buffer.append(prompt)

        df = (
            pd.DataFrame(np.array(returns).reshape(-1, 1), columns=["returns"])
            # .rolling(10)
            # .mean()
            .reset_index().rename(columns=dict(index="episode"))
        )

        alt.Chart(df).mark_line(interpolate="bundle").encode(
            x="episode", y="returns"
        ).save("logs/returns.json")


if __name__ == "__main__":
    main()
