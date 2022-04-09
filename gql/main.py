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
from model import GPT3, Prompt, Q, V


@dataclass
class TimeStep:
    state: int
    action: int
    reward: float
    next_state: Optional[int]


def main(
    batch_size: int = 1,
    goal: int = 4,
    max_trajectory: int = 5,
    prompt_buffer_size: int = 20,
    q_prompt_size: int = 10,
    v_prompt_size: int = 5,
    replay_buffer_size: int = 50,
    seed: int = 0,
    states: int = 8,
    episodes: int = 40,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = Env(states, goal, seed)
    assert batch_size <= replay_buffer_size

    returns = deque()

    def evaluate_trajectory(_trajectory: List[TimeStep]) -> str:
        if not _trajectory:
            return ""
        head, *tail = _trajectory
        if head.next_state is None:
            reward_str = env.reward_str(head.reward)
        else:
            reward_str = ""

        tail_trajectory = evaluate_trajectory(tail)
        sep = " " if tail_trajectory and reward_str else ""
        return Prompt.make(
            head.state, head.action, f"{reward_str}{sep}{tail_trajectory}"
        ).to_string(env)

    buffer = deque(maxlen=prompt_buffer_size)
    with shelve.open("completions/completions.pkl") as db:
        gpt3 = GPT3(db)
        v = V(
            buffer=buffer,
            env=env,
            gpt3=gpt3,
            prompt_size=v_prompt_size,
            rng=rng,
        )
        q = Q(
            buffer=buffer,
            env=env,
            gpt3=gpt3,
            prompt_size=q_prompt_size,
            rng=rng,
        )

        for i in range(episodes):
            done = False
            state = env.reset()
            trajectory: List[TimeStep] = []
            while not done:
                models = [m for m in [q, v] if m.ready()]
                if len(models) == 2:
                    model = rng.choice(models)
                else:
                    model = next(iter(models), q)
                action = model.act(state)
                next_state, reward, done, _ = env.step(action)
                step = TimeStep(state, action, reward, None if done else next_state)
                if done and model.ready():
                    # print(i)
                    # print("state", state)
                    # print("action", action)
                    # print("reward", reward)
                    # breakpoint()
                    returns.append(reward)
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$ Reward:", reward)
                trajectory.append(step)
                state = next_state

            # if model.ready():
            #     _last10 = list(returns) + [0] * (10 - len(returns))
            #     _last10 = sorted(_last10, reverse=True)
            #     # print()
            #     print(i, "".join(["#" if r > 0 else " " for r in _last10]) + "|")

            if len(trajectory) < max_trajectory:
                head, *tail = trajectory
                value = evaluate_trajectory(tail)
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
