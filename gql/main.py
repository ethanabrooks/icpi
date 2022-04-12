import math
import os
import shelve
from collections import deque
from typing import List

import altair as alt
import numpy as np
import openai
import pandas as pd
from env import Env
from model import GPT3, Pi, Prompt, Q, TimeStep


def to_string(_trajectory: List[TimeStep], env) -> str:
    if not _trajectory:
        return ""
    head, *tail = _trajectory
    if head.next_state is None:
        reward_str = env.reward_str(head.reward, next_state=None)
    else:
        reward_str = ""

    tail_trajectory = to_string(tail, env)
    sep = " " if tail_trajectory and reward_str else ""
    return Prompt.make(
        head.state, head.action, f"{reward_str}{sep}{tail_trajectory}"
    ).to_string(env)


def main(
    failure_threshold: float = 0.0,
    gamma: float = 0.9,
    goal: int = 4,
    max_steps: int = 16,
    max_trajectory: int = 8,
    min_successes: int = 3,
    n=10,
    q_prompt_size: int = 10,
    pi_prompt_size: int = 8,
    seed: int = 1,
    states: int = 8,
    episodes: int = 40,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rng = np.random.default_rng(seed)
    env = Env(states, goal, seed)

    regrets = deque()

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
            gamma=gamma,
            gpt3=gpt3,
            max_steps=max_trajectory,
            prompt_size=q_prompt_size,
            rng=rng,
        )

        for i in range(episodes):
            done = False
            state = env.reset()
            optimal = gamma ** (abs(env.goal - state) + 1)
            trajectory: List[TimeStep] = []
            use_pi = i % 2 == 0
            timed_out = False
            t = 0
            while not done:
                value_quantities = [
                    p.to_value_quantity(env, gamma=1) for p in list(buffer)
                ]
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
                    print("episode", i)
                    print("state", state)
                    print("action", action)
                    print("reward", reward)
                    if use_pi:
                        # if reward > 0:
                        #     breakpoint()
                        regrets.append((i, optimal - reward * gamma ** t))
                trajectory.append(step)
                state = next_state

            trajectory = trajectory[-max_trajectory:]
            if not timed_out:
                while trajectory:
                    head, *tail = trajectory
                    value = to_string(tail, env)
                    if not value:
                        value = env.reward_str(head.reward, next_state=None)
                    prompt = Prompt.make(head.state, head.action, value)
                    buffer.append(prompt)
                    trajectory = tail

        df = pd.DataFrame(
            np.array(regrets).reshape(-1, 2), columns=["episode", "regrets"]
        )

        alt.Chart(df).mark_line(interpolate="bundle").encode(
            x="episode", y="regrets"
        ).save("logs/returns.json")


if __name__ == "__main__":
    main()
