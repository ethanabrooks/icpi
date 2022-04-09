import os
import shelve
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, cast

import numpy as np
import openai
from gym.spaces import Discrete

from env import Env


@dataclass
class TimeStep:
    state: int
    action: int
    reward: float
    next_state: Optional[int]


def main(
    batch_size: int = 1,
    goal: int = 3,
    max_trajectory: int = 5,
    prompt_buffer_size: int = 20,
    prompt_size: int = 10,
    replay_buffer_size: int = 50,
    seed: int = 0,
    states: int = 5,
    training_steps: int = 100000,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    env = Env(states, goal, seed)
    assert batch_size <= replay_buffer_size
    rng = np.random.default_rng(seed)

    @dataclass
    class Prompt:
        state: int
        action: int
        value: str

        @staticmethod
        def make(state: int, action: int, value: str):
            return Prompt(state, action, value.lstrip())

        def to_string(self):
            return f"{env.state_str(self.state)} {env.action_str(self.action)} {self.value}"

    class PromptBuffer:
        def __init__(self):
            self.success_buffer = deque(maxlen=prompt_buffer_size)
            self.failure_buffer = deque(maxlen=prompt_buffer_size)

        def add(self, prompt: Prompt):
            buffer = (
                self.failure_buffer
                if env.quantify(prompt.value) == 0
                else self.success_buffer
            )
            buffer.append(prompt)

        def sample(self):
            num_failure = prompt_size - len(self.success_buffer)
            if num_failure > 0:
                failure_idxs = rng.choice(
                    len(self.failure_buffer), size=num_failure, replace=False
                )
                failure_prompts = [
                    self.failure_buffer[k].to_string() for k in failure_idxs
                ]
            else:
                failure_prompts = []
            success_prompts = [p.to_string() for p in self.success_buffer]
            prompts = failure_prompts + success_prompts
            rng.shuffle(prompts)
            return prompts

        def ready(self) -> bool:
            return (
                len(self.success_buffer) + len(self.failure_buffer)
                >= prompt_buffer_size
            )

    replay_buffer: Deque[TimeStep] = deque(maxlen=replay_buffer_size)
    prompt_buffer: PromptBuffer = PromptBuffer()

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
        ).to_string()

    with shelve.open("completions/completions.db") as db:

        def q(state: int, action: int) -> str:
            prompt = "\n".join(
                [
                    *prompt_buffer.sample(),
                    f"{env.state_str(state)} {env.action_str(action)}",
                ]
            )
            if prompt in db:
                value = cast(str, db[prompt])
                # print("Completion:")
                # print(value)
                return value

            print("Prompt:")
            print(prompt)

            # print("Querying...", end=" ")
            choice, *_ = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            ).choices
            # print("Received response.")
            value, *_ = choice.text.split("\n")
            print("Completion:")
            print(value)
            breakpoint()
            # print("Completion:")
            # print(value)
            db[prompt] = value
            return value

        def choose_action(state: int) -> Tuple[int, str]:
            assert isinstance(env.action_space, Discrete)
            actions = range(env.action_space.n)
            values = [q(state, a) for a in actions]
            action, value = max(zip(actions, values), key=lambda x: env.quantify(x[1]))
            print("state", state)
            print("action", action)
            print("value", value)
            breakpoint()
            return action, value

        for i in range(training_steps):
            done = False
            state = env.reset()
            trajectory: List[TimeStep] = []
            while not done:
                if len(replay_buffer) >= batch_size and prompt_buffer.ready():
                    action, _ = choose_action(state)
                else:
                    action = env.action_space.sample()

                next_state, reward, done, _ = env.step(action)
                step = TimeStep(state, action, reward, None if done else next_state)
                if done:
                    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$ Reward:", reward)
                trajectory.append(step)
                state = next_state

            replay_buffer.extend(trajectory)

            def get_last_10():
                count = 0
                for ts in reversed(replay_buffer):
                    if count == 10:
                        return
                    if ts.next_state is None:
                        count += 1
                        yield ts

            last_10 = list(get_last_10())
            print(
                "".join(
                    ["#" for ts in last_10 if ts.reward == 1]
                    + [
                        " "
                        for ts in last_10
                        if ts.next_state is None and ts.reward == 0
                    ]
                )
                + "|"
            )

            if len(trajectory) < max_trajectory:
                head, *tail = trajectory
                value = evaluate_trajectory(tail)
                if not value:
                    value = env.reward_str(head.reward)
                prompt_buffer.add(Prompt.make(head.state, head.action, value))  # TODO

            if len(replay_buffer) >= batch_size:
                sample = rng.choice(len(replay_buffer), size=batch_size, replace=False)
                for i in sample:
                    transition = replay_buffer[i]
                    if prompt_buffer.ready():
                        next_action, next_value = choose_action(transition.next_state)
                        done = transition.next_state is None
                        value = (
                            env.reward_str(transition.reward)
                            if done
                            else Prompt.make(
                                transition.next_state, next_action, next_value
                            ).to_string()
                        )
                        prompt = Prompt.make(
                            transition.state,
                            transition.action,
                            value,
                        )
                        prompt_buffer.add(prompt)


if __name__ == "__main__":
    main()
