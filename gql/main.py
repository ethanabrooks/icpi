import os
import better_partial as p
from collections import deque
from typing import Optional, Tuple, cast
from gym.spaces import Discrete
import numpy as np
import openai
from dollar_lambda import command
from env import Env
from dataclasses import asdict, dataclass, replace
import shelve


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int


@command()
def main(
    batch_size: int,
    goal: int,
    prompt_buffer_size: int,
    prompt_size: int,
    random_steps: int,
    replay_buffer_size: int,
    seed: int,
    states: int,
    training_steps: int,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    env = Env(states, goal)
    assert batch_size <= replay_buffer_size
    replay_buffer = deque(maxlen=replay_buffer_size)
    prompt_buffer = deque(maxlen=prompt_buffer_size)
    rng = np.random.default_rng(seed)

    @dataclass
    class Prompt:
        state: int
        action: int
        value: str

        def to_string(self):
            return f"{env.state_str(self.state)} {env.action_str(self.action)} {self.value}"

    with shelve.open("volume/completions.db") as db:

        def q(state: int, action: int) -> str:
            def get_prompt():
                sample = rng.choice(prompt_buffer_size, size=prompt_size, replace=False)
                for i in sample:
                    yield prompt_buffer[i].to_string()
                yield env.state_str(state), env.action_str(action)

            prompt = "\n".join(get_prompt())
            if prompt in db:
                return cast(str, db[prompt])

            choice, *_ = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            ).choices
            db[prompt] = choice.text
            return choice.text

        def choose_action(state: int) -> Tuple[int, str]:
            actions = range(states)
            values = [q(state, a) for a in actions]
            action, value = max(zip(actions, values), key=lambda x: env.quantify(x[1]))
            return action, value

        for i in range(training_steps):
            done = False
            state = env.reset()
            while not done:
                if i < random_steps:
                    action = env.action_space.sample()
                else:
                    action, _ = choose_action(state)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, next_state))
                state = next_state

                if len(replay_buffer) >= batch_size:
                    sample = rng.choice(
                        len(replay_buffer), size=batch_size, replace=False
                    )
                    for i in sample:
                        transition = replay_buffer[i]
                        next_action, next_value = choose_action(transition.next_state)
                        prompt_buffer.append(
                            Prompt(
                                transition.state,
                                transition.action,
                                env.reward_str(transition.reward)
                                if transition.done
                                else Prompt(
                                    transition.next_state, next_action, next_value
                                ).to_string(),
                            )
                        )


if __name__ == "__main__":
    main()
