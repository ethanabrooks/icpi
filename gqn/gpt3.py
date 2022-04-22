import sys
import time
from dataclasses import dataclass

import openai
from run_logger import HasuraLogger

from gql import gql


def post_completion(
    logger: HasuraLogger, prompt: str, completion: str, temperature: float, top_p: float
):
    return logger.execute(
        query=gql(
            """
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!, $max_tokens: Int) {
insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p, max_tokens: $max_tokens}, on_conflict: {constraint: completions_pkey, update_columns: completion}) {
completion
}
}
"""
        ),
        variable_values=dict(
            prompt=prompt,
            completion=completion,
            temperature=temperature,
            top_p=top_p,
            max_tokens=None,
        ),
    )


@dataclass
class GPT3:
    debug: int
    logger: HasuraLogger
    max_tokens: int
    temperature: float
    top_p: float

    def __call__(self, prompt, pause=True):
        print("<", end="")

        completions = self.get_completions(prompt)
        if completions:
            completion, *_ = completions
            # print("Completion:")
            # print(value)
            print(">", end="")
            return completion["completion"]

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 4:
            breakpoint()
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                choice, *_ = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=int(round((len(prompt) + self.max_tokens) / 4)),
                ).choices

            except openai.error.RateLimitError:
                time.sleep(1)
                continue
            completion = choice.text.lstrip()
            if "." in completion:
                response = post_completion(
                    logger=self.logger,
                    prompt=prompt,
                    completion=completion,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )["insert_completions_one"]["completion"]
                if response != completion:
                    breakpoint()
                print(">", end="")
                self.print("Completion:", completion.split("\n")[0])
                if self.debug >= 4:
                    breakpoint()
                return completion

    def get_completions(self, prompt):
        return self.logger.execute(
            gql(
                """
query get_completion($prompt: String!, $temperature: numeric!, $top_p: numeric!) {
  completions(where: {prompt: {_eq: $prompt}, temperature: {_eq: $temperature}, top_p: {_eq: $top_p}}) {
    completion
    max_tokens
  }
}"""
            ),
            variable_values=dict(
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
            ),
        )["completions"]

    def print(self, *args, **kwargs):
        if self.debug >= 3:
            print(*args, **kwargs)
