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
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!) {
insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p}, on_conflict: {constraint: completions_pkey, update_columns: completion}) {
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
        ),
    )


@dataclass
class GPT3:
    logger: HasuraLogger
    temperature: float
    top_p: float
    debug: bool = False

    def __call__(self, prompt, pause=True):
        print("<", end="")

        completions = self.get_completions(prompt)
        if completions:
            completion, *_ = completions
            # print("Completion:")
            # print(value)
            print(">", end="")
            return completion["completion"]

        # print("Prompt:")
        # print(prompt)
        # breakpoint()
        #
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                choice, *_ = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    temperature=0.1,
                    stop=["\n"],
                ).choices
            except openai.error.RateLimitError as e:
                print(e)
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
                # print("Completion:", completion.split("\n")[0])
                # breakpoint()
                return completion

    def get_completions(self, prompt):
        return self.logger.execute(
            gql(
                """
query get_completion($prompt: String!, $temperature: numeric!, $top_p: numeric!) {
  completions(where: {prompt: {_eq: $prompt}, temperature: {_eq: $temperature}, top_p: {_eq: $top_p}}) {
    completion
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
        if self.debug:
            print(*args, **kwargs)
