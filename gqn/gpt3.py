import sys
import time
from dataclasses import dataclass
from typing import Optional

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
    debug: int
    logger: HasuraLogger
    log_probs: Optional[int]
    temperature: float
    top_p: float

    def __call__(self, prompt):
        return self.get_full_completion(prompt)["completion"]

    def get_full_completion(self, prompt):
        print("<", end="")

        completions = self.get_completions(prompt)
        if completions:
            completion, *_ = completions
            # print("Completion:")
            # print(value)
            print(">", end="")
            return completion

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 6:
            breakpoint()
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                choice, *_ = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    logprobs=self.log_probs,
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
                self.print("Completion:", completion.split("\n")[0])
                if self.debug >= 5:
                    breakpoint()
                return dict(
                    prompt=prompt,
                    completion=completion,
                    log_probs=choice.logprobs.top_logprobs,
                )

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
        if self.debug >= 5:
            print(*args, **kwargs)
