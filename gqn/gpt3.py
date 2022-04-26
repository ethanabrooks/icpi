import sys
import time
from dataclasses import dataclass

import openai
from run_logger import HasuraLogger

from gql import gql


def post_completion(
    logprobs: list,
    logger: HasuraLogger,
    prompt: str,
    completion: str,
    temperature: float,
    top_p: float,
):
    return logger.execute(
        query=gql(
            """
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!, $logprobs: jsonb!) {
  insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p, logprobs: $logprobs}, on_conflict: {constraint: completions_pkey1, update_columns: [completion, logprobs, temperature, top_p]}) {
    completion
  }
}
"""
        ),
        variable_values=dict(
            completion=completion,
            logprobs=logprobs,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        ),
    )


@dataclass
class GPT3:
    debug: int
    logger: HasuraLogger
    logprobs: int
    temperature: float
    top_p: float
    max_tokens: int = 100

    def __call__(self, prompt):
        return self.get_full_completion(prompt)["completion"]

    def get_full_completion(self, prompt, use_cache: bool = True):
        if self.debug >= 0:
            print("<", end="")

        if use_cache:
            completions = self.get_completions(prompt)
            if completions:
                completion, *_ = completions
                # print("Completion:")
                # print(value)
                if self.debug >= 0:
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
                    max_tokens=self.max_tokens,
                    prompt=prompt,
                    logprobs=self.logprobs,
                    temperature=0.1,
                    stop=[".", ":"],
                ).choices
            except openai.error.RateLimitError as e:
                print(e)
                time.sleep(1)
                continue
            logprobs = [l.to_dict() for l in choice.logprobs.top_logprobs]
            completion = choice.text.lstrip()
            response = post_completion(
                logprobs=logprobs,
                logger=self.logger,
                prompt=prompt,
                completion=completion,
                temperature=self.temperature,
                top_p=self.top_p,
            )["insert_completions_one"]["completion"]
            if response != completion:
                breakpoint()
            if self.debug >= 0:
                print(">", end="")
            self.print("Completion:", completion.split("\n")[0])
            if self.debug >= 5:
                breakpoint()
            return dict(
                prompt=prompt,
                completion=completion,
                logprobs=logprobs,
            )

    def get_completions(self, prompt):
        return self.logger.execute(
            gql(
                """
query get_completion($prompt: String!, $temperature: numeric!, $top_p: numeric!) {
  completions(where: {prompt: {_eq: $prompt}, temperature: {_eq: $temperature}, top_p: {_eq: $top_p}}) {
    prompt
    completion
    logprobs
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
