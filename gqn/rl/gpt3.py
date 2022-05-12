import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import openai
from run_logger import HasuraLogger
from util import Colorize

from gql import gql

ENGINE = "code-davinci-002"


def post_completion(
    completion: str,
    logprobs: int,
    logger: HasuraLogger,
    prompt: str,
    stop: List[str],
    temperature: float,
    top_logprobs: list,
    top_p: float,
):
    return logger.execute(
        query=gql(
            """
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!, $logprobs: Int!, $top_logprobs: jsonb!, $stop: jsonb, $model: String!) {
  insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p, logprobs: $logprobs, top_logprobs: $top_logprobs, stop: $stop, model: $model}) {
    completion
    stop
  }
}
"""
        ),
        variable_values=dict(
            completion=completion,
            logprobs=logprobs,
            model=ENGINE,
            prompt=prompt,
            stop=stop,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
        ),
    )


@dataclass
class GPT3:
    debug: int
    logger: HasuraLogger
    logprobs: int
    top_p: float
    max_tokens: int = 100
    require_cache: bool = False
    stop: Optional[List[str]] = None

    def __post_init__(self):
        assert self.logprobs <= 5

    def __call__(
        self, prompt, stop: List[str], temperature: float, use_cache: bool = True
    ):
        return self.get_full_completion(
            prompt, stop=stop, temperature=temperature, use_cache=use_cache
        )["completion"]

    def get_full_completion(
        self, prompt, stop: list[str], temperature: float, use_cache: bool = True
    ):
        if self.debug >= 0:
            print("<", end="")

        if use_cache:
            completions = self.get_completions(
                prompt, stop=stop, temperature=temperature
            )
            if completions:
                completion, *_ = completions
                # print("Completion:")
                # print(value)
                if self.debug >= 0:
                    print(">", end="")
                return completion
            elif self.require_cache:
                print("No completions found in cache for prompt:")
                print(prompt)
                breakpoint()

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 5:
            breakpoint()
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                choice, *_ = openai.Completion.create(
                    engine=ENGINE,
                    max_tokens=self.max_tokens,
                    prompt=prompt,
                    logprobs=self.logprobs,
                    temperature=0.1,
                    stop=stop,
                ).choices
                if not choice.text:
                    print(prompt)
                    Colorize.print_warning("Empty completion!")
                    breakpoint()
                time.sleep(4)
            except openai.error.RateLimitError as e:
                print("Rate limit error:")
                print(e)
                time.sleep(8)
                continue
            except openai.error.InvalidRequestError as e:
                print("Invalid request error:")
                print(e)
                _, *prompts = prompt.split("\n")
                prompt = "\n".join(prompts)
                continue

            top_logprobs = [l.to_dict() for l in choice.logprobs.top_logprobs]
            completion = choice.text.lstrip()
            response = post_completion(
                logger=self.logger,
                logprobs=self.logprobs,
                prompt=prompt,
                completion=completion,
                stop=stop,
                temperature=temperature,
                top_logprobs=top_logprobs,
                top_p=self.top_p,
            )["insert_completions_one"]["completion"]
            if response != completion:
                breakpoint()
            if self.debug >= 0:
                print(">", end="")
            self.print("Completion:", completion.split("\n")[0])
            if self.debug >= 6:
                breakpoint()
            return dict(
                prompt=prompt,
                completion=completion,
                top_logprobs=top_logprobs,
            )

    def get_completions(self, prompt: str, stop: List[str], temperature: float):
        return self.logger.execute(
            gql(
                """
query get_completion($prompt: String!, $temperature: numeric!, $top_p: numeric!, $best_of: Int, $stop: jsonb, $logprobs: Int!, $model: String!) {
  completions(where: {prompt: {_eq: $prompt}, temperature: {_eq: $temperature}, top_p: {_eq: $top_p}, stop: {_eq: $stop}, logprobs: {_eq: $logprobs}, model: {_eq: $model}, best_of: {_is_null: true}}) {
    prompt
    completion
    top_logprobs
  }
}"""
            ),
            variable_values=dict(
                logprobs=self.logprobs,
                model=ENGINE,
                prompt=prompt,
                stop=stop,
                temperature=temperature,
                top_p=self.top_p,
            ),
        )["completions"]

    def print(self, *args, **kwargs):
        if self.debug >= 5:
            print(*args, **kwargs)
