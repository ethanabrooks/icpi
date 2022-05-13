import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import openai
from rl.common import Colorize
from run_logger import HasuraLogger
from transformers import GPT2TokenizerFast

from gql import gql

OPENAI_MODELS = ["code-davinci-002", "text-davinci-002"]


def post_completion(
    completion: str,
    logprobs: int,
    logger: HasuraLogger,
    model: str,
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
            model=model,
            prompt=prompt,
            stop=stop,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
        ),
    )


MAX_TOKENS = 4000


@dataclass
class GPT3:
    debug: int
    logger: HasuraLogger
    logprobs: int
    model_name: str
    top_p: float
    wait_time: float
    max_tokens: int = 100
    require_cache: bool = False
    stop: Optional[List[str]] = None

    def __post_init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        if self.wait_time is None:
            if self.model_name == "code-davinci-002":
                self.wait_time = 4
            elif self.model_name == "text-davinci-002":
                self.wait_time = 0
            else:
                raise ValueError(f"Unknown model {self.model_name}")
        assert self.logprobs <= 5

    def __call__(
        self, prompt: str, stop: List[str], temperature: float, use_cache: bool = True
    ):
        return self.get_full_completion(
            prompt, stop=stop, temperature=temperature, use_cache=use_cache
        )["completion"]

    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        if self.debug >= 0:
            print("<", end="")

        tokens = self.tokenizer(prompt)["input_ids"]
        max_tokens = MAX_TOKENS - self.max_tokens - 100
        tokens = tokens[-max_tokens:]
        prompt = self.tokenizer.decode(tokens)

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
                print(prompt)
                Colorize.print_warning("No completions found in cache for this prompt.")
                breakpoint()
                exit()

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 5:
            breakpoint()
        wait_time = self.wait_time
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            wait_time = min(wait_time, 60)
            sys.stdout.flush()
            try:
                time.sleep(wait_time)
                tick = time.time()
                choice, *_ = openai.Completion.create(
                    engine=self.model_name,
                    max_tokens=self.max_tokens,
                    prompt=prompt,
                    logprobs=self.logprobs,
                    temperature=0.1,
                    stop=stop,
                ).choices

                if self.logger.run_id is not None:
                    self.logger.log(
                        **{
                            "hours": (time.time() - self.start_time) / 3600,
                            "run ID": self.logger.run_id,
                            "seconds per query": time.time() - tick,
                        },
                    )
                # if not choice.text:
                #     print(prompt)
                #     Colorize.print_warning("Empty completion!")
                #     breakpoint()
            except openai.error.RateLimitError as e:
                print("Rate limit error:")
                print(e)
                sys.stdout.flush()
                wait_time *= 2
                continue
            except openai.error.InvalidRequestError as e:
                print("Invalid request error:")
                print(e)
                breakpoint()
                continue

            top_logprobs = [l.to_dict() for l in choice.logprobs.top_logprobs]
            completion = choice.text.lstrip()
            response = post_completion(
                completion=completion,
                logger=self.logger,
                logprobs=self.logprobs,
                model=self.model_name,
                prompt=prompt,
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
                model=self.model_name,
                prompt=prompt,
                stop=stop,
                temperature=temperature,
                top_p=self.top_p,
            ),
        )["completions"]

    def print(self, *args, **kwargs):
        if self.debug >= 5:
            print(*args, **kwargs)
