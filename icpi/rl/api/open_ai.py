import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import openai
from rl.common import Colorize, Debug
from rl.lm import LM, Data
from transformers import GPT2TokenizerFast

OPENAI_MODELS = ["code-davinci-002", "code-cushman-001", "text-davinci-002"]


@dataclass
class OpenAi(LM):
    wait_time: Optional[float]
    completion_count: int = 0
    completion_times: float = 0
    error_count: int = 0
    query_count: int = 0
    query_tick: float = time.time()
    query_times: float = 0

    def __post_init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        self.max_tokens_accepted_by_lm = {
            "code-davinci-002": 4000,
            "code-cushman-001": 2049,
        }[self.model_name]
        assert self.logprobs <= 5

    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        if Debug.print_api_call_indicator.meets_threshold(self.debug):
            print("<", end="")

        prompt = self.clip_prompt(prompt)

        if use_cache:
            completions = self.get_completions(
                prompt, stop=stop, temperature=temperature
            )
            if completions:
                completion, *_ = completions
                # print("Completion:")
                # print(value)
                if Debug.print_api_call_indicator.meets_threshold(self.debug):
                    print(">", end="")
                return completion
            elif self.require_cache:
                print(prompt)
                Colorize.print_warning("No completions found in cache for this prompt.")
                exit()

        if Debug.debug_api_calls.meets_threshold(self.debug):
            print("Prompt:")
            print(prompt)
            breakpoint()
        completion_tick = time.time()
        self.completion_count += 1
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                time_since_last_query = time.time() - self.query_tick
                time.sleep(max(0.0, self.wait_time - time_since_last_query))
                self.query_tick = time.time()
                self.query_count += 1
                choice, *_ = openai.Completion.create(
                    engine=self.model_name,
                    max_tokens=self.max_tokens_in_completion,
                    prompt=prompt,
                    logprobs=self.logprobs,
                    temperature=0.1,
                    stop=stop,
                ).choices
                self.query_times += time.time() - self.query_tick

                if self.logger.run_id is not None:
                    self.logger.log(
                        **{
                            "hours": (time.time() - self.start_time) / 3600,
                            "run ID": self.logger.run_id,
                            "seconds per query": self.query_times / self.query_count,
                            "API error probability": self.error_count
                            / self.query_count,
                        },
                    )
                # if not choice.text:
                #     print(prompt)
                #     Colorize.print_warning("Empty completion!")
                #     breakpoint()
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.APIConnectionError,
            ) as e:
                if isinstance(e, openai.error.RateLimitError):
                    [count] = re.findall(r"Current: (\d+)", e.error.message)
                    print(count, end=" ")
                else:
                    print(type(e))
                    print(e)
                sys.stdout.flush()
                self.error_count += 1
                continue
            except openai.error.InvalidRequestError as e:
                print("Invalid request error:")
                print(e)
                self.max_tokens_accepted_by_lm -= 100
                continue

            top_logprobs = [l.to_dict() for l in choice.logprobs.top_logprobs]
            completion = choice.text.lstrip()
            response = self.post_completion(
                completion=completion,
                prompt=prompt,
                stop=stop,
                temperature=temperature,
                top_logprobs=top_logprobs,
            )["insert_completions_one"]["completion"]
            if response != completion:
                breakpoint()
            if Debug.print_api_call_indicator.meets_threshold(self.debug):
                print(">", end="")
            if Debug.debug_api_calls.meets_threshold(self.debug):
                print("Completion:", completion.split("\n")[0])
                breakpoint()
            self.completion_times += time.time() - completion_tick
            if self.logger.run_id is not None:
                self.logger.log(
                    **{
                        "hours": (time.time() - self.start_time) / 3600,
                        "run ID": self.logger.run_id,
                        "seconds per completion": self.completion_times
                        / self.completion_count,
                    },
                )
            return dict(
                prompt=prompt,
                completion=completion,
                top_logprobs=top_logprobs,
            )

    def max_prompt_tokens(self) -> int:
        return self.max_tokens_accepted_by_lm - self.max_tokens_in_completion - 100

    def trained_on(self) -> Data:
        training_data: dict[str, Data] = {
            "code-cushman-001": Data.code,
            "code-davinci-002": Data.code,
            "text-davinci-002": Data.natural_language,
        }
        return training_data[self.model_name]
