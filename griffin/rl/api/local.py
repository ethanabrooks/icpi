import re
import sys
import time
from dataclasses import dataclass

import requests
from rl.common import Colorize
from rl.lm import LM, Data
from transformers import GPT2TokenizerFast


@dataclass
class Local(LM):
    seed: int
    url: str
    completion_count: int = 0
    completion_times: float = 0
    error_count: int = 0
    query_count: int = 0
    query_tick: float = time.time()
    query_times: float = 0

    def __post_init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        self.max_tokens_accepted_by_lm = 600
        assert self.logprobs == 0

    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        try:
            [stop] = stop
        except ValueError:
            raise RuntimeError("Only one stop token is supported")

        if self.debug >= 0:
            print("<", end="")

        prompt = self.clip_prompt(prompt)

        if use_cache:
            completions = self.get_completions(
                prompt, stop=[stop], temperature=temperature
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
                exit()

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 5:
            breakpoint()
        completion_tick = time.time()
        self.completion_count += 1
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            self.query_tick = time.time()
            self.query_count += 1
            response = requests.post(
                self.url,
                headers={
                    "accept": "application/json",
                    # Already added when you pass json= but not when you pass data=
                    # 'Content-Type': 'application/json',
                },
                json=dict(
                    prompt=prompt,
                    max_tokens=self.max_tokens_in_completion,
                    temperature=0.1,
                    top_p=1,
                    logprobs=self.logprobs,
                    stop=[stop],
                    seed=self.seed,
                ),
            )
            self.query_times += time.time() - self.query_tick

            if self.logger.run_id is not None:
                self.logger.log(
                    **{
                        "hours": (time.time() - self.start_time) / 3600,
                        "run ID": self.logger.run_id,
                        "seconds per query": self.query_times / self.query_count,
                    },
                )

            if not response.ok:
                print(response.text)
                sys.stdout.flush()
                self.error_count += 1
                continue

            choice, *_ = response.json()["choices"]
            completion = choice["text"].lstrip()
            completion = completion[: completion.find(stop)]  # TODO
            completion = re.sub(r"(\S)!=", r"\1 !=", completion)

            top_logprobs = []
            response = self.post_completion(
                completion=completion,
                prompt=prompt,
                stop=[stop],
                temperature=temperature,
                top_logprobs=top_logprobs,
            )["insert_completions_one"]["completion"]
            if response != completion:
                breakpoint()
            if self.debug >= 0:
                print(">", end="")
            self.print("Completion:", completion)
            if self.debug >= 6:
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
            if "state!=" in completion:
                breakpoint()
            return dict(
                prompt=prompt,
                completion=completion,
                top_logprobs=top_logprobs,
            )

    def max_prompt_tokens(self) -> int:
        return self.max_tokens_accepted_by_lm - self.max_tokens_in_completion - 100

    def print(self, *args, **kwargs):
        if self.debug >= 5:
            print(*args, **kwargs)

    def trained_on(self) -> Data:
        return Data.code
