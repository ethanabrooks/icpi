import sys
import time
from dataclasses import dataclass
from typing import Optional

import openai
from rl.common import Colorize
from rl.lm import LM
from transformers import GPT2TokenizerFast

OPENAI_MODELS = ["code-davinci-002", "text-davinci-002"]


@dataclass
class GPT3(LM):
    wait_time: Optional[float]

    def __post_init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        self.max_tokens_accepted_by_lm = 4000
        if self.wait_time is None:
            if self.model_name == "code-davinci-002":
                self.wait_time = 4
            elif self.model_name == "text-davinci-002":
                self.wait_time = 0
            else:
                raise ValueError(f"Unknown model {self.model_name}")
        assert self.logprobs <= 6

    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        if self.debug >= 0:
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
        if self.debug >= 6:
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
                    max_tokens=self.max_tokens_in_completion,
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
            except (
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.APIConnectionError,
            ) as e:
                print(type(e))
                print(e)
                sys.stdout.flush()
                wait_time *= 2
                if wait_time == 0:
                    wait_time = 1
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

    def max_prompt_tokens(self) -> int:
        return self.max_tokens_accepted_by_lm - self.max_tokens_in_completion - 100

    def print(self, *args, **kwargs):
        if self.debug >= 6:
            print(*args, **kwargs)
