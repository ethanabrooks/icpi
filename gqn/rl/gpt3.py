import sys
import time
from dataclasses import dataclass

import openai
from rl.common import Colorize
from rl.lm import LM
from transformers import GPT2TokenizerFast

OPENAI_MODELS = ["code-davinci-002", "text-davinci-002", "gpt3"]


MAX_TOKENS_ACCEPTED_BY_LM = 4000


@dataclass
class GPT3(LM):
    def __post_init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        assert self.logprobs <= 5

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
        if self.debug >= 5:
            breakpoint()
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            try:
                tick = time.time()
                choice, *_ = openai.Completion.create(
                    engine="text-davinci-002"
                    if self.model_name == "gpt3"
                    else self.model_name,
                    max_tokens=self.max_tokens_in_completion,
                    prompt=prompt,
                    logprobs=self.logprobs,
                    temperature=0.1,
                    stop=stop,
                ).choices
                if not use_cache:
                    print("HERE")
                    breakpoint()

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
            ) as e:
                print(type(e))
                print(e)
                sys.stdout.flush()
                continue
            except openai.error.InvalidRequestError as e:
                print("Invalid request error:")
                print(e)
                breakpoint()
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
        return MAX_TOKENS_ACCEPTED_BY_LM - self.max_tokens_in_completion - 200

    def print(self, *args, **kwargs):
        if self.debug >= 5:
            print(*args, **kwargs)
