import re
import sys
import time
from dataclasses import dataclass, field

from rl.common import Colorize, Debug
from rl.lm import LM, Data
from text_generation import Client
from transformers import GPT2TokenizerFast

CODE_ONLY_TEMPLATE = "```python\n{input_text}"
INSTRUCTION_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n" + CODE_ONLY_TEMPLATE
)


@dataclass
class Local(LM):
    seed: int
    url: str
    template: str
    completion_count: int = 0
    completion_times: float = 0
    error_count: int = 0
    query_count: int = 0
    query_tick: float = time.time()
    query_times: float = 0
    client: Client = field(init=False)

    def __post_init__(self):
        self.client = Client(self.url)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start_time = time.time()
        self.max_tokens_accepted_by_lm = 2048
        assert self.logprobs == 0

    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        try:
            [stop] = stop
        except ValueError:
            raise RuntimeError("Only one stop token is supported")

        if Debug.print_api_call_indicator.meets_threshold(self.debug):
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
            self.query_tick = time.time()
            self.query_count += 1
            try:
                response = self.client.generate(
                    self.template.format(input_text=prompt),
                    max_new_tokens=self.max_tokens_in_completion,
                    do_sample=True,
                    temperature=0.1,
                    top_p=1.0,
                    seed=self.seed,
                    stop_sequences=stop,
                    watermark=False,
                )
            except Exception as e:
                print(e)
                sys.stdout.flush()
                self.error_count += 1
                continue

            self.query_times += time.time() - self.query_tick

            if self.logger.run_id is not None:
                self.logger.log(
                    **{
                        "hours": (time.time() - self.start_time) / 3600,
                        "run ID": self.logger.run_id,
                        "seconds per query": self.query_times / self.query_count,
                    },
                )

            completion = response.generated_text.lstrip()
            completion = completion[: completion.rfind(stop)]  # TODO
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
            if Debug.print_api_call_indicator.meets_threshold(self.debug):
                print(">", end="")
            if Debug.debug_api_calls.meets_threshold(self.debug):
                print("Completion:", completion)
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

    def trained_on(self) -> Data:
        return Data.code
