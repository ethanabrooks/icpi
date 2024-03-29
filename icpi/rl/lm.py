import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from run_logger import RunLogger
from transformers import PreTrainedTokenizer

from gql import gql


class Data(Enum):
    code = auto()
    natural_language = auto()


@dataclass
class LM(ABC):
    debug: int
    logger: RunLogger
    logprobs: int
    model_name: str
    top_p: float
    max_tokens_in_completion: int
    require_cache: bool
    tokenizer: PreTrainedTokenizer = field(init=False)

    def __call__(
        self, prompt: str, stop: List[str], temperature: float, use_cache: bool = True
    ):
        return self.get_full_completion(
            prompt, stop=stop, temperature=temperature, use_cache=use_cache
        )["completion"]

    @abstractmethod
    def get_full_completion(
        self, prompt: str, stop: list[str], temperature: float, use_cache: bool = True
    ):
        ...

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

    def clip_prompt(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt)["input_ids"]
        tokens = tokens[-self.max_prompt_tokens() :]
        new_prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)
        new_prompt = re.sub(r"(\S)!=", r"\1 !=", new_prompt)
        l = list(zip(reversed(prompt), reversed(new_prompt)))
        for i, (old, new) in enumerate(reversed(l)):
            if old != new:
                print("old:", prompt[i : (i + 10)])
                print("new:", new_prompt[i : (i + 10)])
                print(f"{i}: {old} -> {new}")
                breakpoint()
        return new_prompt

    @abstractmethod
    def max_prompt_tokens(self) -> int:
        ...

    def post_completion(
        self,
        completion: str,
        prompt: str,
        stop: List[str],
        temperature: float,
        top_logprobs: list,
    ):
        return self.logger.execute(
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
                logprobs=self.logprobs,
                model=self.model_name,
                prompt=prompt,
                stop=stop,
                temperature=temperature,
                top_logprobs=top_logprobs,
                top_p=self.top_p,
            ),
        )

    @abstractmethod
    def trained_on(self) -> Data:
        ...
