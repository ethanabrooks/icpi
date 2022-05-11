import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import deepspeed
import numpy as np
import torch
from deepspeed import DeepSpeedEngine
from run_logger import HasuraLogger
from torch.nn.functional import log_softmax
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.deepspeed import HfDeepSpeedConfig

from gql import gql

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_MODELS = {
    "xglm": "facebook/xglm-564M",
    "gpt2": "gpt2-xl",
    "gptj": "EleutherAI/gpt-j-6B",
    "gptneo": "EleutherAI/gpt-neo-1.3B",
    "fairseq13b": "KoboldAI/fairseq-dense-13B",
    "incoder6b": "facebook/incoder-6B",
}


def post_completion(
    best_of: Optional[int],
    completion: str,
    logprobs: int,
    logger: HasuraLogger,
    prompt: str,
    stop: Optional[List[str]],
    temperature: float,
    top_logprobs: list,
    top_p: float,
    model: str,
):
    return logger.execute(
        query=gql(
            """
mutation post_completion($prompt: String!, $completion: String!, $temperature: numeric!, $top_p: numeric!, $logprobs: Int!, $top_logprobs: jsonb!, $model: String!, $best_of: Int, $stop: jsonb) {
  insert_completions_one(object: {completion: $completion, prompt: $prompt, temperature: $temperature, top_p: $top_p, logprobs: $logprobs, top_logprobs: $top_logprobs, model: $model, best_of: $best_of, stop: $stop}, on_conflict: {constraint: completions_pkey1, update_columns: [completion, logprobs, temperature, top_p, stop, best_of]}) {
    completion
    stop
  }
}
"""
        ),
        variable_values=dict(
            best_of=best_of,
            completion=completion,
            logprobs=logprobs,
            prompt=prompt,
            stop=stop,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
            model=model,
        ),
    )


class TokenStoppingCriteria(StoppingCriteria):
    def __init__(self, token_ids: List[int], device: Union[int, str]):
        self.device = device
        self.token_ids = torch.tensor(token_ids, device=self.device)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return bool(torch.any(input_ids[:, -1] == self.token_ids))


@dataclass
class HuggingFaceModel:
    model_name: str
    debug: int
    logger: HasuraLogger
    logprobs: int
    temperature: float
    top_p: float
    seed: int
    max_tokens: int = 100
    best_of: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop: Optional[List[str]] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    distributed: bool = False
    ds_engine: Optional[DeepSpeedEngine] = None
    hf_ds_conf: Optional[HfDeepSpeedConfig] = None
    local_rank: int = 0
    local_device: Union[int, str] = field(init=False)
    model: PreTrainedModel = field(init=False)
    tokenizer: PreTrainedTokenizer = field(init=False)
    generate_fn: Callable = field(init=False)

    def __post_init__(self):
        assert self.logprobs <= 5
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        world_size = os.getenv("WORLD_SIZE", None)
        if world_size is not None:
            world_size = int(world_size)
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            deepspeed.init_distributed()
            self.local_rank = local_rank
            self.local_device = local_rank
            self.distributed = True
            config = AutoConfig.from_pretrained(self.model_name)
            model_hidden_size = config.d_model
            train_batch_size = 1 * world_size
            ds_config = {
                "fp16": {"enabled": False},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3,
                    # "offload_param": {
                    #     "device": "cpu",
                    #     "pin_memory": True
                    # },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": model_hidden_size * model_hidden_size,
                    "stage3_prefetch_bucket_size": 0.9
                    * model_hidden_size
                    * model_hidden_size,
                    "stage3_param_persistence_threshold": 10 * model_hidden_size,
                },
                "steps_per_print": 2000,
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": 1,
                "wall_clock_breakdown": False,
            }
            self.hf_ds_conf = HfDeepSpeedConfig(ds_config)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.eos_token_id = self.eos_token_id or self.tokenizer.eos_token_id

        if self.distributed:
            self.ds_engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]  # type: ignore
            self.ds_engine.module.eval()
            self.generate_fn = self.ds_engine.module.generate
        else:
            self.local_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model.to(self.local_device)
            self.model.eval()
            self.generate_fn = self.model.generate

        if self.stop:
            self.stopping_criteria = StoppingCriteriaList()
            self.stopping_criteria.append(
                TokenStoppingCriteria(
                    self.tokenizer.convert_tokens_to_ids(self.stop),
                    device=self.local_device,
                )
            )

    def __call__(self, prompt, best_of: bool):
        return self.get_full_completion(prompt, best_of=best_of, stop=self.stop)[
            "completion"
        ]

    def get_full_completion(
        self, prompt, best_of: bool, stop: Optional[List[str]], use_cache: bool = True
    ):
        best_of = 1 if best_of else None
        self.print("<", end="")

        if use_cache and self.local_rank == 0:
            completions = self.get_completions(prompt, best_of=best_of, stop=stop)
            if completions:
                completion, *_ = completions
                # print("Completion:")
                # print(value)
                if self.debug >= 0:
                    self.print(">", end="")
                return completion

        self.print("Prompt:")
        self.print(prompt)
        if self.debug >= 5:
            breakpoint()
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=False,
            )["input_ids"].to(self.local_device)
            top_logprobs = []
            with torch.no_grad():
                result = self.generate_fn(
                    inputs,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.eos_token_id,
                    max_new_tokens=self.max_tokens,
                    stopping_criteria=self.stopping_criteria,
                    synced_gpus=self.distributed,
                )
                if self.logprobs > 0:
                    for token in result.scores:
                        logprobs, token_ids = log_softmax(token, dim=1).topk(
                            self.logprobs
                        )
                        top_logprobs.append(
                            dict(
                                zip(
                                    self.tokenizer.batch_decode(token_ids.T),
                                    logprobs.squeeze().tolist(),
                                )
                            )
                        )
                # Trim the prompt from the output.
                text = result.sequences[:, inputs.size(1) :]
                completion = self.tokenizer.batch_decode(
                    text,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            response = post_completion(
                best_of=best_of,
                logger=self.logger,
                logprobs=self.logprobs,
                prompt=prompt,
                completion=completion,
                stop=stop,
                temperature=self.temperature,
                top_logprobs=top_logprobs,
                top_p=self.top_p,
                model=self.model_name,
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

    def get_completions(self, prompt: str, best_of: bool, stop: Optional[List[str]]):
        return self.logger.execute(
            gql(
                """
query get_completion($prompt: String!, $temperature: numeric!, $top_p: numeric!, $model: String!, $best_of: Int, $stop: jsonb, $logprobs: Int!) {
  completions(where: {prompt: {_eq: $prompt}, temperature: {_eq: $temperature}, top_p: {_eq: $top_p}, stop: {_eq: $stop}, logprobs: {_eq: $logprobs}, model: {_eq: $model}, best_of:"""
                + ("{_is_null: true}" if best_of is None else "{_eq: $best_of}")
                + """}) {
    prompt
    completion
    top_logprobs
  }
}"""
            ),
            variable_values=dict(
                **({} if best_of is None else dict(best_of=best_of)),
                logprobs=self.logprobs,
                prompt=prompt,
                stop=stop,
                temperature=self.temperature,
                top_p=self.top_p,
                model=self.model_name,
            ),
        )["completions"]

    def print(self, *args, **kwargs):
        if self.debug >= 5 and self.local_rank == 0:
            print(*args, **kwargs)
