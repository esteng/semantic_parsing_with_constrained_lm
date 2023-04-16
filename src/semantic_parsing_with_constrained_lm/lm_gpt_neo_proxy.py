# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""IncrementalLanguageModel which uses Huggingface LMs"""
import pdb 
import ast
import asyncio
import collections
import dataclasses
import datetime
import functools
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import httpx
import more_itertools
import torch
from cached_property import cached_property
from httpx import Response
from httpx._types import HeaderTypes
from transformers import GPT2Tokenizer, AutoModelForCausalLM, CodeGenForCausalLM, AutoTokenizer


from semantic_parsing_with_constrained_lm.async_tools import limits
from semantic_parsing_with_constrained_lm.async_tools.batch_helper import BatchingHelper, BatchMaker
from semantic_parsing_with_constrained_lm.cache import CacheClient
from semantic_parsing_with_constrained_lm.lm import IncrementalLanguageModel, TokensWithLogprobs
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer, GPT2ClampTokenizer


from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import (
    Instrumentation, 
    RequestInfo,
    OpenAIAPIError,
    CompletionsParams,
    CompletionsBatchMaker,
    IncrementalOpenAIGPT3,
    EchoBatchMaker,
    NextLogprobsBatchMaker,
    OpenAIGPT3State,
    openai_token_to_id
)
make_default_client = lambda: None
adjust_tokenizer = lambda _1, _2: None

# default_engine = os.environ.get("OPENAI_GPT3_ENGINE", "text-davinci-001")
default_model = "Salesforce/codegen-350M-mono"



@dataclass
class GPTNeoClient:
    # print(default_model)
    model: torch.nn.Module = CodeGenForCausalLM.from_pretrained(default_model)
    tokenizer: torch.nn.Module = AutoTokenizer.from_pretrained(default_model)
    api_key: Optional[str] = None

    cache_client: Optional[CacheClient] = dataclasses.field(
        default_factory=make_default_client
    )
    http_client: httpx.AsyncClient = dataclasses.field(init=False)
    request_limiter: limits.AdaptiveLimiter = dataclasses.field(
        default_factory=functools.partial(
            limits.AdaptiveLimiter, initial_qps=10000, max_qps=100000
        )
    )
    completions_rate_limited: Callable[
        [Dict[str, Any]], Awaitable[httpx.Response]
    ] = dataclasses.field(init=False)
    completions_url: str = dataclasses.field(init=False)

    def _init_api_key(self, env: str) -> str:
        if self.api_key is None:
            self.api_key = os.getenv(env)
        if self.api_key is None:
            raise ValueError(f"{env} was not set")
        return self.api_key

    def __post_init__(self):
        # We have an internal instance which has a different URL, auth token and header.
        # To access that instance, you can use the engine "codex-cushman-sm" -- note that
        # the "-sm" suffix is not part of the actual underlying engine name, just something
        # we use to switch on here.
        # See https://semanticmachines.slack.com/archives/C017P5M1RSL/p1647450366073519?thread_ts=1646782663.584339&cid=C017P5M1RSL
        # Get keys here: https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/b68b2f37-1d37-4c2f-80f6-c23de402792e/resourceGroups/fod/providers/Microsoft.CognitiveServices/accounts/smopenai/cskeys
        auth_header: HeaderTypes

        # Pyright bug forces us to first store the result in `limited`
        # https://github.com/microsoft/pyright/issues/2965
        limited = self.request_limiter(self._completions_with_raise_if_limited)
        self.completions_rate_limited = limited

        self.all_toks = self.tokenizer.convert_ids_to_tokens([i for i in range(51200)])

    async def __aenter__(self):
        pass 
        # await self.http_client.__aenter__()
        # if self.cache_client is not None:
        #     await self.cache_client.__aenter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass 
        # await self.http_client.__aexit__(exc_type, exc_value, traceback)
        # if self.cache_client is not None:
            # await self.cache_client.__aexit__(exc_type, exc_value, traceback)

    async def _completions_with_raise_if_limited(
        self, args_without_engine: Dict[str, Any]
    ) -> httpx.Response:

        # turn into request to a model 
        request_info = RequestInfo.create(args_without_engine)
        Instrumentation.currently_pending_requests += 1
        Instrumentation.record_request(request_info)

        prompt_tensor = torch.tensor(args_without_engine['prompt']).unsqueeze(0)
        prompt_tensor = prompt_tensor.to(self.model.device)
        result = self.model(prompt_tensor) 
        request_info.finish(True)
        return result




@dataclass(frozen=True)
class CompletionsParams:
    max_tokens: int
    temperature: float
    top_p: float
    num_completions: int
    stop: Optional[str]


@dataclass(frozen=True)
class GPTNeoCompletionsBatchMaker(CompletionsBatchMaker):
    client: GPTNeoClient = dataclasses.field(compare=False)
    params: CompletionsParams

    @property
    def max_batch_size(self) -> int:
        return 100

    @property
    def timeout(self) -> float:
        return 0.05

    async def execute(
        self, args: List[Tuple[Sequence[int], CompletionsParams]]
    ) -> List[List[Tuple[List[str], List[float]]]]:
        # Outermost List has length equal to the batch size
        # 2nd level List has length equal to `num_completions`
        # Each Tuple contains two (parallel) lists: tokens and their log probabilities
        batched_tokens = [x[0] for x in args]
        params = {
            "prompt": batched_tokens,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "n": self.params.num_completions,
            "stop": self.params.stop,
            "logprobs": 0,
        }
        response = (
            await self.client.completions_rate_limited(params)  # type: ignore
        )

        result: List[List[Tuple[List[str], List[float]]]] = []
        for choices_per_prompt in more_itertools.chunked(
            response["choices"], self.params.num_completions
        ):
            result.append(
                [
                    (c["logprobs"]["tokens"], c["logprobs"]["token_logprobs"])
                    for c in choices_per_prompt
                ]
            )
        return result



@dataclass
class IncrementalGPTNeo(IncrementalOpenAIGPT3):
    model: str = default_model
    # model: torch.nn.Module = CodeGenForCausalLM.from_pretrained(default_model)

    use_cache: bool = True

    client: GPTNeoClient = dataclasses.field(init=False)
    echo_batch_helper: BatchingHelper[
        Sequence[int], List[List[float]]
    ] = dataclasses.field(init=False)
    next_logprobs_batch_helper: BatchingHelper[
        Sequence[int], List[Dict[str, float]]
    ] = dataclasses.field(init=False)
    completions_batch_helper: BatchingHelper[
        Tuple[Sequence[int], CompletionsParams],
        List[List[Tuple[List[str], List[float]]]],
    ] = dataclasses.field(init=False)

    def __post_init__(self):
        client = GPTNeoClient() #model=self.model)
        client.model.to("cuda:0")
        self.client = client
        self.echo_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda _args: EchoBatchMaker(client),
        )
        self.next_logprobs_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda _args: GPTNeoNextLogprobsBatchMaker(client),
        )
        self.completions_batch_helper = BatchingHelper(
            input_to_batch_maker=lambda args: CompletionsBatchMaker(client, args[1]),
        )
        if self.client.cache_client is None:
            self.use_cache = False

    async def __aenter__(self):
        await self.client.__aenter__()

    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)

    @cached_property
    def vocab_size(self):  # pylint: disable=invalid-overridden-method
        return self.tokenizer.vocab_size

    @cached_property
    def tokenizer(self) -> ClampTokenizer:  # pylint: disable=invalid-overridden-method
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        adjust_tokenizer(self.engine, gpt2_tokenizer)
        return GPT2ClampTokenizer(gpt2_tokenizer)

    @cached_property
    def max_length(self) -> int:
        if self.engine.startswith("davinci-codex"):
            return 4096
        return 2048

    async def execute(
        self,
        tokens: Sequence[int],
        hidden_state: Optional[OpenAIGPT3State] = None,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[OpenAIGPT3State]]:
        # In order to reduce network traffic, this function only returns the
        # logprobs for the last token. It also only returns the top 100 logprobs
        # due to limitations of the OpenAI API.
        if hidden_state is None:
            all_tokens = tuple(tokens)
        else:
            all_tokens = hidden_state.tokens + tuple(tokens)

        if self.use_cache and self.client.cache_client:
            cache_args = {
                "engine": self.engine,
                "prompt": all_tokens,
                "max_tokens": 1,
                "logprobs": 100,
            }
            cached = await self.client.cache_client.get(cache_args)
        else:
            cache_args = None
            cached = None

        if cached:
            next_logprobs = cached["choices"][0]["logprobs"]["top_logprobs"][0]
        else:
            batched_next_logprobs, i = await self.next_logprobs_batch_helper.execute(
                all_tokens
            )
            next_logprobs = batched_next_logprobs[i]
            if self.use_cache and self.client.cache_client:
                assert cache_args is not None
                asyncio.create_task(
                    self.client.cache_client.upload(
                        cache_args,
                        {"choices": [{"logprobs": {"top_logprobs": [next_logprobs]}}]},
                    )
                )

        # result = torch.full(
            # (max(1, len(tokens)), self.tokenizer.vocab_size), -float("inf")
        # )
        # pdb.set_trace()
        # for token, logprob in next_logprobs.items():
        #     # pdb.set_trace()
        #     token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(token)
        #     # token_id = openai_token_to_id(self.tokenizer, token)
        #     result[-1, token_id] = logprob
        result = next_logprobs.unsqueeze(0)
        return (
            result,
            None if drop_next_hidden_state else OpenAIGPT3State(all_tokens),
        )

 
@dataclass(frozen=True)
class GPTNeoNextLogprobsBatchMaker(NextLogprobsBatchMaker):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    async def execute(
        self, batched_tokens: List[Sequence[int]]
    ) -> List[Dict[str, float]]:
        args = {
            "prompt": batched_tokens,
            "max_tokens": 1,
            "logprobs": 100,
        }
        # https://github.com/python/mypy/issues/708
        results = (
            await self.client.completions_rate_limited(args)  # type: ignore
        )
        logits = results['logits'].squeeze(0).squeeze(0)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        # pdb.set_trace()
        # top_logprobs, top_indices = torch.max(logprobs, dim=-1)
        # tokens = self.client.tokenizer.convert_ids_to_tokens(top_indices.tolist())
        # to_ret = {}
        # to_ret = []

        # for i, prob in enumerate(logprobs):
        #     d = {k:p for p, k in zip(prob, self.client.all_toks)}
        #     to_ret.append(d)
        # return [d["logprobs"]["top_logprobs"][0] for d in results["choices"]]   
        # return to_ret 
        return logprobs

   



if Instrumentation.AUTOMATIC_PRINTING_ENABLED:
    threading.Thread(target=Instrumentation.print_loop, daemon=True).start()
