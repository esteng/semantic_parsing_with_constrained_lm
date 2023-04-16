# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import pdb 
import os 
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, List, MutableMapping, Optional, Sequence, Tuple, Dict, Any
import time 
from collections import namedtuple

import torch
import httpx
from cached_property import cached_property

from semantic_parsing_with_constrained_lm.datum import Datum, DatumSub, FullDatum, FullDatumSub
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.fewshot import (
    DataFilter,
    DataRetriever,
    GPT2TokenizerQuirks,
    PromptBuilder,
)
from semantic_parsing_with_constrained_lm.lm import (
    HS,
    AutoregressiveModel,
    IncrementalLanguageModel,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3, OpenAIGPT3State
from semantic_parsing_with_constrained_lm.search import (
    ConstrainedDecodingProblem,
    FullSearchNode,
    LoggingEventListener,
    PackedSearchNode,
    Problem,
    beam_search,
)
from semantic_parsing_with_constrained_lm.speculative_decoding import (
    SpeculativeConstrainedDecodingProblem,
)
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

from ambiguous_parsing.tree.formula import FOLFormula, LispFormula
PartialParseBuilder = Callable[[DatumSub], PartialParse]


@dataclass
class IncrementalLMSimilarityFunction:
    """Computes the similarity between two the utterances in a train and test datum."""

    model: IncrementalLanguageModel

    async def __call__(self, train_datum: FullDatum, test_datum: Datum) -> float:
        prefix_tokens = self.model.tokenizer.encode(f"1. {train_datum.natural}\n2.")
        # TODO: If test_datum.natural begins with something that will cause
        # this leading space to be its own token, should we put the space in prefix_tokens instead?
        completion_tokens = self.model.tokenizer.encode(f" {test_datum.natural}")
        return await self.model.logprob_of_completion(prefix_tokens, completion_tokens)


@dataclass(frozen=True, eq=True)
class DatumPackedSearchNode(Generic[DatumSub], PackedSearchNode):
    test_datum: DatumSub

    def append(self, token: int, logprob: float) -> "DatumPackedSearchNode":
        return DatumPackedSearchNode(
            tokens=self.tokens + (token,), logprobs = self.logprobs + (logprob.item(),), test_datum=self.test_datum
        )

    def extend(self, tokens: Sequence[int], logprobs: Sequence[float]) -> "DatumPackedSearchNode":
        if not tokens:
            return self

        return DatumPackedSearchNode(
            tokens=self.tokens + tuple(tokens), logprobs=self.logprobs + tuple(logprobs), test_datum=self.test_datum
        )


# https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class DecodingSetup(Generic[DatumSub, HS], ABC):
    """Specifies how to start/setup the decoding for a datum, and when to finish.

    This interface encapsulates the knowledge about how to use a model.
    For example, using a fine-tuned seq2seq model requires processing the utterance in the encoder;
    using an autoregressive language model with few-shot in-context examples requires
    finding and assembling the training examples into a prompt and encoding it with the model.

    Instances are used in BeamSearchSemanticParser.
    """

    partial_parse_builder: PartialParseBuilder

    @abstractmethod
    async def unpack_node(
        self, packed_node: DatumPackedSearchNode[DatumSub]
    ) -> Tuple[PartialParse, HS, Sequence[float]]:
        """Prepares a PackedSearchNdoe for decoding.

        The PackedSearchNode is usually empty, but it may contain some tokens
        which is useful for interactive use cases (e.g. an auto-complete server)."""
        pass

    @property
    @abstractmethod
    def eos_tokens(self) -> torch.Tensor:
        """The set of tokens which denotes the end of the search."""
        pass

    @abstractmethod
    def finalize(self, tokens: List[int]) -> str:
        """Converts the result of decoding into a string.

        It's not just tokenizer.decode because further processing may be needed."""
        pass


# https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class ProblemFactory(Generic[DatumSub, HS], ABC):
    """Sets up a Problem and the initial SearchNode for a given Datum.

    Different search strategies are encapsulated as different Problems
    (e.g. decoding token by token, or decoding many tokens at once)
    and this interface specifies which one to use.
    """

    # TODO: Merge this class with the Problem class
    # (unless we need to make `problem` a function of the datum).
    decoding_setup: DecodingSetup[DatumSub, HS]

    def initial(self, datum: DatumSub) -> DatumPackedSearchNode:
        return DatumPackedSearchNode(tokens=(), logprobs=(), test_datum=datum)

    @property
    @abstractmethod
    def problem(self) -> Problem[HS, DatumPackedSearchNode]:
        pass


@dataclass
class ConstrainedDecodingProblemFactory(ProblemFactory[DatumSub, HS]):
    autoregressive_model: AutoregressiveModel[HS]
    length_normalization: float = 0.7
    top_k: Optional[int] = None
    cache: Optional[MutableMapping[PackedSearchNode, List[FullSearchNode[HS]]]] = None

    @cached_property
    def problem(
        self,
    ) -> ConstrainedDecodingProblem[
        HS, DatumPackedSearchNode
    ]:  # pylint: disable=invalid-overridden-method
        return ConstrainedDecodingProblem(
            self.autoregressive_model,
            self.decoding_setup.unpack_node,
            self.decoding_setup.eos_tokens,
            self.length_normalization,
            self.top_k,
            self.cache,
        )


@dataclass
class SpeculativeConstrainedDecodingProblemFactory(
    ProblemFactory[DatumSub, OpenAIGPT3State]
):
    # TODO: Support models other than GPT-3
    gpt3: IncrementalOpenAIGPT3

    # TODO: Avoid duplicating these parameters with SpeculativeConstrainedDecodingProblem
    max_length: int
    num_completions: int
    temperature: float = 1
    top_p: float = 1

    length_normalization: float = 0.7

    @cached_property
    def problem(
        self,
    ) -> SpeculativeConstrainedDecodingProblem[
        DatumPackedSearchNode
    ]:  # pylint: disable=invalid-overridden-method
        return SpeculativeConstrainedDecodingProblem(
            self.gpt3,
            self.decoding_setup.unpack_node,
            self.length_normalization,
            self.decoding_setup.eos_tokens,
            self.max_length,
            self.num_completions,
            self.temperature,
            self.top_p,
        )


@dataclass
class FewShotLMDecodingSetup(
    DecodingSetup[DatumSub, HS], Generic[FullDatumSub, DatumSub, HS]
):
    """Note: we assume prompt based decoding only happening with GPT2Tokenizer.
    TODO: Relax this assumption: make this class use ClampTokenizer interface."""

    train_data: Sequence[FullDatumSub]
    train_retriever: DataRetriever[FullDatumSub, DatumSub]
    train_selectors: Sequence[DataFilter[FullDatumSub, DatumSub]]
    prompt_builder: PromptBuilder[FullDatumSub, DatumSub]

    incremental_lm: IncrementalLanguageModel[HS]
    tokenizer_quirks: GPT2TokenizerQuirks
    _eos_tokens: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        eos_bytes = self.prompt_builder.stop.encode("utf-8")
        tokens_starting_with_eos = {
            i
            for token, i in self.incremental_lm.tokenizer.utf8_token_to_id_map.items()
            if token.startswith(eos_bytes)
        }
        tokens_containing_eos = {
            i
            for token, i in self.incremental_lm.tokenizer.utf8_token_to_id_map.items()
            if eos_bytes in token
        }

        if tokens_starting_with_eos != tokens_containing_eos:
            raise ValueError(
                "Stop from PromptBuilder invalid: occurs at places other than the start of a token"
            )
        if not tokens_starting_with_eos:
            raise ValueError(
                "Stop from PromptBuilder invalid: not at the start of any token"
            )

        self._eos_tokens = torch.tensor(
            sorted(tokens_starting_with_eos), dtype=torch.long
        )
        self.tokenizer_quirks.check_prompt_builder(self.prompt_builder)

    def initial(self, datum: DatumSub) -> DatumPackedSearchNode:
        return DatumPackedSearchNode(tokens=(), test_datum=datum)

    @property
    def eos_tokens(self) -> torch.Tensor:
        return self._eos_tokens

    async def unpack_node(
        self, packed_node: DatumPackedSearchNode[DatumSub]
    ) -> Tuple[PartialParse, HS, Sequence[float]]:
        selected_train_data: Sequence[FullDatumSub] = await self.train_retriever(
            packed_node.test_datum
        )
        for train_selector in self.train_selectors:
            selected_train_data = await train_selector(
                selected_train_data, packed_node.test_datum
            )
        prompt_prefix = self.prompt_builder.assemble(
            selected_train_data, packed_node.test_datum
        )
        print(f"Prompt prefix: {prompt_prefix}")

        prompt_prefix_tokens = self.incremental_lm.tokenizer.encode(
            self.tokenizer_quirks.postprocess_prompt(prompt_prefix)
        )
        all_tokens = prompt_prefix_tokens + list(packed_node.tokens)
        logprobs, hidden_state = await self.incremental_lm.execute(all_tokens)
        assert hidden_state is not None

        # https://github.com/python/mypy/issues/708
        initial_partial_parse = self.partial_parse_builder(packed_node.test_datum)  # type: ignore
        for token in packed_node.tokens:
            initial_partial_parse = initial_partial_parse.append(token)

        # TODO: Figure out how to generalize this for when some tokens are already present
        if not packed_node.tokens:
            allowed_tokens, can_end = initial_partial_parse.allowed_next(
                torch.argsort(logprobs[-1], descending=True)
            )
            # pdb.set_trace()
            self.tokenizer_quirks.check_initial_allowed_tokens(
                set(allowed_tokens.tolist()) if allowed_tokens is not None else None,
                can_end,
            )
        return (
            initial_partial_parse,
            hidden_state,
            # TODO: This won't work for IncrementalOpenAIGPT3 if the packed_node already contains tokens
            logprobs[
                list(range(len(prompt_prefix_tokens), len(all_tokens))),
                list(packed_node.tokens),
            ].tolist(),
        )

    def finalize(self, tokens: List[int]) -> str:
        return self.tokenizer_quirks.postprocess_result(
            self.incremental_lm.tokenizer.decode(tokens)
        )


@dataclass
class GPT3ApiDecodingSetup(FewShotLMDecodingSetup,
    DecodingSetup[DatumSub, HS], Generic[FullDatumSub, DatumSub, HS]): 

    def finalize(self, tokens: List[int]) -> str:
        return self.tokenizer_quirks.postprocess_result(
            self.incremental_lm.tokenizer.decode(tokens)
        )

@dataclass
class FOLLampFewShotLMDecodingSetup(FewShotLMDecodingSetup,
    DecodingSetup[DatumSub, HS], Generic[FullDatumSub, DatumSub, HS]):

    def finalize(self, tokens: List[int], postprocess: bool = True) -> str: 
        if postprocess: 
            decoded = self.tokenizer_quirks.postprocess_result(
                self.incremental_lm.tokenizer.decode(tokens)
            )
        else:
            decoded = "".join(tokens)
        try:
            pdb.set_trace()
            formula = FOLFormula.parse_formula(decoded)
            rerendered = formula.render(ordered_vars=True)
        except (ValueError, IndexError, AssertionError, KeyError) as e:
            print(f"Unbound variable error on {decoded}")
            return decoded
        return rerendered

@dataclass
class LispLampFewShotLMDecodingSetup(FewShotLMDecodingSetup,
    DecodingSetup[DatumSub, HS], Generic[FullDatumSub, DatumSub, HS]):

    def finalize(self, tokens: List[int], postprocess: bool = True) -> str: 
        if postprocess:
            decoded = self.tokenizer_quirks.postprocess_result(
                self.incremental_lm.tokenizer.decode(tokens)
            )
        else:
            decoded = "".join(tokens)
        try:
            formula = LispFormula.parse_formula(decoded)
            rerendered = formula.render(ordered_vars=True)
        except (ValueError, IndexError, AssertionError, KeyError) as e:
            # unbound variable error 
            print(f"Unbound variable error on {decoded}")
            return decoded
        return rerendered


@dataclass
class Seq2SeqDecodingSetup(DecodingSetup[DatumSub, HS]):
    seq2seq_model: Seq2SeqModel[HS]
    _eos_tokens: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        self._eos_tokens = torch.tensor(
            [self.seq2seq_model.decoder_eos_id], dtype=torch.long
        )

    @property
    def eos_tokens(self) -> torch.Tensor:
        return self._eos_tokens

    async def unpack_node(
        self, packed_node: DatumPackedSearchNode
    ) -> Tuple[PartialParse, HS, Sequence[float]]:
        decoder_tokens = self.seq2seq_model.decoder_bos_ids + list(packed_node.tokens)
        logprobs, hidden_state = await self.seq2seq_model.initial(
            self.seq2seq_model.encode_for_encoder(packed_node.test_datum.natural),
            decoder_tokens,
        )
        assert hidden_state is not None
        # https://github.com/python/mypy/issues/708
        initial_partial_parse = self.partial_parse_builder(packed_node.test_datum)  # type: ignore
        for token in packed_node.tokens:
            initial_partial_parse = initial_partial_parse.append(token)

        return (
            initial_partial_parse,
            hidden_state,
            logprobs[
                list(
                    range(len(self.seq2seq_model.decoder_bos_ids), len(decoder_tokens))
                ),
                list(packed_node.tokens),
            ].tolist(),
        )

    def finalize(self, tokens: List[int]) -> str:
        return self.seq2seq_model.decode_output(tokens)


@dataclass
class ModelResult:
    text: str
    tokens: List[int]
    cost: float
    logprobs: List[float]



class Model(Generic[DatumSub], ABC):
    """Performs the decoding for a given datum."""

    @abstractmethod
    async def predict(self, test_datum: DatumSub) -> List[ModelResult]:
        pass


@dataclass
class BeamSearchSemanticParser(Model[DatumSub], Generic[DatumSub, FullDatumSub, HS]):
    problem_factory: ProblemFactory[DatumSub, HS]
    tokenizer: ClampTokenizer

    # Beam search-related parameters.
    # They could be moved to its own class so that we can also parametrize search methods.
    beam_size: int
    max_steps_fn: Optional[Callable[[DatumSub], Optional[int]]] = None

    async def predict(self, test_datum: DatumSub) -> List[ModelResult]:
        """Returns tuple of (hypothesis, whether hypothesis was artificially kept
        alive using force_decokde, k-best list"""
        max_steps = self.max_steps_fn(test_datum) if self.max_steps_fn else None
        # TODO (elias) add way to get token probs from beam search
        try:
            results = await beam_search(
                self.problem_factory.problem,
                self.problem_factory.initial(test_datum),
                self.beam_size,
                event_listener=LoggingEventListener(self.tokenizer, self.beam_size),
                max_steps=max_steps,
            )
        except RuntimeError:
            # NOTE (elias): adding except to avoid out-of-memory for very long inputs (very rare)
            print(f"SKIPPING LONG")
            results = []

        return [
            # TODO (elias): add token probs to model result 
            ModelResult(self.problem_factory.decoding_setup.finalize(n.tokens), 
                        n.tokens, 
                        n.cost, 
                        n.logprobs)  # type: ignore
            for n in results
        ]


@dataclass
class ApiBeamSearchSemanticParser(Model[DatumSub], Generic[DatumSub, FullDatumSub, HS]):
    decoding_setup: DecodingSetup[DatumSub, HS]
    train_retriever: DataRetriever[FullDatumSub, DatumSub]
    train_selectors: Sequence[DataFilter[FullDatumSub, DatumSub]]
    prompt_builder: PromptBuilder[FullDatumSub, DatumSub]
    engine: str

    # Beam search-related parameters.
    # They could be moved to its own class so that we can also parametrize search methods.
    beam_size: int
    max_steps_fn: Optional[Callable[[DatumSub], Optional[int]]] = None
    api_key: str = None 

    def _init_api_key(self, env: str) -> str:
        if self.api_key is None:
            self.api_key = os.getenv(env)
        if self.api_key is None:
            raise ValueError(f"{env} was not set")
        return self.api_key

    def __post_init__(self):
        api_key = self._init_api_key("OPENAI_API_KEY")
        self.completions_url = (
            f"https://api.openai.com/v1/completions"
        )
        self.chat_completions_url = (
            f"https://api.openai.com/v1/chat/completions"
        )
        if self.engine in ['gpt-3.5-turbo', 'gpt-4']: 
            self.url = self.chat_completions_url
            self.is_chat = True
            self.clean_fxn = self.clean_prediction_chat
        else:
            self.url = self.completions_url
            self.is_chat = False
            self.clean_fxn = self.clean_prediction_nonchat

        auth_header = {"Authorization": f"Bearer {api_key}"}

        self.http_client = httpx.AsyncClient(
            headers=auth_header,
            # HTTP/2 should be more efficient, but it appears to be buggy in practice
            http2=False,
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=500),
        )

    def build_request(self, 
                      prompt_prefix: str, 
                      max_steps: int) -> Dict[str, Any]:
        to_ret = {"model": self.engine,
                "max_tokens": max_steps * 3,
                "temperature": 0.0,
                "n": self.beam_size,
                # "stop": "\n\n"
                }
        if self.is_chat: 
            split_prefix = re.split("\n+", prompt_prefix)
            instr = split_prefix[0]
            human_model_turns = split_prefix[1:]
            messages = [{"role": "user", "content": instr}]
            for i, turn in enumerate(human_model_turns[:-1]):
                turn = re.sub("^Human: ", "", turn)
                turn = re.sub("^Computer: ", "", turn)

                if i % 2 == 0:
                    messages.append({"role": "user", "content": turn})
                else:
                    messages.append({"role": "assistant", "content": turn})
            to_ret['messages'] = messages

        else:
            to_ret['prompt'] = prompt_prefix
            to_ret["logprobs"] = self.beam_size,
        return to_ret 

    def clean_prediction_chat(self, 
                                response: Dict[str, Any]):
        """
        clean up predicted text for chat models 
        """
        data = response['choices']
        data = sorted(data, key = lambda x: x['index'])
        # split because finalization joins 
        texts = [[x['message']['content']] for x in data]

        tup = namedtuple('result', ['tokens', 'logprobs', 'cost'])
        to_ret = []
        for tokens in texts:
            to_ret.append(tup(tokens=tokens, logprobs=[0 for _ in range(len(tokens))], cost=0.0))

        return to_ret 

    def clean_prediction_nonchat(self, 
                                response: Dict[str, Any]):
        """
        clean up predicted text by trimming whitespace, taking first non-whitespace character sequence, 
        removing subsequent strings, and obtaining corresponding logprobs.
        Transform into something that can go into a list of ModelResults 
        """
        data = response['choices']
        to_ret = []

        for i in range(len(data)): 
            tokens = data[i]['logprobs']['tokens']
            logprobs = data[i]['logprobs']['token_logprobs']

            toks_to_ret, logprobs_to_ret = [], []
            # trim starting whitespace 
            ws = True
            for tok, lp in zip(tokens, logprobs):
                if re.match('\s+', tok) and ws:
                    continue
                else:
                    ws = False
                    toks_to_ret.append(tok)
                    logprobs_to_ret.append(lp)
            tup = namedtuple('result', ['tokens', 'logprobs', 'cost'])
            to_ret.append(tup(tokens=toks_to_ret, logprobs=logprobs_to_ret, cost=0.0))
        return to_ret

    async def predict(self, test_datum: DatumSub) -> List[ModelResult]:
        """Returns tuple of (hypothesis, whether hypothesis was artificially kept
        alive using force_decokde, kbest list"""
        # get a prompt for the API
        max_steps = self.max_steps_fn(test_datum) if self.max_steps_fn else None
        selected_train_data: Sequence[FullDatumSub] = await self.train_retriever(
            test_datum
        )
        for train_selector in self.train_selectors:
            selected_train_data = await train_selector(
                selected_train_data, test_datum
            )
        prompt_prefix = self.prompt_builder.assemble(
            selected_train_data, test_datum
        )
        # construct API request 
        request = self.build_request(prompt_prefix, max_steps) 
        # get response 
        response = await self.http_client.post(self.url, json=request)
        response = response.json()
        # clean response and return 
        results = self.clean_fxn(response)

        return [
            # TODO (elias): add token probs to model result 
            ModelResult(self.decoding_setup.finalize(n.tokens, postprocess=False), 
                        n.tokens, 
                        n.cost, 
                        n.logprobs)  # type: ignore
            for n in results
        ]
