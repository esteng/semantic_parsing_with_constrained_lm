# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pdb 
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.fewshot import (
    DataRetriever,
    GPT2TokenizerQuirks,
    PromptBuilder,
    ShuffleAndSample,
    TopKSimilar,
    TruncateTokenLength,
)
from semantic_parsing_with_constrained_lm.index.bm25_index import BM25Retriever, LampBM25Retriever
from semantic_parsing_with_constrained_lm.index.exact_match_index import LampGenerator, LampExactMatchRetriever
from semantic_parsing_with_constrained_lm.index.lamp_index import (
    LampGeneralizationPPRetriever, 
    LampGeneralizationScopeRetriever, 
    LampGeneralizationRevscopeRetriever,
    LampGeneralizationBoundRetriever,
    LampGeneralizationConjRetriever
)
from semantic_parsing_with_constrained_lm.lm import (
    HS,
    AutoregressiveModel,
    IncrementalLanguageModel,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer, LlamaClampTokenizer
from semantic_parsing_with_constrained_lm.model import (
    BeamSearchSemanticParser,
    ApiBeamSearchSemanticParser,
    ConstrainedDecodingProblemFactory,
    DecodingSetup,
    FewShotLMDecodingSetup,
    FOLLampFewShotLMDecodingSetup,
    LispLampFewShotLMDecodingSetup,
    IncrementalLMSimilarityFunction,
    ProblemFactory,
    Seq2SeqDecodingSetup,
)


class PromptOrder(Enum):
    # Shuffle the training examples inside the prompt.
    Shuffle = 0
    # Put the best (most similar to test) training example earliest in the prompt.
    BestFirst = 1
    # Put the best training example at the end of the prompt.
    BestLast = 2


class SimilarityMethod:
    pass


class DefaultLM(SimilarityMethod):
    pass


@dataclass
class SeparateLM(SimilarityMethod):
    similarity_lm: IncrementalLanguageModel


@dataclass
class BM25Indexer(SimilarityMethod):
    pass


ALL_RETRIEVERS = {"pp": LampGeneralizationPPRetriever,
                 "scope": LampGeneralizationScopeRetriever,
                "revscope": LampGeneralizationRevscopeRetriever,
                "bound": LampGeneralizationBoundRetriever,
                "conj": LampGeneralizationConjRetriever
}

def make_semantic_parser(
    train_data: Sequence[FullDatumSub],
    lm: AutoregressiveModel[HS],
    use_gpt3: bool,
    use_api: bool, 
    data_id: str,
    global_max_steps: int,
    beam_size: int,
    partial_parse_builder: Callable[[DatumSub], PartialParse],
    max_steps_fn: Optional[Callable[[DatumSub], Optional[int]]],
    prompt_order: PromptOrder = PromptOrder.Shuffle,
    # Settings for using autoregressive models in a few-shot in-context setting
    prompt_builder: Optional[PromptBuilder] = None,
    num_examples_per_prompt: int = 20,
    problem_factory_builder: Optional[
        Callable[[DecodingSetup[DatumSub, HS]], ProblemFactory[DatumSub, HS]]
    ] = None,
    similarity_method: SimilarityMethod = DefaultLM(),
    is_fol: bool = True,
    exp_type: str = "generalize",
    zero_one_ratio: float = 0.5,
    do_shuffle: bool = True,
    baseline_type: str = None,
) -> BeamSearchSemanticParser:
    decoding_setup: DecodingSetup[DatumSub, HS]
    if isinstance(lm, IncrementalLanguageModel):
        if prompt_builder is None and isinstance(lm.tokenizer, GPT2ClampTokenizer):
            prompt_builder = PromptBuilder.for_demo(
                do_include_context=False, use_preamble=True
            )
        elif prompt_builder is None and isinstance(lm.tokenizer, LlamaClampTokenizer):
            prompt_builder = PromptBuilder.for_llama(
                do_include_context=False, use_preamble=True
            )
        else:
            pass 
        similarity_lm = (
            similarity_method.similarity_lm
            if isinstance(similarity_method, SeparateLM)
            else lm
        )

        if use_gpt3:
            if exp_type == "regular":
                train_retriever = LampGenerator(
                    train_data=train_data, 
                    top_k=num_examples_per_prompt,
                    ratio=zero_one_ratio,
                    shuffle=do_shuffle,
                )
            else:
                train_retriever = LampGeneralizationPPRetriever(
                    train_data=train_data, top_k=num_examples_per_prompt, baseline_type=baseline_type
                )
            if prompt_order == PromptOrder.Shuffle:
                # train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                #     LampGenerator(train_data=train_data, top_k=num_examples_per_prompt)
                #     if isinstance(similarity_method, BM25Indexer)
                #     else TopKSimilar[FullDatumSub, DatumSub](
                #         train_data=train_data,
                #         scorer=IncrementalLMSimilarityFunction(similarity_lm),
                #         k=num_examples_per_prompt,
                #     )
                # )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                    ShuffleAndSample(
                        num_per_sample=num_examples_per_prompt,
                        random_seed=0,
                    ),
                ]
            elif prompt_order == PromptOrder.BestFirst:
                # train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                #     LampGenerator(train_data=train_data, top_k=num_examples_per_prompt)
                #     if isinstance(similarity_method, BM25Indexer)
                #     else TopKSimilar[FullDatumSub, DatumSub](
                #         train_data=train_data,
                #         scorer=IncrementalLMSimilarityFunction(similarity_lm),
                #         k=num_examples_per_prompt,
                #     )
                # )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                    ),
                ]
            elif prompt_order == PromptOrder.BestLast:
                # train_retriever: DataRetriever[FullDatumSub, DatumSub] = (
                #     LampGenerator(
                #         train_data=train_data,
                #         top_k=num_examples_per_prompt,
                #         best_first=False,
                #     )
                #     if isinstance(similarity_method, BM25Indexer)
                #     else TopKSimilar[FullDatumSub, DatumSub](
                #         train_data=train_data,
                #         scorer=IncrementalLMSimilarityFunction(similarity_lm),
                #         k=num_examples_per_prompt,
                #         best_first=False,
                #     )
                # )
                train_selectors = [
                    TruncateTokenLength(
                        tokenizer=lm.tokenizer,
                        completion_length=global_max_steps,
                        prompt_builder=prompt_builder,
                        reverse=True,
                    ),
                ]

            if exp_type == "regular":
                train_retriever = LampGenerator(
                    train_data=train_data, 
                    top_k=num_examples_per_prompt,
                    ratio=zero_one_ratio,
                    shuffle=do_shuffle,
                )
            else:
                amb_type = data_id.split("_")[-2]
                train_retriever = ALL_RETRIEVERS[amb_type](
                    train_data=train_data, top_k=num_examples_per_prompt, baseline_type=baseline_type
                )
        else:
            if exp_type == "regular":
                train_retriever = LampGenerator(
                    train_data=train_data, 
                    top_k=num_examples_per_prompt,
                    ratio=zero_one_ratio,
                    shuffle=do_shuffle,
                )
            else:
                amb_type = data_id.split("_")[-2]
                train_retriever = ALL_RETRIEVERS[amb_type](
                    train_data=train_data, top_k=num_examples_per_prompt, baseline_type=baseline_type
                )
            train_selectors = []

        # if use_api: 
        #     pass 
        # else:
        if is_fol:
            decoding_setup = FOLLampFewShotLMDecodingSetup[FullDatumSub, DatumSub, HS](
                # mypy complains that Callable[[FullDatumSub], PartialParse] is
                # expected here, not sure why
                partial_parse_builder=partial_parse_builder,
                train_data=train_data if use_gpt3 else [],
                train_retriever=train_retriever,
                train_selectors=train_selectors,
                prompt_builder=prompt_builder,
                incremental_lm=lm,
                tokenizer_quirks=GPT2TokenizerQuirks(lm.tokenizer),
            )
        else:
            decoding_setup = LispLampFewShotLMDecodingSetup[FullDatumSub, DatumSub, HS](
                # mypy complains that Callable[[FullDatumSub], PartialParse] is
                # expected here, not sure why
                partial_parse_builder=partial_parse_builder,
                train_data=train_data if use_gpt3 else [],
                train_retriever=train_retriever,
                train_selectors=train_selectors,
                prompt_builder=prompt_builder,
                incremental_lm=lm,
                tokenizer_quirks=GPT2TokenizerQuirks(lm.tokenizer),
            )
    elif isinstance(lm, Seq2SeqModel):
        decoding_setup = Seq2SeqDecodingSetup(
            partial_parse_builder=partial_parse_builder, seq2seq_model=lm
        )
    else:
        raise ValueError("Unsupported type for lm")

    if use_api:
        # if using unconstrained API, no need to do constrained decoding, waste of money 
        return ApiBeamSearchSemanticParser(
            decoding_setup=decoding_setup,
            train_retriever=train_retriever,
            train_selectors=train_selectors,
            prompt_builder=prompt_builder,
            engine=lm.engine,
            beam_size=beam_size,
            max_steps_fn=max_steps_fn,
        )
    else:
        problem_factory: ProblemFactory[DatumSub, HS]
        if problem_factory_builder is None:
            problem_factory = ConstrainedDecodingProblemFactory(
                autoregressive_model=lm,
                decoding_setup=decoding_setup,
                length_normalization=0.7,
                top_k=beam_size,
            )
        else:
            problem_factory = problem_factory_builder(decoding_setup)

        return BeamSearchSemanticParser(
            problem_factory=problem_factory,
            tokenizer=lm.tokenizer,
            beam_size=beam_size,
            max_steps_fn=max_steps_fn,
        )
