# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Config to run evaluation experiments with BenchCLAMP with GPT-3 based language models with a few shot prompted
approach.
"""
import pdb 
import copy
import functools
import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.configs.lib.benchclamp import create_partial_parse_builder
from semantic_parsing_with_constrained_lm.configs.lib.common import (
    BM25Indexer,
    PromptOrder,
    make_semantic_parser,
)
from semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.decoding.partial_parse import (
    PartialParse,
    StartsWithSpacePartialParse,
)
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    BENCHCLAMP_DATA_CONFIGS,
    BenchClampDataset,
    BenchClampDatasetConfig,
    ClampDataConfig,
)
from semantic_parsing_with_constrained_lm.domains.lispress_v2.lispress_exp import TopKLispressMatch
from semantic_parsing_with_constrained_lm.domains.overnight import OutputType, OvernightPieces
from semantic_parsing_with_constrained_lm.eval import Metric, TopKExactMatch
from semantic_parsing_with_constrained_lm.fit_max_steps import compute_and_print_fit
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3
from semantic_parsing_with_constrained_lm.paths import OVERNIGHT_DATA_DIR_AZURE, BENCH_CLAMP_PROCESSED_DATA_DIR
from semantic_parsing_with_constrained_lm.run_exp import Experiment
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import TrainExperiment
from semantic_parsing_with_constrained_lm.configs.benchclamp_config import extend_data_configs
from semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config import get_zero_one_ratio

LOG_DIR = Path("/brtx/602-nvme1//estengel/ambiguous_parsing/logs")
VERSION = "1.0"

BEAM_SIZE = 5
# SEARCH_MAX_STEPS = 500
SEARCH_MAX_STEPS = 50


def create_eval_exp(
    open_ai_model_name: str,
    data_config: ClampDataConfig,
    problem_type: Literal["constrained", "unconstrained-beam", "unconstrained-greedy", "unconstrained-api"],
    is_dev: bool,
    prompt_order: PromptOrder,
    num_prompts: Any,
) -> Experiment:
    train_data, dev_data, test_data = data_config.setup_data()
    lm = IncrementalOpenAIGPT3(engine=open_ai_model_name)

    use_api = False 
    if problem_type == "constrained":
        constrained = True
        beam_size = BEAM_SIZE
    elif problem_type == "unconstrained-beam":
        constrained = False
        beam_size = BEAM_SIZE
    elif problem_type == "unconstrained-greedy":
        constrained = False
        beam_size = 1
    elif problem_type == "unconstrained-api": 
        constrained = False
        beam_size = BEAM_SIZE
        use_api = True
    else:
        raise ValueError(f"{problem_type} not allowed")

    eval_data = dev_data if is_dev else test_data

    # truncate 
    eval_data = eval_data[0:10]
    test_data = test_data[0:10]

    if isinstance(data_config, BenchClampDatasetConfig):
        if data_config.dataset_name == BenchClampDataset.Overnight.value:
            # Overnight has a different grammar strategy so handling separately
            pieces = OvernightPieces.from_dir(
                lm.tokenizer,
                OVERNIGHT_DATA_DIR_AZURE,
                data_config.domain,  # type: ignore
                is_dev=is_dev,
                k=BEAM_SIZE,
                output_type=OutputType.MeaningRepresentation,
                simplify_logical_forms=True,
                # TODO: Set prefix_with_space properly by inspecting `lm`
                prefix_with_space=True,
            )
            max_steps = (
                max(
                    len(lm.tokenizer.tokenize(" " + canon))
                    for canon in pieces.denotation_metric.canonical_to_denotation
                )
                + 3  # +3 to be safe
            )

            partial_parse_builder: Callable[[Datum], PartialParse]
            if constrained:
                partial_parse_builder = pieces.partial_parse_builder  # type: ignore
            else:
                partial_parse = StartsWithSpacePartialParse(lm.tokenizer)
                partial_parse_builder = lambda _: partial_parse

            parser = make_semantic_parser(
                train_data,
                lm,  # type: ignore
                True,
                max_steps,
                beam_size,
                partial_parse_builder,
                lambda _datum: max_steps,
                similarity_method=BM25Indexer(),
                prompt_order=prompt_order,
            )

            return Experiment(
                model=parser,
                client=lm,
                metrics={
                    "exact_match": TopKExactMatch(beam_size),
                    "denotation": pieces.denotation_metric,
                },
                test_data=test_data,
                log_dir=LOG_DIR / VERSION,
            )

        else:
            # Everything other than Overnight in BenchClamp
            train_length_pairs = []
            for datum in train_data:
                num_input_tokens = len(lm.tokenizer.tokenize(datum.natural))
                num_output_tokens = len(lm.tokenizer.tokenize(datum.canonical)) + 1
                train_length_pairs.append((num_input_tokens, num_output_tokens))

            print("Computing max steps regression model parameters ...")
            max_steps_intercept, max_steps_slope = compute_and_print_fit(
                train_length_pairs, 10, 1
            )
            print("Done")

            partial_parse_builder = create_partial_parse_builder(
                constrained, data_config, lm.tokenizer
            )
            max_steps_fn = lambda _datum: min(
                int(
                    len(lm.tokenizer.tokenize(_datum.natural)) * max_steps_slope
                    + max_steps_intercept
                ),
                1000,
            )

            is_fol = "_fol" in data_config.data_id
            exp_type = "regular" if data_config.dataset_name[0].isdigit() else "generalize"
            if exp_type == "generalize":
                baseline_type = num_prompts
                num_prompts = 3
            else:
                baseline_type = None
                num_prompts = int(num_prompts)

            zero_one_ratio = get_zero_one_ratio(data_config.dataset_name)

            parser = make_semantic_parser(
                train_data=train_data,  # type: ignore
                lm=lm,  # type: ignore
                use_gpt3=True,
                use_api=use_api,
                data_id=data_config.data_id,
                global_max_steps=SEARCH_MAX_STEPS,
                beam_size=beam_size,
                partial_parse_builder=partial_parse_builder,
                max_steps_fn=max_steps_fn,
                similarity_method=BM25Indexer(),
                prompt_order=prompt_order,
                is_fol=is_fol,
                exp_type=exp_type,
                zero_one_ratio=zero_one_ratio,
                num_examples_per_prompt=num_prompts, # NOTE (elias): increase to 6 once debugging is done,
                baseline_type=baseline_type,
            )
            metrics: Dict[str, Metric[Sequence[str], FullDatum]] = {
                "exact_match": TopKExactMatch(beam_size)
            }
            if data_config.dataset_name in [
                BenchClampDataset.CalFlowV2.value,
                BenchClampDataset.TreeDST.value,
            ]:
                metrics["lispress_match"] = TopKLispressMatch(beam_size)

            return Experiment(
                model=parser,
                metrics=metrics,
                test_data=eval_data,
                client=lm,
                log_dir=LOG_DIR / VERSION,
            )

    raise ValueError("Could not create eval experiment with inputs")


def create_exps_dict() -> Tuple[
    Dict[str, Callable[[], TrainExperiment]], Dict[str, Callable[[], Experiment]]
]:
    data_configs = copy.deepcopy(BENCHCLAMP_DATA_CONFIGS)
    data_configs = extend_data_configs(data_configs, BENCH_CLAMP_PROCESSED_DATA_DIR)

    train_exps_dict: Dict[str, Callable[[], TrainExperiment]] = {}
    eval_exps_dict: Dict[str, Callable[[], Experiment]] = {}
    for (
        data_config,
        open_ai_model,
        is_dev,
        constrained,
        prompt_order,
        num_prompts,
    ) in itertools.product(
        data_configs,
        ("gpt-3.5-turbo", "text-davinci-001", "code-davinci-001", "text-ada-001", "gpt-4"),
        (True, False),
        ("constrained", "unconstrained-beam", "unconstrained-greedy", "unconstrained-api"),
        PromptOrder,
        (6, 10, "baseline_instrument", "baseline_possessive", "full") 
    ):
        dev_or_test = "dev" if is_dev else "test"
        eval_exp_name = (
            f"{open_ai_model}_{data_config.data_id}_{prompt_order.value}_{dev_or_test}_"
            f"eval_{constrained}_bs_{BEAM_SIZE}"
            f"_np_{num_prompts}"
        )
        eval_exps_dict[eval_exp_name] = functools.partial(
            create_eval_exp,
            open_ai_model,
            data_config,
            constrained,  # type: ignore
            is_dev=is_dev,
            prompt_order=prompt_order,
            num_prompts=num_prompts,
        )

    return train_exps_dict, eval_exps_dict


def build_config(
    log_dir,  # pylint: disable=unused-argument
    **kwargs: Any,  # pylint: disable=unused-argument
) -> Dict[str, Callable[[], Union[TrainExperiment, Experiment]]]:
    sys.setrecursionlimit(50000)
    expts: Dict[str, Callable[[], Union[TrainExperiment, Experiment]]] = {}
    train_expts, eval_expts = create_exps_dict()
    expts.update(train_expts)
    expts.update(eval_expts)
    return expts
