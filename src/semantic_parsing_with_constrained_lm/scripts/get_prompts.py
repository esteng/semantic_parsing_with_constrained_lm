from pathlib import Path
import argparse
import pdb 
import json 
import re 
import asyncio
from tqdm import tqdm
from semantic_parsing_with_constrained_lm.fewshot import PromptBuilder
from semantic_parsing_with_constrained_lm.index.exact_match_index import LampGenerator
from semantic_parsing_with_constrained_lm.index.lamp_index import (
    LampGeneralizationBoundRetriever,
    LampGeneralizationPPRetriever,
    LampGeneralizationScopeRetriever,
    LampGeneralizationRevscopeRetriever,
    LampGeneralizationBoundRetriever,
    LampGeneralizationConjRetriever
)
from semantic_parsing_with_constrained_lm.text_to_lispress_autoreg import data_from_textio, run_retriever

def make_generator(train_file, ratio):
    with open(train_file) as f1:
        train_data = data_from_textio(f1)
    generator = LampGenerator(train_data=train_data,
                              top_k = 10,
                              ratio = ratio,
                              shuffle=True)
    return generator


def make_lamp_generator(train_file, zero_shot_type):
    ALL_RETRIEVERS = {"pp": LampGeneralizationPPRetriever,
                "scope": LampGeneralizationScopeRetriever,
            "revscope": LampGeneralizationRevscopeRetriever,
            "bound": LampGeneralizationBoundRetriever,
            "conj": LampGeneralizationConjRetriever
    }
    with open(train_file) as f1:
        train_data = data_from_textio(f1)
    retriever = ALL_RETRIEVERS[zero_shot_type](
                train_data,
                top_k=10, 
                baseline_type=None)
    return retriever


def cut_prompt(prompt):
    split_prompt = re.split("\n+", prompt)
    lines = []
    for line in split_prompt:
        line_type, line_content = re.split(": ", line)
        if line_content == "":
            continue
        lines.append({"line_type": line_type, "line_content": line_content})
    return lines

async def main(args):
    is_zero_shot = args.zero_shot 

    if is_zero_shot:
        # doing zero-shot

        zero_shot_type = Path(args.train_file).parent.stem.split("_")[0]
        retriever = make_lamp_generator(args.train_file, zero_shot_type)
        selectors = []
    else:
        ratio = float(Path(args.train_file).parent.stem.split("_")[0].split("-")[0])/100
        retriever = make_generator(args.train_file, ratio=ratio)
        selectors = []

    prompt_builder = PromptBuilder.for_writing(do_include_context=False, 
                                            use_preamble=False)

    async def preprocess(examples):
        to_ret = []
        for ex in tqdm(examples): 
            prompt = await run_retriever(retriever, selectors, prompt_builder, ex) 
            prompt = cut_prompt(prompt)
            continuation = ex.canonical 
            result = {"prompt": prompt, "test_sent": prompt[-1]['line_content'], "gold_tgt": continuation}
            to_ret.append(result)
        return to_ret

    split = Path(args.validation_file).stem
    with open(args.validation_file) as f1:
        dev_dataset = data_from_textio(f1)

    print(f"Preprocessing...")
    dev_dataset = await preprocess(dev_dataset)
    with open(args.out_file, "w") as f1:
        for ex in dev_dataset:
            f1.write(json.dumps(ex) + "\n")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--zero_shot", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))