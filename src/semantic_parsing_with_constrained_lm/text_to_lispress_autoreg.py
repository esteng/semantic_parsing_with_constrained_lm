import os 
import re 
import asyncio
from pathlib import Path
from tqdm import tqdm 
import json 
import torch 
from typing import List, TextIO
from transformers import Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
import numpy as np 
import pdb 
from typing import Optional
import logging 
import jsons 

from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, FullDatum
from semantic_parsing_with_constrained_lm.configs.lib.common import BM25Retriever, TruncateTokenLength
from semantic_parsing_with_constrained_lm.modeling_codegen import MyCodeGenForCausalLM
from semantic_parsing_with_constrained_lm.text_to_lispress import ModelArguments, DataTrainingArguments
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

logger = logging.getLogger(__name__)

class ConvertDatum(FullDatum):
    """
    class to convert saved BenchClampDatum to FullDatum for retriever
    """
    def __init__(self,
                # dialogue_id: Optional[str],
                utterance,
                plan,
                unfilled_template,
                var_bindings,
                template_idx, 
                type,
                ): 
        super().__init__(dialogue_id="0",
                        turn_part_index="0",
                        agent_context=None,
                         natural=utterance,
                         canonical=plan,
                         unfilled_template=unfilled_template,
                         var_bindings=var_bindings,
                         template_idx=str(template_idx),
                         type=type)

def data_from_textio(data_file: TextIO) -> List[ConvertDatum]:
    return [jsons.loads(line.strip(), cls=ConvertDatum) for line in data_file]

def make_bm25_retriever(train_file, prompt_builder, tokenizer, max_len):
    # turn train file into list of BenchClampDatum
    with open(train_file) as f1:
        train_data = data_from_textio(f1)
    # create retriever
    retriever = BM25Retriever(train_data, top_k = 5, best_first = False)

    train_selectors = [
                    TruncateTokenLength(
                        tokenizer=tokenizer,
                        completion_length=max_len,
                        prompt_builder=prompt_builder,
                        reverse=True,
                    ),
                ]
    return retriever, train_selectors


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

async def run_retriever(retriever, selectors, prompt_builder, test_datum):
    retrieved = await retriever(test_datum)
    for selector in selectors:
        retrieved = await selector(retrieved, test_datum)

    prompt_prefix = prompt_builder.assemble(
        retrieved, test_datum
    )
    return prompt_prefix

async def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer  = GPT2ClampTokenizer.from_pretrained(model_args.model_name_or_path)

    model = MyCodeGenForCausalLM.from_pretrained(model_args.model_name_or_path)
    # maybe parallelize 
    if Path(model_args.model_name_or_path).stem == "codegen-350M" and torch.cuda.device_count() == 2:
        device_map = {0: list(range(16)),
                    1: list(range(16,32))}
        model.parallelize(device_map)

    if Path(model_args.model_name_or_path).stem == "codegen-6B":
        if torch.cuda.device_count() == 8:
            device_map = {0: list(range(4)),
                        1: list(range(4,8)),
                        2: list(range(8,12)),
                        3: list(range(12,16)),
                        4: list(range(16,20)),
                        5: list(range(20,24)),
                        6: list(range(24,28)),
                        7: list(range(28,33))}

        elif torch.cuda.device_count() == 3: 
            device_map = {0: list(range(11)), 1: list(range(11, 22)), 2: list(range(22, 33))}
        else:
            raise AssertionError("Only 3 and 8 GPUs supported")

        model.half()
        model.parallelize(device_map)

    elif Path(model_args.model_name_or_path).stem == "codegen-2B":
        device_map = {0: list(range(15)), 1: list(range(15, 32))}
        # device_map = {0: list(range(10)), 1: list(range(10, 20)), 2: list(range(20, 32))}
        model.half()
        model.parallelize(device_map)

    elif Path(model_args.model_name_or_path).stem == "codegen-16B":
        device_map={0: list(range(9)), 
                    1: list(range(9, 17)), 
                    2: list(range(17, 26)), 
                    3: list(range(26, 34))}
        model.half()
        model.parallelize(device_map)

    else:
        device = torch.device("cuda:0")
        model.to(device)
        device_map = None 

    prompt_builder = PromptBuilder.for_demo(do_include_context=False, 
                                            use_preamble=True)
    print(f"Building retriever...")
    is_zero_shot = False
    try:
        ratio = float(Path(data_args.train_file).parent.stem.split("_")[0].split("-")[0])/100
        retriever = make_generator(data_args.train_file, ratio=ratio)
        selectors = []
    except ValueError:
        is_zero_shot = True
        # doing zero-shot
        zero_shot_type = Path(data_args.train_file).parent.stem.split("_")[0]
        retriever = make_lamp_generator(data_args.train_file, zero_shot_type)
        selectors = []
        #  
    # retriever, selectors = make_bm25_retriever(data_args.train_file, 
    #                                 prompt_builder, 
    #                                 tokenizer, 
    #                                 max_len=data_args.max_source_length)

    def encode_for_encoder(s: str) -> List[int]:
        token_ids = (
            list(tokenizer.encode(s))
        )
        return token_ids

    def encode_for_decoder(s: str) -> List[int]:
        token_ids = tokenizer.encode(s) 
        return token_ids

    async def preprocess(examples):
        to_ret = []
        for ex in tqdm(examples): 
            prompt = await run_retriever(retriever, selectors, prompt_builder, ex) 
            continuation = ex.canonical 
            input_prompt = f"{tokenizer.tokenizer.bos_token}{prompt}{continuation}" 
            input_only = encode_for_encoder(f"{tokenizer.tokenizer.bos_token}{prompt}")
            inp = encode_for_encoder(input_prompt) 

            label_prompt = f"{prompt}{ex.canonical}{tokenizer.tokenizer.eos_token}"
            labels = encode_for_decoder(label_prompt) 

            result = {
                    "input_ids": torch.tensor(inp).unsqueeze(0),
                    "labels": torch.tensor(labels).unsqueeze(0),
                    "inp_length": torch.tensor(len(input_only)),
                    "natural": ex.natural,
                    "template_idx": ex.template_idx,
                }
            to_ret.append(result)
        return to_ret

    _except_keys = ['inp_length', 'natural', 'template_idx']
        # data_files = {"dev": data_args.validation_file}
    # split = Path(data_args.validation_file).stem
    # raw_datasets = load_dataset("json", data_files = data_files)
    # dev_dataset = raw_datasets['dev']
    # pdb.set_trace()
    # dev_dataset = dev_dataset.map(preprocess, batched=True, remove_columns=raw_datasets['dev'].column_names, load_from_cache_file=False )
    split = Path(data_args.validation_file).stem
    with open(data_args.validation_file) as f1:
        dev_dataset = data_from_textio(f1)

    print(f"Preprocessing...")
    dev_dataset = await preprocess(dev_dataset)

    data_collator = DataCollatorForSeq2Seq(tokenizer.tokenizer, model=model)
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=None,
    #     eval_dataset=dev_dataset,
    #     tokenizer=tokenizer.tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=None,
    # )
    
    if data_args.get_logits:
        # eval_dataloader = trainer.get_eval_dataloader()
        print(f"WRiting to {training_args.output_dir}")
        output_logit_file = os.path.join(training_args.output_dir, f"{split}.logits")
        model.eval()
        with open(output_logit_file, "w") as f1:
            # for step, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            for step, inputs in tqdm(enumerate(dev_dataset), total=len(dev_dataset)): 
                cuda_inputs = {k: v.to("cuda:0") for k, v in inputs.items() if k not in _except_keys}
                with torch.no_grad():
                    inputs_to_run = {k:v for k, v in cuda_inputs.items()}
                    outputs = model(**inputs_to_run)
                    logits = outputs.logits
                    input_len = inputs['inp_length'].item()
                    label_logits = logits[:, input_len:,:]
                    torch_logits = torch.exp(torch.log_softmax(label_logits, dim=-1))
                    logits = torch_logits.detach().cpu().numpy()
                    # exp_logits = np.exp(logits)
                    # logits = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    # pdb.set_trace()
                    logits_top_k_idxs = np.argsort(logits, axis=-1)[:, :, -data_args.top_k:]
                    logits_top_k = np.take_along_axis(logits, logits_top_k_idxs, axis=-1)
                    batch_size = logits.shape[0]
                    logits_top_k_idxs = logits_top_k_idxs.tolist()
                    logits_top_k = logits_top_k.tolist()

                    # get logits at label idxs 
                    # handle pad tokens
                    labels = cuda_inputs['labels'][:, input_len:]
                    unsqueezed_labels = labels.unsqueeze(-1)

                    labels_to_gather = unsqueezed_labels.clone()
                    labels_to_gather[unsqueezed_labels == -100] = 0
                    logit_at_label = torch_logits.gather(2, labels_to_gather)
                    logit_at_label[unsqueezed_labels == -100] = -100
                    
                    logit_at_label = logit_at_label.squeeze(-1)
                    logit_at_label = logit_at_label.detach().cpu().numpy().tolist()
                    labels = labels.detach().cpu().numpy().tolist()
                    input_ids = inputs['input_ids'] #.detach().cpu().numpy().tolist()
                    input_str = [tokenizer.decode(x) for x in input_ids] #, skip_special_tokens=True)

                # for batch_idx in range(batch_size): 
                    assert( batch_size == 1) 
                    batch_idx = 0
                    # trim off padding
                    instance_logit_at_label = logit_at_label[batch_idx]
                    instance_logit_at_label = [x for x in instance_logit_at_label if x != -100]
                    instance_labels = labels[batch_idx]
                    instance_labels = [x for x in instance_labels if x != -100]
                    instance_input_str = input_str[batch_idx]
                    instance_input_str = re.sub("<pad>", "", instance_input_str)
                    instance_top_logits = logits_top_k[batch_idx][0:len(instance_labels)]
                    instance_top_logit_idxs = logits_top_k_idxs[batch_idx][0:len(instance_labels)]
                    to_append = {"top_logits": instance_top_logits,
                                "top_logit_idxs": instance_top_logit_idxs,
                                "logit_at_label": instance_logit_at_label,
                                "labels": instance_labels,
                                "input_str": instance_input_str,
                                "natural": inputs['natural'],
                                "template_idx": inputs['template_idx']}

                    f1.write(json.dumps(to_append) + "\n")

                # remove from memory 
                del inputs 
                del outputs
                
if __name__ == "__main__":
    asyncio.run(main()) 
