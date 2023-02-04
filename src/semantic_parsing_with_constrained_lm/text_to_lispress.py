import os 
import re 
from pathlib import Path 
from tqdm import tqdm 
import json 
import torch 
from typing import List
from transformers import Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding, Seq2SeqTrainer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from datasets import load_dataset, load_metric 
import numpy as np 
from dataclasses import dataclass, field
import pdb 
from typing import Optional
import logging 

from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer, T5ClampTokenizer, GPT2ClampTokenizer
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, FullDatum
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import BenchClampDataset, data_from_textio
from semantic_parsing_with_constrained_lm.domains.sql.sequence_creator import CoSqlUtterance
from semantic_parsing_with_constrained_lm.domains.sql.cosql.dialogue import load_cosql_data, convert_cosql_to_datum_format
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import DbSchema, load_schemas
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    BENCH_CLAMP_RAW_DATA_DIR,
)
from semantic_parsing_with_constrained_lm.domains.sql.sql_datum import SqlDatum
logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """


    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})

    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file (a jsonlines)"
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of returned sequences for decoding"
        }
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    use_last_user: Optional[bool] = field(default=False, metadata={"help": "Use last user utterance as input"})
    use_last_agent: Optional[bool] = field(default=True, metadata={"help": "Use last agent utterance as input"})
    get_logits: Optional[bool] = field(default=False, metadata={"help": "Get logits for each token"})
    top_k: Optional[int] = field(default=5, metadata={"help": "Top k for logit storing"})

    def __post_init__(self):
        self.val_max_target_length = self.max_target_length

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast = True) 
    if "bart" in model_args.model_name_or_path:
        tokenizer = GPT2ClampTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer  = T5ClampTokenizer.from_pretrained(model_args.model_name_or_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    is_sql = "spider" in model_args.model_name_or_path or "cosql" in model_args.model_name_or_path
    is_spider = "spider" in model_args.model_name_or_path 

    def encode_for_encoder(s: str) -> List[int]:
        string_to_tokenize = s
        # if self.settings.input_surround.starts_with_space:
        # string_to_tokenize = " " + s
        token_ids = (
            # self.settings.input_surround.bos
            list(tokenizer.encode(string_to_tokenize))
            # + self.settings.input_surround.eos
        )
        return token_ids

    def encode_for_decoder(s: str) -> List[int]:
        string_to_tokenize = s
        token_ids = tokenizer.encode(string_to_tokenize)
        return token_ids

    def preprocess_sql(examples):
        datum_parser = CoSqlUtterance(use_db_val = True,
                                        past_utterances = "all")
        # list of utterances per datapoint 
        for ex in examples: 
            utts = ex['utterance']
            if len(utts) == 1:
                past_utterances = None
            else:
                past_utts = utts[:-1]
                utt = utts[-1]
            

            datum = BenchClampDatum()
        # 

    # TODO: (elias): ensure that this function remains accurate for sql 
    def preprocess(examples):
        input = [datapoint for datapoint in examples['utterance']]
        pdb.set_trace() 
        prefix_last_user = [utt for utt in examples['last_user_utterance']]
        prefix_last_agent = [utt for utt in examples['last_agent_utterance']]

        for i, inp in enumerate(input):
            to_cat = []
            to_cat.append(prefix_last_user[i])
            to_cat.append(prefix_last_agent[i])
            to_cat.append(inp)
            input[i] = ' | '.join(to_cat)
            if "bart" in model_args.model_name_or_path:
                input[i] = f"<s> {input[i]}</s>"
            else:
                input[i] = f" {input[i]}</s>"

        model_inputs = [encode_for_encoder(x) for x in input]

        if "bart" in model_args.model_name_or_path:
            output = [f"<s> {datapoint}</s>" for datapoint in examples['plan']]
        else:
            output = [f" {datapoint}</s>" for datapoint in examples['plan']]
        labels = [encode_for_decoder(x) for x in output]

        result = {
                "input_ids": model_inputs,
                "labels": labels,
                # "labels": [None for x in range(len(model_inputs))],
                "length": [len(x) for x in model_inputs],
            }
        return result


    if is_sql:
        if is_spider: 
            input_sequence_creator = CoSqlUtterance(use_db_val = True, past_utterances = "none")
        else:
            input_sequence_creator = CoSqlUtterance(use_db_val = True, past_utterances = "all")
        with open(data_args.validation_file) as f:
            data = data_from_textio(f)

        data_for_expt = []
        for datum in data:
            input_sequence = input_sequence_creator.create_sequence(datum)
            input_sequence = f" {input_sequence}</s>"
            output_sequence = datum.plan
            output_sequence = f" {output_sequence}</s>"
            print(input_sequence)
            print(len(input_sequence))
            print(output_sequence)
            print(len(output_sequence))
            pdb.set_trace()
    else:
        data_files = {"dev": data_args.validation_file}
        split = Path(data_args.validation_file).stem
        raw_datasets = load_dataset("json", data_files = data_files)

        dev_dataset = raw_datasets['dev']
        dev_dataset = dev_dataset.map(preprocess, batched=True, remove_columns=raw_datasets['dev'].column_names, load_from_cache_file=False )

    data_collator = DataCollatorForSeq2Seq(tokenizer.tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # TODO: (elias): ensure that this function works for SQL, add any extra rules     
    def postprocess_and_clean(pred):
        is_small = "small" in model_args.model_name_or_path
        pred = pred.strip()
        if is_small:
            pred = re.sub("<extra_id_99>", "~", pred)
            pred = re.sub("<extra_id_98>", "^", pred)
        else:
            pred = re.sub("<extra_id_99>", "^", pred)
            pred = re.sub("<extra_id_95>", "~", pred) 
        pred = re.sub("<pad>", "", pred)
        pred = re.sub("<s>", "", pred)
        pred = re.sub("</s>", "", pred)
        return pred 
    
    if data_args.get_logits:
        eval_dataloader = trainer.get_eval_dataloader()
        print(f"WRiting to {training_args.output_dir}")
        output_logit_file = os.path.join(training_args.output_dir, f"{split}.logits")
        trainer.model.eval()
        with open(output_logit_file, "w") as f1:
            for step, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                inputs = {k: v.to(trainer.args.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = trainer.model(**inputs)
                    logits = outputs.logits
                    logits = torch.exp(torch.log_softmax(logits, dim=-1))
                    logits = logits.detach().cpu().numpy()
                    # exp_logits = np.exp(logits)
                    # logits = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    logits_top_k_idxs = np.argsort(logits, axis=-1)[:, :, -data_args.top_k:]
                    logits_top_k = np.take_along_axis(logits, logits_top_k_idxs, axis=-1)
                    batch_size = logits.shape[0]
                    logits_top_k_idxs = logits_top_k_idxs.tolist()
                    logits_top_k = logits_top_k.tolist()

                    # get logits at label idxs 
                    # handle pad tokens
                    unsqueezed_labels = inputs['labels'].unsqueeze(-1)
                    labels_to_gather = unsqueezed_labels.clone()
                    labels_to_gather[unsqueezed_labels == -100] = 0
                    logit_at_label = outputs.logits.gather(2, labels_to_gather)
                    logit_at_label[unsqueezed_labels == -100] = -100
                    
                    logit_at_label = logit_at_label.squeeze(-1)
                    logit_at_label = logit_at_label.detach().cpu().numpy().tolist()
                    labels = inputs['labels'].detach().cpu().numpy().tolist()
                    inputs = inputs['input_ids'].detach().cpu().numpy().tolist()
                    input_str = [tokenizer.decode(x) for x in inputs] #, skip_special_tokens=True)

                for batch_idx in range(batch_size): 
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
                                "input_str": instance_input_str}

                    f1.write(json.dumps(to_append) + "\n")

                # remove from memory 
                del inputs 
                del outputs
                
if __name__ == "__main__":
    main()
