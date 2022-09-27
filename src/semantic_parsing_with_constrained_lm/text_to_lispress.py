import os 
import re 
from pathlib import Path 
from tqdm import tqdm 
import json 
import torch 
from transformers import Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from datasets import load_dataset, load_metric 
import numpy as np 
from dataclasses import dataclass, field
import pdb 
from typing import Optional
import logging 

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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast = True) 

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    def preprocess(examples):
        input = [datapoint for datapoint in examples['utterance']]
        prefix_last_user = [utt for utt in examples['last_user_utterance']]
        prefix_last_agent = [utt for utt in examples['last_agent_utterance']]

        for i, inp in enumerate(input):
            to_cat = []
            # if prefix_last_user[i] != '' and data_args.use_last_user:
            to_cat.append(prefix_last_user[i])
            # if prefix_last_agent[i] != '' and data_args.use_last_agent:
            to_cat.append(prefix_last_agent[i])
            to_cat.append(inp)
            input[i] = ' | '.join(to_cat)

        model_inputs = tokenizer(input, padding=True, truncation=True)

        output = [f" {datapoint}</s>" for datapoint in examples['plan']]
        labels = tokenizer(output, padding=True, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_files = {"dev": data_args.validation_file}
    split = Path(data_args.validation_file).stem
    raw_datasets = load_dataset("json", data_files = data_files)

    dev_dataset = raw_datasets['dev']
    dev_dataset = dev_dataset.map(preprocess, batched=True, remove_columns=raw_datasets['dev'].column_names, load_from_cache_file=False )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    predict_results = trainer.predict(
        dev_dataset, metric_key_prefix="predict", max_length=100, num_beams=5
    )
    if training_args.predict_with_generate:
        predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
        predictions = [pred.strip() for pred in predictions]
        predictions = [re.sub("<.*?>", "", pred) for pred in predictions]
        output_prediction_file = os.path.join(training_args.output_dir, f"{split}.tgt")
        #print(predictions)

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

    
    if data_args.get_logits:
        eval_dataloader = trainer.get_eval_dataloader()
        print(f"WRiting to {training_args.output_dir}")
        output_logit_file = os.path.join(training_args.output_dir, f"{split}.logits")
        with open(output_logit_file, "w") as f1:
            for step, inputs in tqdm(enumerate(eval_dataloader)):
                inputs = {k: v.to(trainer.args.device) for k, v in inputs.items()}
                outputs = trainer.model(**inputs)
                logits = outputs.logits
                logits = torch.exp(torch.log_softmax(logits, dim=-1))
                logits = logits.detach().cpu().numpy()
                # exp_logits = np.exp(logits)
                # logits = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                logits_top_k_idxs = np.argsort(logits, axis=-1)[:, :, -data_args.top_k:]
                logits_top_k = np.take_along_axis(logits, logits_top_k_idxs, axis=-1)
                logits_top_k_idxs = logits_top_k_idxs.tolist()
                logits_top_k = logits_top_k.tolist()

                # get logits at label idxs 
                logit_at_label = outputs.logits.gather(2, inputs['labels'].unsqueeze(-1))
                logit_at_label = logit_at_label.reshape(-1)
                logit_at_label = logit_at_label.detach().cpu().numpy().tolist()
                labels = inputs['labels'].detach().cpu().numpy().tolist()
                input_str = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
                to_append = {"top_logits": logits_top_k, "top_logit_idxs": logits_top_k_idxs, "logit_at_label": logit_at_label, "labels": labels, "input_str": input_str}
                f1.write(json.dumps(to_append) + "\n")

if __name__ == "__main__":
    main()