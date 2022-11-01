import os 
import copy 
from transformers import Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, HammingDiversityLogitsProcessor, LogitsProcessorList
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

    def __post_init__(self):
        self.val_max_target_length = self.max_target_length

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast = True) 

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    def preprocess(examples):
        input = [datapoint for datapoint in examples['plan']]
        prefix_last_user = [utt for utt in examples['last_user_utterance']]
        prefix_last_agent = [utt for utt in examples['last_agent_utterance']]

        for i, inp in enumerate(input):
            to_cat = []
            if prefix_last_user[i] != '':
                to_cat.append(prefix_last_user[i])
            if prefix_last_agent[i] != '':
                to_cat.append(prefix_last_agent[i])
            to_cat.append(inp)
            input[i] = ' | '.join(to_cat)

        model_inputs = tokenizer(input, padding=True, truncation=True)

        output = [f" | {datapoint}</s>" for datapoint in examples['utterance']]
        labels = tokenizer(output, padding=True, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_files = {"dev": data_args.validation_file}
    raw_datasets = load_dataset("json", data_files = data_files)

    dev_dataset = raw_datasets['dev']
    dev_dataset = dev_dataset.map(preprocess, batched=True, remove_columns=raw_datasets['dev'].column_names, load_from_cache_file=False )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        labels_for_metric = [[label.strip()] for label in labels]

        return preds, labels, labels_for_metric

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels, decoded_labels_for_metric = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels_for_metric)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result["exact_match"] = np.mean(
            [decoded_preds[idx] == decoded_labels[idx] for idx in range(len(decoded_preds))])

        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # trainer._gen_kwargs['num_return_sequences'] = data_args.num_return_sequences

    if training_args.do_predict:
        logger.info("*** Predict ***")
        beam_size = max(5, data_args.num_return_sequences)

        # generate from dev dataset 
        flat_generations = []
        batch_iterator = trainer.get_eval_dataloader()
        for batch in batch_iterator:
            inputs = batch['input_ids'].to(trainer.args.device)
            generated = trainer.model.generate(inputs,
                                         max_length=100,
                                         num_beams=beam_size,
                                         num_return_sequences=data_args.num_return_sequences)
            flat_generations.extend(generated)

        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                    flat_generations, skip_special_tokens=False, clean_up_tokenization_spaces=True
                )
                
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            #print(predictions)
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()