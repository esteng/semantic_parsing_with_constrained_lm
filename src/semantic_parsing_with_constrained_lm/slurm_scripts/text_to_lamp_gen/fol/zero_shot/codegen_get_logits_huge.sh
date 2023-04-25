#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/codegen_logits_16B_fol.out
#SBATCH -p ba100
#SBATCH --gpus=4

MODEL="codegen-16B"
TRAIN_FILE="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${SPLIT}/train.jsonl" 
VALIDATION_FILE="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${SPLIT}/test_eval.jsonl" 
OUTPUT_DIR="/brtx/602-nvme1/estengel/ambiguous_parsing/model_outputs/${MODEL}/${SPLIT}/"
BEST_CHECKPOINT="/brtx/601-nvme1/estengel/.cache/${MODEL}"

mkdir -p ${OUTPUT_DIR}/outputs 

python text_to_lispress_autoreg.py \
    --model_name_or_path ${BEST_CHECKPOINT} \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${OUTPUT_DIR}/outputs \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --get_logits 


