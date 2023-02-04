#!/bin/bash 

#SBATCH -o //brtx/601-nvme1/estengel/calflow_calibration/benchclamp/logs/spider_get_logits.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# CHECKPOINT_DIR='/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000'
# VALIDATION_FILE /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_medium.jsonl \

CHECKPOINT_DIR="/brtx/604-nvme2/estengel/calflow_calibration/benchclamp/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_0.0001/checkpoint-10000/"
VALIDATION_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_tiny.jsonl" 

mkdir -p ${CHECKPOINT_DIR}/outputs

python text_to_lispress.py \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${CHECKPOINT_DIR}/outputs \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --get_logits 


