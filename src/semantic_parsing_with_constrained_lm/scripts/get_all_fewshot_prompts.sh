#!/bin/bash 

for split in 0-100 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10 100-0; do # range across all splits 
    for dtype in fol lisp; do # for now, just do FOL 
        for dsplit in dev test; do
            SPLIT="${split}-5k-train-100-perc-ambig_fol_fewshot"
            TRAIN_FILE="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${SPLIT}/train_eval.jsonl" 
            VALIDATION_FILE="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${SPLIT}/${dsplit}_eval.jsonl" 
            OUT_DIR="/brtx/602-nvme1/estengel/ambiguous_parsing/data/with_prompts/${split}_${dytpe}/"
            mkdir -p ${OUT_DIR}
            OUT_FILE="${OUT_DIR}/${dsplit}_eval.jsonl"

            python scripts/get_prompts.py \
            --train_file ${TRAIN_FILE} \
            --validation_file ${VALIDATION_FILE} \
            --out_file ${OUT_FILE}
        done
    done
done