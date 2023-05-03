#!/bin/bash

for atype in pp bound conj scope revscope; do
    for split in dev test; do
        mkdir -p /brtx/602-nvme1/estengel/ambiguous_parsing/data/with_prompts/${atype}_fol/
        python scripts/get_prompts.py \
        --train_file /brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${atype}_fol/train.jsonl \
        --validation_file /brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${atype}_fol/${split}_eval.jsonl \
        --out_file /brtx/602-nvme1/estengel/ambiguous_parsing/data/with_prompts/${atype}_fol/${split}_eval.jsonl \
        --zero_shot 
    done
done
