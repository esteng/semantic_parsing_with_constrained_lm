#!/bin/bash

for t in pp scope revscope bound; do
    for model in small medium large huge; do
        export SPLIT="${t}_fol_hit"
        sbatch slurm_scripts/text_to_lamp_hit/fol/zero_shot/codegen_get_logits_${model}.sh
    done
done
