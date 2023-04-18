#!/bin/bash

for t in scope revscope bound; do
    for model in small medium large huge; do
        sbatch slurm_scripts/text_to_lamp_gen/fol/zero_shot/${t}/decode_codegen_${model}.sh
    done
done
