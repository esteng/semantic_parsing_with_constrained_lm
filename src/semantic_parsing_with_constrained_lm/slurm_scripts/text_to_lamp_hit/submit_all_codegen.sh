#!/bin/bash

for t in scope pp revscope bound; do
    for model in small medium large huge; do
        sbatch slurm_scripts/text_to_lamp_hit/fol/zero_shot/${t}/decode_codegen_${model}.sh
    done
done
