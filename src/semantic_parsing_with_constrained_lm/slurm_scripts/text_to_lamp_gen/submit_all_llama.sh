#!/bin/bash

#for t in scope pp revscope conj bound; do
for t in scope; do
    #for model in small medium large huge; do
    for model in medium; do
        sbatch slurm_scripts/text_to_lamp_gen/fol/zero_shot/${t}/decode_codegen_${model}.sh
    done
done
