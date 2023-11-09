#!/bin/bash

for t in scope revscope conj bound; do
    #for model in small medium large huge; do
    for model in large; do
        sbatch slurm_scripts/text_to_lamp_gen/fol/zero_shot/${t}/decode_vicuna_${model}.sh
    done
done
