#!/bin/bash

#for t in pp scope revscope conj bound; do
for t in conj; do
    #  for model in small medium large huge; do
    for model in medium large huge; do
    #for model in small; do
        export SPLIT="${t}_fol"
        sbatch slurm_scripts/text_to_lamp_gen/fol/zero_shot/codegen_get_logits_${model}.sh
    done
done
