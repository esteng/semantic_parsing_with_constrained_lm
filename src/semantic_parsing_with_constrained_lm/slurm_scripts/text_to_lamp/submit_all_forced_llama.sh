#!/bin/bash

for split in 0-100 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10 100-0; do # range across all splits 
    for dtype in fol; do # for now, just do FOL 
        export SPLIT="${split}-5k-train-100-perc-ambig_fol_fewshot"
        sbatch slurm_scripts/text_to_lamp/fol/llama_get_logits.sh --export 
        sbatch slurm_scripts/text_to_lamp/fol/vicuna_get_logits.sh --export 
    done
done
