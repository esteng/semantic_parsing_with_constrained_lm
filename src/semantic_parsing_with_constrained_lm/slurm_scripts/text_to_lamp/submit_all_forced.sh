#!/bin/bash

#for split in 0-100 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10 100-0; do # range across all splits 
for split in 80-20 ; do # range across all splits 
    for dtype in fol; do # for now, just do FOL 
        #for model in small medium large huge; do 
        for model in medium; do 
            export SPLIT="${split}-5k-train-100-perc-ambig_fol_fewshot"
            sbatch slurm_scripts/text_to_lamp/fol/codegen_get_logits_${model}.sh --export 
        done
    done
done
