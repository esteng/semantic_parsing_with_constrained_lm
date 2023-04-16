#!/bin/bash

#for split in 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10; do
#    for dtype in lisp fol; do
#        for model in small medium large huge; do
#            export SPLIT=${split}
#            sbatch slurm_scripts/text_to_lamp/${dtype}/decode_codegen_${model}.sh
#        done
#    done
#done

for split in 50-50; do # do 50-50 across all models and data 
    for dtype in lisp fol; do
        for model in small medium large huge; do
            export SPLIT=${split}
            sbatch slurm_scripts/text_to_lamp/${dtype}/decode_codegen_${model}.sh
        done
    done
done

for split in 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10; do # range across all splits 
    for dtype in fol; do # for now, just do FOL 
        for model in medium; do # for now, just do medium 
            export SPLIT=${split}
            sbatch slurm_scripts/text_to_lamp/${dtype}/decode_codegen_${model}.sh
        done
    done
done
