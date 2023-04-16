#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/lamp_train_t5_small.out
#SBATCH -p brtx6
#SBATCH --gpus=1

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-small-lm-adapt_lamp_no_context_all_50-50-5k-train-10-perc-ambig_0.0001'

