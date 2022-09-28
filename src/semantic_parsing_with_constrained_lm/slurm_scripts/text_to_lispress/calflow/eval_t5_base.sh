#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_t5_base.out
#SBATCH -p brtx6
#SBATCH --gpus=4


python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-base-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5'
    
