#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/spider_decode_t5_small.out
#SBATCH -p brtx6
#SBATCH --gpus=1

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-small-lm-adapt_spider_past.*tiny.*0.0001.*dev_eval.*'

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-small-lm-adapt_spider_past.*tiny.*0.0001.*test_eval.*'
