#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_codegen_2b.out
#SBATCH -p ba100
#SBATCH --gpus=1


export BENCH_CLAMP_MACRO_TRAIN_DIR=$1
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern 'codegen-350M_calflow_no_context_low_0_2_test_eval_unconstrained-beam_bs_5_0.5'
