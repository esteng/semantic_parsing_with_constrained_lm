#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_codegen_2B_lisp.out
#SBATCH -p brtx6
#SBATCH --gpus=4


#python -m semantic_parsing_with_constrained_lm.run_exp \
#--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
#--exp-name-pattern 'codegen-2B_calflow_no_context_all_2_test_eval_constrained_bs_5'

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern "codegen-2B_lamp_no_context_all_${SPLIT}-5k-train-100-perc-ambig_lisp_fewshot_conj_only_2_test_eval_constrained_bs_5_np_10"
