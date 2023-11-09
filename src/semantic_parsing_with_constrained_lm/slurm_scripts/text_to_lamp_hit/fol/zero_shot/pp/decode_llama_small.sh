#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_llama-7B_fol.out
#SBATCH -p brtx6
#SBATCH --gpus=10
#SBATCH --nodes=1


#python -m semantic_parsing_with_constrained_lm.run_exp \
#--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
#--exp-name-pattern 'codegen-350M_calflow_no_context_all_2_test_eval_constrained_bs_5'

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern 'llama-7B_lamp_no_context_all_pp_fol_0_test_eval_constrained_bs_5_np_full'
