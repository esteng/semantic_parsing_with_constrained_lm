#!/bin/bash
#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_llama_13B_fol.out
#SBATCH -p ba100
#SBATCH --gpus=4


python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern "llama-13B_lamp_no_context_all_${SPLIT}-5k-train-100-perc-ambig_fol_fewshot_2_test_eval_constrained_bs_5_np_10"
