#!/bin/bash

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_gpt3_config \
--exp-name-pattern 'text-ada-001_lamp_no_context_all_50-50-5k-train-100-perc-ambig_fol_fewshot_conj_only_2_test_eval_constrained_bs_5'
#--exp-name-pattern 'gpt-3.5-turbo_lamp_no_context_all_50-50-5k-train-100-perc-ambig_fol_fewshot_2_test_eval_constrained_bs_5'

