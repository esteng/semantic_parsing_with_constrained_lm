#!/bin/bash

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern 'codegen-350M_calflow_no_context_low_0_2_dev_eval_constrained_bs_5'
