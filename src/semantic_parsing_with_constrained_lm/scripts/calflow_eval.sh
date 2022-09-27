#! /bin/bash

# This script evaluates the model on the dev data using official calflow eval 
onmt_text_data_dir=$1
SPLIT=$2
generated_file=$3
#onmt_text_data_dir="/brtx/601-nvme1/estengel/resources/data/tree_dst.agent.data"
python -m dataflow.leaderboard.predict \
    --datum_id_jsonl ${onmt_text_data_dir}/${SPLIT}.datum_id \
    --src_txt ${onmt_text_data_dir}/${SPLIT}.src_tok \
    --ref_txt ${onmt_text_data_dir}/${SPLIT}.tgt \
    --nbest_txt ${generated_file} \
    --nbest 1

prediction_file=${CHECKPOINT_DIR}/outputs/${SPLIT}_pred.jsonl
gold_file="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/valid.answers.jsonl" 
mv predictions.jsonl ${prediction_file}

python -m dataflow.leaderboard.evaluate  \
    --predictions_jsonl ${prediction_file} \
    --gold_jsonl ${gold_file} \
    --scores_json ${CHECKPOINT_DIR}/outputs/${SPLIT}_scores.json
