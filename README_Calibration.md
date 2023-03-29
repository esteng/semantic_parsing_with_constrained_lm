# Calibration via BenchCLAMP 

This fork of BenchCLAMP was used to run experiments for determining the calibration of semantic parsing models. 
For more information, please refer to this paper: [Calibrated Interpretation: Confidence Estimation in Semantic Parsing](https://arxiv.org/abs/2211.07443) 

### Datasets
Calibration experiments were run with the following four datasets
1. CalFlowV2
2. TreeDST (in LispressV2 format, as in BenchCLAMP) 
3. Spider
4. CoSQL

all experiments were run using the `all` training split. 

### Scripts
You can find helpful scripts for running experiments in `src/semantic_parsing_with_constrained_lm/slurm_scripts/`.
All text-to-sql scripts are in `text-to-sql`, while the CalFlow and TreeDST experiments are in `text-to-lispress`. 

### Data 
Instructions for downloading and processing data can be found in the [benchclamp README](README_BenchCLAMP.md). 
Note that by design, paths in BenchCLAMP are coded into the configs. You will need to change these paths in `paths.py`.

### Extracting token confidence scores 
Token-level confidence scores can be extracted by running `text_to_lispress.py` (for seq2seq models) and `text_to_lispress_autoreg.py` (for autoregressive models). 
The `get_logits.sh` scripts show examples of how these scripts are run. 
These scores are extracted under a forced decode of the prefix, i.e. using teacher forcing.

### Extracting test-time confidence scores
To extract test-time confidence scores (rather than teacher-forced scores) you can simply run decoding as normal using this modified branch. 
Confidence scores will be output into the jsonlines files produced by BenchCLAMP, which are written to the log dir.
The token logprobs are written to the `token_logprobs` field in the jsonl, which gives the token logprobs for the top 5 generated sequences. 
Note that the lengths of the top 5 sequences may vary.
The corresponding token predictions (not de-tokenized) are given in the `token_sequence` field, while the text predictions from the beam arge given in the `outputs` field. 
 

