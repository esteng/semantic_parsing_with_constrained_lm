#!/bin/bash

for split in 0-100 10-90 20-80 30-70 40-60 50-50 60-40 70-30 80-20 90-10 100-0; do
    for dtype in lisp fol; do
        in_dir="/brtx/602-nvme1/estengel/ambiguous_parsing/data/raw/${split}-5k-train-100-perc-ambig_${dtype}"
        out_dir="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${split}-5k-train-100-perc-ambig_${dtype}_fewshot"
        python scripts/process_lamp.py --in_dir $in_dir --out_dir $out_dir
    done
done
