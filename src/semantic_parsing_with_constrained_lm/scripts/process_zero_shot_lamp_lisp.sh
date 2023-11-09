#!/bin/bash

for split in scope_lisp; do
    in_dir="/brtx/602-nvme1/estengel/ambiguous_parsing/data/raw/generalization/${split}"
    out_dir="/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/${split}"
    python scripts/process_lamp.py --in_dir $in_dir --out_dir $out_dir
done
