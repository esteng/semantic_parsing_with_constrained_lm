
import json
import pathlib
from collections import defaultdict
import pdb 

orig_data_path = pathlib.Path("/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/")
new_data_path = pathlib.Path("/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2")
out_path = pathlib.Path("/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data.from_benchclamp")
# for split in ["dev_valid.dataflow_dialogues.jsonl", "test_valid.dataflow_dialogues.jsonl",  "train.dataflow_dialogues.jsonl", "valid.dataflow_dialogues.jsonl"]:
for new_split, old_split in [("dev_low.jsonl", "train"), ("dev_medium.jsonl", "train"), ("dev_all.jsonl", "train"), ("train_all.jsonl", "train"), ("test_all.jsonl", "valid")]: 
    print(f"Processing {new_split}")
    new_split_name = new_split.split(".")[0]
    new_split_path = new_data_path.joinpath(new_split)
    with open(new_split_path) as f:
        new_split_data = [json.loads(x) for x in f.readlines()]
        datum_id_data = [{"dialogue_id": x['dialogue_id'], "turn_index": x['turn_part_index']} for x in new_split_data]

    old_split_src = orig_data_path.joinpath(f"{old_split}.src_tok")
    old_split_tgt = orig_data_path.joinpath(f"{old_split}.tgt")
    old_split_datum_id = orig_data_path.joinpath(f"{old_split}.datum_id")
    processed_data = []
    processed_lut = defaultdict(dict)
    with open(old_split_src) as src_f, open(old_split_tgt) as tgt_f, open(old_split_datum_id) as datum_id_f:
        src_data = src_f.readlines()
        tgt_data = tgt_f.readlines()
        old_datum_id_data = [json.loads(x) for x in datum_id_f.readlines()]
        for datum_id, src, tgt in zip(old_datum_id_data, src_data, tgt_data):
            processed_lut[datum_id['dialogue_id']][datum_id['turn_index']] = {"src": src.strip(), "tgt": tgt.strip()}


    # out_path_to_write = out_path.joinpath(new_split_name + ".jsonl") 
    out_path_src = out_path.joinpath(new_split_name + ".src_tok")
    out_path_tgt = out_path.joinpath(new_split_name + ".tgt")
    out_path_datum_id = out_path.joinpath(new_split_name + ".datum_id")
    print(f"Writing to {out_path}")
    with open(out_path_src, "w") as src_f, open(out_path_tgt, "w") as tgt_f, open(out_path_datum_id, "w") as datum_id_f:
        # # for line in processed_data:
        #     # f1.write(json.dumps(line) + "\n")
        for datum_id_dict in datum_id_data:
            did = datum_id_dict['dialogue_id']
            turn_idx = datum_id_dict['turn_index']
            to_write = processed_lut[did][turn_idx]

            src_f.write(to_write['src'] + "\n") 
            tgt_f.write(to_write['tgt'] + "\n")
            datum_id_f.write(json.dumps(datum_id_dict) + "\n")

