import json
import pathlib

parent_path = pathlib.Path("/brtx/602-nvme1/estengel/ambiguous_parsing/data/processed/")
child_dirs_fol = parent_path.glob("*5k-train-100-perc-ambig_fol_fewshot")
child_dirs_lisp = parent_path.glob("*5k-train-100-perc-ambig_lisp_fewshot")

for child_dirs in [child_dirs_fol, child_dirs_lisp]:
    conj_start_idx = 100
    conj_end_idx = 200
    for child_dir in child_dirs:
        
        dev_file = child_dir.joinpath("dev.jsonl")
        test_file = child_dir.joinpath("test.jsonl")
        dev_data = [line for line in open(dev_file, "r")]
        test_data = [line for line in open(test_file, "r")]
        dev_conj_data = dev_data[conj_start_idx:conj_end_idx]
        test_conj_data = test_data[conj_start_idx:conj_end_idx]

        dir_name = f"{child_dir.name}_conj_only"
        new_dir = parent_path.joinpath(dir_name)
        new_dir.mkdir(exist_ok=True)
        with open(new_dir/"dev.jsonl", "w") as f:
            for line in dev_conj_data:
                f.write(line)
        with open(new_dir/"test.jsonl", "w") as f:
            for line in test_conj_data:
                f.write(line)
        # copy train.jsonl, train_eval.jsonl, test_Eval.jsonl, dev_eval.jsonl
        for file_name in ["train.jsonl", "train_eval.jsonl", "test_eval.jsonl", "dev_eval.jsonl"]:
            with open(child_dir/file_name, "r") as f:
                data = [line for line in f]
            with open(new_dir/file_name, "w") as f:
                for line in data:
                    f.write(line)


