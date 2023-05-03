import pathlib 
import re
import datetime 
parent_dir = pathlib.Path("/brtx/602-nvme1/estengel/ambiguous_parsing/logs/1.0/") 
old_dirs = parent_dir.glob("*_fewshot_2_test_eval_constrained_bs_5_np_10")

conj_start_idx = 100
conj_end_idx = 200

for old_dir in old_dirs:

    model_outputs = old_dir.glob("model_outputs*")
    model_output = sorted(model_outputs)[-1]
    with open(model_output) as f1:
        model_output_data = [line for line in f1]
    # model_output_data = model_output_data[conj_start_idx:conj_end_idx]
    
    new_dir = re.sub("_fewshot_", "_fewshot_conj_only_", str(old_dir))
    new_dir = pathlib.Path(new_dir)
    new_model_outputs = new_dir.glob("model_outputs*")
    try:
        new_model_output = sorted(new_model_outputs)[-1]
    except IndexError:
        print(new_dir)
        continue
    with open(new_model_output) as f2:
        new_model_output_data = [line for line in f2]

    model_output_before = model_output_data[:conj_start_idx]
    model_output_after = model_output_data[conj_end_idx:]
    new_model_output_data = model_output_before + new_model_output_data + model_output_after
    # print(old_dir, len(new_model_output_data))

    if len(new_model_output_data)   == 500:
        now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        new_file = new_dir.joinpath(f"model_outputs.{now}.jsonl")
        with open(new_file, "w") as f3:
            for line in new_model_output_data:
                f3.write(line)

