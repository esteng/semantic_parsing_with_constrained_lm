import argparse
import pathlib
import json 

def main(args):
    for model_outputs_file in args.input_dir.glob("model_outputs*.jsonl"):
        filename = model_outputs_file.stem
        name, timestep = filename.split(".") 
        out_filename = f"{name}.{timestep}.tgt"
        with open(model_outputs_file) as f1:
            data = [json.loads(line) for line in f1]
            tgt_lines = [x['outputs'][0] for x in data]
        with open(args.output_dir / out_filename, "w") as f2:
            for line in tgt_lines:
                f2.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()
    main(args)