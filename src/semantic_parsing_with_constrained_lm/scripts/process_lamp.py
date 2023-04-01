import json 
from pathlib import Path
import argparse
import numpy as np
np.random.seed(12)


def read_dir(path):
    data_by_fname = {} 
    for file in path.glob("*.jsonl"):
        with open(file) as f1:
            data_by_fname[file.name] = [json.loads(line) for line in f1]
    return data_by_fname

def write_dir(data_by_fname, path):
    for fname, data in data_by_fname.items():
        with open(path / fname, "w") as f1:
            for line in data:
                f1.write(json.dumps(line) + "\n") 


def convert(data_by_fname):
    for split, data in data_by_fname.items():
        new_data = []
        for i, datum in enumerate(data):
            new_datum = {}
            new_datum['utterance'] = datum['surface']
            new_datum['plan'] = datum['lf']
            new_datum['unfilled_template'] = datum['unfilled_template']
            new_data.append(new_datum)
        data_by_fname[split] = new_data

    np.random.shuffle(data_by_fname['train.jsonl']) 
    return data_by_fname


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)

    args = parser.parse_args()
    data_by_fname = read_dir(args.in_dir)
    data_by_fname = convert(data_by_fname)
    args.out_dir.mkdir(exist_ok=True, parents=True)
    write_dir(data_by_fname, args.out_dir)