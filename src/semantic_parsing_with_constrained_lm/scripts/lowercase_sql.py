import pathlib
import argparse
import json
from tqdm import tqdm

def read_sql_file(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def convert_data(data): 
    new_data = []
    for ex in tqdm(data):
        sql = ex['plan']
        new_sql = sql.lower()
        ex['plan'] = new_sql
        new_data.append(ex)
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args() 

    print("reading data")
    data = read_sql_file(f"/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/{args.split}_all.jsonl")
        
    new_data = convert_data(data) 
    with open(args.out_dir / f"{args.split}_all_lower.jsonl", 'w') as f:
        for ex in new_data:
            f.write(json.dumps(ex) + "\n")