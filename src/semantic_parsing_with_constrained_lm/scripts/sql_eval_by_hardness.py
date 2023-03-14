import json
import jsons 
import sys
import os 
import pathlib
import argparse
from collections import defaultdict

from semantic_parsing_with_constrained_lm.configs.benchclamp_config import TEST_SUITE_PATH
from semantic_parsing_with_constrained_lm.domains.sql.sql_datum import SqlDatum
sys.path.insert(0, str(TEST_SUITE_PATH)) 
from evaluation import Evaluator
from process_sql import get_schema, Schema, get_sql

def build_schema_map(targets):
    schema_map = {} 
    for target in targets:
        schema_name = target.schema_name
        schema_map[target.dialogue_id] = schema_name
    return schema_map 

def main(args):
    path = args.path_to_pred
    target_path = args.path_to_gold 

    with open(path) as f1:
        lines = [json.loads(x) for x in f1]

    with open(target_path) as f1:
        targets =  [jsons.loads(line.strip(), cls=SqlDatum) for line in f1]
    schema_map = build_schema_map(targets)

    evaluator = Evaluator()
    lines_by_difficulty = defaultdict(list)


    for line in lines:
        gold = line['text']
        if gold.count(")") == 1 and gold.count("(") == 0:
                gold = gold.replace(")", "")
        if "faculty_participates_in" in gold:
            gold = gold.replace(
                    "faculty_participates_in", "Faculty_participates_in"
                )
        gold = gold.replace(" . ", ".")
        db = schema_map[line['test_datum_id']]
        db = os.path.join(args.db_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, gold)
        difficulty = evaluator.eval_hardness(g_sql)

        lines_by_difficulty[difficulty].append(line)

    for diff, lines in lines_by_difficulty.items(): 
        with open(args.output_dir / f"{diff}_outputs.jsonl", "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_pred", type=str, required=True, help='path to predicted jsonl')
    parser.add_argument("--path_to_gold", type=str, help='path to gold jsonl', default="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/Spider/test_all.jsonl") 
    parser.add_argument("--db_dir", type=str, help='path to db dir', default="/brtx/601-nvme1/estengel/resources/data/benchclamp/raw/test_sql/database/")
    parser.add_argument("--output_dir", type=pathlib.Path, help='path to output dir', default=None)
    args = parser.parse_args()

    if args.output_dir is None: 
        args.output_dir = pathlib.Path(args.path_to_pred).parent / "by_difficulty"
        args.output_dir.mkdir(exist_ok=True)

    main(args)