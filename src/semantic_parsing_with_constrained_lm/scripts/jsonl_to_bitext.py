import json 
import sys
from pathlib  import Path
path = Path(sys.argv[1])


def prepare_input(datum):
    inputs = []
    if datum['last_user_utterance'] != "":
        inputs.append("__User")
        inputs.append(datum['last_user_utterance'])
    if datum['last_agent_utterance'] != "":
        inputs.append("__Agent")
        inputs.append(datum['last_agent_utterance'])
    inputs.append("__User")
    inputs.append(datum['utterance'])
    return " ".join(inputs).strip() 

with open(path) as f1:
    data = [json.loads(x) for x in f1.readlines()]

parent_path = path.parent
filename = str(parent_path.joinpath(path.stem))
print(f"{filename}")
with open(filename + ".src_tok","w") as srcf,\
    open(filename+".tgt", "w") as tgtf,\
        open(filename+".datum_id", "w") as idf:
    for datum in data:
        input = prepare_input(datum)
        srcf.write(input + "\n")
        tgtf.write(datum["plan"] + "\n")
        datum_id = {"dialogue_id": datum["dialogue_id"], "turn_index": datum["turn_part_index"]}
        idf.write(json.dumps(datum_id) + "\n")