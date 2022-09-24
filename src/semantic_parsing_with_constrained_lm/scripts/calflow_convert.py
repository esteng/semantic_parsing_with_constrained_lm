import json
import pathlib

data_path = pathlib.Path("/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/")

ids = {"train": [], "dev_valid": [], "test_valid": []}
# build LUT for utterances with context 
utterance_context_lut = {"train": {}, "dev_valid": {}, "test_valid": {}}
for split in ["train", "dev_valid", "test_valid"]:
    datum_id_path = data_path.joinpath(f"{split}.datum_id")
    src_path = data_path.joinpath(f"{split}.src")
    with open(datum_id_path) as df, open(src_path) as sf:
        datum_data = df.readlines()
        src_data = sf.readlines()
        for datum, src in zip(datum_data, src_data):
            utterance_context_lut[split][datum] = src.strip()
            datum = json.loads(datum)
            ids[split].append((datum["dialogue_id"], datum["turn_index"]))

out_path = pathlib.Path("/home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/domains/calflow/data")
with open(out_path / "ids_dev_valid_full.txt", "w") as df, open(out_path / "ids_train_full.txt", "w") as tf: 
    for split in ["train", "dev_valid"]:
        for dialogue_id, turn_index in ids[split]:
            if split == "train": 
                tf.write(f"{dialogue_id},{turn_index}\n") 
            else:
                df.write(f"{dialogue_id},{turn_index}\n")
    

# for split in ["dev_valid.dataflow_dialogues.jsonl", "test_valid.dataflow_dialogues.jsonl",  "train.dataflow_dialogues.jsonl"]:
#     splitname = split.split(".")[0]
#     with open(data_path.joinpath(split)) as f1:
#         for line in f1:
#             data = json.loads(line)
#             for turn in data['turns']: 
#                 to_write = {"dialogueId": data['dialogueId'],
#                             "turnIndex": turn['turnIndex'],
#                             "utterance": '',
#                             "canonical_utterance": turn['agent_utterance']['original_text'],
#                             "lispress": turn['lispress'],
#                             "context": ''}

#                 # build datum_id 
#                 datum = {"dialogue_id": data['dialogueId'], "turn_index": turn['turnIndex']}
#                 input_with_context = utterance_context_lut[splitname][json.dumps(datum)]
#                 to_write["utterance"] = input_with_context
