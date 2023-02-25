import re 
import pathlib
import argparse
import pdb 
from collections import defaultdict, Counter
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
np.random.seed(12)

import spacy


def read_sql_file(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def get_vocab(data): 
    nlp = spacy.load("en_core_web_sm")
    vocab = set()
    for ex in tqdm(data):
        sql = ex['plan']
        split_sql = [x.text for x in nlp(sql)]
        vocab.update(split_sql)
    return vocab

def collect_stats(vocab, tokenizer):
    lens = []
    for i, word in enumerate(vocab):
        tokenized_word = tokenizer.tokenize(word)
        lens.append(len(tokenized_word))
    return Counter(lens)

def build_random_word_map(tokenizer):
    nlp = spacy.load("en_core_web_sm")
    # build a mapping from word lengths to random words of that length
    mapping = defaultdict(list)
    dataset = load_dataset("wikitext", 'wikitext-103-v1')

    train_vocab = set()
    # get different documents
    dataset_idxs = [i for i in range(len(dataset['train']))]
    np.random.shuffle(dataset_idxs)
    selected_idxs = dataset_idxs[0:10000]
    
    whole_dataset = [dataset['train'][i]['text'] for i in selected_idxs]
    for text in tqdm(whole_dataset): 
        text = text.strip()
        tokens = [x.text for x in nlp(text)]
        train_vocab.update(tokens)
    for word in tqdm(train_vocab):
        tokenized_word = tokenizer.tokenize(word)
        mapping[len(tokenized_word)].append(word)
    return mapping 

def reassign_vocab(vocab, mapping, tokenizer):
    new_to_old = {}
    old_to_new = {}
    had_to_reduce = 0
    for i, word in tqdm(enumerate(vocab)):
        tokenized_word = tokenizer.tokenize(word)
        l = len(tokenized_word)
        mapping_idxs = [i for i in range(len(mapping[l]))]
        if len(mapping_idxs) == 0:
            # reduce 
            had_to_reduce += 1
            new_l = l
            while(len(mapping[new_l]) == 0):
                new_l -= 1
            mapping_idxs = [i for i in range(len(mapping[new_l]))]
            l = new_l 

        new_word_idx = np.random.choice(mapping_idxs)
        new_word = mapping[l][new_word_idx]
        # remove
        mapping[l].pop(new_word_idx)
        new_to_old[new_word] = word
        old_to_new[word] = new_word
    print(f"had to reduce {had_to_reduce} words")
    return new_to_old, old_to_new

def convert_data(data, old_to_new):
    nlp = spacy.load("en_core_web_sm")
    new_data = []
    for ex in tqdm(data):
        sql = ex['plan']
        split_sql = [x.text for x in nlp(sql)]
        new_sql = [old_to_new.get(word, word) for word in split_sql]
        new_sql = " ".join(new_sql)
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
    if args.override:
        # data = data[0:100]
        print("getting vocab")
        vocab = get_vocab(data)

        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        print("getting mapping")
        mapping  = build_random_word_map(tokenizer)
        print("reassigning vocab")
        new_to_old, old_to_new = reassign_vocab(vocab, mapping, tokenizer)

        print("writing to file")
        with open("scripts/anon_sql/new_to_old.json", 'w') as f:
            json.dump(new_to_old, f)
        with open("scripts/anon_sql/old_to_new.json", 'w') as f:
            json.dump(old_to_new, f)
    else:
        with open("scripts/anon_sql/new_to_old.json", 'r') as f:
            new_to_old = json.load(f)
        with open("scripts/anon_sql/old_to_new.json", 'r') as f:
            old_to_new = json.load(f)
        
        new_data = convert_data(data, old_to_new)
        with open(args.out_dir / f"{args.split}_all_converted.jsonl", 'w') as f:
            for ex in new_data:
                f.write(json.dumps(ex) + "\n")