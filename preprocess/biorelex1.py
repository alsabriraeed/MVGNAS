import os
import json

from constants import *
from os.path import join
from data.base import DataInstance
from data.helpers import tokenize

def read_split(raw_insts, tokenizer):
    # Read raw instances
    # raw_insts = []
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     raw_insts += json.load(f)
    print('The lenght of training/test data',len(raw_insts))
    # Construct data_insts
    data_insts = []
    interaction_types, interaction_implicits, interaction_labels = set(), set(), set()
    for inst in raw_insts:
        id, text = inst['id'], inst['text']
        raw_entites, raw_interactions = inst.get('entities', []), inst.get('interactions', [])
        tokenization = tokenize(tokenizer, text.split(' '))
        startchar2token, endchar2token = tokenization['startchar2token'], tokenization['endchar2token']

        # Compute entities
        entities = []
        if len(raw_entites) > 0:
            appeared, mid = set(), 0
            for eid, e in enumerate(raw_entites):
                entity_mentions = []
                for name in e['names']:
                    for start, end in e['names'][name]['mentions']:
                        if (start, end) in appeared: continue
                        try:
                            new_mention = {
                                'name': name, 'mention_id': mid, 'entity_id': eid,
                                'start_char': start, 'start_token': startchar2token[start],
                                'end_char': end, 'end_token': endchar2token[end],
                            }
                        except:
                            print('skipped {}'.format(id))
                            continue
                        appeared.add((start, end))
                        entity_mentions.append(new_mention)
                        mid += 1
                entities.append({
                    'label': BIORELEX_ENTITY_TYPES.index(e['label']),
                    'entity_id': eid, 'mentions': entity_mentions
                })

        # Compute relations
        relations = []
        if len(raw_interactions) > 0:
            for interaction in raw_interactions:
                p1, p2 = interaction['participants']
                label = interaction['label']
                relations.append({'participants': [p1, p2], 'label': BIORELEX_RELATION_TYPES.index(label)})

        # Create a new DataInstance
        data_insts.append(DataInstance(inst, id, text, tokenization, True, entities, relations))

    return data_insts

def load_biorelex_dataset(base_path, tokenizer):
    # Determine absolute file paths
    train_fp = join(base_path, 'train.json')
    test_fp = join(base_path, 'dev.json')
    
    # adding for Graphnas------------------------------------
    # Read types info
    # with open(types_fp, 'r') as f:
    #     type_info = json.loads(f.read())

    # Read splits
    all_train_data = []
    with open(train_fp, 'r', encoding='utf-8') as f:
        all_train_data += json.load(f)
        
        
    # with open(train_fp, 'r') as f:
    #     all_train_data = json.loads(f.read())
    
    test_data = []
    with open(test_fp, 'r', encoding='utf-8') as f:
        test_data += json.load(f)
    
    # with open(test_fp, 'r') as f:
    #     test_data = json.loads(f.read())
     
    val_size = int(len(all_train_data) * 0.1)
    train_size = len(all_train_data) - val_size
    train_data = all_train_data[:train_size]
    val_data = all_train_data[train_size:]
    
    train = read_split(train_data, tokenizer)
    dev = read_split(val_data, tokenizer)
    test = read_split(test_data, tokenizer)
    # end adding for GraphNas--------------------------------
    
    
    # # Read splits
    # train = read_split(train_fp, tokenizer)
    # dev = read_split(dev_fp, tokenizer)

    return train,test, dev
