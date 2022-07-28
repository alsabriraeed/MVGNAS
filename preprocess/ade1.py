import os
import json

from constants import *
from os.path import join
from data.helpers import tokenize
from data.base import DataInstance

def read_split(data, type_info, tokenizer):
    # with open(file_path, 'r') as f:
    #     data = json.loads(f.read())
     
    # if mode =='train':
    #     val_size = int(len(data) * 0.2)
    #     train_size = len(data) - val_size
    # data length
    print('The lenght of training/test data',len(data))

    # Construct data_insts
    data_insts = []
    for raw_inst in data:
        id = raw_inst['orig_id']
        tokens, raw_entities, raw_relations = raw_inst['tokens'], raw_inst['entities'], raw_inst['relations']
        tokenization = tokenize(tokenizer, tokens)
        startchar2token, endchar2token = tokenization['startchar2token'], tokenization['endchar2token']

        # Initialize inst_data
        inst_data = {
            'text': ' '.join(tokens), 'id': id, 'entities':[], 'interactions': []
        }

        # Compute mappings from original tokens to char offsets
        otoken2startchar, otoken2endchar, char_offset = {}, {}, 0
        for ix in range(len(tokens)):
            otoken2startchar[ix] = char_offset
            otoken2endchar[ix] = char_offset + len(tokens[ix])
            char_offset += (1 + len(tokens[ix]))

        # Compute entities
        entities = []
        if len(raw_entities) > 0:
            mid = 0
            for eid, e in enumerate(raw_entities):
                text = ' '.join(tokens[e['start']:e['end']])
                start, end = otoken2startchar[e['start']], otoken2endchar[e['end']-1]
                start_token, end_token = startchar2token[start], endchar2token[end]
                entities.append({
                    'label': ADE_ENTITY_TYPES.index(e['type']),
                    'entity_id': eid, 'mentions': [{
                        'name': text, 'mention_id': mid, 'entity_id': eid,
                        'start_char': start, 'end_char': end,
                        'start_token': start_token, 'end_token': end_token
                    }]
                })
                inst_data['entities'].append({
                    'label': e['type'],
                    'names': {
                        text: {
                            'is_mentioned': True,
                            'mentions': [[start, end]]
                        }
                    },
                    'is_mentioned': True
                })
                mid += 1

        # Compute relations
        relations = []
        if len(raw_relations) > 0:
            for relation in raw_relations:
                p1, p2 = relation['head'], relation['tail']
                label = relation['type']
                relations.append({
                                  'participants': [p1, p2],
                                  'label': ADE_RELATION_TYPES.index(label)
                                })
                inst_data['interactions'].append({
                                  'participants': [p1, p2],
                                  'label': label
                                })

        # Create a new DataInstance
        text = ' '.join(tokens)
        data_insts.append(DataInstance(inst_data, id, text, tokenization, False, entities, relations))

    return data_insts

def load_ade_dataset(base_path, tokenizer, split_nb):
    # Determine absolute file paths
    assert(split_nb in list(range(10)))
    train_fp = join(base_path, 'ade_split_{}_train.json'.format(split_nb))
    test_fp = join(base_path, 'ade_split_{}_test.json'.format(split_nb))
    types_fp = join(base_path, 'ade_types.json')
    
    # print('train_fp:',train_fp , 'test_fp:  ', test_fp )

    # Read types info
    with open(types_fp, 'r') as f:
        type_info = json.loads(f.read())

    # Read splits
    with open(train_fp, 'r') as f:
        all_train_data = json.loads(f.read())
    
    with open(test_fp, 'r') as f:
        test_data = json.loads(f.read())
     
    val_size = int(len(all_train_data) * 0.1)
    train_size = len(all_train_data) - val_size
    train_data = all_train_data[:train_size]
    val_data = all_train_data[train_size:]
    
    train = read_split(train_data, type_info, tokenizer)
    val = read_split(val_data, type_info, tokenizer)
    test = read_split(test_data, type_info, tokenizer)

    return train, test,val
