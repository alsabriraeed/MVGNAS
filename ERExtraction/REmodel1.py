import os
import copy
import utils
import torch
import random
import math
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utilsRE import *
from constants import *
# from transformers import *    
from transformers import AutoTokenizer
from data import load_data
from scorer import evaluate
# from models import JointModel
from graphnas_module.models.model import JointModel
from argparse import ArgumentParser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Main Functions
def train(configs,actions):
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    train,test, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
    # configs['split_nb'] = configs['split_nb'] +1
    # train, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
    model = JointModel(configs,actions)
    print('Train Size = {} | Dev Size = {}'.format(len(train), len(dev)))
    print('Initialize a new model | {} parameters'.format(get_n_params(model)))
    best_dev_score, best_dev_m_score, best_dev_rel_score = 0, 0, 0
    best_test_score, best_test_m_score, best_test_rel_score = 0, 0, 0
    PRETRAINED_MODEL_Path = join(configs['save_dir'], 'model_{}.pt'.format(configs['split_nb']))
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL_Path):
        checkpoint = torch.load(PRETRAINED_MODEL_Path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded a pretrained model')
        print('Evaluation on the dev set')
        dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
        print('Evaluation on the test set')
        test_m_score, test_rel_score = evaluate(model, test, configs['dataset'])
        best_dev_score = (dev_m_score + dev_rel_score) / 2.0

    # Prepare the optimizer and the scheduler
    num_train_docs = len(train)
    num_epoch_steps = math.ceil(num_train_docs / configs['batch_size'])
    num_train_steps = int(num_epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    
    # added
    param = actions['hyper_param']
    lr = param[0]
    configs['ieg_bignn_dropout'] = param[1]
    configs['task_learning_rate'] = param[0]
    weight_decay = param[2]
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    # added  
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start training
    accumulated_loss = RunningAverage()
    iters, batch_loss = 0, 0
    for i in range(configs['epochs']):
        print('Starting epoch {}'.format(i+1), flush=True)
        model.in_ned_pretraining = i < configs['ned_pretrain_epochs']
        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        # print('train:', train)
        for train_idx in train_indices:
            iters += 1
            # print('train[train_idx].example: ', train[train_idx].example)
            tensorized_example = [b.to(model.device) for b in train[train_idx].example]
            tensorized_example.append(train[train_idx].all_relations)
            tensorized_example.append(train[train_idx])
            tensorized_example.append(True) # is_training
            # print('tensorized_example: ', tensorized_example)
            iter_loss = model(*tensorized_example)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            # print('batch_loss: ',batch_loss)
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, accumulated_loss()), flush=True)
                accumulated_loss = RunningAverage()

        # Evaluation after each epoch
        with torch.no_grad():
            print('Evaluation on the dev set')
            dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
            dev_score = (dev_m_score + dev_rel_score) / 2.0
        with torch.no_grad():
                print('Evaluation on the test set:')
                test_m_score, test_rel_score = evaluate(model, test, configs['dataset'])
                test_score = (dev_m_score + dev_rel_score) / 2.0
        # Save model if it has better dev score
        if test_score > best_test_score:
            best_test_score = test_score
            best_test_m_score = test_m_score
            best_test_rel_score = test_rel_score
            
            
        # if dev_score > best_dev_score:
        #     best_dev_score = dev_score
        #     best_dev_m_score = dev_m_score
        #     best_dev_rel_score = dev_rel_score
            # Save the model
            save_path = join(configs['save_dir'], 'model_{}.pt'.format(configs['split_nb']))
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)
            # Evaluation after all epochs on test set
            # with torch.no_grad():
            #     print('Evaluation on the test set:')
            #     test_m_score, test_rel_score = evaluate(model, test, configs['dataset'])
            #     test_score = (dev_m_score + dev_rel_score) / 2.0
                # print('Mentions F1: ', 'Relations F1')
    # return {'all': best_dev_score, 'mention': best_dev_m_score, 'relation': best_dev_rel_score}
    # return  best_dev_score
    return model, best_test_rel_score,best_test_m_score,best_test_rel_score

# if __name__ == "__main__":
#     # Parse argument
#     parser = ArgumentParser()
#     parser.add_argument('-c', '--config_name', default='basic')
#     parser.add_argument('-d', '--dataset', default=ADE, choices=DATASETS)
#     parser.add_argument('-s', '--split_nb', default=0) # Only affect ADE dataset
#     args = parser.parse_args()
#     args.split_nb = int(args.split_nb)

#     # Start training
#     # prepare the dataset name; file directory; entities; relations; .....
#     configs = prepare_configs(args.config_name, args.dataset, args.split_nb)
#     train(configs)
