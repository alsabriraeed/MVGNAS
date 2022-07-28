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
import time       
from ERExtraction.utilsRE import *
from constants import *
from transformers import AutoTokenizer                  
from transformers import logging     
logging.set_verbosity_warning()      
logging.set_verbosity_error()      
from data import load_data                   
from scorer import evaluate                 
from ERExtraction.model import JointModel     
from argparse import ArgumentParser      
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(configs,actions):           
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])           
    # train,test, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)         
    configs['max_span_width'] = configs['max_span_width']
    train, dev = load_data(configs['dataset'], configs['split_nb'], tokenizer)
    model = JointModel(configs,actions)      
    # print('Train Size = {} | Dev Size = {}'.format(len(train), len(dev)))              
    # print('Initialize a new model | {} parameters'.format(get_n_params(model)))     
    best_dev_score, best_dev_m_score, best_dev_rel_score = 0, 0, 0
    best_test_score, best_test_m_score, best_test_rel_score = 0, 0, 0
    PRETRAINED_MODEL_Path = join(configs['save_dir'], 'model_{}.pt'.format(configs['split_nb']))
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL_Path):
        checkpoint = torch.load(PRETRAINED_MODEL_Path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded a pretrained model')
        print('Evaluation on the dev set')
        dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
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

    # Start training        
    accumulated_loss = RunningAverage()
    iters, batch_loss = 0, 0
    loss_score_file_path = join(configs['save_dir'], 'model_{}.txt'.format(configs['split_nb']))
    loss_score_file = open(loss_score_file_path, 'a')
    for i in range(configs['epochs']):
        print('Starting epoch {}'.format(i+1), flush=True)
        t0 = time.time()
        model.in_ned_pretraining = i < configs['ned_pretrain_epochs']
        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        for train_idx in train_indices:
            iters += 1
            tensorized_example = [b.to(model.device) for b in train[train_idx].example]
            tensorized_example.append(train[train_idx].all_relations)
            tensorized_example.append(train[train_idx])
            tensorized_example.append(True) # is_training

            iter_loss = model(*tensorized_example)[0]       
            iter_loss /= configs['batch_size']      
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()        
                optimizer.zero_grad()
                batch_loss = 0        
            # Report loss      
            if iters % configs['report_frequency'] == 0:
                accumulated_loss = RunningAverage()       
        duration = time.time() - t0
        # print('duration for one training epoch: ', duration)       
        t1 = time.time()
        # Evaluation after each epoch                               
        with torch.no_grad():
            print('Evaluation on the dev set')               
            dev_m_score, dev_rel_score = evaluate(model, dev, configs['dataset'])
            dev_score = (dev_m_score + dev_rel_score) / 2.0
        duration1 = time.time() - t1
        # print('duration for one test epoch: ', duration1)                  
  
        if dev_score > best_dev_score:       
            best_dev_score = dev_score
            best_dev_m_score = dev_m_score
            best_dev_rel_score = dev_rel_score
            # Save the model
            save_path = join(configs['save_dir'], 'model_{}.pt'.format(configs['split_nb']))
            
            torch.save({'model_state_dict': model.state_dict()}, save_path)        
            # print('Saved the model', flush=True)                  
                   
        if i >= 5 and best_dev_score <=0.3:
            break                          
    loss_score_file.writelines("the result of span width 2 :best_dev_m_score: "+ str(best_dev_m_score) +" best_dev_rel_score: "+str(best_dev_rel_score)+ "best_dev_score: "+ \
                                   str( best_dev_score))
    loss_score_file.close()            
    return actions, best_dev_rel_score,best_dev_m_score,best_dev_rel_score       
