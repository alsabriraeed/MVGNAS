from utils.model_utils import EarlyStop, TopAverage, process_action
import torch.nn.functional as F    
import time
import sys
# from torch_geometric.data import Data                                   
import os
import torch.nn as nn
import copy
import utils
import torch
import random
import math
import pyhocon
import warnings
import numpy as np
import torch.optim as optim

from ERExtraction.utilsRE import *
from constants import *
from transformers import *
from ERExtraction.model import JointModel      

from ERExtraction.REmodel import train as trainmodel
class GNNModule(object):

    def __init__(self, args, configs):
        self.args = args
        self.early_stop_manager = EarlyStop(10)     
        # instance for class that imlement top average and average     
        self.reward_manager = TopAverage(10)
        self.configs = configs
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs     

        self.train_graph_index = 0
        self.train_set_length = 10

        self.param_file = args.param_file    
        self.shared_params = None

        # self.in_feats = 300       
        # self.n_classes = 3        
        # self.in_edge_attr = 1             
        
    def trainfun(self, actions=None, format="micro"):
        self.current_action = actions
         
        return self.train1(actions, format=format)

    def record_action_info(self, origin_action, reward, val_acc):
        return self.record_action_info(self.current_action, reward, val_acc)
     
    def train1(self, actions=None, format="two"):       
        origin_action = actions
        actions = process_action(actions, format, self.args)      
        GNNmodel = actions      
        print("Model:", actions)        
        try:          
            GNNmodel, val_acc,best_dev_m_score,best_dev_rel_score = self.run_model(self,actions)

        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                best_dev_m_score = 0
                best_dev_rel_score = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        self.save_param(GNNmodel, update_all=(reward > 0))

        self.record_action_info1(origin_action, reward, val_acc,best_dev_m_score,best_dev_rel_score)

        return reward, val_acc

    def record_action_info1(self, origin_action, reward, val_acc,best_dev_m_score,best_dev_rel_score):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as file:
            file.write(str(origin_action))   
            file.write(";")              
            file.write(str(reward))       
            file.write(";")
            file.write(str(val_acc))       
            # file.write(";")     
            # file.write("Entity:  " + str(best_dev_m_score))
            # file.write(";")
            # file.write("Relaion:  " + str(best_dev_rel_score))
            file.write("\n")    
            

    @staticmethod
    def run_model(self,actions):
        GNNmodel, best_dev_score,best_dev_m_score,best_dev_rel_score = trainmodel(self.configs ,actions)
        return GNNmodel, best_dev_score,best_dev_m_score,best_dev_rel_score
        
    def test_with_param(self, actions=None, format="two", with_retrain=False):
        return self.trainfun(actions, format)
    
    def retrain(self, actions, format="two"):
        return self.trainfun(actions, format)
    def load_param(self):
        # don't share param     
        pass

    def save_param(self, model, update_all=False):
        # don't share param
        pass

