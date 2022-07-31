import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random           
from ERExtraction.utilsRE import *
from constants import *      
from ERExtraction.base import *
from ERExtraction.helpers import *
from ERExtraction.gnn import GNN
from search_space.search_space import gnn_map, act_map


class BiGNNLayer(nn.Module):          
    def __init__(self, retypes, configs, actions):   
        super(BiGNNLayer, self).__init__()
        self.retypes = retypes
        self.num_views = len(retypes)                     
        self.actions = actions      
         
        self.hid_size = configs['span_emb_size']
        # taking the GNN architecture components and dropout and learning rate       
        model_actions = actions['action']        
        param = actions['hyper_param']
        configs['ieg_bignn_dropout'] = param[1]

        if configs['multi_architectures']:     
            
            model_info= len(model_actions)// configs.num_of_model
            models = [model_actions[x:x+model_info] for x in range(0, len(model_actions), model_info)]
    
            gnn2p_fw, gnn2p_bw = [], []
            for i in range(self.num_views):
                gnn2p_fw.append(GNN(models[i], self.hid_size, self.hid_size // 2,
                                    dropout=configs['ieg_bignn_dropout']))
                gnn2p_bw.append(GNN(models[i], self.hid_size, self.hid_size // 2,
                                    dropout=configs['ieg_bignn_dropout']))
        else:               
            models = model_actions              
            gnn2p_fw, gnn2p_bw = [], []       
            for i in range(self.num_views):
                gnn2p_fw.append(GNN(models, self.hid_size, self.hid_size // 2,
                                    dropout=configs['ieg_bignn_dropout']))
                gnn2p_bw.append(GNN(models, self.hid_size, self.hid_size // 2,
                                    dropout=configs['ieg_bignn_dropout']))
            
        self.gnn2p_fw = nn.ModuleList(gnn2p_fw)
        self.gnn2p_bw = nn.ModuleList(gnn2p_bw) 
        self.dropout = nn.Dropout(configs['ieg_bignn_dropout'])
         
        self.act = act_map(model_actions[-2])      
        self.linear1 = nn.Linear(self.hid_size, self.hid_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')
   
    def forward(self, inps, fw_adjs, bw_adjs):
        # No self-loops in fw_adjs or bw_adjs                     
        num_views = self.num_views
        assert(len(fw_adjs) == num_views)
        assert(len(bw_adjs) == num_views)
        outs = []  
        
        for i in range(num_views):   
            # added new to solve the slow         
            fw_adjs_edges = fw_adjs[i].nonzero().t().contiguous()
            bw_adjs_edges = bw_adjs[i].nonzero().t().contiguous()
            if fw_adjs_edges.nelement() == 0 or bw_adjs_edges.nelement() == 0:
                outs.append(self.dropout(inps))       
            else:     
                fw_outs = self.gnn2p_fw[i](inps, fw_adjs_edges)
                bw_outs = self.gnn2p_bw[i](inps, bw_adjs_edges)
                outs.append(self.dropout(torch.cat([bw_outs, fw_outs], dim=-1)))
        outs = torch.cat([o.unsqueeze(0) for o in outs], dim=0)
        feats = self.linear1(self.act(torch.sum(outs, dim=0)))
        feats += inps # Residual connection
        return feats

class BiGNN(nn.Module):
    def __init__(self, retypes, configs, action):
        super(BiGNN, self).__init__()
        self.retypes = retypes         
        self.configs = configs     
        self.action = action
        self.num_hidden_layers = configs['ieg_bignn_hidden_layers']

        bignn_layers = []
        
        bignn_layers.append(BiGNNLayer(retypes, configs,self.action))
        self.bignn_layers = nn.ModuleList(bignn_layers)

    def forward(self, embs, fw_adjs, bw_adjs):
        out = embs
        
        out = self.bignn_layers[0](out, fw_adjs, bw_adjs)
        return out

class EncodingView(nn.Module):
    def __init__(self, configs,action):
        super(EncodingView, self).__init__()
        self.configs = configs      
        self.action = action           
        
        if configs['dataset'] == ADE1 or configs['dataset'] == ADE : nb_ieg_retypes = len(ADE_RELATION_TYPES)
        elif configs['dataset'] == BIORELEX or configs['dataset'] == BIORELEX1: nb_ieg_retypes = len(BIORELEX_RELATION_TYPES)
        self.ieg_retypes = list(range(nb_ieg_retypes))
        self.bignn = BiGNN(self.ieg_retypes, configs,self.action)    
    def forward(self, text, ie_preds):       
        tokenization = ie_preds['tokenization']  
        # Process prior IE predictions
        candidate_starts, candidate_ends = tolist(ie_preds['starts']), tolist(ie_preds['ends'])
        candidate_char_starts = [tokenization['token2startchar'][s] for s in candidate_starts]
        candidate_char_ends = [tokenization['token2endchar'][e] for e in candidate_ends]
        candidate_spans = list(zip(candidate_char_starts, candidate_char_ends))       
        candidate_embs, relation_probs = ie_preds['embs'], ie_preds['relation_probs']
        fw_adjs, bw_adjs = self.adjs_from_preds(relation_probs)     
        
        # Apply BiGNN on the adjacency matrices     
        ieg_out_h = self.bignn(candidate_embs, fw_adjs, bw_adjs)
        ka_span_embs = ieg_out_h
        return ka_span_embs

    def adjs_from_preds(self, relation_probs):
        relation_probs = relation_probs.clone().detach()
        fw_adjs, bw_adjs = [], []
        nb_nodes = relation_probs.size()[0]
        for ix in range(len(self.ieg_retypes)):
            A = relation_probs[:,:,ix]
            A.fill_diagonal_(0)       
            fw_adjs.append(A.to(self.device))
            bw_adjs.append(A.T.to(self.device))
        return fw_adjs, bw_adjs
