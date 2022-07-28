# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:33:37 2022

@author: Raeed
"""
import torch.nn.functional as F
from torch_geometric.nn import CGConv  
from torch_geometric.nn import MessagePassing

class CG_Conv(MessagePassing):    
    def __init__(self,in_dim, out_dim,aggr, edge_dim, concat=False, bias=False):     
        super(CG_Conv, self).__init__(aggr=aggr)         
        self.conv1 = CGConv(in_dim, out_dim)
        # self.conv2 = CGConv(out_dim, out_dim)     
 
       
    def forward(self,x, edge_index,edge_attr):
        x = self.conv1(x, edge_index, edge_attr)  
        # x = self.conv2(x, edge_index, edge_attr)         
        return x