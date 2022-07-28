# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:01:30 2021

@author: Raeed
"""
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import MessagePassing  

 
class Hypergraph_Conv(MessagePassing):    
    def __init__(self,in_dim, out_dim, aggr,  edge_dim, concat=False, bias=False):
        super(Hypergraph_Conv, self).__init__(aggr=aggr)      
        self.conv1 = HypergraphConv(in_dim, out_dim)
        # self.conv2 = HypergraphConv(out_dim, out_dim) 
                    
    def forward(self,x, edge_index,edge_attr):  
        x = self.conv1(x, edge_index, edge_attr)     
        # x = self.conv2(x, edge_index, edge_attr)                    
        return x
