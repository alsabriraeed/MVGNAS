# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:58:39 2022

@author: Raeed
"""
import torch.nn.functional as F   
from torch_geometric.nn import GraphConv  
from torch_geometric.nn import MessagePassing   

 
class Graph_Conv(MessagePassing):    
    def __init__(self,in_dim, out_dim,aggr ='add', edge_dim=1, concat=False, bias=False):        
        super(Graph_Conv, self).__init__(aggr=aggr)           
        self.conv1 = GraphConv(in_dim, out_dim)
        # self.conv2 = GraphConv(out_dim, out_dim)             
       
    def forward(self,x, edge_index,edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        # x = self.conv2(x, edge_index, edge_attr)                     
        return x
