# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:01:30 2021

@author: Raeed
"""
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv    
from torch_geometric.nn import MessagePassing   
from torch.nn import Linear        
 
class Transformer_Conv(MessagePassing):    
    def __init__(self,in_dim, out_dim, aggr, edge_dim, concat=False, bias=False):   
        super(Transformer_Conv, self).__init__(aggr=aggr)                
        self.encode_attr =Linear(edge_dim,in_dim)                 
        self.encode_node =Linear(in_dim,in_dim)                
        self.conv1 = TransformerConv(in_dim, out_dim, heads=1, edge_dim=in_dim)                     
        # self.conv2 = TransformerConv(out_dim, out_dim, edge_dim=in_dim)       
                    
    def forward(self,x, edge_index,edge_attr):  
        
        edge_weight =  self.encode_attr(edge_attr)             
        x= self.encode_node(x)          
        x =self.conv1(x, edge_index, edge_weight)           
        # x = self.conv2(x, edge_index, edge_weight)                

        return x
