# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:59:14 2021

@author: Raeed
"""

import torch
from torch_scatter import scatter_add
# from torch_geometric.nn import MessagePassing
from message_passing import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class Edge_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim,concat=True, bias=True):
        super(Edge_GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        # super(Edge_GCNConv, self).__init__(aggr='max')  # "Max" aggregation.

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.edge_dim = edge_dim  # new
        self.edge_update = torch.nn.Parameter(torch.Tensor(out_channels + edge_dim, out_channels))  # new
        self.concat = concat

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        # 1. Linearly transform node feature matrix  (XÎ˜)
        x = torch.matmul(x, self.weight)  # N x emb(out) = N x emb(in) @ emb(in) x emb(out)
        # print('x', x)
                    
        # 2. Add self-loops to the adjacency matrix   (A' = A + I)
        edge_weight = torch.ones((edge_index.size(1),),
                                 dtype=x.dtype,
                                 device=edge_index.device)   # [E+N]
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0))   # 2 x (E+N), [E+N]
        # print('edge_index', edge_index)
        # print('edge_weight', edge_weight)
                   
        # 2.1 Add node's self information (value=0) to edge_attr 
        # self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)  # N x edge_dim   # new
        # print('self_loop_edges', self_loop_edges)
        # edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # new
        # (E+N) x edge_dim = E x edge_dim + N x edge_dim   
        # print('edge_attr', edge_attr)

        # 3. Compute normalization  ( D^(-0.5) A D^(-0.5) )
        row, col = edge_index  # [E+N], [E+N]
        # print("row", row)           
        # print("col", col)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))  # [n]
        # print("deg", deg)
        deg_inv_sqrt = deg.pow(-0.5)  # [N]
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # [N]  # same to use masked_fill_
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]   # [E+N]
        # print('norm', norm)

        # 4. Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)   # 2 x (E+N), N x emb(out), [E+N], [E+N]

    def message(self, x_i, x_j, size_i, edge_attr, norm):   # 4.1 Normalize node features (concat edge_attr)
        # x_j: after linear x and expand edge  (N+E) x emb(out) = N x emb(in) @ emb(in) x emb(out) (+) E x emb(out)
        # print('x_j', x_j)   
        # print('edge_attr', edge_attr)
        # print('norm', norm)
        # print(x_j.shape,edge_attr.shape )    
        x_j = torch.cat([x_j, edge_attr], dim=-1)   # (N+E) x (emb(out)+edge_dim)   # new
        # print('x_j', x_j)
        # print(f'Norm*x_j: {norm.view(-1, 1) * x_j}')
        return norm.view(-1, 1) * x_j   # (N+E) x (emb(out)+edge_dim)  
        # return: each row is norm(embedding+edge_dim) vector for each edge_index pair

    def update(self, aggr_out):   # 4.2 Return node embeddings
        # print('aggr_out', aggr_out)
        # print('self.edge_update', self.edge_update)          
        # if self.concat is True:     
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:  
        #     aggr_out = aggr_out.mean(dim=1)
            
        aggr_out = torch.mm(aggr_out, self.edge_update)   # new
        # N x emb(out) = N x (emb(out)+edge_dim) @ (emb(out)+edge_dim) x emb(out)  
        # print('aggr_out', aggr_out)
        # for Node 0: Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

        if self.bias is not None:
            return aggr_out + self.bias
        else:
            return aggr_out