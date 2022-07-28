# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:01:30 2021

@author: Raeed
"""
# from message_passing import MessagePassing
import torch
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter
from message_passing import MessagePassing


import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def softmax(src, index, num_nodes):
    """
    Given a value tensor: `src`, this function first groups the values along the first dimension
    based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
    """
    # print('src', src)
    # print('index', index)                 
    # print('num_nodes', num_nodes)            
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    # print('N', N)
    # print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')}")
    # print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')[index]}")
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    # print('out', out)
    out = out.exp()
    # print('out', out)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    # print('out_sum', out_sum)
    # print(f'return: {out / (out_sum + 1e-16)}')
    return out / (out_sum + 1e-16)


class Edge_GATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 edge_dim=1,  # new
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(Edge_GATConv, self).__init__(aggr='add')  # "Add" aggregation.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim  # new
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))    # emb(in) x [H*emb(out)]
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))   # 1 x H x [2*emb(out)+edge_dim]    # new
        # self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))   # [emb(out)+edge_dim] x emb(out)  # new
        self.edge_update = Parameter(torch.Tensor(out_channels *heads, out_channels))                

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)  # new
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        # 1. Linearly transform node feature matrix (XÃŽËœ)     
        # print('x', x.shape)            
        # print(self.heads)
        # print(self.out_channels)         
        # print(self.weight.shape)     
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)   # N x H x emb(out)        
        # print('x', x.shape)                                         
        # print('edge_index: ', edge_index.shape)       
        # print('edge_attr', edge_attr.shape)              

        # 2. Add self-loops to the adjacency matrix (A' = A + I)
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)   # 2 x E
            # print('edge_index', edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # 2 x (E+N)
            # print('edge_index', edge_index)
        # print('hajahajahjahahjahaajah')                
        # 2.1 Add node's self information (value=0) to edge_attr
        # self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)   # N x edge_dim   # new
        # # print('self_loop_edges', self_loop_edges)
        # edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (E+N) x edge_dim  # new
        # print('edge_attr', edge_attr)

        # 3. Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)  # new
                            # 2 x (E+N), N x H x emb(out), (E+N) x edge_dim, None

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):  # Compute normalization (concatenate + softmax)
        # x_i, x_j: after linear x and expand edge (N+E) x H x emb(out)
        # = N x emb(in) @ emb(in) x [H*emb(out)] (+) E x H x emb(out)
        # edge_index_i: the col part of index  [E+N]
        # size_i: number of nodes
        # edge_attr: edge values of 1->0, 2->0, 3->0.   (E+N) x edge_dim
        # print('x_i', x_i)
        # print('x_j', x_j)
        # print('size_i', size_i)

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim  # new
        # print('edge_attr', edge_attr)         
        x_j = torch.cat([x_j, edge_attr], dim=-1)  # (E+N) x H x (emb(out)+edge_dim)   # new
        del edge_attr
        # print('x_j', x_j)                      

        x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)
        # print('x_i', x_i)
        # print(torch.cat([x_i, x_j], dim=-1))   # (E+N) x H x [emb(out)+(emb(out)+edge_dim)]
        # print('self.att', self.att)   # 1 x H x [2*emb(out)+edge_dim]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H
        del x_i
        # print('alpha', alpha)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # print('alpha', alpha)
        alpha = softmax(alpha, edge_index_i, size_i)   # Computes a sparsely evaluated softmax
        # print('alpha', alpha)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # print(f'x_j*alpha {x_j * alpha.view(-1, self.heads, 1)}')
        return x_j * alpha.view(-1, self.heads, 1)   # (E+N) x H x (emb(out)+edge_dim)

    def update(self, aggr_out):   # 4. Return node embeddings (average heads)
        # for Node 0: Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

        # print('aggr_out', aggr_out)   # N x H x (emb(out)+edge_dim)
        # aggr_out = aggr_out.mean(dim=1)
        # if self.concat is True:         
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:      
        #     aggr_out = aggr_out.mean(dim=1)              
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)                
        # self.edge_update = self.edge_update.view(-1, self.heads * self.out_channels)
        # aggr_out = aggr_out.view(-1,  self.out_channels) 
        # aggr_out = aggr_out.mean(dim=1)        
        # print('aggr_out', aggr_out.shape)   # N x (emb(out)+edge_dim)        
        # print('self.edge_update', self.edge_update)   # (emb(out)+edge_dim) x emb(out)
        # print(aggr_out.shape)                    
        # print(self.edge_update.shape)
        aggr_out = torch.mm(aggr_out, self.edge_update)
        # print('aggr_out', aggr_out)   # N x emb(out)  # new
        
        
            
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out