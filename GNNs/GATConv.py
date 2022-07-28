# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:36:23 2021

@author: Raeed
"""
   
import torch
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_geometric.nn import MessagePassing         
from ERExtraction.message_passing import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import math
from torch_scatter import scatter


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


class GATConv1(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1, 
                 concat= False,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(GATConv1, self).__init__(aggr='add')  # "Add" aggregation.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))  # \theta
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))  # \alpha: rather than separate into two parts

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        # 1. Linearly transform node feature matrix (XΘ)
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)  # N x H x emb(out)
        # print('x', x)

        # 2. Add self-loops to the adjacency matrix (A' = A + I)
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)  # 2 x E
            # print('edge_index', edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # 2 x (E+N)
            # print('edge_index', edge_index)

        # 3. Start propagating messages   
        return self.propagate(edge_index, x=x, size=size)  # 2 x (E+N), N x H x emb(out), None

    def message(self, x_i, x_j, size_i, edge_index_i):  # Compute normalization (concatenate + softmax)
        # x_i, x_j: after linear x and expand edge (N+E) x H x emb(out) 
        # = N x H x emb(in) @ emb(in) x emb(out) (+) E x H x emb(out)
        # edge_index_i: the col part of index
        # size_i: number of nodes
        # print('x_i', x_i)
        # print('x_j', x_j)
        # print('size_i', size_i)
        # print('edge_index_i', edge_index_i)

        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H x (emb(out)+ emb(out))
        # print('alpha', alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # LeakReLU only changes those negative.
        # print('alpha', alpha)
        alpha = softmax(alpha, edge_index_i, size_i)  # Computes a sparsely evaluated softmax
        # print('alpha', alpha)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # print(f'x_j*alpha {x_j * alpha.view(-1, self.heads, 1)}')
        return x_j * alpha.view(-1, self.heads, 1)
        # each row is norm(embedding) vector for each edge_index pair (detail in the following)

    def update(self, aggr_out):  # 4. Return node embeddings (average heads)
        # Based on the directed graph, Node 0 gets message from three edges and one self_loop 
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

        # print('aggr_out', aggr_out)  # (E+N) x H x emb(out)
        aggr_out = aggr_out.mean(dim=1)  # to average multi-head
        # print('aggr_out', aggr_out)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out