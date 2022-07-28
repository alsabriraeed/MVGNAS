import inspect
import sys

import torch
import torch_geometric.utils
# from torch_geometric.utils import scatter_
import torch_scatter
from torch_scatter import scatter_add
# from torch import scatter

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


class MessagePassing(torch.nn.Module):

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow  
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """
        size = [None, None] if size is None else list(size)
        assert len(size) == 2
        
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}
        
        message_args = []
        for arg in self.__message_args__:
            
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))
        # print('reached here.....')
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size
        

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        out = self.message(*message_args)
        # print('out shape: ',out.shape )
        # out = message_args           
        if self.aggr in ["add", "mean", "max"]:
            # out = scatter(self.aggr, out, edge_index[i], dim_size=size[i],reduce='add')
             # out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
            #  print('size[i]: ',size[i])               
             out = self.scatter1(self.aggr, message_args[i], edge_index[i], dim_size=size[i])

        else:
            pass
        out = self.update(out, *update_args)
        return out
    
    # @staticmethod
    # def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
    #     if edge_weight is None:
    #         edge_weight = torch.ones((edge_index.size(1), ),
    #                                   dtype=dtype,
    #                                   device=edge_index.device)

    #     fill_value = 1 if not improved else 2
    #     edge_index, edge_weight = add_remaining_self_loops(
    #         edge_index, edge_weight, fill_value, num_nodes)

    #     row, col = edge_index
    #     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    #     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    def scatter1(name, src, index, dim=0,dim1=0, dim_size=None):
        r"""Aggregates all values from the :attr:`src` tensor at the indices
        specified in the :attr:`index` tensor along the first dimension.
        If multiple indices reference the same location, their contributions
        are aggregated according to :attr:`name` (either :obj:`"add"`,
        :obj:`"mean"` or :obj:`"max"`).
    
        Args:
            name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
                :obj:`"min"`, :obj:`"max"`).
            src (Tensor): The source tensor.
            index (LongTensor): The indices of elements to scatter.
            dim (int, optional): The axis along which to index. (default: :obj:`0`)
            dim_size (int, optional): Automatically create output tensor with size
                :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
                minimal sized output tensor is returned. (default: :obj:`None`)
    
        :rtype: :class:`Tensor`
        """
        # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnn')
        # print(name, type(name))
        # print('index: ',index.shape)
        # print('dim_size: ',dim_size)
        # print('src: ',src)
        # print('dim',dim)
        # print('dim1',dim1)
        # print('self.aggr', name.shape)
        assert src in ['add', 'mean', 'min', 'max']

        op = getattr(torch_scatter, 'scatter_{}'.format(src))
        out = op(index, dim, dim1, None, dim_size)
        out = out[0] if isinstance(out, tuple) else out
    
        if name == 'max':
            out[out < -10000] = 0
        elif name == 'min':
            out[out > 10000] = 0
            
        return out


    # def message(self, x_j):  # pragma: no cover     self, edge_index,x_i, x_j,num_nodes ,edge_attr    message(self, x_j):
    #     r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
    #     for each edge in :math:`(i,j) \in \mathcal{E}`.
    #     Can take any argument which was initially passed to :meth:`propagate`.
    #     In addition, features can be lifted to the source node :math:`i` and
    #     target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
    #     variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
    #     print("This is the load:")
    #     print(len(self.x_j))

    #     return x_j  
    # def message(self, edge_index,x_i, x_j,edge_attr,num_nodes,y):  #self, edge_index_i, x_i, x_j, size_i,edgemat

    #     self.var1 = 100
    #     if self.att_type == "const":
    #         if self.training and self.dropout > 0:
    #             x_j = F.dropout(x_j, p=self.dropout, training=True)
    #         neighbor = x_j
    #     elif self.att_type == "gcn":
            
    #         if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):  # 对于不同的图gcn_weight需要重新计算
    #             # print('pppppppppppppppppp')
    #             _, norm = self.norm(edge_index, num_nodes,edge_attr) #, None)
    #             self.gcn_weight = norm
    #         neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
    #     else:
    #         # Compute attention coefficients.
    #         alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
    #         # alpha = softmax(alpha, edge_index[0], num_nodes)
    #         alpha = softmax(alpha, edge_index[0], num_nodes)
    #         # Sample attention coefficients stochastically.
    #         if self.training and self.dropout > 0:
    #             alpha = F.dropout(alpha, p=self.dropout, training=True)

    #         neighbor = x_j * alpha.view(-1, self.heads, 1)
    #     if self.pool_dim > 0:
    #         for layer in self.pool_layer:
    #             neighbor = layer(neighbor)
    #     return neighbor

    # def apply_attention(self, edge_index, num_nodes, x_i, x_j):
    #     if self.att_type == "gat":
    #         alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
    #         alpha = F.leaky_relu(alpha, self.negative_slope)

    #     elif self.att_type == "gat_sym":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
    #         alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
    #         alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

    #     elif self.att_type == "linear":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         al = x_j * wl
    #         ar = x_j * wr
    #         alpha = al.sum(dim=-1) + ar.sum(dim=-1)
    #         alpha = torch.tanh(alpha)
    #     elif self.att_type == "cos":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         alpha = x_i * wl * x_j * wr
    #         alpha = alpha.sum(dim=-1)

    #     elif self.att_type == "generalized_linear":
    #         wl = self.att[:, :, :self.out_channels]  # weight left
    #         wr = self.att[:, :, self.out_channels:]  # weight right
    #         al = x_i * wl
    #         ar = x_j * wr
    #         alpha = al + ar
    #         alpha = torch.tanh(alpha)
    #         alpha = self.general_att_layer(alpha)
    #     else:
    #         raise Exception("Wrong attention type:", self.att_type)
    #     return alpha


    # def update(self, aggr_out):  # pragma: no cover
    #     r"""Updates node embeddings in analogy to
    #     :math:`\gamma_{\mathbf{\Theta}}` for each node
    #     :math:`i \in \mathcal{V}`.
    #     Takes in the output of aggregation as first argument and any argument
    #     which was initially passed to :meth:`propagate`."""

    #     return aggr_out
