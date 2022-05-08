import torch
import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from math import sqrt


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1#. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.normal_(0, 1)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, scale=1):
        super(GCN, self).__init__()

        self.scale = scale#2 / nhid / nhid if not sq else math.sqrt(2) / nhid
        self.nhid= nhid
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.layer = nn.Linear(in_features=nfeat,out_features=nclass,bias=False)

    # def forward(self, x, adj, i): # finite ntk

    #     x = F.relu(self.gc1(x, adj))
    #     # x = x * self.scale
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.gc2(x, adj)
    #     x = x * self.scale
    #     x = torch.spmm(adj, x)
    #     return torch.mean(x[i:i+1, :])

    def forward(self, x, adj, i): # basic

        x = F.relu(self.gc1(x, adj))
        x = x * self.scale
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)#[i:i+1, :]

    # def forward(self, x, adj, i): # no conv

    #     x = torch.spmm(adj, x)
    #     x = self.layer(x)
    #     x = torch.spmm(adj, x)

    #     return F.log_softmax(x, dim=1)#[i:i+1, :]


    # def forward(self, x, adj, i): # one block one agg
    #     x = F.relu(self.gc1(x, adj))
    #     x = torch.spmm(adj, x)
    #     x *= self.scale
    #     return F.log_softmax(x, dim=1)#[i:i+1, :]




