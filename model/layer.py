import torch
import math
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import pdb





class HyperGraphAttentionLayerSparse(nn.Module):
    """
    The transformer message passing proposed by the EMNLP 2020 paper "be more with Less: Hypergraph Attention Networks forInductive Text Classification "
    """

    def __init__(self, in_features, out_features, node_type, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.node_type = node_type

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(
                self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(
            self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(
            self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # node_type num, 64
        # num_etypes = torch.max(self.node_type[:,1]) +1
        # self.edge_emb = nn.Embedding(num_etypes, self.in_features)
        self.weight_e = Parameter(torch.Tensor(
            self.in_features, self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        self.weight_e.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        """
        x: input features
        adj: input incidence matrix
        """
        # pdb.set_trace()
        if len(x.shape) == 2:
            x = x[None, :]
        if len(adj.shape) == 2:
            adj = adj.to_dense()
            adj = adj[None, :]
            # adj =adj.to_sparse()
        # print("x.shape: ",x.shape) # torch.Size([8, 127, 300]) (Batch_size, Node_num, feature_diem)
        # print("adj.shape: ",adj.shape) #torch.Size([8, 63, 127]) (Batch_size, hyperedge, Node_num)
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        # incidence mat #.shape (8,63,127)
        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t()  # (3,21334)
        node_pair = adj[0].nonzero().t()
        x1 = x_4att[0][node_pair[1]]
        q1 = self.word_context.weight[0:].view(
            1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)  # (21334,64)

        ############ Get edge_type embedding for each node ####################
        # e_feature = self.edge_emb(self.node_type[node_pair[1]][:,1])
        # e_4att = e_feature.matmul(self.weight_e)

        ########### Add edge embedding into node attention ####################
        # edge = torch.matmul(adj.float(), x)
        # edge_4att = edge.matmul(self.weight3)
        # q1 = edge_4att[0][node_pair[0]]

        pair_h = torch.cat((q1, x1), dim=-1)  # 200M
        # self.a.size = (2*out_features, 1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout,
                           training=self.training)  # raw attention
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size(
            [x.shape[0], N1, N2])).to_dense()  # 1.5G
        zero_vec = -9e15*torch.ones_like(e)

        # Get edge attention and representation
        attention = torch.where(adj > 0, 1, -9e15)
        attention_edge = F.softmax(attention, dim=2)

        # attention_edge = F.dropout( attention_edge, self.dropout, training=self.training)
        edge = torch.matmul(attention_edge, x)  # Get edge feature
        edge = F.dropout(edge, self.dropout, training=self.training)

        # Agg edge representations to node representation
        edge_4att = edge.matmul(self.weight3)  # (1,3025,64)
        y1 = edge_4att[0][node_pair[0]]  # (21334,64)
        x1 = x_4att[0][node_pair[1]]  # (21334,64)

        pair_h = torch.cat((x1, y1), dim=-1)  # (21334, 128)
        # self.a2.size = (2*out_features, 1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(
            pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()
        # zero_vec = -9e15*torch.ones_like(e)
        # Get node attention and representation, can be changed to one
        attention = torch.where(adj > 0, e,  -9e15*torch.ones_like(e))
        attention_node = F.softmax(attention.transpose(1, 2), dim=2)

        node = torch.matmul(attention_node, edge)

        if self.concat:
            node = F.leaky_relu(node)

        return F.normalize(node, p=2, dim=1), attention_edge
        # logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


