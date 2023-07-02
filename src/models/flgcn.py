"""FRGCN modules"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.LightGCN import LGCNLayer
from models.edge_drop import edge_drop


class FLGCN(nn.Module):

    def __init__(
        self,
        in_feats,
        latent_dim=[32, 32, 32, 32],
        edge_dropout=0.2,
    ):
        super(FLGCN, self).__init__()

        self.edge_dropout = edge_dropout

        self.convs = th.nn.ModuleList()
        # self.lins = th.nn.ModuleList()
        # self.lins.append(
        #     nn.Linear(latent_dim[0], latent_dim[1])
        # )
        self.convs.append(LGCNLayer())
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(LGCNLayer())

        self.lin1 = nn.Linear(2 * in_feats * len(latent_dim), 128)
        self.lin2 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph):
        graph = edge_drop(graph, self.edge_dropout,)

        concat_states = []
        x = graph.ndata["x"].type(
            th.float32
        )  # one hot feature to emb vector : this part fix errors

        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = conv(graph,x)
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)

        users = graph.ndata["nlabel"][:, 0] == 1
        items = graph.ndata["nlabel"][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        return x[:, 0]

    def __repr__(self):
        return self.__class__.__name__

