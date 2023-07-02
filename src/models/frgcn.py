"""FRGCN modules"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv
from models.edge_drop import edge_drop


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class FRGCN(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.

    def __init__(
        self,
        in_feats,
        gconv=RelGraphConv,
        latent_dim=[32, 32, 32, 32],
        num_relations=3,
        num_bases=2,
        regression=False,
        edge_dropout=0.2,
        force_undirected=False,
        side_features=False,
        n_side_features=0,
        multiply_by=1,
    ):
        super(FRGCN, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        print(in_feats, latent_dim, num_relations, num_bases)
        self.convs.append(
            gconv(
                in_feats,
                latent_dim[0],
                num_relations,
                num_bases=num_bases,
                self_loop=True,
            )
        )
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(
                gconv(
                    latent_dim[i],
                    latent_dim[i + 1],
                    num_relations,
                    num_bases=num_bases,
                    self_loop=True,
                )
            )

        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
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
            x = th.tanh(
                conv(
                    graph,
                    x,
                    graph.edata["etype"],
                    norm=graph.edata["edge_mask"].unsqueeze(1),
                )
            )
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)

        users = graph.ndata["nlabel"][:, 0] == 1
        items = graph.ndata["nlabel"][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
