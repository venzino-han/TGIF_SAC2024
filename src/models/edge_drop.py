import torch as th

# def edge_drop(graph, edge_dropout=0.2, training=True):
#     assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

#     if not training:
#         return graph

#     # set edge mask to zero in directional mode
#     src, _ = graph.edges()
#     to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
#     to_drop = th.bernoulli(to_drop).to(th.bool)
#     graph.edata['edge_mask'][to_drop] = 0

#     return graph

import dgl
from dgl import DropEdge

def edge_drop(graph, edge_dropout=0.0, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph
    transform = DropEdge(p=edge_dropout)
    graph = transform(graph)
    graph = dgl.add_self_loop(graph)
    return graph

