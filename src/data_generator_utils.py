# import sys
# sys.path.append()

from typing import Tuple
import dgl
import torch as th


def one_hot(idx, length):
    x = th.zeros([len(idx), length], dtype=th.float32)
    x[th.arange(len(idx)), idx] = 1.0
    return x


def normalize_timestamp(timestamps, standard_ts):
    timestamps[timestamps == -1.0] = standard_ts
    timestamps = abs(timestamps - standard_ts)
    timestamps = 1 - (
        (timestamps - th.min(timestamps))
        / ((th.max(timestamps) - th.min(timestamps) + 1e-9))
    )
    return timestamps


#######################
# Subgraph Extraction
#######################

def get_subgraph_label(
    graph: dgl.graph,
    u_node_idx: th.tensor,
    i_node_idx: th.tensor,
    u_neighbors: th.tensor,
    i_neighbors: th.tensor,
    s_neighbors: th.tensor,
    ) -> dgl.graph:

    nodes = th.cat(
        [
            u_node_idx,
            i_node_idx,
            u_neighbors,
            i_neighbors,
            s_neighbors,
        ],
        dim=0,
    )
    nodes = nodes.type(th.int32)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True)

    node_labels = (
        [0, 1]
        + [2] * len(u_neighbors)
        + [3] * len(i_neighbors)
        + [4] * len(s_neighbors)
    )

    subgraph.ndata["ntype"] = th.tensor(node_labels, dtype=th.int8)
    subgraph.ndata["nlabel"] = one_hot(node_labels, 5)
    subgraph.ndata["x"] = subgraph.ndata["nlabel"]

    # set edge mask to zero as to remove links between target nodes in training process
    subgraph.edata["edge_mask"] = th.ones(subgraph.number_of_edges(), dtype=th.float32)

    try:
        target_edges = subgraph.edge_ids([0, 1], [1, 0], return_uv=False)
        subgraph.remove_edges(target_edges)
    except:
        pass
    
    subgraph.edata["efeat"] = one_hot(subgraph.edata['etype'].tolist(), 3)
    subgraph = dgl.add_self_loop(subgraph)

    return subgraph

def dedup_edges(srcs, dsts) -> Tuple[list, list]:
    dedup_srcs, dedup_dsts = [], []
    edge_set = set()
    for s, d in zip(srcs, dsts):
        if (s, d) in edge_set:
            continue
        edge_set.add((s, d))
        dedup_srcs.append(s)
        dedup_dsts.append(d)

    return dedup_srcs, dedup_dsts

def collate_data(data):
    g_list, label_list = map(list, zip(*data))
    g = dgl.batch(g_list)
    g_label = th.stack(label_list)
    return g, g_label
