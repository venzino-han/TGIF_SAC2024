import pickle
import numpy as np
import pandas as pd

import dgl
import torch as th
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix

import config
from data_generator_utils import collate_data, get_subgraph_label 

class UserOutfitSubgraph(Dataset):
    def __init__(self, 
                 user_item_dict,
                 item_user_dict, 
                 
                 user_itemset_dict, 
                 itemset_user_dict, 

                 item_itemset_dict, 
                 itemset_item_dict, 
                 
                 user_item_df,
                 user_itemset_df,
                 itemset_item_df,
                 
                 itemset_item_query_df,):
        
        ''' 
        prepare for subgraph extraction
        itemset_item_query_df : main iteration pairs
        '''
        self.itemset_item_query_df = itemset_item_query_df
        
        self.user_item_dict = user_item_dict
        self.item_user_dict = item_user_dict
        self.itemset_item_dict = itemset_item_dict 
        self.item_itemset_dict = item_itemset_dict 
        self.user_itemset_dict = user_itemset_dict 
        self.itemset_user_dict = itemset_user_dict 

        self.graph = self._build_graph(user_item_df, user_itemset_df, itemset_item_df)

    def __len__(self):
        return len(self.itemset_item_query_df)
    
    def __getitem__(self, index):
        '''
        build subgraph from neigbors
        '''
        # print(self.itemset_item_query_df.loc[index])
        row = self.itemset_item_query_df.loc[index]
        itemset_id, item_id = row.itemset_id, row.item_id 
        label = row.answer

        u_neighbors = np.array(list(self.item_user_dict.get(item_id, set()))+list(self.itemset_user_dict.get(itemset_id, set())))
        s_neighbors = np.array(list(self.item_itemset_dict.get(item_id, set())))
        i_neighbors = np.array(list(self.itemset_item_dict.get(itemset_id, set())))

        i_neighbors = i_neighbors[i_neighbors!=item_id]
        s_neighbors = s_neighbors[s_neighbors!=itemset_id]

        subgraph = get_subgraph_label(graph = self.graph,
                                        u_node_idx=th.tensor([item_id]),
                                        i_node_idx=th.tensor([itemset_id]),
                                        u_neighbors=th.tensor(u_neighbors),
                                        i_neighbors=th.tensor(i_neighbors),
                                        s_neighbors=th.tensor(s_neighbors)
                                        )
                
        return subgraph, th.tensor(label, dtype=th.float32)
    
    def _build_graph(self, user_item_df, user_itemset_df, itemset_item_df):
        '''
        build entire graph (user, itemset, item)
        '''
        user_ids_item = user_item_df.user_id
        item_ids_user = user_item_df.item_id
        user_ids_itemset = user_itemset_df.user_id
        itemset_ids_user = user_itemset_df.itemset_id
        itemset_ids_item = itemset_item_df.itemset_id
        item_ids_itemset = itemset_item_df.item_id

        num_ui, num_us, num_si = len(user_item_df), len(user_itemset_df), len(itemset_item_df)

        src_nodes = np.concatenate((user_ids_item, item_ids_user, 
                                    user_ids_itemset, itemset_ids_user, 
                                    itemset_ids_item, item_ids_itemset,
                                    ))
        dst_nodes = np.concatenate((item_ids_user, user_ids_item, 
                                    itemset_ids_user, user_ids_itemset, 
                                    item_ids_itemset, itemset_ids_item,
                                    ))

        etypes = np.array([0]*num_ui*2 + [1]*num_us*2 + [2]*num_si*2)

        self.num_nodes = max(src_nodes)+1

        num_users = len(set(np.concatenate((user_ids_item, user_ids_itemset))))
        num_itemsets = len(set(np.concatenate((itemset_ids_user, itemset_ids_item))))
        num_items =  len(set(np.concatenate((item_ids_user, item_ids_itemset))))

        usi_matrix = coo_matrix((etypes, (src_nodes, dst_nodes)), shape=(self.num_nodes, self.num_nodes))
        
        # build graph 
        graph = dgl.from_scipy(sp_mat=usi_matrix, idtype=th.int32)

        graph.ndata['node_id'] = th.tensor(list(range(self.num_nodes)), dtype=th.int32)
        graph.ndata['ntype'] = th.tensor([0]*num_users + [1]*num_itemsets + [2]*num_items ,  dtype=th.int8)
        graph.edata['etype'] = th.tensor(etypes, dtype=th.int8)
        return graph


def get_task2_dataloader(data_path, batch_size, num_workers):
    with open('./processed_data/user_item_dict.pkl', 'rb') as f:
        user_item_dict = pickle.load(f)
    with open('./processed_data/item_user_dict.pkl', 'rb') as f:
        item_user_dict = pickle.load(f)
    with open('./processed_data/user_itemset_train_dict.pkl', 'rb') as f:
        user_itemset_dict = pickle.load(f)
    with open('./processed_data/itemset_user_train_dict.pkl', 'rb') as f:
        itemset_user_dict = pickle.load(f)
    with open('./processed_data/item_itemset_train_dict.pkl', 'rb') as f:
        item_itemset_dict = pickle.load(f)
    with open('./processed_data/itemset_item_train_dict.pkl', 'rb') as f:
        itemset_item_dict = pickle.load(f)

    user_item_df = pd.read_csv('./processed_data/user_item.csv')
    user_itemset_df = pd.read_csv('./processed_data/user_itemset_training.csv')
    itemset_item_df = pd.read_csv('./processed_data/itemset_item_training.csv')
    itemset_item_query_df = pd.read_csv('./processed_data/itemset_item_train_query.csv')
    
    
    user_outfit_subgraph_dataset = UserOutfitSubgraph( 
                                            user_item_dict,
                                            item_user_dict,                  
                                            user_itemset_dict, 
                                            itemset_user_dict, 
                                            item_itemset_dict, 
                                            itemset_item_dict,                  
                                            user_item_df,
                                            user_itemset_df,
                                            itemset_item_df,                 
                                            itemset_item_query_df,)

    train_dataloader = DataLoader(user_outfit_subgraph_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers,
                                  collate_fn=collate_data, 
                                  pin_memory=True
                                  )

    itemset_item_query_df = pd.read_csv('./processed_data/itemset_item_valid_query.csv')
    
    user_outfit_subgraph_dataset = UserOutfitSubgraph( 
                                            user_item_dict,
                                            item_user_dict,                  
                                            user_itemset_dict, 
                                            itemset_user_dict, 
                                            item_itemset_dict, 
                                            itemset_item_dict,                  
                                            user_item_df,
                                            user_itemset_df,
                                            itemset_item_df,                 
                                            itemset_item_query_df,)

    valid_dataloader = DataLoader(user_outfit_subgraph_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers,
                                  collate_fn=collate_data, 
                                  pin_memory=True
                                  )

    itemset_item_query_df = pd.read_csv('./processed_data/itemset_item_test_query.csv')
    
    user_outfit_subgraph_dataset = UserOutfitSubgraph( 
                                            user_item_dict,
                                            item_user_dict,                  
                                            user_itemset_dict, 
                                            itemset_user_dict, 
                                            item_itemset_dict, 
                                            itemset_item_dict,                  
                                            user_item_df,
                                            user_itemset_df,
                                            itemset_item_df,                 
                                            itemset_item_query_df,)

    test_dataloader = DataLoader(user_outfit_subgraph_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  num_workers=num_workers,
                                  collate_fn=collate_data, 
                                  pin_memory=True
                                  )
    

    return train_dataloader, valid_dataloader, test_dataloader


# if __name__=="__main__":
    # with open('./processed_data/user_item_dict.pkl', 'rb') as f:
    #     user_item_dict = pickle.load(f)
    # with open('./processed_data/item_user_dict.pkl', 'rb') as f:
    #     item_user_dict = pickle.load(f)
    # with open('./processed_data/user_itemset_training_dict.pkl', 'rb') as f:
    #     user_itemset_dict = pickle.load(f)
    # with open('./processed_data/itemset_user_training_dict.pkl', 'rb') as f:
    #     itemset_user_dict = pickle.load(f)
    # with open('./processed_data/item_itemset_valid_dict.pkl', 'rb') as f:
    #     item_itemset_dict = pickle.load(f)
    # with open('./processed_data/itemset_item_valid_dict.pkl', 'rb') as f:
    #     itemset_item_dict = pickle.load(f)

    # user_item_df = pd.read_csv('./processed_data/user_item.csv')
    # user_itemset_df = pd.read_csv('./processed_data/user_itemset_training.csv')
    # itemset_item_df = pd.read_csv('./processed_data/itemset_item_valid.csv')
    # itemset_item_query_df = pd.read_csv('./processed_data/user_itemset_train_query.csv')
    
    
    # user_outfit_subgraph_dataset = UserOutfitSubgraph( 
    #                                         user_item_dict,
    #                                         item_user_dict,                  
    #                                         user_itemset_dict, 
    #                                         itemset_user_dict, 
    #                                         item_itemset_dict, 
    #                                         itemset_item_dict,                  
    #                                         user_item_df,
    #                                         user_itemset_df,
    #                                         itemset_item_df,                 
    #                                         itemset_item_query_df,)
    # for subg, label in user_outfit_subgraph_dataset:
    #     print(subg)
    #     print(label)
    #     break

