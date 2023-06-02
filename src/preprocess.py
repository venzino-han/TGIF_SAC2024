"""
pre-processing original data
reset id : user_id --> itemset_id --> item_id
"""


import os
import pickle
import time
import random
from copy import copy

import pandas as pd


def group_seq(df, groupby_key, cols, save_path):
    # build group : user_id(item_id) - seq
    # save as file
    cols = copy(cols)
    cols.remove(groupby_key)
    print(cols)
    group = df.groupby(groupby_key).apply(
        lambda df: tuple([df[c].values for c in cols])
    )
    with open(save_path, "wb") as pick:
        pickle.dump(group, pick)
    del group, df
    return


def get_group_dict(df, key_col, val_col):
    group_dict = dict()
    for key, group_df in df.groupby(key_col):
        group_dict[key] = set(group_df[val_col])
    return group_dict


def save_pkl(obj, path):
    with open(path, "wb") as pick:
        pickle.dump(obj, pick)


if __name__ == "__main__":
    # folder path
    dir_path = "./dataset/"
    res = os.listdir(dir_path)
    df_dicts = {}
    cols_dict = {
        "user_item.csv": ["user_id", "item_id"],
        
        "user_itemset_training.csv": ["user_id", "itemset_id"],
        "user_itemset_valid_query.csv": ["user_id", "itemset_id"],
        "user_itemset_valid_answer.csv": ["answer"],
        "user_itemset_test_query.csv": ["user_id", "itemset_id"],
        # "user_itemset_test_answer.csv": ["answer"],

        "itemset_item_training.csv": ["itemset_id", "item_id"],
        "itemset_item_valid_query.csv": ["itemset_id", "item_id"],
        "itemset_item_valid_answer.csv": ["itemset_id", "item_id"],
        "itemset_item_test_query.csv": ["itemset_id", "item_id"],
        # "itemset_item_test_answer.csv": ["itemset_id", "item_id"],
    }

    for file in res:
        cols = cols_dict.get(file)
        if cols is None:
            print(file)
        df = pd.read_csv(dir_path + file, header=None, names=cols)
        df_dicts[file] = df
    #     df.to_csv("./processed_data/" + file, index=False)

    # get number of user, item, itemset
    df = df_dicts.get("user_item.csv")
    number_user = max(df.user_id) + 1
    number_item = max(df.item_id) + 1

    df = df_dicts.get("user_itemset_training.csv")
    number_user = max(max(df.user_id) + 1, number_user)
    number_itemset = max(df.itemset_id) + 1

    print('number_item :', number_item)
    print('number_itemset :', number_itemset)
    print('number_user :', number_user)


    # get user-item group dict
    file = "user_item.csv"
    df = df_dicts.get(file)
    df.item_id += number_user + number_itemset
    df.to_csv(f"./processed_data/{file}", index=False)
    user_items_dict = get_group_dict(df, "user_id", "item_id")
    item_users_dict = get_group_dict(df, "item_id", "user_id")
    save_pkl(user_items_dict, "./processed_data/user_item_dict.pkl")
    save_pkl(item_users_dict, "./processed_data/item_user_dict.pkl")


    ''' TASK 1 '''
    '''get user-itemset group dict'''
    file = "user_itemset_training.csv"
    train_df = df_dicts.get(file)
    train_df.itemset_id += number_user
    train_df['answer'] = 1
    train_df.to_csv(f"./processed_data/{file}", index=False)
    user_itemsets_dict = get_group_dict(train_df, "user_id", "itemset_id")
    itemset_users_dict = get_group_dict(train_df, "itemset_id", "user_id")
    save_pkl(user_itemsets_dict, "./processed_data/user_itemset_train_dict.pkl")
    save_pkl(itemset_users_dict, "./processed_data/itemset_user_train_dict.pkl")

    file = "user_itemset_valid_query.csv"
    valid_df = df_dicts.get(file)
    df_ans = df_dicts.get("user_itemset_valid_answer.csv")
    valid_df['answer'] = df_ans.answer
    valid_df.itemset_id += number_user
    valid_df.to_csv(f"./processed_data/{file}", index=False)

    file = "user_itemset_test_query.csv"
    test_df = df_dicts.get(file)
    test_df.itemset_id += number_user
    test_df['answer'] = -1
    test_df.to_csv(f"./processed_data/{file}", index=False)

    ''' Generate negative set (random pair not in train, valid, test) '''
    df = pd.concat([train_df,valid_df,test_df])
    print(df)
    existing_user_itemset_pair_set = set(list(zip(df.user_id, df.itemset_id)))
    user_ids = list(range(number_user))
    itemset_ids = list(range(number_user, number_user+number_itemset))
    negative_pair_count = 0
    negative_pair_uids = list()
    negative_pair_isids = list()
    while negative_pair_count < 1343768:
        uid = random.sample(user_ids, 1)[0]
        isid = random.sample(itemset_ids, 1)[0]
        if (uid,isid) not in existing_user_itemset_pair_set:
            negative_pair_uids.append(uid)
            negative_pair_isids.append(isid)
            existing_user_itemset_pair_set.add((uid,isid))
            negative_pair_count += 1
    
    negative_train_df = pd.DataFrame(
        {
            'user_id':negative_pair_uids,
            'itemset_id':negative_pair_isids,
        }
    )

    negative_train_df['answer'] = 0

    train_df = pd.concat([train_df, negative_train_df])
    print('train_df with negative length : ', len(train_df))
    train_df.to_csv('./processed_data/user_itemset_train_query.csv')
    

    ''' TASK 2 '''
    ''' Train '''
    NEGATIVE_SAMPLES_PER_ITEMSET = 20
    # get item-itemset group dict
    file = "itemset_item_training.csv"
    train_df = df_dicts.get(file)
    print(file, len(train_df))
    print('itemset in train df', len(set(train_df.itemset_id)))
    train_df.itemset_id += number_user
    train_df.item_id += number_user + number_itemset
    train_df['answer'] = 1

    item_itemsets_dict = get_group_dict(train_df, "item_id", "itemset_id")
    itemset_items_dict = get_group_dict(train_df, "itemset_id", "item_id")

    ''' Generate negtaive query pairs for training '''
    itemset_ids = list(range(number_user, number_user+number_itemset))
    item_ids = list(range(number_user+number_itemset, number_user+number_itemset+number_item))

    query_pair_iids = list()
    query_pair_isids = list()
    query_pair_answer = list()
    for isid in itemset_ids:
        negative_pair_count = 0
        existing_item_set = itemset_items_dict.get(isid, set())
        if len(existing_item_set) == 0:
            continue
        pos_sample_item = random.choice(list(existing_item_set))
        query_pair_isids.append(isid)
        query_pair_iids.append(pos_sample_item)
        query_pair_answer.append(1)

        #remove from dict 
        item_set = item_itemsets_dict.get(pos_sample_item)
        itemset_set = itemset_items_dict.get(isid)
        item_set.remove(isid)
        itemset_set.remove(pos_sample_item)
        item_itemsets_dict[pos_sample_item] = item_set
        itemset_items_dict[isid] = itemset_set
        
        while negative_pair_count < NEGATIVE_SAMPLES_PER_ITEMSET:
            iid = random.sample(item_ids, 1)[0]
            if iid not in existing_item_set:
                query_pair_isids.append(isid)
                query_pair_iids.append(iid)
                query_pair_answer.append(0)
                existing_item_set.add(iid)
                negative_pair_count += 1

    save_pkl(item_itemsets_dict, "./processed_data/item_itemset_train_dict.pkl")
    save_pkl(itemset_items_dict, "./processed_data/itemset_item_train_dict.pkl")


    query_train_df = pd.DataFrame(
        {
            'itemset_id':query_pair_isids,
            'item_id':query_pair_iids,
            'answer':query_pair_answer,
        }
    )

    print('train_df with negative length : ', len(query_train_df))
    query_train_df.to_csv("./processed_data/itemset_item_train_query.csv", index=False)


    ''' Valid '''
    file = "itemset_item_valid_query.csv"
    valid_df = df_dicts.get(file)
    valid_df['answer'] = 1
    valid_df_ans = df_dicts.get("itemset_item_valid_answer.csv")
    valid_df_ans['answer'] = 1

    valid_df.itemset_id += number_user
    valid_df.item_id += number_user + number_itemset
    valid_df_ans.itemset_id += number_user
    valid_df_ans.item_id += number_user + number_itemset

    ''' 
    entire graph for valid query
    only contains positive pairs (train + valid)
    '''
    valid_graph_df = pd.concat([train_df, valid_df])
    valid_graph_df.to_csv('./processed_data/itemset_item_valid.csv')
    item_itemsets_dict = get_group_dict(valid_graph_df, "item_id", "itemset_id")
    itemset_items_dict = get_group_dict(valid_graph_df, "itemset_id", "item_id")
    save_pkl(item_itemsets_dict, "./processed_data/item_itemset_valid_dict.pkl")
    save_pkl(itemset_items_dict, "./processed_data/itemset_item_valid_dict.pkl")

    
    ''' Generate negtaive query pairs for validation '''
    NEGATIVE_SAMPLES_PER_ITEMSET = 1000
    itemset_ids = set(valid_df_ans.itemset_id)
    query_pair_iids = list()
    query_pair_isids = list()
    query_pair_answer = list()
    for isid, iid in zip(valid_df_ans.itemset_id, valid_df_ans.item_id):
        itemset_items_dict[isid].add(iid)
        query_pair_isids.append(isid)
        query_pair_iids.append(iid)
        query_pair_answer.append(1)

    for isid in itemset_ids:
        existing_item_set = itemset_items_dict.get(isid, set())
        iids = random.sample(item_ids, NEGATIVE_SAMPLES_PER_ITEMSET)
        for iid in iids:
            if iid not in existing_item_set:
                query_pair_isids.append(isid)
                query_pair_iids.append(iid)
                query_pair_answer.append(0)
                existing_item_set.add(iid)
    
    valid_query_df = pd.DataFrame(
        {
            'itemset_id':query_pair_isids,
            'item_id':query_pair_iids,
            'answer':query_pair_answer,
        }
    )
    print('valid query df len: ',len(valid_query_df))
    valid_query_df.to_csv("./processed_data/itemset_item_valid_query.csv", index=False)


    ''' Test '''

    file = "itemset_item_test_query.csv"
    test_df = df_dicts.get(file)
    test_df.itemset_id += number_user
    test_df.item_id += number_user + number_itemset

    ''' 
    entire graph for valid query
    only contains positive pairs (train + valid)
    '''
    test_graph_df = pd.concat([train_df, valid_df, test_df])
    test_graph_df.to_csv('./processed_data/itemset_item_test.csv')
    item_itemsets_dict = get_group_dict(test_graph_df, "item_id", "itemset_id")
    itemset_items_dict = get_group_dict(test_graph_df, "itemset_id", "item_id")
    save_pkl(item_itemsets_dict, "./processed_data/item_itemset_test_dict.pkl")
    save_pkl(itemset_items_dict, "./processed_data/itemset_item_test_dict.pkl")
    
    ''' Generate candidate query pairs for test '''
    itemset_ids = set(test_df.itemset_id)
    query_pair_iids = list()
    query_pair_isids = list()
    query_pair_answer = list()

    for isid in itemset_ids:
        existing_item_set = itemset_items_dict.get(isid, set())
        # for iid in existing_item_set:
        #     isid_set = item_itemsets_dict.get(iid,set())
        #     for isid_ in isid_set:
        #         if isid_ != isid:

        # iids = random.sample(item_ids, 8000)
        for iid in item_ids:
            if iid not in existing_item_set:
                query_pair_isids.append(isid)
                query_pair_iids.append(iid)
                query_pair_answer.append(-1)
                existing_item_set.add(iid)
    
    test_query_df = pd.DataFrame(
        {
            'itemset_id':query_pair_isids,
            'item_id':query_pair_iids,
            'answer':query_pair_answer,
        }
    )
    print('test query df len: ',len(test_query_df))
    test_query_df.to_csv("./processed_data/itemset_item_test_query.csv", index=False)


    print('itemset id start from', number_user)
    print('item id start from', number_user + number_itemset)
