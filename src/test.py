import copy, math
import pickle as pkl
import time
from tqdm import tqdm

import config

import dgl
import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn

from easydict import EasyDict
from models.frgcn import FRGCN
from models.fgat import FGAT
from models.flgcn import FLGCN

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import optim

from utils import get_args_from_yaml, get_logger
from train import get_rank


task2_valid_query_df = pd.read_csv('./processed_data_task2/itemset_item_valid_query.csv')
task2_valid_answer_df = task2_valid_query_df.query('answer==1')
task2_test_query_df = pd.read_csv('./processed_data_task2/itemset_item_test_query.csv')
task2_test_answer_df = task2_test_query_df.query('answer==1')

def evaluate_task2(model, valid_loader, test_loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_preds = []
    for batch in tqdm(valid_loader):
        with th.no_grad():
            preds = model(batch[0].to(device))
        val_preds.extend(preds.cpu().tolist())

    itemset_item_answer_dict={k:v for k,v in zip(task2_valid_answer_df.itemset_id, task2_valid_answer_df.item_id)}

    preds_df = pd.DataFrame({
        'itemset_id': task2_valid_query_df.itemset_id,
        'item_id': task2_valid_query_df.item_id,
        'score': val_preds,
    })
    accs = []
    hit = []
    for itemset_id, sub_df in preds_df.groupby('itemset_id'):
        sub_df = sub_df.sort_values('score', ascending=False)
        pred_iid = list(sub_df.item_id)[0]
        answer_iid = itemset_item_answer_dict[itemset_id]
        if pred_iid != answer_iid:
            accs.append(0)
            hit.append(0)
        else:
            accs.append(1)
            hit.append(1)
    val_result = np.mean(hit)
    val_acc = np.mean(accs)
    print('val hit@3: ', val_result)


    ''' TEST '''
    test_preds = []
    for batch in tqdm(test_loader):
        with th.no_grad():
            preds = model(batch[0].to(device))
        test_preds.extend(preds.cpu().tolist())

    itemset_item_answer_dict={k:v for k,v in zip(task2_test_answer_df.itemset_id, task2_test_answer_df.item_id)}

    preds_df = pd.DataFrame({
        'itemset_id': task2_test_query_df.itemset_id,
        'item_id': task2_test_query_df.item_id,
        'score': test_preds,
    })
    accs = []
    hit = []
    for itemset_id, sub_df in preds_df.groupby('itemset_id'):
        sub_df = sub_df.sort_values('score', ascending=False)
        pred_iid = list(sub_df.item_id)[0]
        answer_iid = itemset_item_answer_dict[itemset_id]
        if pred_iid != answer_iid:
            accs.append(0)
            hit.append(0)
        else:
            accs.append(1)
            hit.append(1)
    test_result = np.mean(hit)
    test_acc = np.mean(accs)
    print('test hit@3: ', test_result)

    return val_result, val_acc, test_preds


task1_test_query_df = pd.read_csv('./processed_data/user_itemset_test_query.csv')

def evaluate_task1(model, valid_loader, test_loader, device):
    # Evaluate AUC, ACC
    model.eval()
    # val_labels = []
    # val_preds = []
    # graphs = []
    # for batch in tqdm(valid_loader):
    #     with th.no_grad():
    #         preds = model(batch[0].to(device))

    #     # graphs.append(batch[0].cpu())
    #     labels = batch[1]
    #     val_labels.extend(labels.cpu().tolist())
    #     val_preds.extend(preds.cpu().tolist())

    # val_result = roc_auc_score(val_labels, val_preds)
    # val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # print('val AUC: ', val_result)
    # print('val ACC: ', val_acc)

    test_preds = []
    test_labels = []
    for batch in tqdm(test_loader):
        with th.no_grad():
            preds = model(batch[0].to(device))
        test_preds.extend(preds.cpu().tolist())
        labels = batch[1]
        test_labels.extend(labels.cpu().tolist())

    test_result = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(list(map(round, test_labels)), list(map(round, test_preds)))
    print('test AUC: ', test_result)
    print('test ACC: ', test_acc)

    return None, None, test_preds


def test(args: EasyDict):
    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    ### prepare data and set model
    in_feats = config.IN_FEATS
    if args.model_type == "FRGCN":
        model = FRGCN(
            in_feats=in_feats,
            latent_dim=args.latent_dims,
            num_relations=args.num_relations,
            num_bases=4,
            regression=True,
            edge_dropout=args.edge_dropout,
        ).to(args.device)

    if args.model_type == "FGAT":
        model = FGAT(
            in_nfeats=in_feats,
            in_efeats=args.num_relations,
            latent_dim=args.latent_dims,
            edge_dropout=args.edge_dropout,
        ).to(args.device)

    if args.model_type == "FLGCN":
        model = FLGCN(
            in_feats=in_feats,
            latent_dim=args.latent_dims,
        ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))

    dataloader_manager = DATALOADER_MAP.get(args.dataset)
    _, valid_loader, test_loader = dataloader_manager(
        data_path=args.dataset,
        batch_size=512,
        num_workers=config.NUM_WORKER,
    )

    evaluate = EVALUATE_MAP.get(args.dataset)
    val_result, val_acc, test_preds= evaluate(
        model, valid_loader, test_loader, args.device
    )

    return val_result, val_acc, test_preds



import yaml
from data_generator_task1 import get_task1_dataloader
from data_generator_task2 import get_task2_dataloader

DATALOADER_MAP = {
    "task1": get_task1_dataloader,
    "task2": get_task2_dataloader,
}

EVALUATE_MAP = {
    "task1": evaluate_task1,
    "task2": evaluate_task2,
}



def main():
    with open("./test_configs/test_list.yaml") as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files["files"]
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info("train args")
        for k, v in args.items():
            logger.info(f"{k}: {v}")

        best_lr = None
        sub_args = args
        best_auc_list = []

        val_result, val_acc, test_preds = test(sub_args)

        print('val_result', val_result)
        print('val_acc', val_acc)


if __name__ == "__main__":
    main()
