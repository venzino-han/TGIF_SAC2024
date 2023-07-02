import copy, math
import pickle as pkl

import time

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

from prettytable import PrettyTable

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_generator_task1 import get_task1_dataloader
from torch import optim

from utils import get_args_from_yaml, get_logger


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# TODO Task 2 evaluation method --> top 100
def get_rank(iids, answer_iid):
    if answer_iid not in set(iids):
        return 101
    else:
        rank = iids.index(answer_iid) +1
        return rank 

task2_valid_query_df = pd.read_csv('./processed_data/itemset_item_valid_query.csv')
task2_valid_answer_df = task2_valid_query_df.query('answer==1')

def evaluate_task2(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        # labels = batch[1].to(device)
        # val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())

    itemset_item_answer_dict={k:v for k,v in zip(task2_valid_answer_df.itemset_id,task2_valid_answer_df.item_id)}

    preds_df = pd.DataFrame({
        'itemset_id': task2_valid_query_df.itemset_id,
        'item_id': task2_valid_query_df.item_id,
        'score': val_preds,
    })
    accs = []
    ranks = []
    for itemset_id, sub_df in preds_df.groupby('itemset_id'):
        sub_df = sub_df.sort_values('score', ascending=False)
        result = sub_df.item_id[:100]
        answer_iid = itemset_item_answer_dict[itemset_id]
        rank = get_rank(list(result), answer_iid)
        ranks.append(rank)
        if rank == 101:
            accs.append(0)
        else:
            accs.append(1)
    val_rank = np.mean(ranks)
    val_acc = np.mean(accs)
    return None, val_rank, val_acc

def evaluate(model, loader, device):
    # Evaluate AUC, ACC
    model.eval()
    val_labels = []
    val_preds = []
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        val_labels.extend(labels.cpu().tolist())
        val_preds.extend(preds.cpu().tolist())

    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    # val_f1 = f1_score(list(map(round,val_labels)), list(map(round,val_preds)))
    return val_auc, None, val_acc


def train_epoch(model, optimizer, loader, device, logger, log_interval):
    model.train()

    epoch_loss = 0.0
    iter_loss = 0.0
    iter_mse = 0.0
    iter_cnt = 0
    iter_dur = []
    mse_loss_fn = nn.MSELoss().to(device)
    # bce_loss_fn = nn.BCELoss().to(device)

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(inputs)
        # print(preds[:8], labels[:8])
        
        optimizer.zero_grad()
        loss = mse_loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(
                f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}"
            )
            iter_loss = 0.0
            iter_mse = 0.0
            iter_cnt = 0
            if iter_idx == 500:
                break

    return epoch_loss / len(loader.dataset)


def train(args: EasyDict, train_loader, test_loader, logger):
    th.manual_seed(0)
    np.random.seed(0)
    # dgl.random.seed(0)

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
            edge_dropout=args.edge_dropout,
        ).to(args.device)

    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))

    optimizer = optim.Adam(
        model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    logger.info("Loading network finished ...\n")

    count_parameters(model)

    best_epoch = 0
    best_auc, best_rank, best_acc = 0, 1000, 0

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    epochs = list(range(1, args.train_epochs + 1))

    eval_func_map = {
        "task1": evaluate,
        "task2": evaluate_task2,
    }
    eval_func = eval_func_map.get(args.dataset, evaluate)
    for epoch_idx in epochs:
        logger.debug(f"Epoch : {epoch_idx}")

        train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            args.device,
            logger,
            log_interval=args.log_interval,
        )
        # train_loss = 0
        test_auc, test_rank, test_acc = eval_func(model, test_loader, args.device)
        if test_auc is None:
            test_auc = -1
        if test_rank is None:
            test_rank = -1
        eval_info = {
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "test_auc": test_auc,
            "test_rank": test_rank,
            "test_acc": test_acc,
        }
        logger.info(
            "=== Epoch {}, train loss {:.4f}, test auc {:.4f}, test rank {:.4f}, test acc {:.4f} ===".format(
                *eval_info.values()
            )
        )

        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param["lr"] = args.lr_decay_factor * param["lr"]
            print("lr : ", param["lr"])

        if test_rank == -1 :
            if best_auc < test_auc:
                logger.info(f'new best test auc {test_auc:.4f} acc {test_acc:.4f} ===')
                best_epoch = epoch_idx
                best_auc = test_auc
                best_acc = test_acc
                best_lr = args.train_lr
                best_state = copy.deepcopy(model.state_dict())
        else:
            if best_rank > test_rank:
                logger.info(f'new best test rank {test_rank:.4f} acc {test_acc:.4f} ===')
                best_epoch = epoch_idx
                best_rank = test_rank
                best_acc = test_acc
                best_lr = args.train_lr
                best_state = copy.deepcopy(model.state_dict())

    th.save(best_state, f'./parameters/{args.key}_{args.dataset}_{best_auc:.4f}.pt')
    logger.info(f"Training ends. The best testing auc {best_auc:.4f} acc {best_acc:.4f} at epoch {best_epoch}")
    return best_auc, best_acc, best_lr


import yaml
from data_generator_task1 import get_task1_dataloader
from data_generator_task2 import get_task2_dataloader

DATALOADER_MAP = {
    "task1": get_task1_dataloader,
    "task2": get_task2_dataloader,
}


def main():
    with open("./train_configs/train_list.yaml") as f:
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
        best_auc_acc_lr_list = []

        dataloader_manager = DATALOADER_MAP.get(sub_args.dataset)

        train_loader, valid_loader, _ = dataloader_manager(
            data_path=sub_args.dataset,
            batch_size=sub_args.batch_size,
            num_workers=config.NUM_WORKER,
        )

        for lr in args.train_lrs:
            sub_args["train_lr"] = lr
            best_auc_acc_lr = train(sub_args, train_loader, valid_loader, logger=logger)
            best_auc_acc_lr_list.append(best_auc_acc_lr)
        
        best_auc, best_acc, best_lr = max(best_auc_acc_lr_list, key = lambda x: x[0])
        best_auc_list = [x[0] for x in best_auc_acc_lr_list]
        best_acc_list = [x[1] for x in best_auc_acc_lr_list]
        if sub_args.dataset == 'task2':
            logger.info(f"**********The final best testing RANK {best_auc:.4f} ACC {best_acc:.4f} at lr {best_lr}********")
            logger.info(f"**********The mean testing RANK {np.mean(best_auc_list):.4f}, {np.std(best_auc_list):.4f} ********")
        else:
            logger.info(f"**********The final best testing AUC {best_auc:.4f} ACC {best_acc:.4f} at lr {best_lr}********")
            logger.info(f"**********The mean testing AUC {np.mean(best_auc_list):.4f}, {np.std(best_auc_list):.4f} ********")
        logger.info(f"**********The mean testing ACC {np.mean(best_acc_list):.4f}, {np.std(best_acc_list):.4f} ********")
 


if __name__ == "__main__":
    main()
