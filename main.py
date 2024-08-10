import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
import os
import json
import numpy as np
import logging
import wandb
import time
import pickle
import cpuinfo
import networkx as nx

from data_analysis.utils import readCNF, sample
from argparser import parsearg
from model import IndependentModel, HMM, inhHMM, HMMPC

import random


def calc_error(a, b, exp=False):
    return abs(math.exp(a) - math.exp(b)) if exp else abs(a - b)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    ds_root = "benchmarks/altogether"
    
    # configurations
    args = parsearg()
    config = vars(args)
    model_cf = args.model
    if config['model'] != 'ind':
        model_cf += "(" + str(args.num_state) + ")"
    if config['model'] == 'pchmm' and args.reordered:
        model_cf += "reordered"
    config_str = f"{config['file_name']}-{model_cf}-bs{config['batch_size']}-lr{config['lr']}"

    # logger
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("logs/" + config_str + ".log", mode='w'), logging.StreamHandler()])
    logger = logging.getLogger()

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Using device: {device}')
    logger.info(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

    # model path
    modelpth = os.path.join("models", config['ds_class'], config_str + ".pth")
    logger.info(f"Model path: {modelpth}")

    # wandb config
    if not args.debug and not args.wandb_deac:
        wandb.init(project="approxWMC", config=config)
        wandb.config.update(config)

    # exact WMC
    exact_respth = os.path.join(ds_root, "easy_logans.json")
    exact_ans = json.load(open(exact_respth))
    log_exact_prob = exact_ans[config['file_name']]

    # Dataset
    cnf_path = os.path.join(ds_root, config['ds_class'], config['file_name'])
    with open(cnf_path) as f:
        cnf, weights, _ = readCNF(f, mode=args.format)
    clscnt, varcnt = len(cnf), len(weights)

    logger.info(f"Start sampling...")
    t0 = time.time()
    y = sample(cnf, weights, sample_size=config['sample_size'], device=device)  # 512.26s for cpu
    t1 = time.time()
    logger.info(f"Sampling finished in {t1 - t0:.2f}")
    
    ds = TensorDataset(torch.from_numpy(y))
    train_ds, val_ds = random_split(ds, [0.8, 0.2])

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # model & optimisers
    if config['model'] == 'ind':
        model = IndependentModel(dim=clscnt, device=device)
    elif config['model'] == 'hmm':
        model = HMM(dim=clscnt, device=device, num_states=config['num_state'])
    elif config['model'] == 'inh':
        model = inhHMM(dim=clscnt, device=device, num_states=config['num_state'])
    else:
        order = None
        if config['reordered']:
            graphpth = os.path.join(ds_root, config['ds_class'] + "_primal_graphs", config['file_name'] + ".pkl")
            with open(graphpth, "rb") as f:
                G = pickle.load(f)
            order = list(nx.dfs_preorder_nodes(G, source=0))
        model = HMMPC(dim=clscnt, device=device, num_states=config['num_state'], order=order)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    epoch = 0
    # early stopping param
    patience = 5
    best_val_loss = float('inf')
    counter = 0

    # training loop
    while True:
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch[0].to(device)
            lls = model(batch)
            nll = - torch.mean(lls)
            nll.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += nll.item()
            
            if (i + 1) % 100 == 0:
                logger.info(f"Batch {i + 1} loss: {nll}")
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            alltrue = torch.ones((1, clscnt), device=device)
            lls = model(alltrue)
            loglogMAE = calc_error(lls.item(), log_exact_prob)
            MAE = calc_error(lls.item(), log_exact_prob)

            for batch in val_loader:
                batch = batch[0].to(device)
                lls = model(batch)
                nll = - torch.mean(lls)
                val_loss += nll.item()
        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, log-log MAE: {loglogMAE:.4f}')
        
        if not args.debug and not args.wandb_deac:
            wandb.log({'Epoch': epoch, 'Train Loss': train_loss, 'Validation Loss': val_loss, 'log-log MAE': loglogMAE, 'MAE': MAE})

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), modelpth)
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping")
                break

        epoch += 1

    # approx WMC
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    with torch.no_grad():
        alltrue = torch.ones((1, clscnt), device=device)
        lls = model(alltrue)

    logger.info(f'Approx WMC: {math.exp(lls.item())}')
    logger.info(f'Exact WMC: {math.exp(log_exact_prob)}')

    # log sacle error
    loglogMAE = calc_error(lls.item(), log_exact_prob)
    logger.info(f'log-log MAE: {loglogMAE}')

    if not args.debug and not args.wandb_deac:
        wandb.finish()
