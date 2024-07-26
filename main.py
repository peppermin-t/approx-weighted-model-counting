import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.bernoulli import Bernoulli
import math
import os
import json
import numpy as np
import logging
import cpuinfo
import wandb

from data_analysis.utils import readCNF
from argparser import parsearg
from model import IndependentModel, HMM, inhHMM, HMMPC

import random


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    ds_root = "benchmarks/altogether"
    
    # configurations
    args = parsearg()
    config = {
        'ds_class': args.dsclass,
        'file_name': args.filename,

	    'sample_size': args.sample_size,
	    'batch_size': args.batch_size,
	    'learning_rate': args.lr,
	    'model': args.model
    }
    model_cf = args.model
    if config['model'] != 'ind':
        config['num_state'] = args.num_state
        model_cf += "(" + str(args.num_state) + ")"
    config_str = f"{config['file_name']}-{model_cf}-bs{config['batch_size']}-lr{config['learning_rate']}"

    # logger
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/" + config_str + ".log", mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logger.info(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    logger.info(f'Using device: {device}')

    # model path
    modelpth = os.path.join(args.modelpth, config_str + ".pth")
    logger.info(f"Model path: {modelpth}")

    # wandb config
    if not args.debug:
        wandb.init(project="approxWMC", config=config)
        wandb.config.update(config)

    # exact WMC
    exact_result_path = os.path.join(ds_root, "easy_logans.json")
    with open(exact_result_path) as ans:
        exact_ans = json.load(ans)
    log_exact_prob = exact_ans[config['file_name']]

    # Dataset
    # cnf_path = os.path.join(ds_root, config['ds_class'], config['file_name'])
    # with open(cnf_path) as f:
    #     cnf, weights, _ = readCNF(f, mode=args.format)
    # clscnt, varcnt = len(cnf), len(weights)
    sample_path = os.path.join(ds_root, config['ds_class'] + "_samples", config['file_name'] + ".npy")
    with open(sample_path, "rb") as f:
        y = torch.from_numpy(np.load(f))[:config['sample_size'], ]
    clscnt = y.shape[1]

    ds = TensorDataset(y)
    train_ds, val_ds = random_split(ds, [0.8, 0.2])

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # model & optimisers
    if config['model'] == 'ind':
        model = IndependentModel(dim=clscnt, device=device).to(device)
    elif config['model'] == 'hmm':
        model = HMM(dim=clscnt, device=device, num_states=config['num_state']).to(device)
    elif config['model'] == 'inh':
        model = inhHMM(dim=clscnt, device=device, num_states=config['num_state']).to(device)
    else:
        model = HMMPC(dim=clscnt, device=device, num_states=config['num_state']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

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
            if i == 0:
                logger.debug(f"log_likelihood: {lls}")
                logger.debug(f"NLL: {nll}")
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
            loglogMAE = abs(lls.item() - log_exact_prob)

            for batch in val_loader:
                batch = batch[0].to(device)
                lls = model(batch)
                nll = - torch.mean(lls)
                val_loss += nll.item()
        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, log-log MAE: {loglogMAE:.4f}')
        
        if not args.debug:
            wandb.log({'Epoch': epoch, 'Train Loss': train_loss, 'Validation Loss': val_loss, 'log-log MAE': loglogMAE})

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
    loglogMAE = abs(lls.item() - log_exact_prob)
    logger.info(f'log-log MAE: {loglogMAE}')

    if not args.debug:
        wandb.finish()
