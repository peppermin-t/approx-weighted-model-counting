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
# import wandb

from data_analysis.utils import readCNF
from argparser import parsearg
from model import IndependentModel, HMM

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    ds_root = "benchmarks/altogether"
    
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
    if config['model'] == 'hmm':
        config['num_state'] = args.num_state
        model_cf += "(" + str(args.num_state) + ")"
    config_str = f"{config['file_name']}-{model_cf}-bs{config['batch_size']}-lr{config['learning_rate']}"

    # logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/" + config_str + ".log", mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # device & seed
    info = cpuinfo.get_cpu_info()
    logger.info(f"CPU: {info['brand_raw']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    modelpth = os.path.join(args.modelpth, config_str + ".pth")
    logger.info(f"Model path: {modelpth}")

    # wandb.init(project="approxWMC", config=config)
    # wandb.config.update(config)

    # Dataset
    with open(os.path.join(ds_root, config['ds_class'], config['file_name'])) as f:
        cnf, weights, _ = readCNF(f, mode=args.format)
    clscnt, varcnt = len(cnf), len(weights)
    with open(os.path.join(ds_root, config['ds_class'] + "_samples", config['file_name'] + ".npy"), "rb") as f:
        y = torch.from_numpy(np.load(f))[:config['sample_size'], ]

    ds = TensorDataset(y)
    train_ds, val_ds = random_split(ds, [0.8, 0.2])

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # model & optimiser
    if config['model'] == 'ind':
        model = IndependentModel(dim=clscnt).to(device)
    else:
        model = HMM(dim=clscnt, num_states=config['num_state']).to(device)

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
        for y_batch in train_loader:
            optimizer.zero_grad()
            nll = - model(y_batch[0].to(device))
            nll.backward()
            optimizer.step()
            train_loss += nll.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for y_batch in val_loader:
                nll = - model(y_batch[0].to(device))
                val_loss += nll.item()
        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # wandb.log({'Epoch': epoch, 'Train Loss': train_loss, 'Validation Loss': val_loss})

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), modelpth)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

        epoch += 1

    # approx WMC
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    with torch.no_grad():
        log_prob = model.log_p(torch.ones(clscnt).unsqueeze(0)).item()
    logger.info(f'Approx WMC: {math.exp(log_prob)}')

    # exact WMC
    with open(os.path.join(ds_root, "easy_logans.json")) as ans:
        exact_ans = json.load(ans)
    log_exact_prob = exact_ans[config['file_name']]
    logger.info(f'Exact WMC: {math.exp(log_exact_prob)}')
    
    # log sacle error
    loglogMAE = abs(log_prob - log_exact_prob)
    logger.info(f'log-log MAE: {loglogMAE}')

    # wandb.finish()
