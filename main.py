import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.bernoulli import Bernoulli
from utils import readCNF, parsearg, evalCNF
from model import IndependentModel, HMM
import math
# import wandb
import os
import json
import time
import numpy as np
import logging

# def sample_y(probs, cnf, size):  # Train loss: 104.1607, Val loss: 100.7948
# 	dist_x = Bernoulli(torch.from_numpy(probs))
# 	x = dist_x.sample(torch.tensor([size]))
# 	return torch.from_numpy(evalCNF(cnf, x.numpy()))

def sample_y(probs, cnf, size):
    x = np.random.binomial(1, probs, (size, len(probs)))
    return torch.from_numpy(evalCNF(cnf, x))

if __name__ == "__main__":
    args = parsearg()
    config = {
	    'sample_size': args.sample_size,
	    'batch_size': args.batch_size,
	    'learning_rate': args.lr,
	    'model': args.model
    }
    model_cf = args.model
    if config['model'] == 'hmm':
        config['num_state'] = args.num_state 
        model_cf += "(" + str(args.num_state) + ")"
    filename = args.filename.split("/")[-1]
    modelpth = os.path.join(args.modelpth, filename + ".pth")

    # logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{filename}-{model_cf}-bs{config['batch_size']}-lr{config['learning_rate']}.log", mode='w'),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    # device & seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    np.random.seed(0)

    logger.info(f"Model path: {modelpth}")

    # wandb.init(project="approxWMC", config=config)
    # wandb.config.update(config)

    # sample dataset
    with open(args.filename) as f:
        cnf, weights, _ = readCNF(f, mode=args.format)
    clscnt, varcnt = len(cnf), len(weights)
    probs = (weights / weights.sum(axis=1, keepdims=True))[:, 0]
    t0 = time.time()
    y = sample_y(probs, cnf, size=config['sample_size'])
    t1 = time.time()
    logger.info(f"Sample time: {t1 - t0:.2f}")
    
    # Dataset
    ds = TensorDataset(y)
    train_size = int(0.8 * len(y))
    val_size = len(y) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

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
            log_prob = model(y_batch[0].to(device))
            vloss = -log_prob
            vloss.backward()
            optimizer.step()
            train_loss += vloss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for y_batch in val_loader:
                log_prob = model(y_batch[0].to(device))
                vloss = -log_prob
                val_loss += vloss.item()
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
    with open("benchmarks/altogether/easy_logans.json") as ans:
        exact_ans = json.load(ans)
    log_exact_prob = exact_ans[filename]
    logger.info(f'Exact WMC: {math.exp(log_exact_prob)}')
    
    # log sacle error
    loglogMAE = abs(log_prob - log_exact_prob)
    logger.info(f'log-log MAE: {loglogMAE}')

    # wandb.finish()
