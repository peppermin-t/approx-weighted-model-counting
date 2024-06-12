import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.bernoulli import Bernoulli
from utils import readCNF, parsearg, evalCNF
from model import IndependentModel, HMM
import math
import wandb
import os
import json

def sample_y(probs, cnf, size):
	dist_x = Bernoulli(probs[:, 0])
	x = dist_x.sample(torch.tensor([size]))
	return torch.from_numpy(evalCNF(cnf, x))

if __name__ == "__main__":
    args = parsearg()
    filename = args.filename.split("/")[-1]
    modelpth = os.path.join(args.modelpth, filename + ".pth")
    print(f"Model path: {modelpth}")
    torch.manual_seed(0)

    # wandb config
    config = {
        'sample_size': args.sample_size, 
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    wandb.init(project="approxWMC", config=config)
    wandb.config.update(config)

    # sample dataset
    with open(args.filename) as f:
        cnf, weights, _ = readCNF(f, mode=args.format)
    clscnt, varcnt = len(cnf), len(weights)
    probs = weights / weights.sum(axis=1, keepdims=True)
    y = sample_y(torch.from_numpy(probs), cnf, size=args.sample_size)
    
    # Dataset
    ds = TensorDataset(y)
    train_size = int(0.8 * len(y))
    val_size = len(y) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # model & optimiser
    if args.model == 'ind':
        model = IndependentModel(dim=clscnt)  # epoch < 1000
    else:
        model = HMM(dim=clscnt, num_states=4)  # epoch < 2000

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
            y_batch = y_batch[0]
            log_prob = model(y_batch)
            vloss = -log_prob
            vloss.backward()
            optimizer.step()
            train_loss += vloss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for y_batch in val_loader:
                y_batch = y_batch[0]
                log_prob = model(y_batch)
                vloss = -log_prob
                val_loss += vloss.item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        wandb.log({'Epoch': epoch, 'Train Loss': train_loss, 'Validation Loss': val_loss})

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
        prob = torch.exp(log_prob)
    print(f'Approx WMC: {prob}')

    # exact WMC
    with open("benchmarks/altogether/easy_logans.json") as ans:
        exact_ans = json.load(ans)
    log_exact_prob = exact_ans[filename]
    print(f'Exact WMC: {math.exp(log_exact_prob)}')
    
    # log sacle error
    loglogMAE = abs(log_prob - log_exact_prob)
    print(f'log-log MAE: {loglogMAE}')

    wandb.finish()
