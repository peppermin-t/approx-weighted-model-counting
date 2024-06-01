import torch
import torch.optim as optim
from utils import readCNF, parsearg
from model import IndependentModel, HMM
from exact_wmc import compute_exact_WMC_from_file

import wandb

args = parsearg()

config = {
    'num_epochs': args.num_epoch,
    'batch_size': args.batch,
    'learning_rate': args.lr
}
wandb.init(project="approxWMC", config=config)
wandb.config.update(config)

with open(args.filename) as f: 
    cnf, weights, _ = readCNF(f, mode=args.format)
clscnt, varcnt = len(cnf), len(weights)
probs = weights / weights.sum(axis=1, keepdims=True)
print(f'Exact WMC: {compute_exact_WMC_from_file(args.filename, probs)}')

torch.manual_seed(0)
if args.model == 'ind':
    model = IndependentModel(dim=clscnt, cnf=cnf)  # epoch < 1000
else:
    model = HMM(dim=clscnt, cnf=cnf, num_states=4)  # epoch < 2000

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print(trainable_params)
# losses = []
# k = 1.4
# eps = varcnt * 0.1 * (trainable_params / 30) ** k  # variance?
# csc_epochs = 500

# training loop
for epoch in range(config['num_epochs']):
    model.train()
    optimizer.zero_grad()

    log_prob = model(torch.from_numpy(probs), config['batch_size'])
    vloss = -log_prob
    vloss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], NLL: {vloss.item():.4f}')

    # losses.append(vloss)
    
    # if len(losses) > csc_epochs:
    #     if abs(losses[-(csc_epochs + 1)] - vloss) < eps:
    #         print(f"Training has stabilized at epoch {epoch + 1}.")
    #         break
    
    wandb.log({'epoch': epoch, 'loss': vloss.item()})

log_prob = model.log_p(torch.ones(clscnt).unsqueeze(0))
prob = torch.exp(log_prob)
print(f'Approx WMC: {prob}')

wandb.finish()
