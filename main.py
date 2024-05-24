import torch
import torch.optim as optim
from parseCNF import readCNF
from model import IndependentModel, HMMModel
from answers import compute_WMC


file_str1 = "data/example.cnf"
file_str2 = "./benchmarks/altogether/bayes/50-10-1-q.cnf"

with open(file_str1) as f:
	cnf, weights = readCNF(f, mode="CAC")

clscnt = len(cnf)

torch.manual_seed(0)
# model = IndependentModel(dim=clscnt, cnf=cnf)
model = HMMModel(dim=clscnt, cnf=cnf)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# training parameters
num_epochs = 10000
eps = 1e-4
prev_loss = float('inf')
patience = 10
patience_counter = 0

# training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    log_prob = model(torch.from_numpy(weights))
    vloss = -log_prob.sum()
    vloss.backward()

    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], NLL: {vloss.item():.4f}')

    if abs(prev_loss - vloss) < eps:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Training converged at epoch {epoch + 1}')
            break
    else:
        patience_counter = 0

    prev_loss = vloss

log_prob = model.log_p(torch.ones(clscnt))
prob = torch.exp(log_prob)
print(f'Weighted model count: {prob}')
print(f'Exact WMC: {compute_WMC(weights, cnf)}')
