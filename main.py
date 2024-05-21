import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli

from parseCNF import readCNF, evalCNF

with open("./benchmarks/altogether/bayes/50-10-1-q.cnf") as f:
	cnf, weights = readCNF(f)

class GaussianModel(torch.nn.Module):
	def __init__(self, dim, cnf):
		super(GaussianModel, self).__init__()
		self.cnf = cnf
		self.mu = nn.Parameter(torch.randn(dim))
		self.log_std = nn.Parameter(torch.randn(dim))

	def forward(self, weight):
		dist_x = Bernoulli(weight)
		x = dist_x.sample()
		y = evalCNF(self.cnf, x)

		std = torch.exp(self.log_std)
		cov_mat = torch.diag(std ** 2)
		dist = MultivariateNormal(self.mu, cov_mat)
		log_prob = dist.log_prob(y)
		return log_prob
	

torch.manual_seed(0)
model = GaussianModel(dim=1, cnf=cnf)
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    log_prob = model(weights)
    vloss = -log_prob.sum()
    vloss.backward()

    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], NLL: {vloss.item():.4f}')

print(f'Estimated mu: {model.mu.data}')
print(f'Estimated log_std: {model.log_std.data}')
print(f'Estimated std: {torch.exp(model.log_std).data}')
