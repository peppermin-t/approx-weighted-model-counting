import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from parseCNF import evalCNF

def sample_y(ws, cnf):
	probs = F.normalize(ws, p=1, dim=1)
	dist_x = Bernoulli(probs[:, 0])
	x = dist_x.sample()
	return evalCNF(cnf, x)

class IndependentModel(torch.nn.Module):
	def __init__(self, dim, cnf) -> None:
		self.cnf = cnf
		super(IndependentModel, self).__init__()
		self.logit_theta = nn.Parameter(torch.randn(dim))

	def forward(self, ws):
		y = sample_y(ws, self.cnf)
		log_prob = self.log_p(y)
		return log_prob
	
	def log_p(self, y):
		dist = Bernoulli(torch.sigmoid(self.logit_theta))
		return dist.log_prob(y).sum()

	
class HMMModel(torch.nn.Module):
	def __init__(self, dim, cnf) -> None:
		super(HMMModel, self).__init__()
		self.cnf = cnf
		self.dim = dim
		self.logit_theta0 = nn.Parameter(torch.randn(1))
		self.log_std = nn.Parameter(torch.randn(dim))

	def forward(self, ws):
		y = sample_y(ws, self.cnf)
		log_prob = self.log_p(y)
		return log_prob
	
	def log_p(self, y):
		log_prob = 0
		logit_theta = self.logit_theta0
		for d in range(self.dim):
			dist = Bernoulli(torch.sigmoid(logit_theta))
			log_prob += dist.log_prob(y[d])
			logit_theta = Normal(logit_theta, torch.exp(self.log_std[d])).sample()

		return log_prob