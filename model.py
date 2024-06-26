import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from abc import abstractmethod

class ApproxWMC(torch.nn.Module):
	def __init__(self, dim) -> None:
		self.dim = dim
		super(ApproxWMC, self).__init__()

	def forward(self, y):
		log_prob = self.log_p(y)
		return log_prob

	@abstractmethod
	def log_p(self, y):
		pass

class IndependentModel(ApproxWMC):
	def __init__(self, dim) -> None:
		super(IndependentModel, self).__init__(dim)
		self.logit_theta = nn.Parameter(torch.randn(self.dim))
	
	def log_p(self, y):
		dist = Bernoulli(torch.sigmoid(self.logit_theta))
		return dist.log_prob(y).sum(dim=-1).mean()


class HMM(ApproxWMC):
	def __init__(self, dim, num_states=3) -> None:
		super(HMM, self).__init__(dim)
		self.num_states = num_states
		
		self.transition_probs = nn.Parameter(torch.randn(num_states, num_states))
		self.emission_probs = nn.Parameter(torch.randn(num_states, 2))  # binary obs
		self.start_probs = nn.Parameter(torch.randn(num_states))
	
	def log_p(self, y):
		batch_size = y.shape[0]

		transition_probs = torch.log_softmax(self.transition_probs, dim=-1)
		emission_probs = torch.log_softmax(self.emission_probs, dim=-1)
		start_probs = torch.log_softmax(self.start_probs, dim=-1)

		log_alpha = torch.zeros(batch_size, self.dim, self.num_states)
		log_alpha[:, 0, :] = start_probs + emission_probs[:, y[:, 0].long()].T

		# for t in range(1, self.dim):
		# 	for j in range(self.num_states):
		# 		log_alpha[:, t, j] = torch.logsumexp(
		# 			log_alpha[:, t - 1, :] + transition_probs[:, j], dim=-1
		# 		) + emission_probs[j, y[:, t].long()]

		for t in range(1, self.dim):
			log_alpha[:, t, :] = torch.logsumexp(
				log_alpha[:, t - 1, :].unsqueeze(2) + transition_probs, dim=1
			) + emission_probs[:, y[:, t].long()].T

		log_prob = torch.logsumexp(log_alpha[:, self.dim - 1, :], dim=-1)
		return log_prob.mean()


# class RandomWalk(torch.nn.Module):
# 	def __init__(self, dim, cnf) -> None:
# 		super(RandomWalk, self).__init__()
# 		self.cnf = cnf
# 		self.dim = dim
# 		self.logit_theta0 = nn.Parameter(torch.randn(1))
# 		self.log_std = nn.Parameter(torch.randn(dim))

# 	def forward(self, ws):
# 		y = sample_y(ws, self.cnf)
# 		log_prob = self.log_p(y)
# 		return log_prob
	
# 	def log_p(self, y):
# 		log_prob = 0
# 		logit_theta = self.logit_theta0
# 		for d in range(self.dim):
# 			dist = Bernoulli(torch.sigmoid(logit_theta))
# 			log_prob += dist.log_prob(y[d])
# 			logit_theta = Normal(logit_theta, torch.exp(self.log_std[d])).sample()

# 		return log_prob
