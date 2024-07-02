import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F

from abc import abstractmethod

class ApproxWMC(torch.nn.Module):
	def __init__(self, dim, device) -> None:
		self.dim = dim
		self.device = device
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
	def __init__(self, dim, device, num_states=3) -> None:
		super(HMM, self).__init__(dim, device)
		self.num_states = num_states
		
		self.transition_probs = nn.Parameter(torch.randn(num_states, num_states))
		self.emission_probs = nn.Parameter(torch.randn(num_states, 2))  # binary obs
		self.start_probs = nn.Parameter(torch.randn(num_states))
	
	def log_p(self, y):
		batch_size = y.shape[0]

		transition_probs = torch.log_softmax(self.transition_probs, dim=-1)
		emission_probs = torch.log_softmax(self.emission_probs, dim=-1)
		start_probs = torch.log_softmax(self.start_probs, dim=-1)

		log_alpha = torch.zeros(batch_size, self.dim, self.num_states, device=self.device)
		log_alpha[:, 0, :] = start_probs + emission_probs[:, y[:, 0].long()].T

		for t in range(1, self.dim):
			log_alpha[:, t, :] = torch.logsumexp(
				log_alpha[:, t - 1, :].unsqueeze(2) + transition_probs, dim=1
			) + emission_probs[:, y[:, t].long()].T

		log_prob = torch.logsumexp(log_alpha[:, self.dim - 1, :], dim=-1)
		return log_prob.mean()
