import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from cirkit.symbolic.circuit import Circuit
from cirkit.pipeline import PipelineContext
from cirkit_factories import categorical_layer_factory, hadamard_layer_factory, dense_layer_factory

from abc import ABC, abstractmethod

class ApproxWMC(torch.nn.Module, ABC):
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
	def __init__(self, dim, device) -> None:
		super(IndependentModel, self).__init__(dim, device)
		self.logit_theta = nn.Parameter(torch.randn(self.dim))
	
	def log_p(self, y):
		dist = Bernoulli(torch.sigmoid(self.logit_theta))
		return dist.log_prob(y).sum(dim=-1)


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
		return log_prob


class inhHMM(ApproxWMC):
	def __init__(self, dim, device, num_states=3) -> None:
		super(inhHMM, self).__init__(dim, device)
		self.num_states = num_states
		
		self.transition_probs = nn.Parameter(torch.randn(self.dim - 1, num_states, num_states))
		self.emission_probs = nn.Parameter(torch.randn(self.dim, num_states, 2))  # binary obs
		self.start_probs = nn.Parameter(torch.randn(num_states))
	
	def log_p(self, y):
		batch_size = y.shape[0]

		transition_probs = torch.log_softmax(self.transition_probs, dim=-1)
		emission_probs = torch.log_softmax(self.emission_probs, dim=-1)
		start_probs = torch.log_softmax(self.start_probs, dim=-1)

		log_alpha = torch.zeros(batch_size, self.dim, self.num_states, device=self.device)
		log_alpha[:, 0, :] = start_probs + emission_probs[0, :, y[:, 0].long()].T

		for t in range(1, self.dim):
			log_alpha[:, t, :] = torch.logsumexp(
				log_alpha[:, t - 1, :].unsqueeze(2) + transition_probs[t - 1, :, :], dim=1
			) + emission_probs[t, :, y[:, t].long()].T

		log_prob = torch.logsumexp(log_alpha[:, self.dim - 1, :], dim=-1)
		return log_prob

class HMMPC(ApproxWMC):
    def __init__(self, dim, device, num_states=50, order=None) -> None:
        super(HMMPC, self).__init__(dim, device)
        self.num_state = num_states
        if order is None:
            order = list(range(self.dim))
        order.reverse()

        symbolic_circuit = Circuit.from_hmm(
            order=order,
            num_units=num_states,
            input_factory=categorical_layer_factory,
            sum_factory=dense_layer_factory,
            prod_factory=hadamard_layer_factory
        )

        ctx = PipelineContext(
            backend='torch',
            fold=True,
            semiring='lse-sum'
        )
        self.model = ctx.compile(symbolic_circuit)
        self.pf_model = ctx.integrate(self.model)
        
    def log_p(self, y):
        y = y.unsqueeze(dim=1)
        return self.model(y) - self.pf_model()
