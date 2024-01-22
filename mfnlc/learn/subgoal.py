import torch
from torch import nn
import numpy as np

""" Actor """
class GaussianPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(GaussianPolicy, self).__init__()
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
		self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

		self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

	def set_training_mode(self, train):
		return

	def forward(self, state, goal):
		if type(state) == dict:
			goal = state["desired_goal"]
			state = state["observation"]
			_, _, mean = self.sample(state, goal)
			return mean
		x = self.fc(torch.cat([state, goal], -1))
		mean = self.mean_linear(x)
		log_std = self.logstd_linear(x)
		std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
		normal = torch.distributions.Normal(mean, std)
		return normal

	def sample(self, state, goal):
		normal = self.forward(state, goal)
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)
		mean = torch.tanh(normal.mean)
		return action, log_prob, mean

""" Critic """
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(Critic, self).__init__()
		fc = [nn.Linear(2*state_dim + action_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, action, goal):
		x = torch.cat([state, action, goal], -1)
		return self.fc(x)
	
class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], n_Q=2):
		super(EnsembleCritic, self).__init__()
		ensemble_Q = [Critic(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def set_training_mode(self, train):
		return

	def forward(self, state, action, goal):
		Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q

""" High-level policy """
class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, goal_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()	
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], goal_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], goal_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):	
		h = self.fc( torch.cat([state, goal], -1) )	
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.laplace.Laplace(mean, scale)
		return distribution