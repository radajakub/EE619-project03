"""Agent for DMControl Walker-Run task."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
from os.path import abspath, dirname, realpath, join
from dm_env import TimeStep
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.
    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32)

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self, policy_type='layered') -> None:
        if policy_type == 'layered':
            self.policy = GaussianPolicyLayered(24, 6, hidden_dim=256)
        else:
            self.policy = GaussianPolicyParametrized(24, 6, hidden_dim=256)
        self.path = join(ROOT, 'trained_model.pt')

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        state = flatten_and_concat(time_step.observation)
        action = self.policy.act(state)
        return action

    def load(self):
        """Loads network parameters if there are any."""
        self.policy.load_state_dict(torch.load(self.path))

class GaussianPolicyLayered(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=64, nonlinearity: str='tanh') -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc_layer = nn.Linear(hidden_dim, action_dim)
        self.scale_layer = nn.Linear(hidden_dim, action_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.loc_layer.weight)

        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif nonlinearity == 'relu':
            self.activation = nn.ReLU()
        else:
            raise TypeError("The nonlinearity was specified wrongly - choose 'relu' or 'tanh'!")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.activation(self.fc1(state))
        val = self.activation(self.fc2(val))

        loc = self.loc_layer(val)
        scale = self.scale_layer(val).exp()
        scale = torch.clamp(scale, min=-20, max=2)
        scale = scale.exp()

        return loc, scale

    def act(self, state: np.ndarray) -> np.ndarray:
        loc, scale = self(to_tensor(state).unsqueeze(0))
        action = self.sample(loc, scale)
        return action.detach().numpy()

    # squas action so it is in interval (-1, 1)
    def squash(self, action, log_prob):
        regularizer = torch.sum(torch.log(1 - torch.tanh(action) ** 2 + 1e-6), dim=1)
        return torch.tanh(action), log_prob - regularizer


    def sample(self, locs, scales):
        action = Independent(Normal(locs, scales), 1).sample().squeeze(0)
        return action

    def rsample(self, locs, scales):
        action = Independent(Normal(locs, scales), 1).rsample().squeeze(0)
        return action

    def hard_update(self, other: GaussianPolicyLayered) -> None:
        self.load_state_dict(other.state_dict())

class GaussianPolicyParametrized(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=64, nonlinearity: str='tanh') -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc_layer = nn.Linear(hidden_dim, action_dim)
        # self.scale_layer = nn.Linear(hidden_dim, action_dim)
        self.scale = nn.Parameter(torch.zeros(action_dim))
        torch.nn.init.constant_(self.scale, -0.5)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.loc_layer.weight)

        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif nonlinearity == 'relu':
            self.activation = nn.ReLU()
        else:
            raise TypeError("The nonlinearity was specified wrongly - choose 'relu' or 'tanh'!")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.activation(self.fc1(state))
        val = self.activation(self.fc2(val))

        loc = self.loc_layer(val)
        scale = self.scale.exp().expand_as(loc)

        return loc, scale

    def act(self, state: np.ndarray) -> np.ndarray:
        loc, scale = self(to_tensor(state).unsqueeze(0))
        action = self.sample(loc, scale)
        return action.detach().numpy()

    # squas action so it is in interval (-1, 1)
    def squash(self, action, log_prob):
        regularizer = torch.sum(torch.log(1 - torch.tanh(action) ** 2 + 1e-6), dim=1)
        return torch.tanh(action), log_prob - regularizer


    def sample(self, locs, scales):
        action = Independent(Normal(locs, scales), 1).sample().squeeze(0)
        return action

    def rsample(self, locs, scales):
        action = Independent(Normal(locs, scales), 1).rsample().squeeze(0)
        return action

    def hard_update(self, other: GaussianPolicyParametrized) -> None:
        self.load_state_dict(other.state_dict())
