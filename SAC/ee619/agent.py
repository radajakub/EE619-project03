from __future__ import annotations
from typing import Dict, Tuple
from os.path import abspath, dirname, realpath, join
from copy import deepcopy

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=256, nonlinearity: str='relu', max_action: float=1.0) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc_layer = nn.Linear(hidden_dim, action_dim)
        self.log_sig_layer = nn.Linear(hidden_dim, action_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.loc_layer.weight)
        torch.nn.init.xavier_uniform_(self.log_sig_layer.weight)

        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif nonlinearity == 'relu':
            self.activation = nn.ReLU()
        else:
            raise TypeError("The nonlinearity was specified wrongly - choose 'relu' or 'tanh'!")

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # backpropagate through the net
        val = self.activation(self.fc1(state))
        val = self.activation(self.fc2(val))

        mu = self.loc_layer(val)
        log_sig = self.log_sig_layer(val)
        log_sig = torch.clamp(log_sig, min=-20, max=2)
        sig = torch.exp(log_sig)

        return mu, sig

    def act_deterministic(self, state: torch.tensor) -> torch.tensor:
        mu, _ = self(to_tensor(state).unsqueeze(0))
        return self.squash_action(mu).squeeze(0)

    def act(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        mu, sig = self(state)

        distribution = Normal(mu, sig)
        action = distribution.rsample()

        return self.squash(action, distribution)

    # squas action so it is in interval (-1, 1)
    def squash(self, action, distribution):
        # compute log probabilities (fixed by stable version from OpenAI)
        log_prob = distribution.log_prob(action).sum(axis=-1)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)

        squashed_scaled = self.squash_action(action)

        return squashed_scaled, log_prob

    def squash_action(self, action):
        return self.max_action * torch.tanh(action)

    def hard_update(self, other: GaussianPolicy) -> None:
        self.load_state_dict(deepcopy(other.state_dict()))

    def clone(self, trainable=False) -> GaussianPolicy:
        new_pi = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_dim, self.nonlinearity, self.max_action)
        new_pi.load_state_dict(deepcopy(self.state_dict))
        new_pi.train(trainable)
        return new_pi

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self, deterministic=True)-> None:
        self.policy = GaussianPolicy(24, 6, hidden_dim=256)
        self.path = join(ROOT, 'trained_model.pt')
        self.deterministic = deterministic

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        state = to_tensor(flatten_and_concat(time_step.observation)).unsqueeze(0)
        # consider only means of actions, don't explore during evaluation
        with torch.no_grad():
            if self.deterministic:
                action = self.policy.act_deterministic(state)
            else:
                action, _ = self.policy.act(state)
        return action.numpy()

    def load(self):
        """Loads network parameters if there are any."""
        self.policy.load_state_dict(torch.load(self.path))
