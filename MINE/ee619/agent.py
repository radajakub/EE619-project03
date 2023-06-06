"""Agent for DMControl Walker-Run task."""
from __future__ import annotations
from typing import Dict, Tuple
from os.path import abspath, dirname, realpath
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
    def __init__(self) -> None:
        pass

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        step_type = time_step.step_type
        observation = flatten_and_concat(time_step.observation)
        reward = time_step.reward
        discount = time_step.discount
        return np.ones(6)

    def load(self):
        """Loads network parameters if there are any."""
        # Example:
        #   path = join(ROOT, 'model.pt')
        #   self.policy.load_state_dict(torch.load(path))

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super.__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.scale = nn.Parameter(torch.zeros(action_dim))
        torch.nn.init.constant_(self.scale, -0.5)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc = torch.tanh(self.fc1(input_))
        loc = torch.tanh(self.fc2(loc))
        loc = self.fc3(loc)
        scale = self.scale.exp().expand_as(loc)
        return loc, scale

    def act(self, state: np.ndarray) -> np.ndarray:
        loc, scale = self(to_tensor(state).unsqueeze(0))
        action = Independent(Normal(loc, scale), 1).sample().squeeze(0).numpy()
        return action
